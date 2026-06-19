import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats


# ============================================================================
# GRID  — aligned to Baseline and V1 configs
#
#  Symmetric: EIT = EMG = CAP = hidden  (no asymmetric ratio)
#  shared_size and decoder_hidden scale with hidden (same as Baseline/V1)
#
#  hidden=64,  layers=2  →  shared=128, dec=64   matches Baseline structure
#  hidden=64,  layers=3  →  shared=128, dec=64   Baseline hidden + V1 depth
#  hidden=128, layers=2  →  shared=256, dec=128  V1 hidden + Baseline depth
#  hidden=128, layers=3  →  shared=256, dec=128  matches V1 structure
#
#  Reference (asymmetric):
#    V2 Asym  EIT:128 EMG:64  CAP:32   L3  shared=256  dec=128  MAE=0.0691
#    Baseline EIT:64  EMG:64  CAP:64   L2  shared=128  dec=64   MAE=0.0800
#    V1       EIT:128 EMG:128 CAP:128  L3  shared=256  dec=128  MAE=0.0696
# ============================================================================

GRID = [
    {
        "label":          "sym64_L2",
        "hidden":         64,
        "layers":         2,
        "shared_size":    128,
        "decoder_hidden": 64,
        "note":           "= Baseline (symmetric version)",
    },
    {
        "label":          "sym64_L3",
        "hidden":         64,
        "layers":         3,
        "shared_size":    128,
        "decoder_hidden": 64,
        "note":           "Baseline hidden + V1 depth",
    },
    {
        "label":          "sym128_L2",
        "hidden":         128,
        "layers":         2,
        "shared_size":    256,
        "decoder_hidden": 128,
        "note":           "V1 hidden + Baseline depth",
    },
    {
        "label":          "sym128_L3",
        "hidden":         128,
        "layers":         3,
        "shared_size":    256,
        "decoder_hidden": 128,
        "note":           "= V1 ScaleUp (symmetric version)",
    },
]

REFERENCE = {
    "Baseline": {"hidden": "64(asym)",  "layers": 2, "shared": 128, "dec": 64,  "params": 253_697,   "mae": 0.0800},
    "V1_ScaleUp": {"hidden": "128(asym)", "layers": 3, "shared": 256, "dec": 128, "params": 1_395_201, "mae": 0.0696},
    "V2_Asym":  {"hidden": "128/64/32", "layers": 3, "shared": 256, "dec": 128, "params": 737_665,   "mae": 0.0691},
}

print("SYMMETRIC GRID  (all modalities same hidden)")
print(f"\n  {'Label':<14} {'Hidden':>8} {'Layers':>7} {'Shared':>8} {'Dec':>6}  Note")
print("  " + "-" * 72)
for g in GRID:
    print(f"  {g['label']:<14} {g['hidden']:>8} {g['layers']:>7} "
          f"{g['shared_size']:>8} {g['decoder_hidden']:>6}  {g['note']}")

print(f"\n  Reference:")
for name, r in REFERENCE.items():
    print(f"  {name:<14} {str(r['hidden']):>8} {r['layers']:>7} "
          f"{r['shared']:>8} {r['dec']:>6}  MAE={r['mae']}")


# ============================================================================
# DATASET
# ============================================================================

class CrossModalForceDatasetNoSeg(Dataset):
    def __init__(self, csv_path, seq_length=50, stride=10, drop_nan_target=True):
        self.seq_length = int(seq_length)
        self.stride     = int(stride)

        df = pd.read_csv(csv_path)
        force_col = next(
            (c for c in ["adc_avg", "force", "force_n", "adc_mean"] if c in df.columns), None
        )
        if force_col is None:
            raise ValueError(f"No force column. Available: {list(df.columns)}")

        eit_cols = [f"eit_ch{i}" for i in range(1, 9)]
        emg_cols = [f"emg{i}" for i in range(4)]
        cap_cols = ["cap0_ma"]
        all_cols = eit_cols + emg_cols + cap_cols + [force_col]

        for c in all_cols:
            if c not in df.columns:
                raise ValueError(f"Missing column: {c}")

        if "time_sec" in df.columns:
            df = df.sort_values("time_sec").reset_index(drop=True)
        for c in all_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if drop_nan_target:
            df = df[df[force_col].notna()].copy()

        eit   = df[eit_cols].to_numpy(dtype=np.float32)
        emg   = df[emg_cols].to_numpy(dtype=np.float32)
        cap   = df[cap_cols].to_numpy(dtype=np.float32)
        force = df[[force_col]].to_numpy(dtype=np.float32)

        valid = (
            np.isfinite(eit).any(axis=1) |
            np.isfinite(emg).any(axis=1) |
            np.isfinite(cap).any(axis=1)
        )
        eit, emg, cap, force = eit[valid], emg[valid], cap[valid], force[valid]

        self.sequences = []
        n = len(force)
        if n < self.seq_length:
            return
        for i in range((n - self.seq_length) // self.stride + 1):
            s, e = i * self.stride, i * self.stride + self.seq_length
            if not np.isfinite(force[s:e]).any():
                continue
            self.sequences.append({
                "eit": eit[s:e], "emg": emg[s:e],
                "cap": cap[s:e], "force": force[s:e]
            })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        s = self.sequences[idx]
        return {k: torch.tensor(v, dtype=torch.float32) for k, v in s.items()}


# ============================================================================
# MODEL
# ============================================================================

class ModalityEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]


class FusionLayer(nn.Module):
    def __init__(self, fusion_input_size, shared_size, dropout=0.3):
        super().__init__()
        self.fc      = nn.Linear(fusion_input_size, shared_size)
        self.bn      = nn.BatchNorm1d(shared_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, feats):
        x = torch.cat(feats, dim=-1)
        return self.dropout(torch.relu(self.bn(self.fc(x))))


class ForceDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, seq_length, num_layers=2, dropout=0.2):
        super().__init__()
        self.seq_length = seq_length
        self.fc_in  = nn.Linear(input_size, hidden_size)
        self.lstm   = nn.LSTM(hidden_size, hidden_size, num_layers,
                              batch_first=True,
                              dropout=dropout if num_layers > 1 else 0.0)
        self.fc_out = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = torch.relu(self.fc_in(x))
        x = x.unsqueeze(1).repeat(1, self.seq_length, 1)
        out, _ = self.lstm(x)
        return self.fc_out(out)


class SymmetricForceModel(nn.Module):
    def __init__(self, input_modalities, seq_length,
                 hidden, layers, shared_size, decoder_hidden,
                 decoder_layers=2, dropout=0.3):
        super().__init__()
        self.input_modalities = list(input_modalities)
        self.eit_encoder = ModalityEncoder(8, hidden, layers, dropout)
        self.emg_encoder = ModalityEncoder(4, hidden, layers, dropout)
        self.cap_encoder = ModalityEncoder(1, hidden, layers, dropout)

        fusion_in = hidden * len(self.input_modalities)
        self.fusion  = FusionLayer(fusion_in, shared_size, dropout)
        self.decoder = ForceDecoder(shared_size, decoder_hidden,
                                    seq_length, decoder_layers, dropout)

    def forward(self, eit, emg, cap):
        feats = []
        if 'eit' in self.input_modalities: feats.append(self.eit_encoder(eit))
        if 'emg' in self.input_modalities: feats.append(self.emg_encoder(emg))
        if 'cap' in self.input_modalities: feats.append(self.cap_encoder(cap))
        return self.decoder(self.fusion(feats))


# ============================================================================
# TRAIN / VAL / EVAL
# ============================================================================

def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()
    criterion = nn.MSELoss()
    losses = []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            eit   = batch["eit"].to(device)
            emg   = batch["emg"].to(device)
            cap   = batch["cap"].to(device)
            force = batch["force"].to(device)
            pred  = model(eit, emg, cap)
            loss  = criterion(pred, force)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


def evaluate(model, loader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            eit   = batch["eit"].to(device)
            emg   = batch["emg"].to(device)
            cap   = batch["cap"].to(device)
            force = batch["force"].to(device)
            preds.append(model(eit, emg, cap).cpu().numpy())
            gts.append(force.cpu().numpy())

    p = np.concatenate(preds).reshape(-1)
    g = np.concatenate(gts).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(g, p)))
    mae  = float(mean_absolute_error(g, p))
    corr = float(stats.pearsonr(p, g)[0]) if (
        np.std(p) > 1e-12 and np.std(g) > 1e-12) else 0.0
    ss_res = np.sum((g - p) ** 2)
    ss_tot = np.sum((g - g.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2, "corr": corr}


# ============================================================================
# RESULTS TABLE
# ============================================================================

def print_results(sym_results):
    print("\n" + "=" * 78)
    print("FINAL COMPARISON  —  Symmetric vs Reference")
    print("=" * 78)
    print(f"\n  {'Model':<16} {'Type':>8} {'Hidden':>8} {'L':>3} "
          f"{'Shared':>8} {'Params':>10}  {'MAE':>8}  Note")
    print("  " + "-" * 75)

    # reference rows
    for name, r in REFERENCE.items():
        print(f"  {name:<16} {'asym':>8} {str(r['hidden']):>8} {r['layers']:>3} "
              f"{r['shared']:>8} {r['params']:>10,}  {r['mae']:>8.4f}")

    print("  " + "-" * 75)

    # sym results
    for cfg in GRID:
        label = cfg["label"]
        if label not in sym_results:
            continue
        res = sym_results[label]
        m   = res["metrics"]

        # find best reference MAE to compare against
        ref_mae = REFERENCE["Baseline"]["mae"] if cfg["hidden"] == 64 \
                  else REFERENCE["V1_ScaleUp"]["mae"]
        delta   = m["mae"] - ref_mae
        verdict = f"Δ{delta:+.4f} vs ref"

        print(f"  {label:<16} {'sym':>8} {cfg['hidden']:>8} {cfg['layers']:>3} "
              f"{cfg['shared_size']:>8} {res['params']:>10,}  {m['mae']:>8.4f}  {verdict}")

    print("\n" + "=" * 78)
    print("KEY QUESTION: sym128_L3 vs V1_ScaleUp")
    print("  Same hidden/layers/shared/decoder — only difference is")
    print("  V1 uses uniform hidden=128 for all modalities (= sym128_L3)")
    print("  V1 was originally built with uniform hidden — this confirms it.")
    print("=" * 78)


# ============================================================================
# RUN
# ============================================================================

INPUT_MODALITIES = ['eit', 'emg']

SEQ_LENGTH     = 30
STRIDE         = 2
BATCH_SIZE     = 32
MAX_EPOCHS     = 200
PATIENCE       = 15
LR             = 1e-3
DROPOUT        = 0.3

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/content/drive/MyDrive/Multi_Modal_Rehab/synced/group_1_processed.csv"

print(f"\nDevice : {device}")
print(f"Grid   : {len(GRID)} runs  (early stopping patience={PATIENCE})\n")

dataset = CrossModalForceDatasetNoSeg(DATA_PATH, seq_length=SEQ_LENGTH, stride=STRIDE)

n       = len(dataset)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)
n_test  = n - n_train - n_val

train_ds, val_ds, test_ds = random_split(
    dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False)

print(f"Windows — train:{len(train_ds)}  val:{len(val_ds)}  test:{len(test_ds)}\n")

sym_results = {}

for cfg in GRID:
    label = cfg["label"]
    print(f"\n{'='*55}")
    print(f"RUN: {label}  [{cfg['note']}]")
    print(f"  EIT=EMG=CAP={cfg['hidden']}  L{cfg['layers']}  "
          f"shared={cfg['shared_size']}  dec={cfg['decoder_hidden']}")
    print(f"{'='*55}")

    model = SymmetricForceModel(
        input_modalities = INPUT_MODALITIES,
        seq_length       = SEQ_LENGTH,
        hidden           = cfg["hidden"],
        layers           = cfg["layers"],
        shared_size      = cfg["shared_size"],
        decoder_hidden   = cfg["decoder_hidden"],
        dropout          = DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params : {n_params:,}")

    optimizer  = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    save_path  = f"best_{label}.pth"
    best_val   = float("inf")
    no_improve = 0

    for epoch in range(MAX_EPOCHS):
        tr  = run_epoch(model, train_loader, optimizer, device, train=True)
        val = run_epoch(model, val_loader,   optimizer, device, train=False)

        if val < best_val:
            best_val, no_improve = val, 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1

        if (epoch + 1) % 20 == 0 or no_improve == PATIENCE:
            print(f"  Ep {epoch+1:>3}  train={tr:.5f}  val={val:.5f}"
                  + ("  ← best" if no_improve == 0 else
                     f"  (no improve {no_improve}/{PATIENCE})"))

        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break

    print(f"  Best val : {best_val:.6f}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    metrics = evaluate(model, test_loader, device)
    print(f"  Test     : MAE={metrics['mae']:.4f}  R2={metrics['r2']:.4f}")

    sym_results[label] = {"params": n_params, "metrics": metrics}

print_results(sym_results)
