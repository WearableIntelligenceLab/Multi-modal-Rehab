import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats


# ============================================================================
# CHANNEL GROUPING
# ============================================================================

def group_eit(arr):
    """8ch → 2ch: ring1=mean(ch1-4), ring2=mean(ch5-8)"""
    ring1 = arr[:, 0:4].mean(axis=1, keepdims=True)
    ring2 = arr[:, 4:8].mean(axis=1, keepdims=True)
    return np.concatenate([ring1, ring2], axis=1)

def group_emg(arr):
    """4ch → 2ch: siteA=mean(emg0-1), siteB=mean(emg2-3)"""
    siteA = arr[:, 0:2].mean(axis=1, keepdims=True)
    siteB = arr[:, 2:4].mean(axis=1, keepdims=True)
    return np.concatenate([siteA, siteB], axis=1)


# ============================================================================
# EXPERIMENT GRID
#
#   modality combos (7):
#     single : [eit], [emg], [cap]
#     double : [eit,emg], [eit,cap], [emg,cap]
#     triple : [eit,emg,cap]
#
#   channel modes (2): full | grouped
#
#   total: 14 runs
# ============================================================================

MODALITY_COMBOS = []
for r in [1, 2, 3]:
    for combo in combinations(['eit', 'emg', 'cap'], r):
        MODALITY_COMBOS.append(list(combo))

CHANNEL_MODES = ['full', 'grouped']


# ============================================================================
# DATASET
# ============================================================================

class ForceDataset(Dataset):
    def __init__(self, csv_path,
                 seq_length=50, stride=10,
                 active_modalities=('eit', 'emg', 'cap'),
                 channel_mode='full',
                 drop_nan_target=True):

        self.seq_length  = int(seq_length)
        self.stride      = int(stride)
        self.active_mods = list(active_modalities)
        self.ch_mode     = channel_mode

        df = pd.read_csv(csv_path)

        # force column
        force_col = next(
            (c for c in ["adc_avg", "force", "force_n", "adc_mean"] if c in df.columns),
            None
        )
        if force_col is None:
            raise ValueError(f"No force column found. Available: {list(df.columns)}")

        raw_eit_cols = [f"eit_ch{i}" for i in range(1, 9)]
        raw_emg_cols = [f"emg{i}" for i in range(4)]
        raw_cap_cols = ["cap0_ma"]
        all_cols     = raw_eit_cols + raw_emg_cols + raw_cap_cols + [force_col]

        for c in all_cols:
            if c not in df.columns:
                raise ValueError(f"Missing column: {c}")

        if "time_sec" in df.columns:
            df = df.sort_values("time_sec").reset_index(drop=True)

        for c in all_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        if drop_nan_target:
            df = df[df[force_col].notna()].copy()

        # load raw arrays
        eit_raw = df[raw_eit_cols].to_numpy(dtype=np.float32)  # [N,8]
        emg_raw = df[raw_emg_cols].to_numpy(dtype=np.float32)  # [N,4]
        cap_raw = df[raw_cap_cols].to_numpy(dtype=np.float32)  # [N,1]
        force   = df[[force_col]].to_numpy(dtype=np.float32)   # [N,1]

        valid = (
            np.isfinite(eit_raw).any(axis=1) |
            np.isfinite(emg_raw).any(axis=1) |
            np.isfinite(cap_raw).any(axis=1)
        )
        eit_raw, emg_raw, cap_raw, force = (
            eit_raw[valid], emg_raw[valid], cap_raw[valid], force[valid]
        )

        # apply channel grouping
        eit = group_eit(eit_raw) if channel_mode == 'grouped' else eit_raw  # [N, 2or8]
        emg = group_emg(emg_raw) if channel_mode == 'grouped' else emg_raw  # [N, 2or4]
        cap = cap_raw                                                         # [N, 1]

        # store channel counts for model building
        self.ch = {'eit': eit.shape[1], 'emg': emg.shape[1], 'cap': cap.shape[1]}

        # build windows
        self.data   = {'eit': eit, 'emg': emg, 'cap': cap, 'force': force}
        self.sequences = []
        n = len(force)
        if n < self.seq_length:
            return

        for i in range((n - self.seq_length) // self.stride + 1):
            s, e = i * self.stride, i * self.stride + self.seq_length
            if not np.isfinite(force[s:e]).any():
                continue
            self.sequences.append(s)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        s, e = self.sequences[idx], self.sequences[idx] + self.seq_length
        out = {m: torch.tensor(self.data[m][s:e], dtype=torch.float32)
               for m in self.active_mods}
        out['force'] = torch.tensor(self.data['force'][s:e], dtype=torch.float32)
        return out


# ============================================================================
# MODEL COMPONENTS
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
        return h[-1]   # [B, hidden_size]


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
        self.fc_in      = nn.Linear(input_size, hidden_size)
        self.lstm       = nn.LSTM(hidden_size, hidden_size, num_layers,
                                  batch_first=True,
                                  dropout=dropout if num_layers > 1 else 0.0)
        self.fc_out     = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc_in(x))
        x = x.unsqueeze(1).repeat(1, self.seq_length, 1)
        out, _ = self.lstm(x)
        return self.fc_out(out)


class MultiModalForceModel(nn.Module):
    """
    Builds encoders only for active modalities.
    Asymmetric hidden sizes: EIT=128/L3, EMG=64/L2, CAP=32/L1
    """
    HIDDEN = {'eit': 128, 'emg': 64, 'cap': 32}
    LAYERS = {'eit': 3,   'emg': 2,  'cap': 1}

    def __init__(self, active_modalities, ch_dict,
                 seq_length=50, shared_size=256,
                 decoder_hidden=128, decoder_layers=2, dropout=0.3):
        super().__init__()
        self.active_mods = active_modalities

        self.encoders = nn.ModuleDict({
            mod: ModalityEncoder(
                input_size  = ch_dict[mod],
                hidden_size = self.HIDDEN[mod],
                num_layers  = self.LAYERS[mod],
                dropout     = dropout
            )
            for mod in active_modalities
        })

        fusion_in = sum(self.HIDDEN[m] for m in active_modalities)
        self.fusion  = FusionLayer(fusion_in, shared_size, dropout)
        self.decoder = ForceDecoder(shared_size, decoder_hidden,
                                    seq_length, decoder_layers, dropout)

    def forward(self, batch):
        feats  = [self.encoders[m](batch[m]) for m in self.active_mods]
        shared = self.fusion(feats)
        return self.decoder(shared)


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
            batch  = {k: v.to(device) for k, v in batch.items()}
            pred   = model(batch)
            loss   = criterion(pred, batch['force'])
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
            batch = {k: v.to(device) for k, v in batch.items()}
            preds.append(model(batch).cpu().numpy())
            gts.append(batch['force'].cpu().numpy())

    p = np.concatenate(preds).reshape(-1)
    g = np.concatenate(gts).reshape(-1)

    rmse = float(np.sqrt(mean_squared_error(g, p)))
    mae  = float(mean_absolute_error(g, p))
    corr = float(stats.pearsonr(p, g)[0]) if (
        np.std(p) > 1e-12 and np.std(g) > 1e-12) else 0.0
    ss_res = np.sum((g - p) ** 2)
    ss_tot = np.sum((g - g.mean()) ** 2)
    r2   = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'corr': corr}


# ============================================================================
# RESULTS TABLE
# ============================================================================

def print_results_table(results):
    """
    results: list of dicts with keys:
      modalities, ch_mode, ch_eit, ch_emg, ch_cap, n_params, metrics
    """
    print("\n" + "=" * 95)
    print("FULL ABLATION RESULTS")
    print("=" * 95)
    print(f"  Channel modes:  full    = EIT(8ch)  EMG(4ch)  CAP(1ch)")
    print(f"                  grouped = EIT(2ch)  EMG(2ch)  CAP(1ch)  [anatomical avg]")
    print()

    # group by modality combo
    seen_combos = []
    for r in results:
        key = tuple(r['modalities'])
        if key not in seen_combos:
            seen_combos.append(key)

    header = (f"  {'Modalities':<20} {'CH_mode':<10} "
              f"{'EIT':>5} {'EMG':>5} {'CAP':>5} "
              f"{'Params':>9} "
              f"{'RMSE':>8} {'MAE':>8} {'R2':>7} {'Corr':>7}")
    print(header)
    print("  " + "-" * 91)

    for combo in seen_combos:
        combo_rows = [r for r in results if tuple(r['modalities']) == combo]
        mod_label  = '+'.join(combo).upper()
        for i, row in enumerate(combo_rows):
            label = mod_label if i == 0 else ''
            m = row['metrics']
            eit_str = f"{row['ch_eit']}ch" if 'eit' in row['modalities'] else '  -'
            emg_str = f"{row['ch_emg']}ch" if 'emg' in row['modalities'] else '  -'
            cap_str = f"{row['ch_cap']}ch" if 'cap' in row['modalities'] else '  -'
            print(
                f"  {label:<20} {row['ch_mode']:<10} "
                f"{eit_str:>5} {emg_str:>5} {cap_str:>5} "
                f"{row['n_params']:>9,} "
                f"{m['rmse']:>8.4f} {m['mae']:>8.4f} "
                f"{m['r2']:>7.4f} {m['corr']:>7.4f}"
            )
        print("  " + "-" * 91)

    print("=" * 95)
    print("\nINTERPRETATION:")
    print("  full ≈ grouped within same modality combo  →  channel count is not the driver")
    print("  full >> grouped                             →  fine-grained channels matter")
    print("  single modality close to triple             →  other modalities add little")


# ============================================================================
# MAIN
# ============================================================================

DATA_PATH     = "/content/drive/MyDrive/Multi_Modal_Rehab/synced/group_1_processed.csv"
SEQ_LENGTH    = 30
STRIDE        = 2
BATCH_SIZE    = 32
NUM_EPOCHS    = 200
LR            = 1e-3
SHARED_SIZE   = 256
DEC_HIDDEN    = 128
DEC_LAYERS    = 2
DROPOUT       = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Experiments: {len(MODALITY_COMBOS)} modality combos × {len(CHANNEL_MODES)} channel modes = "
      f"{len(MODALITY_COMBOS) * len(CHANNEL_MODES)} runs\n")

all_results = []

for modalities in MODALITY_COMBOS:
    for ch_mode in CHANNEL_MODES:

        run_label = f"[{'+'.join(modalities).upper()}] {ch_mode}"
        print(f"\n{'='*60}")
        print(f"RUN: {run_label}")
        print(f"{'='*60}")

        # ---- dataset ----
        dataset = ForceDataset(
            DATA_PATH,
            seq_length=SEQ_LENGTH,
            stride=STRIDE,
            active_modalities=modalities,
            channel_mode=ch_mode,
        )

        ch_info = {
            'eit': dataset.ch.get('eit', 0),
            'emg': dataset.ch.get('emg', 0),
            'cap': dataset.ch.get('cap', 0),
        }

        print(f"  Windows : {len(dataset)}")
        for m in modalities:
            mode_str = ''
            if m == 'eit' and ch_mode == 'grouped':
                mode_str = ' (ring1_avg, ring2_avg)'
            elif m == 'emg' and ch_mode == 'grouped':
                mode_str = ' (siteA_avg, siteB_avg)'
            print(f"  {m.upper():>4} : {ch_info[m]}ch{mode_str}")

        if len(dataset) == 0:
            print("  SKIP: 0 windows")
            continue

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

        # ---- model ----
        model = MultiModalForceModel(
            active_modalities = modalities,
            ch_dict           = ch_info,
            seq_length        = SEQ_LENGTH,
            shared_size       = SHARED_SIZE,
            decoder_hidden    = DEC_HIDDEN,
            decoder_layers    = DEC_LAYERS,
            dropout           = DROPOUT,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params  : {n_params:,}")

        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

        # ---- train ----
        best_val  = float('inf')
        save_path = f"best_{'_'.join(modalities)}_{ch_mode}.pth"

        for epoch in range(NUM_EPOCHS):
            tr_loss = run_epoch(model, train_loader, optimizer, device, train=True)
            val_loss = run_epoch(model, val_loader,  optimizer, device, train=False)

            if (epoch + 1) % 50 == 0:
                print(f"  Ep {epoch+1:>3}/{NUM_EPOCHS}  train={tr_loss:.5f}  val={val_loss:.5f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), save_path)

        print(f"  Best val: {best_val:.6f}")

        # ---- evaluate ----
        model.load_state_dict(torch.load(save_path, map_location=device))
        metrics = evaluate(model, test_loader, device)
        print(f"  Test    : rmse={metrics['rmse']:.4f}  mae={metrics['mae']:.4f}  "
              f"r2={metrics['r2']:.4f}  corr={metrics['corr']:.4f}")

        all_results.append({
            'modalities': modalities,
            'ch_mode':    ch_mode,
            'ch_eit':     ch_info['eit'],
            'ch_emg':     ch_info['emg'],
            'ch_cap':     ch_info['cap'],
            'n_params':   n_params,
            'metrics':    metrics,
        })

# ---- final table ----
print_results_table(all_results)
