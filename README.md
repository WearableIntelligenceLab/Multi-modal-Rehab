# Multi-modal Rehabilitation Sensing and Modeling

This repository contains the data-processing, visualization, and machine-learning code for a multimodal wearable rehabilitation system.

The system combines multiple sensing modalities, including:

- Electrical impedance sensing (EIS)
- Electromyography (EMG)
- Capacitive sensing (CAP)

These signals are used to estimate rehabilitation-related outcomes such as grip force, joint angles, and arm-reach height.

## Repository Structure

```text
Multi-modal-Rehab/
├── Data Processing/
│   ├── analysis_cap.ipynb
│   ├── analysis_emg.ipynb
│   ├── data_to_video.ipynb
│   ├── sync.ipynb
│   └── visualization.ipynb
│
├── Model/
│   ├── Architecture Comparason.py
│   ├── Arm_reach.ipynb
│   ├── Channel Ablation.py
│   ├── Cross_Model_Prediction_with_SHAP.py
│   ├── Grasp and bicep curl.ipynb
│   ├── Power_grip.ipynb
│   ├── Shared_Embedding_Model_with_SHAP.py
│   └── Supination.ipynb
│
└── README.md
