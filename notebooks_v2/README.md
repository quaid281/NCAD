# Mamba Anomaly Classifier (MAC)

## Research Premise: "Context is King"

A novel approach to time series anomaly detection using **Selective State Space Models (Mamba)** that learns contextual patterns directly from raw signals without feature engineering or memory bank clustering.

## Key Innovations

1. **Mamba Architecture**: Leverages selective state space models for efficient long-range context modeling with linear computational complexity
2. **Raw Signal Input**: No hand-crafted features - the model learns temporal patterns end-to-end
3. **Self-Supervised Training**: Uses synthetic anomaly injection for training without labeled data
4. **Single-Pass Inference**: Streaming-capable detection with constant memory footprint

## Architecture Overview

```
Raw Signal → Mamba Encoder → Classification Head → Anomaly Probability
   x_t           h_t              MLP                  p(anomaly)
```

## Directory Structure

```
notebooks_v2/
├── models/
│   ├── __init__.py
│   ├── mamba_encoder.py      # Selective State Space Model
│   ├── classifier.py         # Classification head
│   └── anomaly_injector.py   # Synthetic anomaly generation
├── utils/
│   ├── __init__.py
│   ├── data_loader.py        # Raw signal loading
│   └── metrics.py            # Evaluation metrics
├── results/                  # Experiment outputs
├── train.py                  # Training loop
├── evaluate.py               # Evaluation script
└── README.md
```

## Training Objective

Self-supervised binary classification:
- **Input**: Windows of raw telemetry signal
- **Labels**: 0 (clean) or 1 (synthetically corrupted)
- **Loss**: Binary Cross Entropy

## Anomaly Types (Synthetic Injection)

1. **Spike**: Single-point extreme deviation
2. **Level Shift**: Sustained offset from baseline
3. **Variance Change**: Altered signal volatility
4. **Stuck Value**: Constant value (sensor freeze)

## Requirements

- Python 3.10+
- PyTorch 2.0+
- mamba-ssm (or custom implementation)
- numpy, pandas, matplotlib

## Usage

```bash
# Training
python train.py --channel A-1 --epochs 100

# Evaluation
python evaluate.py --channel A-1 --checkpoint best_model.pt
```

## References

- Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
- SMAP/MSL Anomaly Detection Dataset (NASA)
