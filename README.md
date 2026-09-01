# NCAD-CS: Neural Contextual Anomaly Detection with Counterfactual Successor Memory

[![Python](https://img.shields.io/badge/python-3.13%2B-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/architecture-TS--JEPA%20%2B%20VICReg-purple.svg)](src/models/jepa/ts_jepa.py)
[![Version](https://img.shields.io/badge/version-5.0.0-blue.svg)](src/__init__.py)

**NCAD-CS** is a non-contrastive self-supervised deep learning framework for unsupervised multivariate time-series anomaly detection. The core architecture is **TS-JEPA** (Time-Series Joint-Embedding Predictive Architecture), which predicts future system states directly in representation space rather than reconstructing raw noisy waveforms or relying on hand-crafted synthetic anomaly injections. This is extended with Flow Matching, Patch Tokenization, Multi-Scale prediction, Graph Attention, and Selective State-Space encoder variants.

---

## Key Innovations

1. **Latent World Model Dynamics (Zero Signal Reconstruction):**
   Predicts future latent states from context embeddings via a non-contrastive VICReg (Invariance, Variance hinge, Covariance decorrelation) objective.

2. **Conditional Flow Matching (OT-CFM) Variants:**
   `FlowTSJEPA` and `PatchFlowJEPA` replace deterministic latent regression with continuous Optimal Transport Flow Matching, learning a neural velocity field that transports Gaussian prior noise to target latents along straight ODE paths.

3. **Covariance-Whitened (Mahalanobis) Discrepancy Scoring:**
   Normalizes prediction residuals through the learned empirical precision tensor, capturing directional uncertainty across multi-sensor axes and eliminating threshold collapse.

4. **Extreme Value Theory (EVT / SPOT) Tail Calibration:**
   Fits asymptotic Generalized Pareto Distributions (GPD) on normal latent predictive residuals to establish principled, false-alarm-free anomaly thresholds under the Pickands-Balkema-de Haan theorem.

5. **Spatial-Temporal Relational Graph Attention:**
   Optional multi-head graph attention layers (`RelationalGAT_JEPAModel`) to capture dynamic inter-sensor topological dependencies in multi-sensor networks.

6. **Multi-Scale Hierarchical Prediction:**
   `MultiScaleTSJEPA` predicts target representations at multiple horizons simultaneously, capturing both short-term and long-term dynamical structure.

---

## Model Variants

| Model | Registry Name | Architecture | Encoder | Description |
| :--- | :--- | :--- | :--- | :--- |
| **TS-JEPA** | `ts_jepa` | JEPA + VICReg | TCN | Primary model: latent predictive coding with VICReg loss |
| **Patch-TS-JEPA** | `patch_ts_jepa` | JEPA + VICReg | Patch Transformer | Patch-tokenized Transformer encoder variant |
| **GAT-JEPA** | `gat_jepa` | JEPA + VICReg | TCN + GAT | Relational graph-attention for inter-sensor dependencies |
| **Flow-JEPA** | `flow_jepa` | OT-CFM + VICReg | TCN | Conditional Flow Matching with velocity field predictor |
| **Patch-Flow-JEPA** | `patch_flow_jepa` | OT-CFM + VICReg | Patch Transformer | Patch-tokenized Flow Matching with cross-attention predictor |
| **MultiScale-TS-JEPA** | `multiscale_ts_jepa` | JEPA + VICReg | Multi-Scale TCN | Multi-horizon hierarchical prediction heads |
| **NCAD** | `ncad` | Contrastive | TCN | Legacy NCAD contrastive encoder baseline |

### SOTA Baselines (for comparison)

| Baseline | Year | Architecture |
| :--- | :--- | :--- |
| **TranAD** | VLDB 2022 | Adversarial dual-decoder Transformer autoencoder |
| **TimesNet** | ICLR 2023 | 2D temporal variation decomposition with Inception blocks |
| **Anomaly Transformer** | ICLR 2022 | Association discrepancy with minimax attention |
| **DCdetector** | KDD 2023 | Dual-attention contrastive representation learning |

---

## Benchmark Datasets

The `mTSBench_data/` directory contains 19 benchmark datasets spanning diverse physical, cyber-physical, and distributed domains:

| Category | Datasets | Description |
| :--- | :--- | :--- |
| **IoT / Sensors** | CalIt2, cicids, Daphnet, GECCO, Genesis, GHL, metro, room-occupancy | Wearable sensors, building/traffic monitors, water quality, industrial pick-and-place, gas heating loop, metro train compressor, environmental |
| **Remote Sensing** | MSL, SMAP, swan | Mars rover telemetry, satellite telemetry, solar flare observatory |
| **IT / Server** | PSM, SMD, Exathlon | Pooled server metrics, server machine dataset, distributed cloud logs |
| **Financial** | creditcard | Credit card fraud detection |
| **Physiological** | MITDB, SVDB, OPPORTUNITY | ECG arrhythmia, supraventricular arrhythmia, wearable activity |
| **Synthetic** | GutenTAG | Controlled synthetic time series with injected anomalies |

---

## Repository Structure

```
NCAD_CS/
├── train.py                      # Root CLI entry point (delegates to src.cli)
├── mTSBench_data/                # 19 benchmark datasets
├── paper/                        # Manuscript LaTeX sources and figures
├── references/                   # Research literature and baseline docs
├── reports/                      # Benchmark CSVs and evaluation reports
├── scripts/                      # Benchmark runners and visualization
│   ├── run_modern_baselines.py       # SOTA benchmark (9 models, all datasets)
│   ├── run_benchmark.py              # Unified multi-dataset benchmark
│   ├── run_full_benchmark_suite.py   # Full cross-domain suite
│   ├── run_jepa_ablations.py         # JEPA architectural ablations
│   ├── visualize_attractor.py        # Phase-space attractor plotting
│   └── ...
├── src/
│   ├── cli.py                    # CLI argument parsing and entry point
│   ├── config.py                 # CSMConfig dataclass + validation
│   ├── data/                     # Data loaders and pipeline
│   ├── engine/                   # Training, evaluation, orchestration
│   │   ├── trainer.py                # Encoder/model builders + training
│   │   ├── evaluator.py              # Scoring + EVT calibration
│   │   └── orchestrator.py           # Full experiment pipeline
│   ├── features/                 # Feature extraction
│   ├── models/
│   │   ├── __init__.py               # Top-level model re-exports
│   │   ├── _jepa_utils.py            # JEPABase mixin + covariance fitting
│   │   ├── registry.py               # Central model type registry
│   │   ├── baselines/                # SOTA baselines (TranAD, TimesNet, AT, DCdetector)
│   │   ├── encoders/                 # Encoder backbones (TCN, MultiScale, GAT, SSM)
│   │   ├── jepa/                     # JEPA-family models
│   │   │   ├── ts_jepa.py                # Primary TS-JEPA + VICReg loss
│   │   │   ├── patch_ts_jepa.py          # Patch Transformer JEPA
│   │   │   ├── flow_ts_jepa.py           # Conditional Flow Matching JEPA
│   │   │   ├── patch_flow_jepa.py        # Patch Flow Matching JEPA
│   │   │   ├── multiscale_ts_jepa.py     # Multi-horizon hierarchical JEPA
│   │   │   ├── gat_jepa.py               # Graph Attention JEPA
│   │   │   └── ncad_jepa.py              # NCAD-JEPA variant
│   │   ├── losses/                   # Anomaly injection + FEI-SigReg loss
│   │   ├── memory/                   # Counterfactual successor memory + SINDy
│   │   └── legacy/                   # Backward-compat shims
│   ├── scoring/                  # EVT calibration + event-level fusion
│   │   ├── evt_calibrator.py         # GPD tail fitting (Grimshaw/numerical MLE)
│   │   └── event_fusion.py           # Window aggregation, metrics, filtering
│   ├── experimental/             # Research branches (Hopfield, causal, SSM)
│   └── utils/                    # Logging + lazy plotting
└── tests/                        # 125+ tests (pytest)
```

---

## Quick Start

### 1. Training on a Telemetry Channel

```bash
# Train TS-JEPA on a target channel (e.g. SMAP A-1)
python train.py --channel A-1 --encoder hybrid_tcn --epochs 15

# Train with the patch-tokenized Transformer JEPA variant
python train.py --channel A-1 --model-type patch_ts_jepa --patch-size 16

# Train with Flow Matching JEPA
python train.py --channel A-1 --model-type flow_jepa --epochs 50

# Train with MultiScale JEPA
python train.py --channel A-1 --model-type multiscale_ts_jepa

# Train with covariance-whitened (Mahalanobis) scoring
python train.py --channel A-1 --scoring mahalanobis
```

### 2. Running SOTA Benchmarks

```bash
# Run all 9 models across IoT + remote sensing datasets
python scripts/run_modern_baselines.py \
  --dataset CalIt2 cicids Daphnet GECCO Genesis GHL metro room-occupancy MSL SMAP swan \
  --models tranad timesnet anomaly_transformer dcdetector ncad ts_jepa patch_ts_jepa flow_jepa patch_flow_jepa \
  --epochs 50 --batch_size 32 --device auto \
  --output_csv reports/grand_sota_publication_benchmark.csv

# Run on a single dataset
python scripts/run_benchmark.py --dataset Daphnet --encoder tcn

# Run full cross-domain benchmark
python scripts/run_full_benchmark_suite.py --dataset all
```

### 3. Visualizing Phase-Space Attractors

```bash
python scripts/visualize_attractor.py --channel S01R01E1 --epochs 15
```

---

## Testing

```bash
# Run the full test suite (requires Pillow for EVT plotting tests)
python -m pytest tests/ -q

# Run without Pillow-dependent tests
python -m pytest tests/ --ignore=tests/test_evt_calibrator.py --ignore=tests/test_evt_pipeline.py -q

# Run focused audit tests
python -m pytest tests/test_audit_fixes.py tests/test_baselines.py -v
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
