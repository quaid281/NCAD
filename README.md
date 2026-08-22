# NCAD-CS: Neural Contextual Anomaly Detection with Counterfactual Successor Memory

[![Tests](https://img.shields.io/badge/pytest-38%20passed-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

NCAD-CS is a self-supervised deep learning framework for unsupervised multi-sensor time-series anomaly detection. It integrates **Spatial-Temporal Relational Graph Attention Networks (GAT)**, **Counterfactual Successor Memory (CSM)** with RPCA sanitization, non-contrastive **Joint Embedding Predictive Architectures (TS-JEPA)**, and **Extreme Value Theory (EVT / SPOT)** tail calibration.

---

## Key Features

- **Spatial-Temporal Relational GAT**: Causal dilated temporal convolutions fused with multi-head inter-variable attention to capture cross-sensor dependencies.
- **Counterfactual Successor Memory**: KNN memory mapping normal contexts to plausible normal successors to detect physical and contextual anomalies without synthetic labels.
- **EVT / SPOT Tail Calibration**: Unsupervised threshold calibration using Generalized Pareto Distribution (GPD) extreme value modeling, avoiding heuristic threshold collapse.
- **Self-Supervised JEPA & FEI-SIGReg**: Non-contrastive invariance, variance, and covariance regularization eliminating artificial time-domain injection biases.
- **SINDy Dynamical Consistency**: Discovers explicit sparse non-linear differential equations governing normal latent trajectories.

---

## Repository Structure

```
NCAD_CS/
├── docs/                 # Research papers and technical documentation
├── mTSBench_data/        # Benchmark datasets (Daphnet, OPPORTUNITY, CalIt2, GECCO, etc.)
├── notebooks/            # Dataset-specific demonstration and evaluation notebooks
│   ├── CalIt2/
│   ├── Daphnet/
│   ├── GECCO/
│   ├── Genesis/
│   ├── GHL/
│   ├── OPPORTUNITY/
│   ├── cicids/
│   ├── creditcard/
│   ├── exploratory/
│   └── room-occupancy/
├── paper/                # IEEE conference/journal paper LaTeX source & diagrams
├── pyproject.toml        # PEP 621 build configuration and tool settings
├── pytest.ini            # Pytest test suite configuration
├── references/           # Dataset specifications and bibliography
├── reports/              # Benchmark reports and comparison summaries
├── requirements.txt      # Pinned dependency specification
├── results/              # Run logs, model checkpoints, and evaluation metrics
├── scratch/              # Experimental sweeps, converters, and diagnostic scripts
│   ├── analysis/
│   ├── benchmarks/
│   ├── converters/
│   └── legacy_tests/
├── scripts/              # Benchmark evaluation runners across mTSBench datasets
├── src/                  # Core NCAD-CS library
│   ├── data/             # Data loading and windowing pipelines
│   ├── experimental/     # Novelty research prototypes (Hopfield, Causal, Memoryless, SSM)
│   ├── features/         # Multi-domain signal feature extraction
│   ├── models/           # Encoders, CSM, TS-JEPA, SINDy, Anomaly Injectors
│   └── utils/            # Event fusion, EVT tail calibrators, logging, and plotting
├── tests/                # Comprehensive unit test suite (100% passing)
└── train.py              # Root CLI entry point for training and evaluation
```

---

## Installation

```bash
# Clone and install in editable mode
git clone https://github.com/quaid281/NCAD.git
cd NCAD_CS
pip install -e .
```

Or install with development dependencies:
```bash
pip install -e ".[dev]"
```

---

## Quick Start

### 1. Training & Evaluation CLI
```bash
# Run training on a specific channel with EVT thresholding
python train.py --channel A-1 --encoder relational_gat --threshold-method evt

# View all configuration options
python train.py --help
```

### 2. Running Benchmarks
```bash
# Run full multi-dataset benchmark suite across mTSBench
python scripts/run_full_benchmark_suite.py --dataset Daphnet --encoder relational_gat
```

### 3. Running Unit Tests
```bash
pytest -v
```

---

## Citation

If you use NCAD-CS in your research, please cite our paper:

```bibtex
@article{ncad_cs_2026,
  title={Neural Contextual Anomaly Detection with Counterfactual Successor Memory and Spatial-Temporal Graph Attention},
  author={NCAD-CS Research Team},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2026}
}
```
