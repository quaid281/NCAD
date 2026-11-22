# Laminar-JEPA-Lite: Quorum Voting over Latent Prediction Residuals for Label-Free Anomaly Detection in Multivariate Time Series

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/architecture-Laminar--JEPA--Lite-purple.svg)](src/models/laminar_lite.py)

Official implementation and reproducible evaluation suite for the manuscript *"Quorum Voting over Latent Prediction Residuals for Label-Free Anomaly Detection in Multivariate Time Series"* (submitted to IEEE Transactions on Neural Networks and Learning Systems).

---

## Overview

**Laminar-JEPA-Lite** adapts fluid-mechanics residual diagnostics to self-supervised latent predictive anomaly detection without imposing physics-loss penalty terms on uncalibrated sensors:

1. **Multi-Horizon Spherical Latent Forecasting (`src/models/laminar_lite.py`):**
   A lightweight 4-block causal TCN with an rFFT dyadic-shell multiscale front-end (`LittlewoodPaleyDyadicBlock`) forecasts nested short-to-long future latent states trained strictly on nominal history via an EMA target encoder and VICReg-style regularization.
2. **ZCA-Whitened Reynolds Decomposition (`src/scoring/laminar_lite_panel.py`):**
   Nominal prediction residuals are whitened by the empirical precision operator and decomposed across horizons into sustained mean-flow drift and transient coordinate fluctuations.
3. **Coordinate Circuit-Breaker Quorum Panel (`src/scoring/circuit_breaker_panel.py`):**
   Instead of collapsing all latent coordinates into a single scalar score, each coordinate and drift channel maintains its own healthy-calibration tripwire and alarms when a coherent quorum trips together.
4. **Label-Free Calibration & Readiness Check (`src/scoring/evt_calibrator.py`, `src/scoring/event_fusion.py`):**
   Evaluated strictly without point adjustment under both strictly causal (`trailing`) and buffered (`smear`) window-to-point mappings, alongside Extreme Value Theory (SPOT/GPD), empirical quantile, conformal, and streaming calibrators.

---

## Key Models & Baselines

| Model | Registry Name | Module | Role |
| :--- | :--- | :--- | :--- |
| **Laminar-JEPA-Lite (128-Dial)** | `laminar_jepa_lite` | `src/models/laminar_lite.py` | Primary 3-horizon spherical JEPA + P+D circuit-breaker panel |
| **Laminar-JEPA (224-Dial)** | `laminar_jepa_224` / `soliton_reynolds_jepa` | `src/models/laminar_lite.py`, `src/models/jepa/alien_jepas.py` | 4-horizon 224-dial diagnostic tier |
| **TS-JEPA** | `ts_jepa` | `src/models/jepa/ts_jepa.py` | Single-horizon scalar JEPA baseline |
| **MultiScale-TS-JEPA** | `multiscale_ts_jepa` | `src/models/jepa/multiscale_ts_jepa.py` | Multi-horizon scalar JEPA baseline |
| **NCAD** | `ncad` | `src/models/encoders/tcn_encoder.py` | Contrastive TCN + contextual anomaly injection baseline |
| **TranAD / TimesNet / Anomaly Transformer / DCdetector** | `tranad`, `timesnet`, `anomaly_transformer`, `dcdetector` | `src/models/baselines/` | External SOTA reconstruction/attention baselines |

---

## Repository Structure

```text
NCAD_CS/
├── train.py                          # Root CLI entry point (delegates to src.cli)
├── app/                              # FastAPI interactive GPU stream explorer
│   ├── server.py                     # Local API server (uvicorn app.server:app)
│   ├── gpu_pipeline.py               # End-to-end Laminar-JEPA-Lite inference & panel trace
│   └── static/index.html             # Single-page interactive visualization UI
├── paper/NCAD_CS/                    # Self-contained IEEE TNNLS manuscript
│   ├── NCAD_CS_standalone.tex        # Generated standalone LaTeX source
│   ├── NCAD_CS_standalone.pdf        # Compiled manuscript (<= 13 pages)
│   ├── export.bib                    # BibTeX references
│   ├── sections/                     # Modular manuscript sections and active tables
│   └── figures/                      # Publication-grade vector figures (.pdf, .svg)
├── reports/                          # Stream manifests and canonical diagnostic CSVs
├── scripts/                          # Reproducible benchmark, audit, and figure scripts
│   ├── build_and_verify_manuscript.py  # Build + compile + verify page cap, spills, acronyms
│   ├── tnnls_revision/               # Full-archive (118-stream) & TNNLS audit runners
│   ├── reviewer_analyses/            # Matched-budget, horizon, calibration, and LODO audits
│   ├── circuit_breaker_19stream/     # Perspective expansion ladder & component ablations
│   └── run_multiseed_heldout_benchmark.py  # Core multi-seed held-out benchmark runner
├── src/                              # Core library
│   ├── cli.py & config.py            # CLI argument parser & validated CSMConfig
│   ├── data/                         # Telemetry stream loaders & windowing pipeline
│   ├── engine/                       # Model builder, trainer, evaluator, orchestrator
│   ├── features/                     # Time-series feature utilities
│   ├── models/                       # Laminar-JEPA-Lite, JEPA, geometric layers, baselines
│   ├── scoring/                      # Circuit-breaker panels, EVT/SPOT, conformal & fusion
│   ├── experimental/                 # Auxiliary 224-dial WSP & Legendre phase ablations
│   └── utils/                        # Structured logging & lazy plotting helpers
└── tests/                            # Modular pytest suite (laminar_lite, models, scoring, engine, archive)
```

---

## Installation & Verification

```bash
pip install -e ".[dev]"

# Lint check
ruff check src tests

# Run full test suite
pytest tests/ -q

# CLI smoke check
python train.py --help
```

---

## Quick Start

### 1. Train on a Telemetry Stream

Place benchmark datasets under `mTSBench_data/<dataset>/` (stream lists are provided in `reports/streams_*.txt` and indexed in `reports/stream_manifest.csv`).

```bash
# Train Laminar-JEPA-Lite on a telemetry channel
python train.py --channel A-1 --model-type laminar_jepa_lite --epochs 15

# Train scalar TS-JEPA with hybrid TCN encoder
python train.py --channel A-1 --model-type ts_jepa --encoder hybrid_tcn --epochs 15
```

### 2. Interactive GPU Explorer

```bash
uvicorn app.server:app --host 127.0.0.1 --port 8000
```

---

## Reproducing Manuscript Tables, Figures & Diagnostics

| Artifact | Command |
| :--- | :--- |
| Held-out 118-stream deployment benchmark | `python scripts/tnnls_revision/run_deployment_benchmark.py` |
| Archive drift, MBB & readiness audit | `python scripts/tnnls_revision/run_archive_118_drift_and_mbb.py` |
| External baselines & LODO audit | `python scripts/tnnls_revision/run_r2_archive_audit_and_baselines.py` |
| Whitening & quorum structure audit | `python scripts/tnnls_revision/run_r3_whitening_structure_audit.py` |
| Perspective ladder & component ablations | `python scripts/circuit_breaker_19stream/run_perspective_ladder.py` |
| Core multi-seed held-out benchmark | `python scripts/run_multiseed_heldout_benchmark.py --seeds 42 123 456 7 2024 --streams reports/streams_19_nonghl.txt --out reports/heldout.csv` |
| Aggregate paired statistics | `python scripts/analyze_multiseed.py --inputs reports/heldout_45stream.csv` |
| Stream-to-split manifest | `python scripts/build_stream_manifest.py` |
| Architecture diagram (Fig. 1) | `python scripts/generate_opentropy_architecture_diagram.py` |
| ZCA-quorum geometry figure (Fig. 2) | `python scripts/generate_zca_quorum_geometry_figure.py` |
| Operating & horizon tradeoffs figure (Fig. 3) | `python scripts/generate_horizon_operating_figure.py` |
| Panel interpretability case studies (Fig. 4) | `python scripts/tnnls_revision/generate_panel_interpretability_figure.py` |
| Build & verify standalone manuscript PDF | `python scripts/build_and_verify_manuscript.py` |

---

## License

This project is licensed under the MIT License; see [LICENSE](LICENSE) for details.
