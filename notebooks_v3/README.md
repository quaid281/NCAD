# NCAD-CS v3

This directory implements the method described in `paper/NCAD_CS.sty` as the source of truth.

## What This Version Implements

- Robust multi-domain feature engineering from the primary univariate telemetry channel.
- A shared hybrid-pooling TCN encoder trained with the paper's contrastive objective.
- Synthetic contextual anomaly injection into the 16-sample suspect window for self-supervised training.
- A Context Memory Bank built from normal training context embeddings using K-Means prototypes.
- Storage of raw representative context windows for each memory prototype.
- Context contamination detection using memory distance, embedding change, and transition confidence.
- Confidence-weighted contextual substitution scoring:

```text
S_win = (1 - S_conf) * S_orig + S_conf * (beta * S_sub)
```

- Point-level aggregation over suspect regions, dynamic weighted smoothing, robust thresholding, and temporal consistency filtering.

## Structure

```text
notebooks_v3/
├── main.ipynb
├── reviewer_support.py
├── train.py
├── models/
│   ├── anomaly_injector.py
│   ├── memory_bank.py
│   └── tcn_encoder.py
└── utils/
    ├── data_loader.py
    ├── features.py
    ├── scoring.py
    └── visualization.py
```

## Reviewer 2 Response Workflow

The v3 directory includes support for the two directions we are keeping in this implementation: novelty framing and dataset breadth.

### 1. Generate paper assets

```bash
python reviewer_support.py
```

This writes `paper_assets/` with:

- `dataset_inventory.csv`: one row per channel-level dataset with train/test lengths and anomaly counts.
- `dataset_summary.md`: manuscript-ready framing of the benchmark as many channel-level datasets, not one isolated time series.
- `novelty_comparison_table.md`: a table that contrasts NCAD-CS against forecasting, contrastive NCAD, graph/attention reconstruction, memory-enhanced reconstruction, and classical prototype detectors.
- `novelty_comparison_table.tex`: a LaTeX version of the novelty table.
- `experiment_protocol.md`: commands and reviewer-concern mapping for the retained NCAD-CS experiments.

### 2. Run NCAD-CS across all channel-level datasets

```bash
python train.py --all --epochs 40
```

Each telemetry channel has its own train file, test file, operating behavior, and anomaly annotations. The all-channel run produces a `summary.csv` that can be used for aggregate reporting.

### 3. Fast sanity check

```bash
python train.py --channel A-1 --epochs 1 --max-train-windows 128 --max-test-windows 128 --no-plots
```

## Quick Smoke Run

From `notebooks_v3`:

```bash
python train.py --channel A-1 --epochs 1 --max-train-windows 128 --max-test-windows 128 --no-plots
```

## Results Analysis And Visualizations

After any run completes, generate an aggregate report and summary figures from the saved CSV artifacts:

```bash
python analyze_results.py --run-dir results/20260509_093529
```

If `--run-dir` is omitted, the script analyzes the newest folder under `results/`.

It writes an `analysis/` subfolder inside the selected run with:

- `analysis_summary.csv`: the original summary with behavior labels such as `strong`, `over_substitution`, and `conservative`.
- `analysis_report.md`: manuscript-friendly aggregate interpretation.
- `performance_overview.png`: aggregate precision/recall, F1, and substitution-rate plots.
- `archetype_diagnostics.png`: a compact panel of representative channels from the run.

## Dynamic Thresholding

The pipeline uses one settled dynamic thresholding strategy. It does not use channel-specific hardcoded thresholds or user-selected thresholding modes. For each channel, the pipeline:

1. estimates a distribution-derived noise floor from the channel's own valid scores;
2. generates self-supervised clean/contaminated calibration windows from the training split;
3. scores those synthetic calibration windows and learns a separating threshold;
4. uses the synthetic threshold only when it does not undercut the observed score-distribution floor.

This keeps thresholding adaptive while preventing synthetic calibration from accepting too many false positives.

The selected threshold path is saved in each channel's `metrics.json` under `threshold.threshold_method`.

## Full Single-Channel Run

```bash
python train.py --channel D-3 --epochs 40
```

## Multiple Channels

```bash
python train.py --channels A-1 A-2 D-3 P-3 --epochs 40
```

## Outputs

Each run writes a timestamped folder under `notebooks_v3/results/` unless `--output-dir` is provided. Per channel, the pipeline saves:

- `metrics.json`
- `point_predictions.csv`
- `window_scores.csv`
- `encoder.pt`
- `memory_bank.npz`
- `feature_metadata.json`
- diagnostic plot when plots are enabled
