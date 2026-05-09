# Reviewer-Response Experiment Protocol

## Reviewer Concern Coverage

- Novelty: use `novelty_comparison_table.md` or `novelty_comparison_table.tex` to clarify that NCAD-CS is not simply clustering. The novelty is the use of normal prototypes as a context-reliability gate plus raw context substitution before contrastive scoring.
- Dataset breadth: use `dataset_inventory.csv` and `dataset_summary.md` to frame the benchmark as channel-level datasets, each with distinct train/test telemetry and labels.
- Manuscript positioning: use the generated comparison table to explain that clustering is not the detector by itself; it is the context validation and substitution mechanism inside NCAD-CS.

## Commands

Generate paper assets:

```bash
python reviewer_support.py
```

Run NCAD-CS across every available channel-level dataset:

```bash
python train.py --all --epochs 40
```

Fast sanity check before a full experiment:

```bash
python train.py --channel A-1 --epochs 1 --max-train-windows 128 --max-test-windows 128 --no-plots
```
