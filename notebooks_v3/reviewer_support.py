"""Generate reviewer-response artifacts for the NCAD-CS paper.

The desk-reject feedback asks for clearer novelty and broader dataset framing.
This script creates reusable paper assets that are kept near the v3 experiments
so the manuscript claims stay tied to the code.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


NOVELTY_ROWS = [
    {
        "method_family": "LSTM forecasting with dynamic thresholding",
        "representative_methods": "LSTM-NDT / spacecraft telemetry forecasting",
        "core_signal": "Prediction residual",
        "prototype_or_memory_role": "None",
        "context_reliability_gate": "No",
        "raw_context_substitution": "No",
        "sustained_anomaly_protection": "Limited; contaminated recent history can suppress residuals",
        "ncad_cs_distinction": "NCAD-CS validates the context before scoring and replaces contaminated context with a stored normal prototype.",
    },
    {
        "method_family": "Contrastive contextual anomaly detection",
        "representative_methods": "NCAD-style TCN context/suspect comparison",
        "core_signal": "Distance between context and suspect embeddings",
        "prototype_or_memory_role": "None in the standard formulation",
        "context_reliability_gate": "No",
        "raw_context_substitution": "No",
        "sustained_anomaly_protection": "Limited; context contamination makes anomalous context and suspect embeddings similar",
        "ncad_cs_distinction": "NCAD-CS keeps the contrastive encoder but adds an external normal-context bank and confidence-weighted substitution.",
    },
    {
        "method_family": "Graph/attention reconstruction",
        "representative_methods": "MTAD-GAT, TranAD, Anomaly Transformer, LGAT",
        "core_signal": "Reconstruction or association discrepancy",
        "prototype_or_memory_role": "Usually none or implicit latent state",
        "context_reliability_gate": "No explicit normal-context gate",
        "raw_context_substitution": "No",
        "sustained_anomaly_protection": "Depends on model sensitivity; long anomalies may become reconstructable",
        "ncad_cs_distinction": "NCAD-CS separates representation learning from context validation and forces comparison to known-good context when needed.",
    },
    {
        "method_family": "Memory-enhanced reconstruction",
        "representative_methods": "Memory networks, DiMER-like contrast memory, HYMAN-like global memory",
        "core_signal": "Memory-aided reconstruction error or latent contrast",
        "prototype_or_memory_role": "Internal representation enrichment",
        "context_reliability_gate": "Typically no external context normality test",
        "raw_context_substitution": "No",
        "sustained_anomaly_protection": "Improved memory can help, but memory is not used to replace contaminated context",
        "ncad_cs_distinction": "NCAD-CS uses memory as an external normal-context validator and stores raw reference windows for substitution.",
    },
    {
        "method_family": "Classical prototype or clustering anomaly detection",
        "representative_methods": "K-Means distance, one-class clustering, prototype nearest-neighbor detectors",
        "core_signal": "Distance to normal cluster/prototype",
        "prototype_or_memory_role": "Primary detector",
        "context_reliability_gate": "Sometimes, as direct anomaly score",
        "raw_context_substitution": "No",
        "sustained_anomaly_protection": "No learned contrastive context/suspect scoring",
        "ncad_cs_distinction": "NCAD-CS does not use clustering as the detector alone; clustering decides whether the learned contextual comparison is trustworthy.",
    },
    {
        "method_family": "Proposed NCAD-CS",
        "representative_methods": "Hybrid-pooling TCN + Context Memory Bank + contextual substitution",
        "core_signal": "Confidence-weighted original/substituted embedding distance",
        "prototype_or_memory_role": "External bank of normal context prototypes plus raw reference windows",
        "context_reliability_gate": "Yes",
        "raw_context_substitution": "Yes",
        "sustained_anomaly_protection": "Designed specifically to prevent score suppression from contaminated context windows",
        "ncad_cs_distinction": "Novel contribution is the operational use of normal prototypes to validate and replace context, not clustering itself.",
    },
]


def parse_sequences(value: object) -> list:
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
        return parsed if isinstance(parsed, list) else []
    if isinstance(value, list):
        return value
    return []


def count_anomaly_points(sequences: Iterable[object]) -> int:
    total = 0
    for sequence in sequences:
        if isinstance(sequence, (list, tuple)) and len(sequence) == 2:
            total += max(0, int(sequence[1]) - int(sequence[0]))
        elif isinstance(sequence, int):
            total += 1
    return total


def npy_length(path: Path) -> int | None:
    if not path.exists():
        return None
    array = np.load(path, mmap_mode="r")
    return int(array.shape[0])


def generate_dataset_inventory(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    labels_path = data_dir / "processed" / "labeled_anomalies.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    labels_df = pd.read_csv(labels_path)
    rows = []
    for _, row in labels_df.iterrows():
        channel_id = str(row["chan_id"])
        train_path = data_dir / "raw" / "train" / f"{channel_id}.npy"
        test_path = data_dir / "raw" / "test" / f"{channel_id}.npy"
        sequences = parse_sequences(row.get("anomaly_sequences", "[]"))
        rows.append(
            {
                "channel_id": channel_id,
                "spacecraft": row.get("spacecraft", "unknown"),
                "anomaly_class": row.get("class", "unknown"),
                "num_labeled_values": int(row.get("num_values", 0)),
                "train_samples": npy_length(train_path),
                "test_samples": npy_length(test_path),
                "num_anomaly_sequences": len(sequences),
                "num_anomaly_points": count_anomaly_points(sequences),
                "has_train_file": train_path.exists(),
                "has_test_file": test_path.exists(),
            }
        )

    inventory = pd.DataFrame(rows)
    inventory["usable_as_channel_dataset"] = inventory["has_train_file"] & inventory["has_test_file"]
    inventory.to_csv(output_dir / "dataset_inventory.csv", index=False)
    write_dataset_summary(inventory, output_dir / "dataset_summary.md")
    return inventory


def write_dataset_summary(inventory: pd.DataFrame, output_path: Path) -> None:
    usable = inventory[inventory["usable_as_channel_dataset"]]
    spacecraft_counts = usable["spacecraft"].value_counts().to_dict()
    total_anomaly_points = int(usable["num_anomaly_points"].sum())
    total_test_points = int(usable["test_samples"].fillna(0).sum())
    anomaly_rate = total_anomaly_points / total_test_points if total_test_points else 0.0

    lines = [
        "# Dataset Inventory Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Usable channel-level datasets: {len(usable)}",
        f"Spacecraft/channel groups: {spacecraft_counts}",
        f"Total training samples across usable channels: {int(usable['train_samples'].fillna(0).sum()):,}",
        f"Total test samples across usable channels: {total_test_points:,}",
        f"Total labeled anomalous test points: {total_anomaly_points:,}",
        f"Overall labeled anomaly rate: {100 * anomaly_rate:.2f}%",
        "",
        "Suggested manuscript framing:",
        "",
        (
            "Although the telemetry comes from a public spacecraft benchmark, each channel is an independent "
            "univariate anomaly-detection dataset with its own train split, test split, operating regime, and anomaly annotations. "
            f"The experimental suite therefore evaluates NCAD-CS over {len(usable)} channel-level datasets rather than a single time series."
        ),
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(dataframe: pd.DataFrame) -> str:
    header = "| " + " | ".join(dataframe.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(dataframe.columns)) + " |"
    rows: List[str] = []
    for _, row in dataframe.iterrows():
        rows.append("| " + " | ".join(str(row[column]).replace("\n", " ") for column in dataframe.columns) + " |")
    return "\n".join([header, separator, *rows]) + "\n"


def latex_table(dataframe: pd.DataFrame) -> str:
    columns = [
        "method_family",
        "prototype_or_memory_role",
        "context_reliability_gate",
        "raw_context_substitution",
        "sustained_anomaly_protection",
    ]
    display = dataframe[columns].copy()
    display.columns = ["Method family", "Prototype/memory role", "Context gate", "Raw substitution", "Sustained anomaly behavior"]
    return display.to_latex(index=False, escape=True, longtable=True)


def generate_novelty_tables(output_dir: Path) -> pd.DataFrame:
    novelty_df = pd.DataFrame(NOVELTY_ROWS)
    novelty_df.to_csv(output_dir / "novelty_comparison_table.csv", index=False)
    (output_dir / "novelty_comparison_table.md").write_text(markdown_table(novelty_df), encoding="utf-8")
    (output_dir / "novelty_comparison_table.tex").write_text(latex_table(novelty_df), encoding="utf-8")
    return novelty_df


def write_experiment_protocol(output_dir: Path) -> None:
    protocol = """# Reviewer-Response Experiment Protocol

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
"""
    (output_dir / "experiment_protocol.md").write_text(protocol, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate NCAD-CS reviewer-response paper assets.")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve() if args.data_dir else Path(__file__).resolve().parents[1] / "data"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path(__file__).resolve().parent / "paper_assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory = generate_dataset_inventory(data_dir, output_dir)
    novelty_df = generate_novelty_tables(output_dir)
    write_experiment_protocol(output_dir)

    usable_count = int(inventory["usable_as_channel_dataset"].sum())
    print(f"Generated reviewer assets in: {output_dir}")
    print(f"Usable channel-level datasets: {usable_count}")
    print(f"Novelty comparison rows: {len(novelty_df)}")


if __name__ == "__main__":
    main()
