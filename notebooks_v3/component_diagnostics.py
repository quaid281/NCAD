"""Analyze saved NCAD-CS scoring components for completed runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.scoring import aggregate_suspect_scores, calculate_threshold, dynamic_weighted_smoothing, temporal_consistency_filter


COMPONENT_COLUMNS = [
    "window_score",
    "original_score",
    "substituted_score",
    "substitution_confidence",
    "context_distance",
    "change_score",
    "transition_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved NCAD-CS window-score components.")
    parser.add_argument("--run-dir", required=True, type=str)
    parser.add_argument("--channels", nargs="+", default=None)
    return parser.parse_args()


def metrics_from_flags(labels: np.ndarray, predictions: np.ndarray) -> dict:
    labels = labels.astype(bool)
    predictions = predictions.astype(bool)
    tp = int(np.sum(labels & predictions))
    tn = int(np.sum(~labels & ~predictions))
    fp = int(np.sum(~labels & predictions))
    fn = int(np.sum(labels & ~predictions))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def oracle_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict]:
    unique_scores = np.unique(scores[np.isfinite(scores)])
    if len(unique_scores) > 500:
        thresholds = np.percentile(unique_scores, np.linspace(1.0, 99.9, 500))
    elif len(unique_scores):
        thresholds = unique_scores
    else:
        thresholds = np.array([0.0])

    best_threshold = float(thresholds[0])
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for threshold in thresholds:
        predictions = temporal_consistency_filter(scores > threshold, scores, float(threshold))
        metrics = metrics_from_flags(labels, predictions)
        if (metrics["f1"], metrics["recall"], metrics["precision"]) > (best_metrics["f1"], best_metrics["recall"], best_metrics["precision"]):
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def load_config(run_dir: Path) -> dict:
    path = run_dir / "run_config.json"
    if not path.exists():
        return {"context_size": 284, "suspect_size": 16, "step": 1, "short_smoothing_window": 50, "long_smoothing_window": 200}
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_channel(run_dir: Path, config: dict, channel: str, args: argparse.Namespace) -> list[dict]:
    channel_dir = run_dir / channel
    points = pd.read_csv(channel_dir / "point_predictions.csv")
    windows = pd.read_csv(channel_dir / "window_scores.csv")
    labels = points["label"].fillna(0.0).to_numpy(dtype=float) > 0
    valid_mask_reference = points["valid_score"].to_numpy(dtype=float) > 0
    rows = []

    substitutions = windows["context_substituted"].to_numpy(dtype=float) > 0
    for component in COMPONENT_COLUMNS:
        if component not in windows.columns:
            continue
        component_scores = windows[component].to_numpy(dtype=np.float32)
        point_scores, substitution_map, valid_mask = aggregate_suspect_scores(
            component_scores,
            substitutions,
            n_points=len(points),
            context_size=int(config.get("context_size", 284)),
            suspect_size=int(config.get("suspect_size", 16)),
            step=int(config.get("step", 1)),
        )
        smoothed_scores = dynamic_weighted_smoothing(
            point_scores,
            substitution_map,
            short_window=int(config.get("short_smoothing_window", 50)),
            long_window=int(config.get("long_smoothing_window", 200)),
        )
        scoring_mask = valid_mask & valid_mask_reference
        threshold_info = calculate_threshold(
            smoothed_scores,
            substitution_map,
            scoring_mask,
        )
        predictions = temporal_consistency_filter(
            (smoothed_scores > threshold_info.final_threshold) & scoring_mask,
            smoothed_scores,
            threshold_info.final_threshold,
        )
        current_metrics = metrics_from_flags(labels[scoring_mask], predictions[scoring_mask])
        oracle_value, oracle_metrics = oracle_threshold(labels[scoring_mask], smoothed_scores[scoring_mask])

        rows.append(
            {
                "channel": channel,
                "component": component,
                "threshold": threshold_info.final_threshold,
                "tail_guard_applied": threshold_info.tail_guard_applied,
                "sparse_tail_guard_applied": threshold_info.sparse_tail_guard_applied,
                "sparse_tail_guard_percentile": threshold_info.sparse_tail_guard_percentile,
                "sparse_tail_guard_multiplier": threshold_info.sparse_tail_guard_multiplier,
                "precision": current_metrics["precision"],
                "recall": current_metrics["recall"],
                "f1": current_metrics["f1"],
                "tp": current_metrics["tp"],
                "tn": current_metrics["tn"],
                "fp": current_metrics["fp"],
                "fn": current_metrics["fn"],
                "oracle_threshold": oracle_value,
                "oracle_precision": oracle_metrics["precision"],
                "oracle_recall": oracle_metrics["recall"],
                "oracle_f1": oracle_metrics["f1"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    summary = pd.read_csv(run_dir / "summary.csv")
    channels = args.channels or summary["channel"].dropna().astype(str).tolist()
    config = load_config(run_dir)
    rows = []
    for channel in channels:
        rows.extend(analyze_channel(run_dir, config, channel, args))

    output_dir = run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "component_diagnostics.csv"
    diagnostics = pd.DataFrame(rows).sort_values(["channel", "oracle_f1", "f1"], ascending=[True, False, False])
    diagnostics.to_csv(output_path, index=False)
    print(f"Component diagnostics written to: {output_path}")
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()