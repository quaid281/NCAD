"""Threshold and score diagnostics for completed NCAD-CS runs.

This script is intentionally post-hoc. It reads saved run artifacts and compares
threshold choices without changing the training or inference pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.scoring import calculate_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose NCAD-CS score and threshold behavior for saved runs.")
    parser.add_argument("--run-dir", required=True, type=str, help="Results directory containing summary.csv and channel folders.")
    parser.add_argument("--channels", nargs="+", default=None, help="Optional subset of channel IDs to analyze.")
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


def temporal_filter(flags: np.ndarray, scores: np.ndarray, threshold: float, min_run: int = 3, extreme_factor: float = 1.8) -> np.ndarray:
    flags = flags.astype(bool)
    filtered = np.zeros_like(flags, dtype=bool)
    index = 0
    while index < len(flags):
        if not flags[index]:
            index += 1
            continue
        start = index
        while index < len(flags) and flags[index]:
            index += 1
        end = index
        run_scores = scores[start:end]
        if (end - start) >= min_run or np.any(run_scores > extreme_factor * threshold):
            filtered[start:end] = True
    return filtered


def evaluate_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = temporal_filter(scores > threshold, scores, threshold)
    result = metrics_from_flags(labels, predictions)
    result["threshold"] = float(threshold)
    result["predicted_points"] = int(np.sum(predictions))
    return result


def oracle_threshold(labels: np.ndarray, scores: np.ndarray) -> dict:
    unique_scores = np.unique(scores[np.isfinite(scores)])
    if len(unique_scores) == 0:
        return evaluate_threshold(labels, scores, 0.0)
    if len(unique_scores) > 500:
        candidates = np.percentile(unique_scores, np.linspace(1.0, 99.9, 500))
    else:
        candidates = unique_scores
    best = None
    for threshold in candidates:
        result = evaluate_threshold(labels, scores, float(threshold))
        if best is None or (result["f1"], result["recall"], result["precision"]) > (best["f1"], best["recall"], best["precision"]):
            best = result
    return best or evaluate_threshold(labels, scores, float(np.max(scores)))


def candidate_thresholds(scores: np.ndarray, substitution_map: np.ndarray, current_threshold: float) -> dict[str, float]:
    finite_mask = np.isfinite(scores)
    normal_mask = finite_mask & ~substitution_map.astype(bool)
    baseline = scores[normal_mask] if np.sum(normal_mask) >= 20 else scores[finite_mask]
    if len(baseline) == 0:
        baseline = np.array([0.0], dtype=np.float32)

    q25, q75 = np.percentile(baseline, [25, 75])
    robust = float(np.median(baseline) + 3.0 * (q75 - q25))
    p95 = float(np.percentile(baseline, 95.0))
    p99 = float(np.percentile(baseline, 99.0))
    p995 = float(np.percentile(baseline, 99.5))
    return {
        "current": float(current_threshold),
        "robust_median_3iqr": robust,
        "p95_baseline": p95,
        "p99_baseline": p99,
        "p995_baseline": p995,
        "min_robust_p99": min(robust, p99),
        "max_robust_p99": max(robust, p99),
        "mean_robust_p99": float(0.5 * (robust + p99)),
    }


def score_quantiles(labels: np.ndarray, scores: np.ndarray) -> dict:
    quantiles = [0.0, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99, 1.0]
    result = {f"all_q{int(q * 1000):03d}": float(np.quantile(scores, q)) for q in quantiles}
    if np.any(labels):
        positive_scores = scores[labels.astype(bool)]
        negative_scores = scores[~labels.astype(bool)]
        for q in quantiles:
            result[f"pos_q{int(q * 1000):03d}"] = float(np.quantile(positive_scores, q))
            result[f"neg_q{int(q * 1000):03d}"] = float(np.quantile(negative_scores, q))
    return result


def analyze_channel(run_dir: Path, channel: str) -> tuple[list[dict], dict]:
    channel_dir = run_dir / channel
    points = pd.read_csv(channel_dir / "point_predictions.csv")
    metrics = json.loads((channel_dir / "metrics.json").read_text(encoding="utf-8"))

    valid = points["valid_score"].to_numpy(dtype=float) > 0
    labels = points.loc[valid, "label"].fillna(0.0).to_numpy(dtype=float) > 0
    substitution_map = points.loc[valid, "context_substituted"].to_numpy(dtype=float) > 0
    current_threshold = float(metrics["threshold"]["final_threshold"])

    rows = []
    channel_summary = {
        "channel": channel,
        "valid_points": int(np.sum(valid)),
        "positive_points": int(np.sum(labels)),
        "substitution_rate": float(np.mean(substitution_map)) if len(substitution_map) else 0.0,
        "current_threshold": current_threshold,
        "current_f1": float(metrics["metrics"].get("f1", 0.0)),
        "current_precision": float(metrics["metrics"].get("precision", 0.0)),
        "current_recall": float(metrics["metrics"].get("recall", 0.0)),
    }

    for score_column in ["point_score", "smoothed_score"]:
        scores = points.loc[valid, score_column].to_numpy(dtype=float)
        thresholds = candidate_thresholds(scores, substitution_map, current_threshold)
        adaptive_threshold = calculate_threshold(
            scores.astype(np.float32),
            substitution_map,
            np.ones_like(substitution_map, dtype=bool),
        ).final_threshold
        thresholds["adaptive_v2"] = adaptive_threshold
        for name, threshold in thresholds.items():
            result = evaluate_threshold(labels, scores, threshold)
            result.update({"channel": channel, "score_column": score_column, "candidate": name})
            rows.append(result)

        oracle = oracle_threshold(labels, scores)
        oracle.update({"channel": channel, "score_column": score_column, "candidate": "oracle_label_only"})
        rows.append(oracle)

        for key, value in score_quantiles(labels, scores).items():
            channel_summary[f"{score_column}_{key}"] = value

    return rows, channel_summary


def write_markdown(output_path: Path, summary: pd.DataFrame, candidates: pd.DataFrame) -> None:
    lines = ["# Threshold Diagnostics", ""]
    for _, row in summary.iterrows():
        channel = row["channel"]
        channel_candidates = candidates[candidates["channel"] == channel]
        best = channel_candidates.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
        lines.extend([
            f"## {channel}",
            "",
            f"- Current F1: {row['current_f1']:.4f}",
            f"- Current precision: {row['current_precision']:.4f}",
            f"- Current recall: {row['current_recall']:.4f}",
            f"- Substitution rate: {row['substitution_rate']:.4f}",
            f"- Best diagnostic candidate: {best['candidate']} on {best['score_column']} with F1={best['f1']:.4f}, precision={best['precision']:.4f}, recall={best['recall']:.4f}",
            "",
        ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    summary = pd.read_csv(run_dir / "summary.csv")
    channels = args.channels or summary["channel"].dropna().astype(str).tolist()
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_summaries = []
    for channel in channels:
        rows, channel_summary = analyze_channel(run_dir, channel)
        all_rows.extend(rows)
        all_summaries.append(channel_summary)

    candidates = pd.DataFrame(all_rows)
    score_summary = pd.DataFrame(all_summaries)
    candidates_path = analysis_dir / "threshold_candidates.csv"
    summary_path = analysis_dir / "score_distribution_summary.csv"
    report_path = analysis_dir / "threshold_diagnostics.md"
    candidates.to_csv(candidates_path, index=False)
    score_summary.to_csv(summary_path, index=False)
    write_markdown(report_path, score_summary, candidates)

    print(f"Threshold diagnostics written to: {analysis_dir}")
    print(f"Candidate table: {candidates_path}")
    print(f"Score summary: {summary_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()