"""Aggregate analysis and visualization for NCAD-CS result folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BEHAVIOR_COLORS = {
    "strong": "#15803d",
    "over_substitution": "#dc2626",
    "conservative": "#1d4ed8",
    "mixed": "#7c3aed",
    "failed": "#9ca3af",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze NCAD-CS result folders and generate report assets.")
    parser.add_argument("--run-dir", type=str, default=None, help="Specific results directory. Defaults to the newest run under results/.")
    parser.add_argument("--top-k", type=int, default=10, help="How many channels to show in the top-F1 chart.")
    return parser.parse_args()


def resolve_run_dir(run_dir: str | None) -> Path:
    if run_dir is not None:
        path = Path(run_dir).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Run directory does not exist: {path}")
        return path

    results_root = Path(__file__).resolve().parent / "results"
    candidates = [path for path in results_root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {results_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def classify_behavior(row: pd.Series) -> str:
    precision = float(row.get("precision", 0.0) or 0.0)
    recall = float(row.get("recall", 0.0) or 0.0)
    f1 = float(row.get("f1", 0.0) or 0.0)
    substitution_rate = float(row.get("substitution_rate_point", 0.0) or 0.0)

    if f1 >= 0.70:
        return "strong"
    if substitution_rate >= 0.85 or (recall >= 0.95 and precision < 0.30):
        return "over_substitution"
    if precision >= 0.80 and recall < 0.30:
        return "conservative"
    if f1 < 0.10:
        return "failed"
    return "mixed"


def load_summary(run_dir: Path) -> pd.DataFrame:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.csv in {run_dir}")
    summary = pd.read_csv(summary_path)
    if "error" not in summary.columns:
        summary["error"] = np.nan
    summary = summary[summary["error"].isna() | (summary["error"] == "")].copy()
    for column in ["precision", "recall", "f1"]:
        if column in summary.columns:
            summary[column] = pd.to_numeric(summary[column], errors="coerce")
    metric_available = summary[["precision", "recall", "f1"]].notna().all(axis=1)
    excluded_metric_channels = summary.loc[~metric_available, "channel"].dropna().astype(str).tolist()
    summary = summary[metric_available].copy()
    for column in ["substitution_rate_window", "substitution_rate_point", "elapsed_seconds"]:
        if column in summary.columns:
            summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0.0)
    summary["behavior"] = summary.apply(classify_behavior, axis=1)
    summary["precision_recall_gap"] = summary["precision"] - summary["recall"]
    summary = summary.sort_values(["f1", "precision", "recall"], ascending=[False, False, False]).reset_index(drop=True)
    summary.attrs["excluded_metric_channels"] = excluded_metric_channels
    return summary


def summarize_metrics(summary: pd.DataFrame) -> dict:
    excluded_metric_channels = summary.attrs.get("excluded_metric_channels", [])
    metrics = {
        "n_channels": int(len(summary)),
        "n_excluded_metric_channels": int(len(excluded_metric_channels)),
        "excluded_metric_channels": excluded_metric_channels,
        "mean_precision": float(summary["precision"].mean()) if len(summary) else 0.0,
        "mean_recall": float(summary["recall"].mean()) if len(summary) else 0.0,
        "mean_f1": float(summary["f1"].mean()) if len(summary) else 0.0,
        "median_f1": float(summary["f1"].median()) if len(summary) else 0.0,
        "mean_substitution_rate_point": float(summary["substitution_rate_point"].mean()) if len(summary) else 0.0,
        "mean_substitution_rate_window": float(summary["substitution_rate_window"].mean()) if len(summary) else 0.0,
        "mean_elapsed_seconds": float(summary["elapsed_seconds"].mean()) if len(summary) else 0.0,
    }
    metrics["behavior_counts"] = summary["behavior"].value_counts().to_dict()
    return metrics


def select_archetypes(summary: pd.DataFrame) -> list[str]:
    selections: list[str] = []

    def add_channel(frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        channel = str(frame.iloc[0]["channel"])
        if channel not in selections:
            selections.append(channel)

    add_channel(summary.sort_values("f1", ascending=False))
    add_channel(summary.sort_values("substitution_rate_point", ascending=False))
    conservative = summary[(summary["precision"] >= 0.80) & (summary["recall"] < 0.30)]
    add_channel(conservative.sort_values("precision", ascending=False))
    add_channel(summary.sort_values("f1", ascending=True))
    return selections[:4]


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows_"

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(run_dir: Path, analysis_dir: Path, summary: pd.DataFrame, metrics: dict, archetypes: Iterable[str]) -> Path:
    report_path = analysis_dir / "analysis_report.md"
    top_rows = summary.nlargest(min(10, len(summary)), "f1")[["channel", "f1", "precision", "recall", "substitution_rate_point", "behavior"]]
    bottom_rows = summary.nsmallest(min(10, len(summary)), "f1")[["channel", "f1", "precision", "recall", "substitution_rate_point", "behavior"]]

    lines = [
        "# NCAD-CS Results Analysis",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Aggregate Metrics",
        "",
        f"- Channels analyzed: {metrics['n_channels']}",
        f"- Channels excluded from metric aggregates: {metrics['n_excluded_metric_channels']}",
        f"- Mean precision: {metrics['mean_precision']:.4f}",
        f"- Mean recall: {metrics['mean_recall']:.4f}",
        f"- Mean F1: {metrics['mean_f1']:.4f}",
        f"- Median F1: {metrics['median_f1']:.4f}",
        f"- Mean point substitution rate: {metrics['mean_substitution_rate_point']:.4f}",
        f"- Mean window substitution rate: {metrics['mean_substitution_rate_window']:.4f}",
        f"- Mean elapsed seconds: {metrics['mean_elapsed_seconds']:.2f}",
        "",
        "## Behavior Counts",
        "",
    ]
    for behavior, count in sorted(metrics["behavior_counts"].items()):
        lines.append(f"- {behavior}: {count}")

    if metrics["excluded_metric_channels"]:
        lines.extend([
            "",
            "## Excluded From Metric Aggregates",
            "",
            ", ".join(metrics["excluded_metric_channels"]),
        ])

    lines.extend([
        "",
        "## Selected Archetypes",
        "",
    ])
    for channel in archetypes:
        row = summary.loc[summary["channel"] == channel].iloc[0]
        lines.append(
            f"- {channel}: behavior={row['behavior']}, F1={row['f1']:.4f}, precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}, substitution_rate_point={row['substitution_rate_point']:.4f}"
        )

    lines.extend([
        "",
        "## Top Channels By F1",
        "",
        dataframe_to_markdown(top_rows),
        "",
        "## Lowest Channels By F1",
        "",
        dataframe_to_markdown(bottom_rows),
        "",
        "Generated assets:",
        "- analysis_summary.csv",
        "- performance_overview.png",
        "- archetype_diagnostics.png",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def plot_performance_overview(summary: pd.DataFrame, output_path: Path, top_k: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    behaviors = summary["behavior"].tolist()
    colors = [BEHAVIOR_COLORS.get(behavior, "#6b7280") for behavior in behaviors]
    sizes = 120 + 800 * summary["substitution_rate_point"].to_numpy()

    axes[0, 0].scatter(summary["recall"], summary["precision"], s=sizes, c=colors, alpha=0.80, edgecolors="#111827", linewidths=0.6)
    for _, row in summary.iterrows():
        axes[0, 0].text(row["recall"] + 0.01, row["precision"] + 0.01, row["channel"], fontsize=8)
    axes[0, 0].set_xlabel("Recall")
    axes[0, 0].set_ylabel("Precision")
    axes[0, 0].set_title("Precision vs Recall (size = point substitution rate)")
    axes[0, 0].grid(alpha=0.20)

    top = summary.nlargest(min(top_k, len(summary)), "f1").iloc[::-1]
    bar_colors = [BEHAVIOR_COLORS.get(behavior, "#6b7280") for behavior in top["behavior"]]
    axes[0, 1].barh(top["channel"], top["f1"], color=bar_colors, alpha=0.90)
    axes[0, 1].set_xlabel("F1")
    axes[0, 1].set_title(f"Top {min(top_k, len(summary))} Channels by F1")
    axes[0, 1].grid(axis="x", alpha=0.20)

    axes[1, 0].scatter(summary["substitution_rate_point"], summary["f1"], s=140, c=colors, alpha=0.85, edgecolors="#111827", linewidths=0.6)
    for _, row in summary.iterrows():
        axes[1, 0].text(row["substitution_rate_point"] + 0.01, row["f1"] + 0.01, row["channel"], fontsize=8)
    axes[1, 0].set_xlabel("Point substitution rate")
    axes[1, 0].set_ylabel("F1")
    axes[1, 0].set_title("Substitution Rate vs F1")
    axes[1, 0].grid(alpha=0.20)

    bins = min(10, max(4, len(summary)))
    axes[1, 1].hist(summary["f1"], bins=bins, color="#2563eb", alpha=0.85, edgecolor="#111827")
    axes[1, 1].axvline(summary["f1"].mean(), color="#dc2626", linestyle="--", linewidth=1.2, label=f"mean={summary['f1'].mean():.3f}")
    axes[1, 1].axvline(summary["f1"].median(), color="#15803d", linestyle=":", linewidth=1.2, label=f"median={summary['f1'].median():.3f}")
    axes[1, 1].set_xlabel("F1")
    axes[1, 1].set_title("F1 Distribution")
    axes[1, 1].legend(loc="upper right")
    axes[1, 1].grid(alpha=0.20)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_channel_metrics(run_dir: Path, channel: str) -> dict:
    path = run_dir / channel / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def plot_archetype_diagnostics(run_dir: Path, summary: pd.DataFrame, channels: list[str], output_path: Path) -> None:
    if not channels:
        return

    fig, axes = plt.subplots(len(channels), 1, figsize=(16, 3.8 * len(channels)), sharex=False)
    if len(channels) == 1:
        axes = [axes]

    for axis, channel in zip(axes, channels):
        points = pd.read_csv(run_dir / channel / "point_predictions.csv")
        metrics = load_channel_metrics(run_dir, channel)
        row = summary.loc[summary["channel"] == channel].iloc[0]
        x_axis = points["time_index"].to_numpy()
        telemetry = points["telemetry"].to_numpy()
        scores = points["smoothed_score"].to_numpy()
        predictions = points["prediction"].to_numpy()
        labels = points["label"].fillna(0.0).to_numpy()
        substitutions = points["context_substituted"].to_numpy()
        threshold = float(metrics["threshold"]["final_threshold"])

        axis.plot(x_axis, telemetry, color="#1f2937", linewidth=0.9, alpha=0.75, label="telemetry")
        if np.any(labels > 0):
            axis.fill_between(x_axis, np.min(telemetry), np.max(telemetry), where=labels > 0, color="#f59e0b", alpha=0.16, label="label")
        scaled_scores = np.interp(scores, (scores.min(), scores.max() + 1e-8), (telemetry.min(), telemetry.max())) if len(scores) else scores
        axis.plot(x_axis, scaled_scores, color="#2563eb", linewidth=1.0, alpha=0.9, label="smoothed score scaled")
        axis.plot(x_axis, predictions * telemetry.max(), color="#dc2626", linewidth=1.0, drawstyle="steps-post", alpha=0.85, label="prediction")
        axis.plot(x_axis, substitutions * telemetry.max(), color="#7c3aed", linewidth=0.9, drawstyle="steps-post", alpha=0.70, label="substitution")
        axis.set_title(
            f"{channel} | behavior={row['behavior']} | F1={row['f1']:.3f} | P={row['precision']:.3f} | R={row['recall']:.3f} | "
            f"tau={threshold:.3f}"
        )
        axis.grid(alpha=0.15)
        axis.legend(loc="upper right", ncol=5, fontsize=8)

    axes[-1].set_xlabel("time index")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(run_dir)
    metrics = summarize_metrics(summary)
    archetypes = select_archetypes(summary)

    enriched_path = analysis_dir / "analysis_summary.csv"
    summary.to_csv(enriched_path, index=False)
    excluded_path = analysis_dir / "excluded_metric_channels.txt"
    excluded_channels = summary.attrs.get("excluded_metric_channels", [])
    excluded_path.write_text("\n".join(excluded_channels), encoding="utf-8")

    overview_path = analysis_dir / "performance_overview.png"
    plot_performance_overview(summary, overview_path, top_k=args.top_k)

    archetypes_path = analysis_dir / "archetype_diagnostics.png"
    plot_archetype_diagnostics(run_dir, summary, archetypes, archetypes_path)

    report_path = write_report(run_dir, analysis_dir, summary, metrics, archetypes)

    print(f"Analyzed run: {run_dir}")
    print(f"Channels: {metrics['n_channels']}")
    print(f"Excluded metric channels: {metrics['n_excluded_metric_channels']}")
    print(f"Mean F1: {metrics['mean_f1']:.4f}")
    print(f"Report: {report_path}")
    print(f"Overview figure: {overview_path}")
    print(f"Archetype figure: {archetypes_path}")


if __name__ == "__main__":
    main()