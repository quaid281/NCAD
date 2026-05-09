"""Diagnostic plots for NCAD-CS runs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_channel_diagnostics(
    channel_id: str,
    telemetry: np.ndarray,
    smoothed_scores: np.ndarray,
    threshold: float,
    predictions: np.ndarray,
    labels: Optional[np.ndarray],
    substitution_map: np.ndarray,
    context_distances: np.ndarray,
    output_path: str | Path,
) -> None:
    """Create a compact four-panel diagnostic figure."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_axis = np.arange(len(telemetry))

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(x_axis, telemetry, color="#243b53", linewidth=1.0, label="telemetry")
    if labels is not None:
        label_mask = labels[: len(telemetry)] > 0
        axes[0].fill_between(x_axis, np.min(telemetry), np.max(telemetry), where=label_mask, color="#f59e0b", alpha=0.20, label="ground truth")
    axes[0].set_title(f"{channel_id} telemetry")
    axes[0].legend(loc="upper right")

    axes[1].plot(smoothed_scores, color="#2563eb", linewidth=1.0, label="NCAD-CS score")
    axes[1].axhline(threshold, color="#dc2626", linestyle="--", linewidth=1.0, label="threshold")
    axes[1].set_title("Smoothed anomaly score")
    axes[1].legend(loc="upper right")

    axes[2].plot(predictions, color="#dc2626", drawstyle="steps-post", label="prediction")
    if labels is not None:
        axes[2].plot(labels[: len(predictions)] * 0.85, color="#16a34a", alpha=0.8, drawstyle="steps-post", label="label")
    axes[2].set_ylim(-0.05, 1.15)
    axes[2].set_title("Point-level detections")
    axes[2].legend(loc="upper right")

    axes[3].plot(substitution_map.astype(float), color="#7c3aed", drawstyle="steps-post", label="substitution map")
    if len(context_distances) > 0:
        distance_axis = np.linspace(0, len(telemetry) - 1, len(context_distances))
        scaled_distances = context_distances / (np.max(context_distances) + 1e-8)
        axes[3].plot(distance_axis, scaled_distances, color="#0f766e", alpha=0.8, label="context distance scaled")
    axes[3].set_ylim(-0.05, 1.15)
    axes[3].set_title("Context substitution diagnostics")
    axes[3].legend(loc="upper right")
    axes[3].set_xlabel("time step")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
