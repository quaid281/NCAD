"""Diagnostic plots for NCAD-CS runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

# Set matplotlib backend to Agg to prevent headless GUI crashes
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.data_loader import ChannelData

logger = logging.getLogger(__name__)


def plot_channel_diagnostics(
    channel_dir: Path,
    channel_data: ChannelData,
    smoothed_scores: np.ndarray,
    threshold: float,
    predictions: np.ndarray,
    context_ood_map: np.ndarray,
) -> None:
    """Create a compact four-panel diagnostic figure and save it to the channel directory."""
    try:
        labels = channel_data.labels if channel_data.labels is not None else np.zeros_like(predictions)
        fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
        time_index = np.arange(len(channel_data.test_raw))

        # Panel 1: Telemetry & ground truth labels
        axes[0].plot(time_index, channel_data.test_raw, color="#111827", linewidth=0.8, label="telemetry")
        if labels is not None:
            anomaly_points = labels[: len(time_index)] > 0
            axes[0].scatter(time_index[anomaly_points], channel_data.test_raw[anomaly_points], s=8, color="#dc2626", label="label")
        axes[0].set_title(f"{channel_data.channel_id} telemetry")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        # Panel 2: Counterfactual successor score & calibrated threshold
        axes[1].plot(smoothed_scores, color="#2563eb", linewidth=1.0, label="successor event score")
        axes[1].axhline(threshold, color="#dc2626", linestyle="--", linewidth=1.0, label="threshold")
        axes[1].set_title("Counterfactual successor score")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        # Panel 3: Point-level detections vs labels
        axes[2].plot(predictions, color="#059669", drawstyle="steps-post", label="prediction")
        axes[2].plot(labels[: len(predictions)], color="#dc2626", alpha=0.45, drawstyle="steps-post", label="label")
        axes[2].set_ylim(-0.05, 1.15)
        axes[2].set_title("Point-level detections")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

        # Panel 4: Context substitution diagnostics
        axes[3].plot(context_ood_map, color="#7c3aed", drawstyle="steps-post", label="context substitution")
        axes[3].set_ylim(-0.05, 1.15)
        axes[3].set_title("Context substitution mapping (OOD)")
        axes[3].legend(loc="upper right")
        axes[3].set_xlabel("time step")
        axes[3].grid(True, alpha=0.3)

        fig.tight_layout()
        output_path = channel_dir / "diagnostics.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"Saved diagnostic plot to {output_path}")
    except Exception as e:
        logger.error(f"Failed to generate diagnostic plot for {channel_data.channel_id}: {e}", exc_info=True)
