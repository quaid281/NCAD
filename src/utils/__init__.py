"""Utilities package for NCAD-CS."""

from src.utils.event_fusion import (
    AdaptiveScoreFloor,
    RobustStats,
    adaptive_elbow_score_floor,
    aggregate_window_scores,
    calibrate_evt_threshold,
    compute_metrics,
    confidence_over_threshold,
    dispersion_confidence,
    dynamic_weighted_smoothing,
    event_level_filter,
    fuse_evidence_scores,
    local_deviation_scores,
    moving_average,
    percentile_score_floor,
    positive_robust_z,
    robust_dispersion_floor,
    robust_stats,
    successor_manifold_uncertainty_scores,
)
from src.utils.evt_calibrator import EVTCalibrator, EVTThresholdResult
from src.utils.logging_utils import setup_logging
from src.utils.plotting import plot_channel_diagnostics

__all__ = [
    "AdaptiveScoreFloor",
    "RobustStats",
    "EVTCalibrator",
    "EVTThresholdResult",
    "adaptive_elbow_score_floor",
    "aggregate_window_scores",
    "calibrate_evt_threshold",
    "compute_metrics",
    "confidence_over_threshold",
    "dispersion_confidence",
    "dynamic_weighted_smoothing",
    "event_level_filter",
    "fuse_evidence_scores",
    "local_deviation_scores",
    "moving_average",
    "percentile_score_floor",
    "positive_robust_z",
    "robust_dispersion_floor",
    "robust_stats",
    "successor_manifold_uncertainty_scores",
    "setup_logging",
    "plot_channel_diagnostics",
]
