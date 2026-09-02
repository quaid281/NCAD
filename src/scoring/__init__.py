"""Core scoring, calibration, and event-level fusion logic.

This subpackage contains the statistical scoring pipeline:
- EVT/GPD tail calibration (``evt_calibrator``)
- Window-to-point score aggregation and event-level filtering (``event_fusion``)
"""

from src.scoring.event_fusion import (
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
from src.scoring.conformal_calibrator import ConformalThresholdResult, SplitConformalCalibrator
from src.scoring.evt_calibrator import EVTCalibrator, EVTThresholdResult

__all__ = [
    "AdaptiveScoreFloor",
    "RobustStats",
    "ConformalThresholdResult",
    "SplitConformalCalibrator",
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
]
