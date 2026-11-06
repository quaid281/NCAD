"""Core scoring, calibration, and event-level fusion logic.

This subpackage contains the statistical scoring pipeline:
- 224-Bit Multi-Perspective Circuit-Breaker Panel (``circuit_breaker_panel``)
- EVT/GPD tail calibration (``evt_calibrator``)
- Novel drift-resilient calibrators (``novel_calibrators``)
- Window-to-point score aggregation and event-level filtering (``event_fusion``)
"""

from src.scoring.circuit_breaker_panel import (
    CircuitBreakerStreamDecision,
    MultiPerspectiveCircuitBreakerPanel,
    aggregate_prefix_224_stream,
    aggregate_vector_stream,
    build_224_perspective_dials,
    extract_multihorizon_4subblocks,
    stabilize_whitened_subblocks,
)
from src.scoring.conformal_calibrator import ConformalThresholdResult, SplitConformalCalibrator
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
from src.scoring.evt_calibrator import EVTCalibrator, EVTThresholdResult
from src.scoring.novel_calibrators import (
    HalpernPrimalDualCalibrator,
    PMIConformalCalibrator,
    StoppingBandCalibrator,
)
from src.scoring.streaming_calibrator import (
    RollingSPOTCalibrator,
    StreamingQuantileCalibrator,
    compute_rolling_spot_thresholds,
    compute_streaming_quantile_thresholds,
)

__all__ = [
    "CircuitBreakerStreamDecision",
    "MultiPerspectiveCircuitBreakerPanel",
    "aggregate_prefix_224_stream",
    "aggregate_vector_stream",
    "build_224_perspective_dials",
    "extract_multihorizon_4subblocks",
    "stabilize_whitened_subblocks",
    "AdaptiveScoreFloor",
    "RobustStats",
    "ConformalThresholdResult",
    "SplitConformalCalibrator",
    "EVTCalibrator",
    "EVTThresholdResult",
    "HalpernPrimalDualCalibrator",
    "PMIConformalCalibrator",
    "StoppingBandCalibrator",
    "RollingSPOTCalibrator",
    "StreamingQuantileCalibrator",
    "compute_rolling_spot_thresholds",
    "compute_streaming_quantile_thresholds",
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
