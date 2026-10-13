"""Unit tests for Online Streaming Calibrators (Rolling SPOT & Streaming Quantiles).

Validates:
1. Stationary baseline calibration matching theoretical design risk.
2. Inter-block distribution drift adaptation (mean shifts, variance scale).
3. Anomaly contamination gating preventing threshold inflation.
4. Equivalence of point-by-point update() vs process_stream().
5. Edge cases: short sequences, empty inputs, non-finite values, and parameter validation.
"""

import math
import numpy as np
import pytest

from src.scoring.streaming_calibrator import (
    RollingSPOTCalibrator,
    StreamingQuantileCalibrator,
    compute_rolling_spot_thresholds,
    compute_streaming_quantile_thresholds,
)


def test_streaming_quantile_stationary():
    """Verify StreamingQuantileCalibrator estimates correct quantile on stationary data."""
    np.random.seed(42)
    n = 2000
    scores = np.random.exponential(scale=1.0, size=n)
    q = 0.01  # 99th percentile

    cal = StreamingQuantileCalibrator(q=q, window_size=1000, method="trailing_window")
    thresholds = cal.process_stream(scores)

    assert len(thresholds) == n
    assert np.all(np.isfinite(thresholds))

    # After warmup, trailing quantile should be close to theoretical -log(0.01) ~= 4.605
    final_thr = thresholds[-1]
    assert 3.5 < final_thr < 6.0


def test_streaming_quantile_drift_adaptation():
    """Verify StreamingQuantileCalibrator tracks sudden distribution shift."""
    np.random.seed(42)
    # Block 1: mean 2.0, Block 2: mean 8.0 (inter-block drift)
    blk1 = np.random.normal(loc=2.0, scale=0.5, size=500)
    blk2 = np.random.normal(loc=8.0, scale=0.5, size=500)
    stream = np.concatenate([blk1, blk2])

    cal = StreamingQuantileCalibrator(q=0.01, window_size=200, step=1)
    thresholds = cal.process_stream(stream)

    thr_early = thresholds[450]   # Before shift
    thr_late = thresholds[950]    # Well after shift

    assert thr_early < 4.0
    assert thr_late > 7.0
    assert thr_late > thr_early + 3.0


def test_streaming_quantile_stochastic_approximation():
    """Verify Robbins-Monro stochastic approximation mode operates cleanly."""
    np.random.seed(42)
    scores = np.random.exponential(scale=2.0, size=1000)
    cal = StreamingQuantileCalibrator(q=0.05, method="stochastic_approximation", learning_rate=0.1)
    thresholds = cal.process_stream(scores)

    assert len(thresholds) == 1000
    assert np.all(np.isfinite(thresholds))
    assert thresholds[-1] > 2.0


def test_rolling_spot_stationary_and_gpd_fit():
    """Verify RollingSPOTCalibrator fits GPD and computes extreme quantile thresholds."""
    np.random.seed(42)
    # Heavy-tailed Pareto distributed scores
    scores = (np.random.pareto(a=3.0, size=1500) + 1.0) * 0.5
    cal = RollingSPOTCalibrator(
        risk_level=1e-3,
        init_percentile=95.0,
        window_size=800,
        step=50,
        min_periods=100,
    )
    thresholds = cal.process_stream(scores)

    assert len(thresholds) == 1500
    assert np.all(np.isfinite(thresholds))

    diag = cal.get_diagnostics()
    assert diag["total_points"] == 1500
    assert diag["buffer_len"] == 800
    assert diag["fit_converged"] is True
    assert diag["current_threshold"] > float(np.percentile(scores[-800:], 95.0))


def test_rolling_spot_inter_block_drift_tracking():
    """Verify RollingSPOTCalibrator adapts threshold when variance dilates."""
    np.random.seed(42)
    # Regime 1: normal scale 1.0; Regime 2: normal scale 3.0 (3x variance scale)
    regime1 = np.abs(np.random.normal(loc=1.0, scale=1.0, size=600))
    regime2 = np.abs(np.random.normal(loc=3.0, scale=3.0, size=600))
    stream = np.concatenate([regime1, regime2])

    cal = RollingSPOTCalibrator(
        risk_level=1e-3,
        init_percentile=95.0,
        window_size=300,
        step=20,
        min_periods=50,
    )
    thresholds = cal.process_stream(stream)

    thr_regime1 = thresholds[550]
    thr_regime2 = thresholds[1150]

    assert thr_regime2 > thr_regime1 * 1.5


def test_anomaly_contamination_gating():
    """Verify that contamination gating prevents the 'Hazard of Naive Rolling SPOT'."""
    np.random.seed(42)
    # Clean baseline
    nominal = np.random.exponential(scale=1.0, size=500)
    # Sustained anomaly burst (score surges by 15x)
    anomaly_burst = np.random.uniform(low=15.0, high=25.0, size=100)
    # Return to normal
    post_anom = np.random.exponential(scale=1.0, size=200)

    stream = np.concatenate([nominal, anomaly_burst, post_anom])

    # 1. Unconstrained naive rolling SPOT (gated=False)
    cal_naive = RollingSPOTCalibrator(
        risk_level=1e-3,
        init_percentile=95.0,
        window_size=300,
        step=10,
        gated=False,
    )
    thrs_naive = cal_naive.process_stream(stream)

    # 2. Gated rolling SPOT (gated=True)
    cal_gated = RollingSPOTCalibrator(
        risk_level=1e-3,
        init_percentile=95.0,
        window_size=300,
        step=10,
        gated=True,
        gate_mode="filter",
    )
    thrs_gated = cal_gated.process_stream(stream)

    # In naive mode, the anomaly burst enters the buffer and severely inflates the threshold
    diag_naive = cal_naive.get_diagnostics()
    diag_gated = cal_gated.get_diagnostics()

    assert diag_gated["gated_points"] > 50  # Filtered out anomaly points
    assert diag_naive["gated_points"] == 0

    # During/immediately after burst, naive threshold should be much higher than gated
    assert thrs_naive[580] > thrs_gated[580] * 2.0


def test_sequential_update_matches_batch_stream():
    """Verify point-by-point .update() identically matches .process_stream()."""
    np.random.seed(123)
    scores = np.random.uniform(0.1, 5.0, size=300)

    # Batch
    cal1 = RollingSPOTCalibrator(risk_level=1e-2, window_size=100, step=10, min_periods=30)
    batch_thrs = cal1.process_stream(scores)

    # Sequential point-by-point
    cal2 = RollingSPOTCalibrator(risk_level=1e-2, window_size=100, step=10, min_periods=30)
    seq_thrs = np.array([cal2.update(s) for s in scores])

    assert np.allclose(batch_thrs, seq_thrs)


def test_convenience_functions_and_warm_start():
    """Verify compute_rolling_spot_thresholds and compute_streaming_quantile_thresholds with pre-fit."""
    np.random.seed(42)
    train_scores = np.random.exponential(scale=1.0, size=300)
    test_scores = np.random.exponential(scale=1.0, size=200)

    # Rolling SPOT
    thrs_spot = compute_rolling_spot_thresholds(
        test_scores=test_scores,
        train_scores=train_scores,
        risk_level=1e-3,
        window_size=200,
        step=20,
    )
    assert len(thrs_spot) == 200
    assert np.all(np.isfinite(thrs_spot))

    # Streaming Quantile
    thrs_q = compute_streaming_quantile_thresholds(
        test_scores=test_scores,
        train_scores=train_scores,
        q=1e-3,
        window_size=200,
    )
    assert len(thrs_q) == 200
    assert np.all(np.isfinite(thrs_q))


def test_edge_cases_and_validation():
    """Verify input validation, empty streams, and NaN handling."""
    cal = RollingSPOTCalibrator()

    # Empty array
    empty_res = cal.process_stream(np.array([]))
    assert len(empty_res) == 0

    # Non-finite scores ignored gracefully
    nan_thr = cal.update(float("nan"))
    assert math.isfinite(nan_thr)

    # Invalid risk levels
    with pytest.raises(ValueError):
        RollingSPOTCalibrator(risk_level=-0.1)
    with pytest.raises(ValueError):
        StreamingQuantileCalibrator(q=1.5)
    with pytest.raises(ValueError):
        StreamingQuantileCalibrator(method="invalid_method")
