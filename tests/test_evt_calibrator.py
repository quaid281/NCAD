"""Unit tests for EVT / SPOT adaptive threshold calibrator."""

import numpy as np
import pytest

from src.scoring.evt_calibrator import (
    EVTCalibrator,
    _grimshaw_gpd_fit,
    _mle_gpd_fit,
    fit_gpd,
)


def test_fit_gpd_exponential():
    """Test GPD fit on exponential tail (gamma ~ 0)."""
    np.random.seed(42)
    # Standard normal scores squared (chi-square 1, exponential tail)
    raw_scores = np.random.randn(2000) ** 2
    fit = fit_gpd(raw_scores, init_percentile=95.0)

    assert fit.fit_converged
    assert fit.n_excess > 50
    assert fit.sigma > 0
    # For exponential tails, gamma should be close to 0 (typically between -0.2 and 0.2)
    assert abs(fit.gamma) < 0.4


def test_fit_gpd_heavy_tailed_pareto():
    """Test GPD fit on heavy-tailed Pareto distribution (gamma > 0)."""
    np.random.seed(42)
    # Pareto samples: shape alpha = 2.0 (gamma = 1/alpha = 0.5)
    pareto_samples = (np.random.pareto(a=2.0, size=2000) + 1.0)
    fit = fit_gpd(pareto_samples, init_percentile=95.0)

    assert fit.fit_converged
    assert fit.sigma > 0
    assert fit.gamma > 0.0  # Heavy tail produces positive gamma


def test_evt_calibrator_threshold_monotonicity():
    """Higher risk level q (e.g. 0.05) should produce a lower threshold than lower risk level q (e.g. 1e-4)."""
    np.random.seed(42)
    scores = np.random.exponential(scale=2.0, size=3000)

    calibrator = EVTCalibrator(init_percentile=95.0)
    calibrator.fit(scores)

    res_loose = calibrator.compute_threshold(risk_level=0.01)
    res_tight = calibrator.compute_threshold(risk_level=1e-4)

    assert res_tight.threshold > res_loose.threshold
    assert res_loose.threshold >= calibrator.gpd_fit_.threshold_init


def test_evt_calibrator_probability_calibration():
    """Test that anomaly probabilities are bounded in [0, 1] and monotonic with score."""
    np.random.seed(42)
    scores = np.random.normal(loc=0.0, scale=1.0, size=2000) ** 2
    calibrator = EVTCalibrator(risk_level=1e-3, init_percentile=95.0)
    calibrator.fit(scores)

    test_scores = np.array([0.1, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0])
    p_anomaly = calibrator.predict_anomaly_probability(test_scores)

    assert np.all(p_anomaly >= 0.0)
    assert np.all(p_anomaly <= 1.0)
    # Probabilities should be strictly increasing with score
    assert np.all(np.diff(p_anomaly) >= 0.0)
    # Extreme outlier (score=50.0) should have high anomaly probability (> 0.99)
    assert p_anomaly[-1] > 0.99


def test_evt_calibrator_degenerate_flatline():
    """Test robust fallback on constant / near-zero variance scores."""
    scores = np.zeros(500, dtype=np.float64)
    calibrator = EVTCalibrator(risk_level=1e-3)
    calibrator.fit(scores)

    res = calibrator.compute_threshold()
    assert res.threshold >= calibrator.degenerate_epsilon
    assert np.isfinite(res.threshold)
