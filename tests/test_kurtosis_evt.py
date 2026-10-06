"""Tests for Kurtosis-Adaptive Extreme Value Theory (EVT) Calibrator."""

import math
import numpy as np
import pytest

from src.scoring.evt_calibrator import (
    EVTCalibrator,
    compute_excess_kurtosis,
    kurtosis_adaptive_params,
)
from src.scoring.event_fusion import calibrate_evt_threshold


def test_compute_excess_kurtosis_distributions():
    """Verify excess kurtosis computation across normal, uniform, and spike distributions."""
    np.random.seed(42)
    N = 10000

    # 1. Normal distribution: excess kurtosis ~ 0
    normal_data = np.random.normal(loc=0.0, scale=1.0, size=N)
    k_normal = compute_excess_kurtosis(normal_data)
    assert abs(k_normal) < 0.5, f"Expected near-zero kurtosis for normal, got {k_normal}"

    # 2. Uniform distribution: excess kurtosis ~ -1.2
    uniform_data = np.random.uniform(low=-1.0, high=1.0, size=N)
    k_uniform = compute_excess_kurtosis(uniform_data)
    assert -1.5 < k_uniform < -0.9, f"Expected ~ -1.2 for uniform, got {k_uniform}"

    # 3. Leptokurtic distribution: normal background + 1% massive spikes
    spike_data = np.random.normal(loc=0.0, scale=1.0, size=N)
    spike_indices = np.random.choice(N, size=int(0.01 * N), replace=False)
    spike_data[spike_indices] += 25.0  # massive anomaly spikes
    k_spike = compute_excess_kurtosis(spike_data)
    assert k_spike > 20.0, f"Expected high kurtosis for spike data, got {k_spike}"


def test_kurtosis_adaptive_params():
    """Verify that kurtosis_adaptive_params tunes percentile and risk level based on peakedness."""
    # Leptokurtic scores with high kurtosis
    scores_spike = np.concatenate([np.random.normal(0, 1, 5000), np.full(50, 30.0)])
    p_adapt, q_adapt, kurt = kurtosis_adaptive_params(scores_spike, base_risk=1e-3, base_percentile=95.0)

    assert kurt > 10.0
    assert p_adapt > 96.0  # init_percentile raised
    assert q_adapt < 1e-3  # risk level tightened

    # Flat scores with negative kurtosis
    scores_flat = np.random.uniform(0, 1, 5000)
    p_flat, q_flat, kurt_flat = kurtosis_adaptive_params(scores_flat, base_risk=1e-3, base_percentile=95.0)

    assert kurt_flat < -0.5
    assert p_flat <= 95.0
    assert q_flat == 1e-3


def test_evt_calibrator_adaptive_vs_static():
    """Verify that adaptive EVT prevents false alarm flood on heavy-tailed distributions."""
    np.random.seed(42)
    # 5000 normal points + 20 distinct extreme spikes
    normal_bulk = np.random.exponential(scale=1.0, size=5000)
    scores = np.concatenate([normal_bulk, np.random.uniform(20.0, 30.0, size=20)])

    # Static calibrator
    cal_static = EVTCalibrator(risk_level=1e-3, init_percentile=95.0, adaptive_kurtosis=False)
    cal_static.fit(scores)
    th_static = cal_static.threshold_

    # Adaptive calibrator
    cal_adaptive = EVTCalibrator(risk_level=1e-3, init_percentile=95.0, adaptive_kurtosis=True)
    cal_adaptive.fit(scores)
    th_adaptive = cal_adaptive.threshold_

    # The adaptive calibrator should detect high kurtosis and set a sharper tail threshold
    assert cal_adaptive.kurtosis_ > 10.0
    assert cal_adaptive.effective_init_percentile_ > 95.0
    assert th_adaptive > th_static


def test_calibrate_evt_threshold_event_fusion_wrapper():
    """Verify that calibrate_evt_threshold wrapper passes adaptive_kurtosis cleanly."""
    scores = np.random.exponential(scale=1.0, size=1000)
    res = calibrate_evt_threshold(scores, risk_level=1e-3, init_percentile=95.0, adaptive_kurtosis=True)

    assert res.threshold > 0.0
    assert res.kurtosis is not None
    assert res.effective_init_percentile is not None
    d = res.to_dict()
    assert "kurtosis" in d
    assert "effective_init_percentile" in d
