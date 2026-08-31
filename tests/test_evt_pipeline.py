"""End-to-end pipeline test comparing EVT threshold calibration with legacy heuristics."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.data_loader import ChannelData, NormalizationStats
from src.models.train_model import CSMConfig, run_channel


def create_synthetic_channel() -> ChannelData:
    """Create a synthetic telemetry channel with nominal training and anomalous test segment."""
    np.random.seed(42)
    # Train: 800 timesteps of clean harmonic signal + Gaussian noise
    t_train = np.linspace(0, 8 * np.pi, 800)
    train_clean = np.sin(t_train) + 0.1 * np.random.randn(800)

    # Test: 1000 timesteps with sustained anomalous burst at [600:750]
    t_test = np.linspace(0, 10 * np.pi, 1000)
    test_raw = np.sin(t_test) + 0.1 * np.random.randn(1000)
    # Inject sustained square shift / anomaly burst
    test_raw[600:750] += 2.5

    labels = np.zeros(1000, dtype=np.float32)
    labels[600:750] = 1.0

    mean = float(np.mean(train_clean))
    std = float(np.std(train_clean))
    norm_stats = NormalizationStats(mean=mean, std=std)

    return ChannelData(
        channel_id="SYNTH-1",
        train_raw=train_clean.astype(np.float32),
        test_raw=test_raw.astype(np.float32),
        train_normalized=norm_stats.transform(train_clean),
        test_normalized=norm_stats.transform(test_raw),
        labels=labels,
        anomaly_sequences=[(600, 750)],
        norm_stats=norm_stats,
    )


def test_evt_vs_legacy_thresholding(tmp_path: Path):
    """Verify that EVT calibrator correctly runs through the training pipeline and achieves high F1."""
    channel_data = create_synthetic_channel()
    device = torch.device("cpu")

    # Run 1: EVT Adaptive Calibration
    evt_dir = tmp_path / "evt_run"
    config_evt = CSMConfig(
        model_type="ncad",
        epochs=3,
        batch_size=16,
        filters=32,
        tcn_layers=2,
        context_size=64,
        suspect_size=16,
        threshold_method="evt",
        evt_risk_level=1e-3,
        evt_init_percentile=95.0,
        max_train_windows=200,
        save_plots=False,
    )
    result_evt = run_channel(channel_data, evt_dir, config_evt, device)

    assert "metrics" in result_evt
    metrics_evt = result_evt["metrics"]
    assert metrics_evt["f1"] > 0.35
    assert metrics_evt["tp"] > 40
    assert result_evt["calibration"]["threshold_method"].startswith("evt_gpd")
    assert result_evt["calibration"]["evt_details"] is not None

    # Check that point_predictions.csv contains calibrated anomaly probabilities
    pred_file = evt_dir / "SYNTH-1" / "point_predictions.csv"
    assert pred_file.exists()
    import pandas as pd
    df = pd.read_csv(pred_file)
    assert "anomaly_probability" in df.columns
    assert np.all(df["anomaly_probability"] >= 0.0)
    assert np.all(df["anomaly_probability"] <= 1.0)
    # The anomalous interval should have high anomaly probability
    anomaly_prob_slice = df["anomaly_probability"].iloc[620:730]
    assert anomaly_prob_slice.mean() > 0.70

    # Run 2: Legacy Adaptive Elbow Floor (backward compatibility verification)
    legacy_dir = tmp_path / "legacy_run"
    config_legacy = CSMConfig(
        model_type="ncad",
        epochs=3,
        batch_size=16,
        filters=32,
        tcn_layers=2,
        context_size=64,
        suspect_size=16,
        threshold_method="adaptive_elbow",
        max_train_windows=200,
        save_plots=False,
    )
    result_legacy = run_channel(channel_data, legacy_dir, config_legacy, device)
    assert "metrics" in result_legacy
    assert result_legacy["calibration"]["threshold_method"].startswith("counterfactual_successor_training")
