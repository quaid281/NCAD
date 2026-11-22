import numpy as np
import pandas as pd
import pytest

from src.data.data_loader import DataLoader, NormalizationStats


def test_normalization_stats():
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    stats = NormalizationStats(mean=2.0, std=1.0)

    transformed = stats.transform(values)
    assert np.allclose(transformed, np.array([-1.0, 0.0, 1.0], dtype=np.float32))

    reverted = stats.inverse_transform(transformed)
    assert np.allclose(reverted, values)


def test_create_windows():
    # 1D array
    values = np.arange(10, dtype=np.float32)
    windows = DataLoader.create_windows(values, window_size=4, step=2)

    # Expected shape: ((10 - 4)//2 + 1, 4, 1) = (4, 4, 1)
    assert windows.shape == (4, 4, 1)
    assert np.allclose(windows[0, :, 0], [0, 1, 2, 3])
    assert np.allclose(windows[1, :, 0], [2, 3, 4, 5])


def test_create_windows_zero_copy():
    values = np.arange(20, dtype=np.float32)
    windows_copy = DataLoader.create_windows(values, window_size=5, step=2, copy=True)
    windows_view = DataLoader.create_windows(values, window_size=5, step=2, copy=False)

    assert np.allclose(windows_copy, windows_view)
    # Verify that windows_view shares underlying memory buffer
    assert windows_view.base is not None


def test_data_loader_load(tmp_path):
    # Setup mock data directory
    raw_dir = tmp_path / "raw"
    train_dir = raw_dir / "train"
    test_dir = raw_dir / "test"
    processed_dir = tmp_path / "processed"

    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    # Create a 2D dataset with 3 columns
    train_array = np.random.randn(100, 3).astype(np.float32)
    test_array = np.random.randn(150, 3).astype(np.float32)

    np.save(train_dir / "MOCK-1.npy", train_array)
    np.save(test_dir / "MOCK-1.npy", test_array)

    # Create labels file
    labels_df = pd.DataFrame([
        {
            "chan_id": "MOCK-1",
            "num_values": 150,
            "anomaly_sequences": "[[10, 20]]"
        }
    ])
    labels_df.to_csv(processed_dir / "labeled_anomalies.csv", index=False)

    # Instantiate DataLoader
    loader = DataLoader(data_dir=tmp_path)

    # Verify channel listing
    channels = loader.list_channels()
    assert channels == ["MOCK-1"]

    # Load channel (default signal index 0)
    channel_data_0 = loader.load_channel("MOCK-1", normalize=True, signal_index=0)
    assert channel_data_0.channel_id == "MOCK-1"
    assert channel_data_0.train_raw.shape == (100,)
    assert channel_data_0.test_raw.shape == (150,)
    assert np.allclose(channel_data_0.train_raw, train_array[:, 0])

    # Verify labels loading
    assert channel_data_0.labels.shape == (150,)
    assert channel_data_0.labels[15] == 1.0
    assert channel_data_0.labels[5] == 0.0

    # Load channel with signal index 2
    channel_data_2 = loader.load_channel("MOCK-1", normalize=True, signal_index=2)
    assert np.allclose(channel_data_2.train_raw, train_array[:, 2])

    # Expect error for invalid index
    with pytest.raises(ValueError):
        loader.load_channel("MOCK-1", normalize=True, signal_index=5)


def test_non_finite_input_guards(tmp_path):
    """Verify that passing NaN or Inf to DataLoader or create_windows raises ValueError."""
    # 1. create_windows guard
    arr_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        DataLoader.create_windows(arr_nan, window_size=3)

    arr_inf = np.array([1.0, 2.0, np.inf, 4.0, 5.0, 6.0], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        DataLoader.create_windows(arr_inf, window_size=3)

    # 2. load_channel guard
    raw_dir = tmp_path / "raw"
    train_dir = raw_dir / "train"
    test_dir = raw_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    np.save(train_dir / "NAN_CHAN.npy", arr_nan)
    np.save(test_dir / "NAN_CHAN.npy", np.ones(6, dtype=np.float32))

    dl = DataLoader(data_dir=tmp_path)
    with pytest.raises(ValueError, match="non-finite"):
        dl.load_channel("NAN_CHAN")


def test_malformed_label_metadata():
    """Verify that malformed anomaly_sequences raise ValueError instead of silently returning empty labels."""
    dl = DataLoader()

    # Invalid Python syntax in string
    with pytest.raises(ValueError, match="Malformed anomaly_sequences"):
        dl._parse_anomaly_sequences("[(10, 20], [30, 40)", channel_id="CHAN-1")

    # Non-list structure
    with pytest.raises(ValueError, match="Expected list"):
        dl._parse_anomaly_sequences("12345", channel_id="CHAN-1")

    # Valid empty or nan strings return clean empty lists
    assert dl._parse_anomaly_sequences("[]") == []
    assert dl._parse_anomaly_sequences("") == []
    assert dl._parse_anomaly_sequences("nan") == []


def test_endpoint_semantics():
    """Verify that anomaly intervals follow strict half-open [start, end) semantics."""
    dl = DataLoader()
    mock_df = pd.DataFrame([{"chan_id": "MOCK-1", "num_values": 50, "anomaly_sequences": "[(10, 20)]"}])
    dl._labels_df = mock_df

    labels, sequences = dl._load_labels("MOCK-1", test_length=50)

    assert len(labels) == 50
    assert sequences == [(10, 20)]
    # Points inside [10, 20) must be 1.0
    assert labels[10] == 1.0
    assert labels[19] == 1.0
    # Boundary points outside [10, 20) must be 0.0
    assert labels[9] == 0.0
    assert labels[20] == 0.0


def test_window_parameter_validation():
    """Verify create_windows enforces positive window_size, positive step, and valid dimensions."""
    values = np.random.randn(100).astype(np.float32)

    with pytest.raises(ValueError, match="window_size must be positive"):
        DataLoader.create_windows(values, window_size=0)

    with pytest.raises(ValueError, match="step must be positive"):
        DataLoader.create_windows(values, window_size=10, step=-1)

    with pytest.raises(ValueError, match="Expected 1D or 2D"):
        DataLoader.create_windows(np.zeros((10, 10, 10), dtype=np.float32), window_size=5)


def test_purged_train_val_isolation():
    """Verify that training and validation partitions have zero raw sample overlap."""
    from src.engine.trainer import split_train_validation

    n_points = 300
    window_size = 40
    step = 2
    raw_series = np.arange(n_points, dtype=np.float32)
    windows = DataLoader.create_windows(raw_series, window_size=window_size, step=step)

    train_windows, val_windows = split_train_validation(
        windows,
        val_split=0.15,
        seed=42,
        window_size=window_size,
        step=step,
    )

    assert len(train_windows) > 0
    assert len(val_windows) > 0

    max_train_sample = train_windows.max()
    min_val_sample = val_windows.min()

    # Zero sample overlap: min_val_sample must be strictly greater than max_train_sample
    assert min_val_sample > max_train_sample, (
        f"Train/Val leakage detected: max train sample {max_train_sample} >= min val sample {min_val_sample}"
    )


def test_uniform_window_subsampling():
    """Verify limit_windows samples uniformly across the full temporal extent."""
    from src.engine.trainer import limit_windows

    windows = np.arange(100)[:, None, None]
    limited = limit_windows(windows, max_windows=10)

    assert len(limited) == 10
    assert limited[0, 0, 0] == 0
    assert limited[-1, 0, 0] == 99
    diffs = np.diff(limited[:, 0, 0])
    assert np.all(diffs >= 9) and np.all(diffs <= 11)

