import pytest
import numpy as np
import pandas as pd
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


def test_sliding_window_dataset():
    from src.data.data_loader import SlidingWindowDataset
    import torch
    
    values = np.arange(20, dtype=np.float32).reshape(10, 2)
    dataset = SlidingWindowDataset(values, window_size=4, step=2)
    
    assert len(dataset) == (10 - 4) // 2 + 1  # 4 windows
    
    sample_0 = dataset[0]
    assert isinstance(sample_0, torch.Tensor)
    assert sample_0.shape == (4, 2)
    assert np.allclose(sample_0.numpy(), values[:4])
    
    sample_1 = dataset[1]
    assert np.allclose(sample_1.numpy(), values[2:6])


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
