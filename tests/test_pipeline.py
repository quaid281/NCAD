import pytest
import numpy as np

from src.data.pipeline import NCADPipeline


def test_pipeline_univariate():
    pipeline = NCADPipeline(max_features_per_channel=8)
    
    # Generate mock training dataset (univariate)
    train_data = np.sin(np.linspace(0, 10, 500)).astype(np.float32)
    test_data = np.sin(np.linspace(10, 15, 300)).astype(np.float32)
    
    # Fit and prepare training windows
    # Window size: 100, step: 2
    train_windows = pipeline.fit_prepare_train(train_data, window_size=100, step=2)
    
    # Expected number of windows: (500 - 100) // 2 + 1 = 201
    # Expected features shape per step: 8
    assert train_windows.shape == (201, 100, 8)
    
    # Transform test dataset using fitted pipeline parameters
    test_windows = pipeline.prepare_windows(test_data, window_size=100, step=2)
    
    # Expected test windows: (300 - 100) // 2 + 1 = 101
    assert test_windows.shape == (101, 100, 8)


def test_pipeline_multivariate():
    pipeline = NCADPipeline(max_features_per_channel=4)
    
    # Mock multivariate dataset with 3 features
    train_data = np.random.randn(500, 3).astype(np.float32)
    test_data = np.random.randn(300, 3).astype(np.float32)
    
    # Select columns 0 and 2
    signal_indices = [0, 2]
    
    train_windows = pipeline.fit_prepare_train(train_data, window_size=100, step=5, signal_indices=signal_indices)
    
    # Concatenated feature dimension: 2 columns * 4 features each = 8 features
    assert train_windows.shape == (81, 100, 8)
    
    test_windows = pipeline.prepare_windows(test_data, window_size=100, step=5, signal_indices=signal_indices)
    
    assert test_windows.shape == (41, 100, 8)


def test_pipeline_zero_copy():
    pipeline = NCADPipeline(max_features_per_channel=4)
    data = np.random.randn(200, 2).astype(np.float32)
    
    windows_view = pipeline.fit_prepare_train(data, window_size=50, step=1, copy=False)
    assert windows_view.flags.owndata is False
    assert windows_view.shape == (151, 50, 8)
    
    test_view = pipeline.prepare_windows(data, window_size=50, step=1, copy=False)
    assert test_view.flags.owndata is False
    assert test_view.shape == (151, 50, 8)

