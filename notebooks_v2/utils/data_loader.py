"""
Data Loader for Mamba Anomaly Classifier

Simple, clean data loading utilities for raw univariate time series.
No feature engineering - just the signal.

Design Philosophy:
    - Load raw signals directly from .npy files
    - Simple z-score normalization (fit on train, transform both)
    - Efficient sliding window creation using stride tricks
    - Ground truth label loading for evaluation

Data Structure Expected:
    data/
    ├── raw/
    │   ├── train/
    │   │   ├── A-1.npy    # Shape: (n_samples,) or (n_samples, n_features)
    │   │   └── ...
    │   └── test/
    │       ├── A-1.npy
    │       └── ...
    └── processed/
        └── labeled_anomalies.csv  # Ground truth labels
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Union
from dataclasses import dataclass
from pathlib import Path


@dataclass
class NormalizationStats:
    """Statistics for z-score normalization."""
    mean: float
    std: float
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization."""
        return (data - self.mean) / (self.std + 1e-8)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse normalization."""
        return data * (self.std + 1e-8) + self.mean


@dataclass  
class ChannelData:
    """Container for a single channel's data."""
    channel_id: str
    train_raw: np.ndarray          # Raw training signal
    test_raw: np.ndarray           # Raw test signal
    train_normalized: np.ndarray   # Normalized training signal
    test_normalized: np.ndarray    # Normalized test signal
    labels: Optional[np.ndarray]   # Ground truth labels (if available)
    norm_stats: NormalizationStats # Normalization statistics
    anomaly_sequences: List[Tuple[int, int]]  # List of (start, end) anomaly ranges


class DataLoader:
    """
    Data loader for time series anomaly detection.
    
    Handles:
        - Loading raw .npy files
        - Z-score normalization (fit on train)
        - Sliding window creation
        - Ground truth label loading
    
    Usage:
        loader = DataLoader(data_dir='../data')
        
        # Load a single channel
        channel_data = loader.load_channel('A-1')
        
        # Create training windows
        windows = loader.create_windows(channel_data.train_normalized, window_size=100)
        
        # Get list of available channels
        channels = loader.list_channels()
    """
    
    def __init__(
        self,
        data_dir: str = '../data',
        labels_file: str = 'labeled_anomalies.csv',
    ):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the data directory containing raw/ and processed/
            labels_file: Name of the CSV file with ground truth labels
        """
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / 'raw' / 'train'
        self.test_dir = self.data_dir / 'raw' / 'test'
        self.labels_path = self.data_dir / 'processed' / labels_file
        
        # Cache for loaded labels
        self._labels_df: Optional[pd.DataFrame] = None
    
    def list_channels(self) -> List[str]:
        """
        List all available channels.
        
        Returns:
            List of channel IDs (e.g., ['A-1', 'A-2', 'D-1', ...])
        """
        if not self.train_dir.exists():
            raise FileNotFoundError(f"Training data directory not found: {self.train_dir}")
        
        channels = []
        for f in self.train_dir.glob('*.npy'):
            channel_id = f.stem  # Remove .npy extension
            # Check if test file also exists
            test_file = self.test_dir / f.name
            if test_file.exists():
                channels.append(channel_id)
        
        return sorted(channels)
    
    def load_channel(
        self,
        channel_id: str,
        normalize: bool = True,
    ) -> ChannelData:
        """
        Load data for a single channel.
        
        Args:
            channel_id: Channel identifier (e.g., 'A-1')
            normalize: Whether to apply z-score normalization
            
        Returns:
            ChannelData object with train/test signals and labels
        """
        # Load raw data
        train_path = self.train_dir / f'{channel_id}.npy'
        test_path = self.test_dir / f'{channel_id}.npy'
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")
        
        train_full = np.load(train_path)
        test_full = np.load(test_path)
        
        # Extract univariate signal (first column if multi-dimensional)
        train_raw = self._extract_signal(train_full)
        test_raw = self._extract_signal(test_full)
        
        print(f"Loaded channel {channel_id}:")
        print(f"  Train: {len(train_raw):,} samples")
        print(f"  Test:  {len(test_raw):,} samples")
        
        # Normalize
        if normalize:
            norm_stats = NormalizationStats(
                mean=float(np.mean(train_raw)),
                std=float(np.std(train_raw))
            )
            train_normalized = norm_stats.transform(train_raw)
            test_normalized = norm_stats.transform(test_raw)
            print(f"  Normalization: mean={norm_stats.mean:.4f}, std={norm_stats.std:.4f}")
        else:
            norm_stats = NormalizationStats(mean=0.0, std=1.0)
            train_normalized = train_raw.copy()
            test_normalized = test_raw.copy()
        
        # Load ground truth labels
        labels, anomaly_sequences = self._load_labels(channel_id, len(test_raw))
        if labels is not None:
            n_anomalous = int(np.sum(labels))
            print(f"  Labels: {n_anomalous:,} anomalous points ({100*n_anomalous/len(labels):.2f}%)")
            print(f"  Anomaly sequences: {len(anomaly_sequences)}")
        else:
            print(f"  Labels: Not available")
        
        return ChannelData(
            channel_id=channel_id,
            train_raw=train_raw,
            test_raw=test_raw,
            train_normalized=train_normalized,
            test_normalized=test_normalized,
            labels=labels,
            norm_stats=norm_stats,
            anomaly_sequences=anomaly_sequences,
        )
    
    def _extract_signal(self, data: np.ndarray) -> np.ndarray:
        """Extract univariate signal from potentially multi-dimensional data."""
        if data.ndim == 1:
            return data.astype(np.float32)
        elif data.ndim == 2 and data.shape[1] > 0:
            # Take first column (primary telemetry signal)
            return data[:, 0].astype(np.float32)
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")
    
    def _load_labels(
        self,
        channel_id: str,
        test_length: int,
    ) -> Tuple[Optional[np.ndarray], List[Tuple[int, int]]]:
        """Load ground truth labels for a channel."""
        anomaly_sequences = []
        
        # Try to load labels CSV
        if self._labels_df is None:
            if self.labels_path.exists():
                try:
                    self._labels_df = pd.read_csv(self.labels_path)
                except Exception as e:
                    print(f"  Warning: Could not load labels file: {e}")
                    return None, anomaly_sequences
            else:
                # Try alternative path
                alt_path = self.data_dir / 'processed' / 'final_predictions.csv'
                if alt_path.exists():
                    try:
                        self._labels_df = pd.read_csv(alt_path)
                    except Exception as e:
                        print(f"  Warning: Could not load labels file: {e}")
                        return None, anomaly_sequences
                else:
                    return None, anomaly_sequences
        
        # Find channel in labels
        channel_info = self._labels_df[self._labels_df['chan_id'] == channel_id]
        
        if channel_info.empty:
            return None, anomaly_sequences
        
        # Parse anomaly sequences
        try:
            anomaly_seq_str = channel_info['anomaly_sequences'].iloc[0]
            if isinstance(anomaly_seq_str, str):
                anomaly_sequences = eval(anomaly_seq_str)
            else:
                anomaly_sequences = anomaly_seq_str if anomaly_seq_str else []
        except Exception:
            anomaly_sequences = []
        
        # Get expected length from labels file
        try:
            num_values = int(channel_info['num_values'].iloc[0])
        except Exception:
            num_values = test_length
        
        # Create binary label array
        labels = np.zeros(num_values, dtype=np.float32)
        
        for seq in anomaly_sequences:
            if isinstance(seq, (list, tuple)) and len(seq) == 2:
                start, end = int(seq[0]), int(seq[1])
                start = max(0, start)
                end = min(num_values, end)
                labels[start:end] = 1.0
            elif isinstance(seq, int):
                if 0 <= seq < num_values:
                    labels[seq] = 1.0
        
        # Adjust length if needed
        if len(labels) != test_length:
            if len(labels) < test_length:
                labels = np.pad(labels, (0, test_length - len(labels)), 'constant')
            else:
                labels = labels[:test_length]
        
        return labels, anomaly_sequences
    
    @staticmethod
    def create_windows(
        data: np.ndarray,
        window_size: int,
        step: int = 1,
    ) -> np.ndarray:
        """
        Create sliding windows from a 1D signal.
        
        Uses numpy stride tricks for memory efficiency.
        
        Args:
            data: 1D array of shape (n_samples,)
            window_size: Size of each window
            step: Step size between windows (1 = fully overlapping)
            
        Returns:
            Windows array of shape (n_windows, window_size)
        """
        if data.ndim != 1:
            raise ValueError(f"Expected 1D data, got shape {data.shape}")
        
        n_samples = len(data)
        
        if n_samples < window_size:
            raise ValueError(
                f"Data length ({n_samples}) < window size ({window_size})"
            )
        
        n_windows = (n_samples - window_size) // step + 1
        
        # Use stride tricks for efficient windowing
        data = np.ascontiguousarray(data)
        windows = np.lib.stride_tricks.as_strided(
            data,
            shape=(n_windows, window_size),
            strides=(step * data.strides[0], data.strides[0])
        )
        
        # Return a copy to avoid memory issues with strided arrays
        return windows.copy()
    
    @staticmethod
    def create_window_labels(
        labels: np.ndarray,
        window_size: int,
        step: int = 1,
        threshold: float = 0.0,
    ) -> np.ndarray:
        """
        Create window-level labels from point-level labels.
        
        A window is labeled anomalous if ANY point in it is anomalous.
        
        Args:
            labels: 1D binary labels array
            window_size: Size of each window
            step: Step size between windows
            threshold: Fraction of points that must be anomalous (0 = any)
            
        Returns:
            Window labels array of shape (n_windows,)
        """
        windows = DataLoader.create_windows(labels, window_size, step)
        
        if threshold == 0.0:
            # Any anomalous point -> anomalous window
            return (np.sum(windows, axis=1) > 0).astype(np.float32)
        else:
            # Fraction of points must exceed threshold
            return (np.mean(windows, axis=1) > threshold).astype(np.float32)


def get_channel_stats(loader: DataLoader, channels: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Get statistics for multiple channels.
    
    Args:
        loader: DataLoader instance
        channels: List of channel IDs (default: all available)
        
    Returns:
        DataFrame with channel statistics
    """
    if channels is None:
        channels = loader.list_channels()
    
    stats = []
    for channel_id in channels:
        try:
            data = loader.load_channel(channel_id)
            stats.append({
                'channel_id': channel_id,
                'train_length': len(data.train_raw),
                'test_length': len(data.test_raw),
                'train_mean': data.norm_stats.mean,
                'train_std': data.norm_stats.std,
                'n_anomalies': int(np.sum(data.labels)) if data.labels is not None else None,
                'anomaly_ratio': float(np.mean(data.labels)) if data.labels is not None else None,
                'n_sequences': len(data.anomaly_sequences),
            })
        except Exception as e:
            print(f"Error loading {channel_id}: {e}")
    
    return pd.DataFrame(stats)


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    print("Testing DataLoader...")
    
    # Initialize loader (assumes running from notebooks_v2/)
    loader = DataLoader(data_dir='../data')
    
    # List available channels
    try:
        channels = loader.list_channels()
        print(f"\nAvailable channels: {len(channels)}")
        print(f"First 5: {channels[:5]}")
    except FileNotFoundError as e:
        print(f"Data directory not found: {e}")
        print("Creating synthetic test data...")
        
        # Create synthetic data for testing
        np.random.seed(42)
        t = np.linspace(0, 100 * np.pi, 10000)
        train_data = np.sin(t) + np.random.randn(len(t)) * 0.1
        test_data = np.sin(t[:5000]) + np.random.randn(5000) * 0.1
        
        # Inject a synthetic anomaly
        test_data[2000:2200] += 2.0  # Level shift
        
        print(f"\nSynthetic train data: {train_data.shape}")
        print(f"Synthetic test data: {test_data.shape}")
        
        # Test windowing
        windows = DataLoader.create_windows(train_data, window_size=100, step=10)
        print(f"\nCreated windows: {windows.shape}")
        print(f"  Window size: 100")
        print(f"  Step: 10")
        print(f"  Number of windows: {windows.shape[0]}")
        
        print("\n✓ DataLoader test passed (with synthetic data)!")
        exit(0)
    
    # Load a channel
    if channels:
        channel_id = channels[0]
        print(f"\nLoading channel: {channel_id}")
        data = loader.load_channel(channel_id)
        
        # Test windowing
        window_size = 100
        step = 1
        windows = loader.create_windows(data.train_normalized, window_size, step)
        print(f"\nCreated training windows:")
        print(f"  Shape: {windows.shape}")
        print(f"  Window size: {window_size}")
        print(f"  Step: {step}")
        
        # Test window labels
        if data.labels is not None:
            window_labels = loader.create_window_labels(data.labels, window_size, step)
            print(f"\nWindow labels:")
            print(f"  Shape: {window_labels.shape}")
            print(f"  Anomalous windows: {int(window_labels.sum())} ({100*window_labels.mean():.2f}%)")
    
    print("\n✓ DataLoader test passed!")
