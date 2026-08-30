"""Data loading helpers for SMAP/MSL style telemetry channels.

The loader extracts the specified signal column, normalizes with training statistics, and
loads point-level anomaly labels when they are available.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


@dataclass
class NormalizationStats:
    """Z-score statistics fitted on the training signal."""

    mean: np.ndarray | float
    std: np.ndarray | float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / (self.std + 1e-8)).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return (values * (self.std + 1e-8) + self.mean).astype(np.float32)


@dataclass
class ChannelData:
    """Container for one telemetry channel."""

    channel_id: str
    train_raw: np.ndarray
    test_raw: np.ndarray
    train_normalized: np.ndarray
    test_normalized: np.ndarray
    labels: Optional[np.ndarray]
    anomaly_sequences: List[Tuple[int, int]]
    norm_stats: NormalizationStats


class DataLoader:
    """Load channel data, labels, and sliding windows."""

    def __init__(self, data_dir: Optional[str | Path] = None, labels_file: str = "labeled_anomalies.csv"):
        if data_dir is None:
            # Go up two levels to get to project root from src/data/data_loader.py
            data_dir = Path(__file__).resolve().parents[2] / "data"

        self.data_dir = Path(data_dir).resolve()
        self.train_dir = self.data_dir / "raw" / "train"
        self.test_dir = self.data_dir / "raw" / "test"
        self.labels_path = self.data_dir / "processed" / labels_file
        self._labels_df: Optional[pd.DataFrame] = None

    def list_channels(self) -> List[str]:
        if not self.train_dir.exists():
            raise FileNotFoundError(f"Training data directory not found: {self.train_dir}")

        channels = []
        for train_file in self.train_dir.glob("*.npy"):
            if (self.test_dir / train_file.name).exists():
                channels.append(train_file.stem)
        return sorted(channels)

    def load_channel(
        self,
        channel_id: str,
        normalize: bool = True,
        signal_index: Optional[int] = 0,
    ) -> ChannelData:
        train_path = self.train_dir / f"{channel_id}.npy"
        test_path = self.test_dir / f"{channel_id}.npy"
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")

        train_raw = np.load(train_path)
        test_raw = np.load(test_path)

        if not np.all(np.isfinite(train_raw)):
            raise ValueError(f"Training telemetry for channel {channel_id} contains non-finite values (NaN/Inf).")
        if not np.all(np.isfinite(test_raw)):
            raise ValueError(f"Test telemetry for channel {channel_id} contains non-finite values (NaN/Inf).")

        train_raw = self._extract_signal(train_raw, signal_index)
        test_raw = self._extract_signal(test_raw, signal_index)

        if normalize:
            axis = 0 if train_raw.ndim == 2 else None
            mean_val = np.mean(train_raw, axis=axis)
            std_val = np.std(train_raw, axis=axis)
            std_val = np.where(std_val < 1e-8, 1.0, std_val)
            norm_stats = NormalizationStats(mean=mean_val, std=std_val)
            train_normalized = norm_stats.transform(train_raw)
            test_normalized = norm_stats.transform(test_raw)
        else:
            norm_stats = NormalizationStats(mean=0.0, std=1.0)
            train_normalized = train_raw.astype(np.float32)
            test_normalized = test_raw.astype(np.float32)

        labels, anomaly_sequences = self._load_labels(channel_id, len(test_raw))
        return ChannelData(
            channel_id=channel_id,
            train_raw=train_raw.astype(np.float32),
            test_raw=test_raw.astype(np.float32),
            train_normalized=train_normalized,
            test_normalized=test_normalized,
            labels=labels,
            anomaly_sequences=anomaly_sequences,
            norm_stats=norm_stats,
        )

    @staticmethod
    def create_windows(values: np.ndarray, window_size: int, step: int = 1, copy: bool = True) -> np.ndarray:
        """Create overlapping windows using stride tricks with strict parameter validation."""
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if step <= 0:
            raise ValueError(f"step must be positive, got {step}")
        if values.ndim not in (1, 2):
            raise ValueError(f"Expected 1D or 2D values array, got {values.ndim}D array with shape {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("Cannot create sliding windows from array containing non-finite values (NaN/Inf)")

        if values.ndim == 1:
            values = values[:, None]

        values = np.ascontiguousarray(values, dtype=np.float32)
        n_samples, n_features = values.shape
        if n_samples < window_size:
            return np.empty((0, window_size, n_features), dtype=np.float32)

        n_windows = (n_samples - window_size) // step + 1
        windows = np.lib.stride_tricks.as_strided(
            values,
            shape=(n_windows, window_size, n_features),
            strides=(step * values.strides[0], values.strides[0], values.strides[1]),
        )
        return windows.copy() if copy else windows

    @staticmethod
    def _extract_signal(values: np.ndarray, signal_index: Optional[int]) -> np.ndarray:
        if values.ndim == 1:
            if signal_index is not None and signal_index != 0:
                raise ValueError(f"Cannot extract signal index {signal_index} from 1D array of shape {values.shape}")
            return values.astype(np.float32)
        if values.ndim == 2:
            if signal_index is None:
                return values.astype(np.float32)
            if signal_index < 0 or signal_index >= values.shape[1]:
                raise ValueError(f"Signal index {signal_index} out of bounds for 2D array of shape {values.shape}")
            return values[:, signal_index].astype(np.float32)
        raise ValueError(f"Unexpected telemetry shape: {values.shape}")

    def _load_labels(self, channel_id: str, test_length: int) -> Tuple[Optional[np.ndarray], List[Tuple[int, int]]]:
        if self._labels_df is None:
            if not self.labels_path.exists():
                return None, []
            self._labels_df = pd.read_csv(self.labels_path)

        rows = self._labels_df[self._labels_df["chan_id"] == channel_id]
        if rows.empty:
            return None, []

        row = rows.iloc[0]
        try:
            expected_length = int(row.get("num_values", test_length))
        except Exception:
            expected_length = test_length

        labels = np.zeros(expected_length, dtype=np.float32)
        seq_str = row.get("anomaly_sequences", "[]")
        anomaly_sequences = self._parse_anomaly_sequences(seq_str, channel_id=channel_id)

        normalized_sequences: List[Tuple[int, int]] = []
        for sequence in anomaly_sequences:
            if isinstance(sequence, (list, tuple)) and len(sequence) == 2:
                start, end = int(sequence[0]), int(sequence[1])
            elif isinstance(sequence, int):
                start, end = int(sequence), int(sequence) + 1
            else:
                continue

            start = max(0, min(start, len(labels)))
            end = max(start, min(end, len(labels)))
            labels[start:end] = 1.0
            normalized_sequences.append((start, end))

        if len(labels) < test_length:
            labels = np.pad(labels, (0, test_length - len(labels))).astype(np.float32)
        elif len(labels) > test_length:
            labels = labels[:test_length].astype(np.float32)

        return labels, normalized_sequences

    @staticmethod
    def _parse_anomaly_sequences(value: object, channel_id: str = "") -> list:
        if isinstance(value, str):
            value = value.strip()
            if not value or value == "nan":
                return []
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError) as err:
                raise ValueError(f"Malformed anomaly_sequences for channel '{channel_id}': {value}") from err
            if not isinstance(parsed, list):
                raise ValueError(f"Expected list for anomaly_sequences in channel '{channel_id}', got: {type(parsed)}")
            return parsed
        if isinstance(value, list):
            return value
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return []
        raise ValueError(f"Unexpected anomaly_sequences type for channel '{channel_id}': {type(value)}")


class SlidingWindowDataset(torch.utils.data.Dataset):
    """Lazy zero-copy sliding window dataset for PyTorch DataLoader.

    Avoids materializing all sliding windows into RAM simultaneously.
    """

    def __init__(
        self,
        values: np.ndarray,
        window_size: int,
        step: int = 1,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        if values.ndim == 1:
            values = values[:, None]
        self.values = np.ascontiguousarray(values, dtype=np.float32)
        self.window_size = window_size
        self.step = step
        self.transform = transform
        n_samples = len(self.values)
        self.n_windows = max(0, (n_samples - window_size) // step + 1) if n_samples >= window_size else 0

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0 or index >= self.n_windows:
            raise IndexError(f"Window index {index} out of range for {self.n_windows} windows")
        start = index * self.step
        end = start + self.window_size
        window = self.values[start:end]
        if self.transform is not None:
            window = self.transform(window)
        return torch.from_numpy(window).float()
