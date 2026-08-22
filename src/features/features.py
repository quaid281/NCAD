"""Robust multi-domain feature engineering for NCAD-CS.

The paper transforms each raw univariate telemetry stream into a richer feature
matrix before windowing. This module keeps that step explicit and reproducible:
rolling statistics, trend, spectral, STFT-like, and wavelet-energy descriptors
are extracted, normalized from training statistics, clipped, and selected by
variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import pywt
from scipy import signal as scipy_signal


@dataclass
class FeatureConfig:
    rolling_windows: tuple[int, ...] = (10, 30, 60)
    long_window: int = 150
    fft_window: int = 64
    stft_window: int = 128
    slope_windows: tuple[int, ...] = (16, 64)
    max_features: int = 64
    clip_value: float = 10.0
    use_delay_embedding: bool = True
    delay_embedding_dim: int = 5
    delay_lag: int = 4


class NCADFeatureExtractor:
    """Fit/transform feature engineering pipeline."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.selected_indices_: Optional[np.ndarray] = None
        self.feature_names_: List[str] = []
        self.selected_feature_names_: List[str] = []

    def fit(self, signal: np.ndarray) -> "NCADFeatureExtractor":
        self.fit_transform(signal)
        return self

    def fit_transform(self, signal: np.ndarray) -> np.ndarray:
        features = self._extract_all(signal)
        self.mean_ = np.nanmean(features, axis=0)
        self.std_ = np.nanstd(features, axis=0)
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)

        normalized = self._normalize(features)
        variances = np.nanvar(normalized, axis=0)
        order = np.argsort(variances)[::-1]

        keep = [0]
        for index in order:
            if index != 0:
                keep.append(int(index))
            if len(keep) >= min(self.config.max_features, normalized.shape[1]):
                break

        self.selected_indices_ = np.array(keep, dtype=np.int64)
        self.selected_feature_names_ = [self.feature_names_[i] for i in self.selected_indices_]
        return normalized[:, self.selected_indices_].astype(np.float32)

    def transform(self, signal: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None or self.selected_indices_ is None:
            raise RuntimeError("Feature extractor must be fitted before calling transform().")
        features = self._extract_all(signal)
        normalized = self._normalize(features)
        return normalized[:, self.selected_indices_].astype(np.float32)

    def to_metadata(self) -> dict:
        return {
            "feature_names": self.feature_names_,
            "selected_feature_names": self.selected_feature_names_,
            "selected_indices": self.selected_indices_.tolist() if self.selected_indices_ is not None else None,
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "std": self.std_.tolist() if self.std_ is not None else None,
            "config": vars(self.config),
        }

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        normalized = (features - self.mean_) / (self.std_ + 1e-8)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=self.config.clip_value, neginf=-self.config.clip_value)
        return np.clip(normalized, -self.config.clip_value, self.config.clip_value)

    def _extract_all(self, signal: np.ndarray) -> np.ndarray:
        values = np.asarray(signal, dtype=np.float64).reshape(-1)
        n_points = len(values)
        if n_points == 0:
            raise ValueError("Cannot extract features from an empty signal.")

        feature_arrays: List[np.ndarray] = []
        names: List[str] = []

        def add(name: str, array: np.ndarray) -> None:
            clean = np.asarray(array, dtype=np.float64).reshape(-1)
            if len(clean) != n_points:
                clean = np.interp(np.arange(n_points), np.linspace(0, n_points - 1, len(clean)), clean)
            clean = np.nan_to_num(clean, nan=0.0, posinf=0.0, neginf=0.0)
            feature_arrays.append(clean)
            names.append(name)

        series = pd.Series(values)
        add("raw", values)
        add("diff_1", np.r_[0.0, np.diff(values)])
        add("diff_2", np.r_[0.0, 0.0, np.diff(values, n=2)] if n_points > 2 else np.zeros(n_points))

        for window in self.config.rolling_windows:
            rolling = series.rolling(window=window, min_periods=1)
            mean = rolling.mean().to_numpy()
            std = rolling.std(ddof=0).fillna(0.0).to_numpy()
            min_value = rolling.min().to_numpy()
            max_value = rolling.max().to_numpy()
            median = rolling.median().to_numpy()
            add(f"roll_mean_{window}", mean)
            add(f"roll_std_{window}", std)
            add(f"roll_min_{window}", min_value)
            add(f"roll_max_{window}", max_value)
            add(f"roll_median_{window}", median)
            add(f"roll_skew_{window}", rolling.skew().fillna(0.0).to_numpy())
            add(f"roll_kurt_{window}", rolling.kurt().fillna(0.0).to_numpy())
            add(f"roll_z_{window}", (values - mean) / (std + 1e-8))
            add(f"roll_range_pos_{window}", (values - min_value) / (max_value - min_value + 1e-8))

        long_rolling = series.rolling(window=self.config.long_window, min_periods=2)
        add(f"long_mean_{self.config.long_window}", long_rolling.mean().bfill().fillna(0.0).to_numpy())
        add(f"long_std_{self.config.long_window}", long_rolling.std(ddof=0).fillna(0.0).to_numpy())

        for window in self.config.slope_windows:
            add(f"slope_{window}", self._rolling_slope(values, window))

        self._add_fft_features(values, add)
        self._add_stft_features(values, add)
        self._add_wavelet_features(values, add)
        self._add_complexity_features(values, add)
        if self.config.use_delay_embedding:
            self._add_delay_features(values, add)

        self.feature_names_ = names
        return np.column_stack(feature_arrays)

    def _add_delay_features(self, values: np.ndarray, add) -> None:
        n_points = len(values)
        dim = max(1, self.config.delay_embedding_dim)
        lag = max(1, self.config.delay_lag)
        for k in range(1, dim):
            total_lag = k * lag
            if total_lag < n_points:
                lagged = np.r_[np.full(total_lag, values[0]), values[:-total_lag]]
            else:
                lagged = np.full(n_points, values[0])
            add(f"delay_coord_m{k}_lag{total_lag}", lagged)

    @staticmethod
    def _rolling_slope(values: np.ndarray, window: int) -> np.ndarray:
        """Vectorized rolling linear regression slope via 1D causal convolution."""
        n_points = len(values)
        slopes = np.zeros(n_points, dtype=np.float64)
        if n_points < 2 or window < 2:
            return slopes

        w = min(window, n_points)
        x = np.arange(w, dtype=np.float64)
        x_centered = x - np.mean(x)
        denom = np.sum(x_centered ** 2)

        if denom > 1e-12:
            kernel = x_centered / denom
            conv = np.convolve(values, kernel[::-1], mode="full")[:n_points]
            slopes[w - 1 :] = conv[w - 1 :]

        # Initial warmup points for t < w - 1
        for index in range(1, min(w - 1, n_points)):
            y_values = values[: index + 1]
            x_val = np.arange(len(y_values), dtype=np.float64)
            x_c = x_val - np.mean(x_val)
            d = np.sum(x_c ** 2)
            if d > 1e-12:
                slopes[index] = np.sum(x_c * y_values) / d

        return slopes

    def _add_fft_features(self, values: np.ndarray, add) -> None:
        """Vectorized batch FFT spectral descriptors using strided sliding windows."""
        n_points = len(values)
        window = min(self.config.fft_window, n_points)
        dominant_freq = np.zeros(n_points, dtype=np.float64)
        spectral_entropy = np.zeros(n_points, dtype=np.float64)
        spectral_power = np.zeros(n_points, dtype=np.float64)

        if n_points < 8:
            add("fft_dominant_freq", dominant_freq)
            add("fft_spectral_entropy", spectral_entropy)
            add("fft_power", spectral_power)
            return

        # 1. Warmup for boundary points
        for index in range(7, min(window - 1, n_points)):
            segment = values[: index + 1]
            fft_values = np.fft.rfft(segment - np.mean(segment))
            power = np.abs(fft_values) ** 2
            spectral_power[index] = float(np.sum(power))
            if len(power) > 1:
                dominant_freq[index] = float(np.argmax(power[1:]) + 1)
            total_power = np.sum(power)
            if total_power > 1e-12:
                normalized_power = power / total_power
                spectral_entropy[index] = -float(np.sum(normalized_power * np.log2(normalized_power + 1e-12)))

        # 2. Vectorized batch FFT for full windows
        if n_points >= window:
            from numpy.lib.stride_tricks import sliding_window_view
            strided_windows = sliding_window_view(values, window_shape=window)
            centered = strided_windows - np.mean(strided_windows, axis=1, keepdims=True)
            fft_vals = np.fft.rfft(centered, axis=1)
            power = np.abs(fft_vals) ** 2

            tot_power = np.sum(power, axis=1)
            spectral_power[window - 1 :] = tot_power

            if power.shape[1] > 1:
                dominant_freq[window - 1 :] = np.argmax(power[:, 1:], axis=1) + 1

            eps = 1e-12
            norm_power = np.divide(power, tot_power[:, None] + eps, out=np.zeros_like(power), where=tot_power[:, None] > eps)
            log_term = np.log2(norm_power + eps)
            spectral_entropy[window - 1 :] = -np.sum(np.where(norm_power > 0, norm_power * log_term, 0.0), axis=1)

        add("fft_dominant_freq", dominant_freq)
        add("fft_spectral_entropy", spectral_entropy)
        add("fft_power", spectral_power)

    def _add_stft_features(self, values: np.ndarray, add) -> None:
        n_points = len(values)
        if n_points < 16:
            for name in ("stft_low", "stft_mid", "stft_high", "stft_centroid"):
                add(name, np.zeros(n_points))
            return

        nperseg = min(self.config.stft_window, n_points)
        noverlap = max(0, nperseg // 2)
        freqs, times, zxx = scipy_signal.stft(values, nperseg=nperseg, noverlap=noverlap, boundary=None)
        power = np.abs(zxx) ** 2
        if power.size == 0:
            for name in ("stft_low", "stft_mid", "stft_high", "stft_centroid"):
                add(name, np.zeros(n_points))
            return

        band_edges = np.array_split(np.arange(len(freqs)), 3)
        low = np.mean(power[band_edges[0]], axis=0) if len(band_edges[0]) else np.zeros(power.shape[1])
        mid = np.mean(power[band_edges[1]], axis=0) if len(band_edges[1]) else np.zeros(power.shape[1])
        high = np.mean(power[band_edges[2]], axis=0) if len(band_edges[2]) else np.zeros(power.shape[1])
        centroid = np.sum(freqs[:, None] * power, axis=0) / (np.sum(power, axis=0) + 1e-12)
        sample_times = np.clip(times, 0, n_points - 1)

        add("stft_low", np.interp(np.arange(n_points), sample_times, low))
        add("stft_mid", np.interp(np.arange(n_points), sample_times, mid))
        add("stft_high", np.interp(np.arange(n_points), sample_times, high))
        add("stft_centroid", np.interp(np.arange(n_points), sample_times, centroid))

    @staticmethod
    def _add_wavelet_features(values: np.ndarray, add) -> None:
        n_points = len(values)
        max_level = pywt.dwt_max_level(max(n_points, 2), pywt.Wavelet("db2").dec_len)
        level = max(1, min(3, max_level))
        try:
            coeffs = pywt.wavedec(values, "db2", level=level, mode="periodization")
            reconstructed_energies = []
            for coeff_index in range(1, len(coeffs)):
                detail_coeffs = [np.zeros_like(coeff) for coeff in coeffs]
                detail_coeffs[coeff_index] = coeffs[coeff_index]
                detail = pywt.waverec(detail_coeffs, "db2", mode="periodization")[:n_points]
                reconstructed_energies.append(detail * detail)
            total_energy = np.sum(reconstructed_energies, axis=0) + 1e-12
            for index, energy in enumerate(reconstructed_energies):
                add(f"wavelet_energy_ratio_{index + 1}", energy / total_energy)
        except Exception:
            for index in range(3):
                add(f"wavelet_energy_ratio_{index + 1}", np.zeros(n_points))

    @staticmethod
    def _add_complexity_features(values: np.ndarray, add) -> None:
        diff = np.r_[0.0, np.diff(values)]
        abs_diff = np.abs(diff)
        rolling_abs_diff = pd.Series(abs_diff).rolling(window=32, min_periods=1).mean().to_numpy()
        centered = values - pd.Series(values).rolling(window=32, min_periods=1).mean().to_numpy()
        signs = np.sign(centered)
        sign_changes = np.r_[0.0, np.abs(np.diff(signs)) > 0].astype(np.float64)
        zero_crossing = pd.Series(sign_changes).rolling(window=32, min_periods=1).mean().to_numpy()
        add("mean_abs_change_32", rolling_abs_diff)
        add("zero_crossing_rate_32", zero_crossing)
