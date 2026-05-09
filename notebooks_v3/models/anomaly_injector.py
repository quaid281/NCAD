"""Synthetic contextual anomaly injection for NCAD-CS contrastive training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class AnomalyInjectionConfig:
    injection_ratio: float = 0.70
    spike_factor_range: Tuple[float, float] = (3.0, 7.0)
    level_shift_factor_range: Tuple[float, float] = (1.5, 4.0)
    variance_factor_range: Tuple[float, float] = (1.5, 4.0)
    min_anomaly_len: int = 5
    max_anomaly_len: int = 15


class ContextualAnomalyInjector:
    """Inject anomalies into the suspect region while scaling by context statistics."""

    def __init__(self, config: Optional[AnomalyInjectionConfig] = None, seed: Optional[int] = None):
        self.config = config or AnomalyInjectionConfig()
        self.rng = np.random.default_rng(seed)
        self.anomaly_types = np.array(["spike", "level_shift", "variance_change", "stuck_value"])

    def inject_batch(self, windows: np.ndarray, context_size: int) -> tuple[np.ndarray, np.ndarray]:
        if windows.ndim != 3:
            raise ValueError("Expected windows with shape (batch, length, features).")

        modified = windows.copy()
        labels = np.zeros(len(windows), dtype=np.float32)
        batch_size, full_window_size, n_features = modified.shape
        suspect_size = full_window_size - context_size
        if batch_size == 0 or suspect_size <= 0:
            return modified, labels

        n_inject = max(1, int(batch_size * self.config.injection_ratio))
        inject_indices = self.rng.choice(batch_size, size=n_inject, replace=False)

        for batch_index in inject_indices:
            context = modified[batch_index, :context_size]
            anomaly_type = str(self.rng.choice(self.anomaly_types))

            if anomaly_type == "spike":
                anomaly_len = 1
            else:
                anomaly_len = int(self.rng.integers(self.config.min_anomaly_len, self.config.max_anomaly_len + 1))
                anomaly_len = min(anomaly_len, suspect_size)

            max_offset = max(0, suspect_size - anomaly_len)
            offset = int(self.rng.integers(0, max_offset + 1)) if max_offset > 0 else 0
            start = context_size + offset
            end = start + anomaly_len

            for feature_index in range(n_features):
                context_values = context[:, feature_index]
                context_std = max(float(np.std(context_values)), 1e-6)
                context_mean = float(np.mean(context_values))
                segment = modified[batch_index, start:end, feature_index]

                if anomaly_type == "spike":
                    factor = self.rng.uniform(*self.config.spike_factor_range)
                    direction = self.rng.choice([-1.0, 1.0])
                    segment[0] = segment[0] + direction * factor * context_std
                elif anomaly_type == "level_shift":
                    factor = self.rng.uniform(*self.config.level_shift_factor_range)
                    direction = self.rng.choice([-1.0, 1.0])
                    segment[:] = segment + direction * factor * context_std
                elif anomaly_type == "variance_change":
                    factor = self.rng.uniform(*self.config.variance_factor_range)
                    if bool(self.rng.integers(0, 2)):
                        factor = 1.0 / factor
                    segment[:] = context_mean + factor * (segment - context_mean)
                elif anomaly_type == "stuck_value":
                    segment[:] = segment[0] if len(segment) else context_mean

            labels[batch_index] = 1.0

        return modified, labels
