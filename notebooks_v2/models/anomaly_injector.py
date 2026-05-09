"""
Anomaly Injector for Self-Supervised Training

Generates synthetic anomalies for training the Mamba Anomaly Classifier.
The key insight: we don't need labeled anomaly data if we can generate
realistic synthetic anomalies and train the model to distinguish them
from normal patterns.

Anomaly Types:
    1. Spike: Single-point extreme deviation (sensor glitch, cosmic ray)
    2. Level Shift: Sustained offset from baseline (calibration drift, mode change)
    3. Variance Change: Altered signal volatility (sensor degradation, environmental change)
    4. Stuck Value: Constant value (sensor freeze, communication dropout)
    5. Trend: Gradual drift over time (thermal drift, degradation)
    6. Noise Burst: Sudden increase in high-frequency noise

Design Philosophy:
    - Anomalies are injected relative to the LOCAL context (std, mean)
    - This makes them "contextually anomalous" rather than just extreme values
    - The model learns: "given this context, is this pattern expected?"
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass
from enum import Enum


class AnomalyType(Enum):
    """Enumeration of supported anomaly types."""
    SPIKE = "spike"
    LEVEL_SHIFT = "level_shift"
    VARIANCE_CHANGE = "variance_change"
    STUCK_VALUE = "stuck_value"
    TREND = "trend"
    NOISE_BURST = "noise_burst"


@dataclass
class AnomalyConfig:
    """Configuration for anomaly injection parameters."""
    # Injection ratio: what fraction of windows get anomalies
    injection_ratio: float = 0.5
    
    # Spike parameters
    spike_factor_range: Tuple[float, float] = (4.0, 10.0)  # Multiplier of local std
    
    # Level shift parameters
    level_shift_factor_range: Tuple[float, float] = (2.0, 6.0)  # Multiplier of local std
    level_shift_min_len: int = 10
    level_shift_max_len: int = 50
    
    # Variance change parameters
    variance_factor_range: Tuple[float, float] = (2.0, 5.0)  # Multiplier/divisor
    variance_min_len: int = 15
    variance_max_len: int = 60
    
    # Stuck value parameters
    stuck_min_len: int = 10
    stuck_max_len: int = 40
    
    # Trend parameters
    trend_factor_range: Tuple[float, float] = (1.0, 4.0)  # Final offset as multiplier of std
    trend_min_len: int = 20
    trend_max_len: int = 80
    
    # Noise burst parameters
    noise_factor_range: Tuple[float, float] = (2.0, 5.0)  # Multiplier of local std
    noise_min_len: int = 10
    noise_max_len: int = 30


class AnomalyInjector:
    """
    Injects synthetic anomalies into time series windows for self-supervised training.
    
    Usage:
        injector = AnomalyInjector()
        
        # For training: inject anomalies into half the batch
        modified_windows, labels = injector.inject_batch(windows)
        
        # For visualization: inject specific anomaly type
        modified, label, info = injector.inject_single(window, anomaly_type='spike')
    
    Args:
        config: AnomalyConfig with injection parameters
        anomaly_types: List of anomaly types to use (default: all)
        anomaly_probs: Probability for each type (default: uniform)
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        config: Optional[AnomalyConfig] = None,
        anomaly_types: Optional[List[str]] = None,
        anomaly_probs: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ):
        self.config = config or AnomalyConfig()
        self.rng = np.random.default_rng(seed)
        
        # Default to all anomaly types
        if anomaly_types is None:
            self.anomaly_types = [t.value for t in AnomalyType]
        else:
            self.anomaly_types = anomaly_types
        
        # Default to uniform probability
        if anomaly_probs is None:
            self.anomaly_probs = np.ones(len(self.anomaly_types)) / len(self.anomaly_types)
        else:
            assert len(anomaly_probs) == len(self.anomaly_types), \
                "anomaly_probs must match length of anomaly_types"
            assert np.isclose(sum(anomaly_probs), 1.0), \
                "anomaly_probs must sum to 1.0"
            self.anomaly_probs = np.array(anomaly_probs)
    
    def inject_batch(
        self,
        windows: np.ndarray,
        injection_ratio: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject anomalies into a batch of windows.
        
        Args:
            windows: Input windows of shape (batch_size, seq_len) or (batch_size, seq_len, 1)
            injection_ratio: Override default injection ratio
            
        Returns:
            modified_windows: Windows with anomalies injected
            labels: Binary labels (0=normal, 1=anomaly)
        """
        # Handle input shape
        squeeze_output = False
        if windows.ndim == 2:
            windows = windows[:, :, np.newaxis]
            squeeze_output = True
        
        batch_size, seq_len, n_features = windows.shape
        assert n_features == 1, "AnomalyInjector expects univariate signals (n_features=1)"
        
        modified = windows.copy()
        labels = np.zeros(batch_size, dtype=np.float32)
        
        # Determine which windows to inject
        ratio = injection_ratio if injection_ratio is not None else self.config.injection_ratio
        n_inject = max(1, int(batch_size * ratio)) if batch_size > 0 else 0
        inject_indices = self.rng.choice(batch_size, n_inject, replace=False)
        
        for idx in inject_indices:
            window = modified[idx, :, 0]  # Shape: (seq_len,)
            modified_window, label, _ = self.inject_single(window)
            modified[idx, :, 0] = modified_window
            labels[idx] = label
        
        if squeeze_output:
            modified = modified[:, :, 0]
        
        return modified, labels
    
    def inject_single(
        self,
        window: np.ndarray,
        anomaly_type: Optional[str] = None,
        return_mask: bool = False,
    ) -> Tuple[np.ndarray, float, Dict]:
        """
        Inject a single anomaly into a window.
        
        Args:
            window: Input window of shape (seq_len,) or (seq_len, 1)
            anomaly_type: Specific type to inject (random if None)
            return_mask: If True, include point-level anomaly mask in info
            
        Returns:
            modified_window: Window with anomaly injected
            label: 1.0 (anomaly was injected)
            info: Dictionary with injection details
        """
        # Handle input shape
        squeeze_output = False
        if window.ndim == 2:
            window = window[:, 0]
            squeeze_output = True
        
        seq_len = len(window)
        modified = window.copy()
        
        # Choose anomaly type
        if anomaly_type is None:
            anomaly_type = self.rng.choice(self.anomaly_types, p=self.anomaly_probs)
        
        # Calculate local statistics for context-relative injection
        local_std = np.std(window)
        local_mean = np.mean(window)
        local_std = max(local_std, 1e-6)  # Prevent division by zero
        
        # Initialize info dict
        info = {
            'anomaly_type': anomaly_type,
            'local_std': local_std,
            'local_mean': local_mean,
        }
        
        # Inject based on type
        if anomaly_type == AnomalyType.SPIKE.value:
            modified, info = self._inject_spike(modified, local_std, info)
        elif anomaly_type == AnomalyType.LEVEL_SHIFT.value:
            modified, info = self._inject_level_shift(modified, local_std, seq_len, info)
        elif anomaly_type == AnomalyType.VARIANCE_CHANGE.value:
            modified, info = self._inject_variance_change(modified, local_mean, seq_len, info)
        elif anomaly_type == AnomalyType.STUCK_VALUE.value:
            modified, info = self._inject_stuck_value(modified, seq_len, info)
        elif anomaly_type == AnomalyType.TREND.value:
            modified, info = self._inject_trend(modified, local_std, seq_len, info)
        elif anomaly_type == AnomalyType.NOISE_BURST.value:
            modified, info = self._inject_noise_burst(modified, local_std, seq_len, info)
        else:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")
        
        if squeeze_output:
            modified = modified[:, np.newaxis]
        
        return modified, 1.0, info
    
    def _inject_spike(
        self,
        window: np.ndarray,
        local_std: float,
        info: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """Inject a single-point spike anomaly."""
        seq_len = len(window)
        
        # Random position (avoid edges for visibility)
        margin = max(1, seq_len // 10)
        pos = self.rng.integers(margin, seq_len - margin)
        
        # Random magnitude and direction
        factor = self.rng.uniform(*self.config.spike_factor_range)
        direction = self.rng.choice([-1, 1])
        magnitude = direction * factor * local_std
        
        window[pos] += magnitude
        
        info.update({
            'position': pos,
            'magnitude': magnitude,
            'factor': factor,
        })
        
        return window, info
    
    def _inject_level_shift(
        self,
        window: np.ndarray,
        local_std: float,
        seq_len: int,
        info: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """Inject a level shift (sustained offset)."""
        # Determine length and position
        min_len = min(self.config.level_shift_min_len, seq_len // 2)
        max_len = min(self.config.level_shift_max_len, seq_len - 1)
        if max_len <= min_len:
            max_len = min_len + 1
        
        anomaly_len = self.rng.integers(min_len, max_len)
        start_pos = self.rng.integers(0, seq_len - anomaly_len)
        end_pos = start_pos + anomaly_len
        
        # Random magnitude and direction
        factor = self.rng.uniform(*self.config.level_shift_factor_range)
        direction = self.rng.choice([-1, 1])
        magnitude = direction * factor * local_std
        
        window[start_pos:end_pos] += magnitude
        
        info.update({
            'start_pos': start_pos,
            'end_pos': end_pos,
            'length': anomaly_len,
            'magnitude': magnitude,
            'factor': factor,
        })
        
        return window, info
    
    def _inject_variance_change(
        self,
        window: np.ndarray,
        local_mean: float,
        seq_len: int,
        info: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """Inject a variance change (amplified or dampened fluctuations)."""
        # Determine length and position
        min_len = min(self.config.variance_min_len, seq_len // 2)
        max_len = min(self.config.variance_max_len, seq_len - 1)
        if max_len <= min_len:
            max_len = min_len + 1
        
        anomaly_len = self.rng.integers(min_len, max_len)
        start_pos = self.rng.integers(0, seq_len - anomaly_len)
        end_pos = start_pos + anomaly_len
        
        # Random factor (increase or decrease variance)
        base_factor = self.rng.uniform(*self.config.variance_factor_range)
        increase = self.rng.choice([True, False])
        factor = base_factor if increase else 1.0 / base_factor
        
        # Scale around local mean
        segment = window[start_pos:end_pos]
        window[start_pos:end_pos] = local_mean + factor * (segment - local_mean)
        
        info.update({
            'start_pos': start_pos,
            'end_pos': end_pos,
            'length': anomaly_len,
            'factor': factor,
            'increased': increase,
        })
        
        return window, info
    
    def _inject_stuck_value(
        self,
        window: np.ndarray,
        seq_len: int,
        info: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """Inject a stuck value (sensor freeze)."""
        # Determine length and position
        min_len = min(self.config.stuck_min_len, seq_len // 2)
        max_len = min(self.config.stuck_max_len, seq_len - 1)
        if max_len <= min_len:
            max_len = min_len + 1
        
        anomaly_len = self.rng.integers(min_len, max_len)
        start_pos = self.rng.integers(0, seq_len - anomaly_len)
        end_pos = start_pos + anomaly_len
        
        # Choose stuck value: start of segment, or a boundary value
        stuck_options = [
            window[start_pos],  # Value at start of segment
            np.min(window),     # Min value in window
            np.max(window),     # Max value in window
            np.mean(window),    # Mean value
            0.0,                # Zero (common failure mode)
        ]
        stuck_value = self.rng.choice(stuck_options)
        
        window[start_pos:end_pos] = stuck_value
        
        info.update({
            'start_pos': start_pos,
            'end_pos': end_pos,
            'length': anomaly_len,
            'stuck_value': stuck_value,
        })
        
        return window, info
    
    def _inject_trend(
        self,
        window: np.ndarray,
        local_std: float,
        seq_len: int,
        info: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """Inject a gradual trend (drift)."""
        # Determine length and position
        min_len = min(self.config.trend_min_len, seq_len // 2)
        max_len = min(self.config.trend_max_len, seq_len - 1)
        if max_len <= min_len:
            max_len = min_len + 1
        
        anomaly_len = self.rng.integers(min_len, max_len)
        start_pos = self.rng.integers(0, seq_len - anomaly_len)
        end_pos = start_pos + anomaly_len
        
        # Random final offset and direction
        factor = self.rng.uniform(*self.config.trend_factor_range)
        direction = self.rng.choice([-1, 1])
        final_offset = direction * factor * local_std
        
        # Linear ramp
        ramp = np.linspace(0, final_offset, anomaly_len)
        window[start_pos:end_pos] += ramp
        
        info.update({
            'start_pos': start_pos,
            'end_pos': end_pos,
            'length': anomaly_len,
            'final_offset': final_offset,
            'factor': factor,
        })
        
        return window, info
    
    def _inject_noise_burst(
        self,
        window: np.ndarray,
        local_std: float,
        seq_len: int,
        info: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """Inject a burst of high-frequency noise."""
        # Determine length and position
        min_len = min(self.config.noise_min_len, seq_len // 2)
        max_len = min(self.config.noise_max_len, seq_len - 1)
        if max_len <= min_len:
            max_len = min_len + 1
        
        anomaly_len = self.rng.integers(min_len, max_len)
        start_pos = self.rng.integers(0, seq_len - anomaly_len)
        end_pos = start_pos + anomaly_len
        
        # Random noise amplitude
        factor = self.rng.uniform(*self.config.noise_factor_range)
        noise_std = factor * local_std
        
        # Add Gaussian noise
        noise = self.rng.normal(0, noise_std, anomaly_len)
        window[start_pos:end_pos] += noise
        
        info.update({
            'start_pos': start_pos,
            'end_pos': end_pos,
            'length': anomaly_len,
            'noise_std': noise_std,
            'factor': factor,
        })
        
        return window, info
    
    def visualize_anomaly_types(
        self,
        window: np.ndarray,
        figsize: Tuple[int, int] = (14, 10),
    ):
        """
        Visualize all anomaly types applied to the same window.
        
        Args:
            window: Input window to demonstrate on
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        
        n_types = len(self.anomaly_types)
        fig, axes = plt.subplots(n_types, 1, figsize=figsize, sharex=True)
        
        if n_types == 1:
            axes = [axes]
        
        for ax, anomaly_type in zip(axes, self.anomaly_types):
            modified, _, info = self.inject_single(window.copy(), anomaly_type=anomaly_type)
            
            ax.plot(window, 'b-', alpha=0.5, label='Original')
            ax.plot(modified, 'r-', alpha=0.8, label='With Anomaly')
            
            # Highlight anomaly region if applicable
            if 'start_pos' in info:
                ax.axvspan(info['start_pos'], info['end_pos'], 
                          alpha=0.2, color='red', label='Anomaly Region')
            elif 'position' in info:
                ax.axvline(info['position'], color='red', linestyle='--', 
                          alpha=0.5, label='Spike Location')
            
            ax.set_ylabel(anomaly_type)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time Step')
        fig.suptitle('Anomaly Type Demonstrations', fontsize=12)
        plt.tight_layout()
        
        return fig


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    print("Testing AnomalyInjector...")
    
    # Create injector
    injector = AnomalyInjector(seed=42)
    
    # Test batch injection
    batch_size = 16
    seq_len = 100
    
    # Generate synthetic "normal" data (sine wave with noise)
    t = np.linspace(0, 4 * np.pi, seq_len)
    base_signal = np.sin(t)
    windows = np.array([base_signal + np.random.randn(seq_len) * 0.1 
                        for _ in range(batch_size)])
    
    print(f"\nInput shape: {windows.shape}")
    
    # Inject anomalies
    modified, labels = injector.inject_batch(windows)
    
    print(f"Output shape: {modified.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Anomalies injected: {int(labels.sum())} / {batch_size}")
    print(f"Injection ratio: {labels.mean():.2%}")
    
    # Test single injection with each type
    print("\nTesting individual anomaly types:")
    for anomaly_type in injector.anomaly_types:
        test_window = base_signal + np.random.randn(seq_len) * 0.1
        modified_single, label, info = injector.inject_single(test_window, anomaly_type=anomaly_type)
        print(f"  {anomaly_type}: {info}")
    
    print("\n✓ AnomalyInjector test passed!")
