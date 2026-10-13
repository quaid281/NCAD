"""Online Streaming Threshold Calibrators for Time-Series Anomaly Detection.

Addresses inter-block temporal drift (baseline shifts, variance dilation, and
autocorrelation) via online streaming calibration:
1. ``RollingSPOTCalibrator``: Sliding-window Peaks-Over-Threshold (SPOT / EVT)
   fitting Generalized Pareto Distribution (GPD) parameters periodically over a
   trailing buffer with anomaly-contamination gating.
2. ``StreamingQuantileCalibrator``: Online streaming quantile tracking via
   sliding empirical windows or adaptive Robbins-Monro stochastic approximation.

Contamination Gating:
Unconstrained ("naive") rolling calibration buffers get contaminated by sustained
anomalies, causing threshold inflation and subsequent false-negative surges. Both
calibrators implement anomaly gating to preserve clean nominal baseline tracking.
"""

from __future__ import annotations

import collections
import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from src.scoring.evt_calibrator import GPDFitResult, fit_gpd

logger = logging.getLogger(__name__)


class StreamingQuantileCalibrator:
    """Online streaming quantile calibrator tracking temporal score drift.

    Provides continuous threshold updates for target tail probability q:
    theta_t = Quantile_{1 - q}(Buffer_t).

    Parameters:
        q: Target tail risk level (default: 1e-3, corresponding to the 99.9th percentile).
        window_size: Trailing window capacity for local baseline estimation.
        method: 'trailing_window' for exact sliding quantile, or
            'stochastic_approximation' for adaptive Robbins-Monro tracking.
        step: Update step frequency for threshold recalculation (for throughput).
        min_periods: Minimum samples required before activating empirical quantile.
        gated: Whether to exclude detected anomalies from contaminating the nominal buffer.
        gate_multiplier: Safety factor on threshold for contamination gating.
        learning_rate: Initial step size for stochastic approximation mode.
        fallback_threshold: Initial default threshold during warmup.
    """

    def __init__(
        self,
        q: float = 1e-3,
        window_size: int = 1000,
        method: str = "trailing_window",
        step: int = 1,
        min_periods: int = 30,
        gated: bool = False,
        gate_multiplier: float = 1.0,
        max_consecutive_gated: Optional[int] = 100,
        learning_rate: float = 0.05,
        fallback_threshold: float = 1.0,
    ):
        if not (0.0 < q < 1.0):
            raise ValueError(f"q must be in (0, 1), got {q}")
        if window_size < 10:
            raise ValueError(f"window_size must be >= 10, got {window_size}")
        if method not in ("trailing_window", "stochastic_approximation"):
            raise ValueError(f"Unknown method {method!r}")

        self.q = float(q)
        self.target_percentile = (1.0 - self.q) * 100.0
        self.window_size = int(window_size)
        self.method = method
        self.step = max(1, int(step))
        self.min_periods = max(5, int(min_periods))
        self.gated = bool(gated)
        self.gate_multiplier = float(gate_multiplier)
        self.max_consecutive_gated = max_consecutive_gated
        self.learning_rate = float(learning_rate)
        self.fallback_threshold = float(fallback_threshold)

        # Internal state
        self.buffer_: collections.deque = collections.deque(maxlen=self.window_size)
        self.current_threshold_: float = self.fallback_threshold
        self.total_points_: int = 0
        self.gated_points_: int = 0
        self.consecutive_gated_: int = 0
        self.last_update_idx_: int = 0

        # Stochastic approximation tracking state
        self.sa_estimate_: float = self.fallback_threshold
        self.sa_scale_: float = 1.0

    def reset(self) -> None:
        """Reset online tracking state."""
        self.buffer_.clear()
        self.current_threshold_ = self.fallback_threshold
        self.total_points_ = 0
        self.gated_points_ = 0
        self.consecutive_gated_ = 0
        self.last_update_idx_ = 0
        self.sa_estimate_ = self.fallback_threshold
        self.sa_scale_ = 1.0

    def fit(self, initial_scores: np.ndarray) -> "StreamingQuantileCalibrator":
        """Initialize the rolling buffer with baseline calibration scores."""
        clean = np.asarray(initial_scores, dtype=np.float64).ravel()
        clean = clean[np.isfinite(clean)]
        self.reset()
        if len(clean) > 0:
            seed_slice = clean[-self.window_size :]
            self.buffer_.extend(seed_slice.tolist())
            self.total_points_ = len(seed_slice)
            if len(self.buffer_) >= self.min_periods:
                self.current_threshold_ = float(np.percentile(self.buffer_, self.target_percentile))
            else:
                self.current_threshold_ = float(np.max(self.buffer_)) if len(self.buffer_) else self.fallback_threshold
            self.sa_estimate_ = self.current_threshold_
            std = float(np.std(self.buffer_)) if len(self.buffer_) > 1 else 1.0
            self.sa_scale_ = max(std, 1e-4)
        return self

    def update(self, score: float) -> float:
        """Ingest a single score point and return the updated dynamic threshold."""
        if not math.isfinite(score):
            return self.current_threshold_

        self.total_points_ += 1

        # Contamination gating check with regime-shift recovery
        is_anomaly = score > (self.current_threshold_ * self.gate_multiplier)
        if self.gated and len(self.buffer_) >= self.min_periods:
            if is_anomaly:
                self.consecutive_gated_ += 1
                if self.max_consecutive_gated is None or self.consecutive_gated_ <= self.max_consecutive_gated:
                    self.gated_points_ += 1
                    return self.current_threshold_
                else:
                    self.consecutive_gated_ = 0
            else:
                self.consecutive_gated_ = 0

        # Ingest into buffer
        self.buffer_.append(float(score))

        if self.method == "trailing_window":
            # Recalculate trailing empirical quantile periodically
            if (self.total_points_ - self.last_update_idx_ >= self.step) or (len(self.buffer_) == self.min_periods):
                if len(self.buffer_) >= self.min_periods:
                    self.current_threshold_ = float(np.percentile(self.buffer_, self.target_percentile))
                else:
                    self.current_threshold_ = float(np.max(self.buffer_))
                self.last_update_idx_ = self.total_points_

        elif self.method == "stochastic_approximation":
            # Robbins-Monro online quantile tracking:
            # theta_{t+1} = theta_t + eta_t * (1_{s_t > theta_t} - q)
            # Scaled by running scale estimate
            t = max(self.total_points_, 1)
            eta = self.learning_rate / math.sqrt(1.0 + 0.005 * t)
            indicator = 1.0 if score > self.sa_estimate_ else 0.0
            step_size = eta * (indicator - self.q) * self.sa_scale_
            self.sa_estimate_ += step_size
            self.current_threshold_ = max(self.sa_estimate_, 1e-6)

        return self.current_threshold_

    def process_stream(self, scores: np.ndarray) -> np.ndarray:
        """Process an entire chronological 1D stream and return time-varying thresholds."""
        arr = np.asarray(scores, dtype=np.float64).ravel()
        n = len(arr)
        if n == 0:
            return np.array([], dtype=np.float64)

        thresholds = np.empty(n, dtype=np.float64)
        for i in range(n):
            thresholds[i] = self.update(float(arr[i]))
        return thresholds

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Return boolean anomaly predictions (score > dynamic_threshold)."""
        thr = self.process_stream(scores)
        arr = np.asarray(scores, dtype=np.float64).ravel()
        return arr > thr

    def get_diagnostics(self) -> Dict[str, Union[float, int, str]]:
        """Return operational diagnostics of the streaming calibrator."""
        return {
            "method": self.method,
            "q": self.q,
            "window_size": self.window_size,
            "buffer_len": len(self.buffer_),
            "total_points": self.total_points_,
            "gated_points": self.gated_points_,
            "current_threshold": self.current_threshold_,
        }


class RollingSPOTCalibrator:
    """Sliding-window Peaks-Over-Threshold (SPOT / EVT) online calibrator.

    Adapts the operational anomaly threshold continuously to non-stationary distribution
    drift using Extreme Value Theory (EVT) fitted over a trailing window of observations.

    Parameters:
        risk_level: Target design risk level q (default: 1e-3, target nominal FPR 0.1%).
        init_percentile: Baseline tail threshold percentile u (default: 98.0).
        window_size: Length of trailing observation window W (default: 1000).
        step: Frequency of GPD refitting (refits every `step` observations for efficiency).
        min_excesses: Minimum tail exceedances required for GPD estimation (default: 10).
        min_periods: Minimum observations before activating GPD; uses empirical quantile prior.
        gated: Whether to activate anomaly contamination gating.
        gate_mode: 'filter' (discards anomalous observations from nominal buffer),
                   'freeze' (pauses buffer ingestion during an alarm episode), or
                   'none' (unconstrained buffer ingestion).
        gate_multiplier: Multiplier on threshold for anomaly classification.
        adaptive_kurtosis: Whether to adapt init_percentile dynamically based on buffer kurtosis.
        fallback_threshold: Initial threshold value before warmup completes.
    """

    def __init__(
        self,
        risk_level: float = 1e-3,
        init_percentile: float = 98.0,
        window_size: int = 1000,
        step: int = 100,
        min_excesses: int = 10,
        min_periods: int = 50,
        gated: bool = False,
        gate_mode: str = "filter",
        gate_multiplier: float = 1.0,
        max_consecutive_gated: Optional[int] = 100,
        adaptive_kurtosis: bool = False,
        fallback_threshold: float = 1.0,
    ):
        if not (0.0 < risk_level < 1.0):
            raise ValueError(f"risk_level must be in (0, 1), got {risk_level}")
        if not (50.0 <= init_percentile < 100.0):
            raise ValueError(f"init_percentile must be in [50, 100), got {init_percentile}")
        if gate_mode not in ("filter", "freeze", "none"):
            raise ValueError(f"Unknown gate_mode {gate_mode!r}")

        self.risk_level = float(risk_level)
        self.init_percentile = float(init_percentile)
        self.window_size = int(window_size)
        self.step = max(1, int(step))
        self.min_excesses = max(3, int(min_excesses))
        self.min_periods = max(20, int(min_periods))
        self.gated = bool(gated)
        self.gate_mode = gate_mode
        self.gate_multiplier = float(gate_multiplier)
        self.max_consecutive_gated = max_consecutive_gated
        self.adaptive_kurtosis = bool(adaptive_kurtosis)
        self.fallback_threshold = float(fallback_threshold)

        # Internal state
        self.buffer_: collections.deque = collections.deque(maxlen=self.window_size)
        self.current_threshold_: float = self.fallback_threshold
        self.last_gpd_fit_: Optional[GPDFitResult] = None
        self.total_points_: int = 0
        self.gated_points_: int = 0
        self.consecutive_gated_: int = 0
        self.last_refit_idx_: int = 0
        self.in_alarm_: bool = False

    def reset(self) -> None:
        """Reset internal streaming state."""
        self.buffer_.clear()
        self.current_threshold_ = self.fallback_threshold
        self.last_gpd_fit_ = None
        self.total_points_ = 0
        self.gated_points_ = 0
        self.consecutive_gated_ = 0
        self.last_refit_idx_ = 0
        self.in_alarm_ = False

    def fit(self, initial_scores: np.ndarray) -> "RollingSPOTCalibrator":
        """Seed the sliding buffer with baseline calibration observations and compute initial threshold."""
        clean = np.asarray(initial_scores, dtype=np.float64).ravel()
        clean = clean[np.isfinite(clean)]
        self.reset()
        if len(clean) > 0:
            seed_slice = clean[-self.window_size :]
            self.buffer_.extend(seed_slice.tolist())
            self.total_points_ = len(seed_slice)
            self._refit_threshold()
            self.last_refit_idx_ = self.total_points_
        return self

    def _refit_threshold(self) -> None:
        """Fit GPD and compute the EVT extreme quantile on current buffer."""
        n = len(self.buffer_)
        if n < self.min_periods:
            if n > 0:
                p_val = (1.0 - self.risk_level) * 100.0
                self.current_threshold_ = float(np.percentile(self.buffer_, p_val))
            else:
                self.current_threshold_ = self.fallback_threshold
            return

        buf_arr = np.array(self.buffer_, dtype=np.float64)

        eff_percentile = self.init_percentile
        if self.adaptive_kurtosis:
            # Kurtosis-adapted initial percentile
            var = float(np.var(buf_arr))
            if var > 1e-12:
                centered = buf_arr - float(np.mean(buf_arr))
                m4 = float(np.mean(centered**4))
                kurt = m4 / (var**2 + 1e-12) - 3.0
                if kurt > 0:
                    eff_percentile = min(99.0, self.init_percentile + 4.0 * (kurt / (kurt + 15.0)))
                else:
                    eff_percentile = max(90.0, self.init_percentile + 0.5 * kurt)

        fit = fit_gpd(buf_arr, init_percentile=eff_percentile, min_excesses=self.min_excesses)
        self.last_gpd_fit_ = fit

        t = fit.threshold_init
        n_total = max(fit.n_total, 1)
        n_excess = max(fit.n_excess, 1)
        gamma = fit.gamma
        sigma = max(fit.sigma, 1e-8)

        if not fit.fit_converged or n_excess < 3:
            # Fallback to empirical high percentile
            p_val = (1.0 - self.risk_level) * 100.0
            self.current_threshold_ = float(np.percentile(buf_arr, p_val))
            return

        # GPD Extreme Quantile Formula:
        # z_q = t + (sigma / gamma) * [ ((q * N) / N_t)^(-gamma) - 1 ]
        ratio = (self.risk_level * n_total) / n_excess
        ratio = max(ratio, 1e-12)

        if abs(gamma) < 1e-5:
            excess_q = -sigma * math.log(ratio)
        else:
            excess_q = (sigma / gamma) * (math.pow(ratio, -gamma) - 1.0)

        # Anti-explosion damping
        max_excess = sigma * (10.0 + math.log(max(float(n_total) / max(float(n_excess), 1.0), 1.0)))
        if excess_q > max_excess:
            excess_q = max_excess

        thr = float(t + excess_q)

        # Baseline safety clamping
        if thr < t or not math.isfinite(thr):
            thr = float(t + sigma * math.log(max(float(n_excess) / (self.risk_level * n_total), 1.1)))

        self.current_threshold_ = max(thr, 1e-6)

    def update(self, score: float) -> float:
        """Ingest a single telemetry score and return the updated dynamic threshold."""
        if not math.isfinite(score):
            return self.current_threshold_

        self.total_points_ += 1
        is_anomaly = score > (self.current_threshold_ * self.gate_multiplier)

        # Handle contamination gating
        if self.gated and len(self.buffer_) >= self.min_periods:
            if is_anomaly:
                self.consecutive_gated_ += 1
                if self.max_consecutive_gated is None or self.consecutive_gated_ <= self.max_consecutive_gated:
                    self.gated_points_ += 1
                    if self.gate_mode == "freeze":
                        self.in_alarm_ = True
                    return self.current_threshold_
                else:
                    # Persistent excursion indicates inter-block regime shift rather than transient anomaly
                    self.consecutive_gated_ = 0
                    self.in_alarm_ = False
            else:
                self.consecutive_gated_ = 0
                self.in_alarm_ = False

        # Nominal point: add to buffer
        self.buffer_.append(float(score))

        # Check if periodic refit is due
        if (self.total_points_ - self.last_refit_idx_ >= self.step) or (len(self.buffer_) == self.min_periods):
            self._refit_threshold()
            self.last_refit_idx_ = self.total_points_

        return self.current_threshold_

    def process_stream(self, scores: np.ndarray) -> np.ndarray:
        """Process a sequence of scores and return the dynamic threshold array."""
        arr = np.asarray(scores, dtype=np.float64).ravel()
        n = len(arr)
        if n == 0:
            return np.array([], dtype=np.float64)

        thresholds = np.empty(n, dtype=np.float64)
        for i in range(n):
            thresholds[i] = self.update(float(arr[i]))
        return thresholds

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Return boolean anomaly detections (score > dynamic_threshold)."""
        thr = self.process_stream(scores)
        arr = np.asarray(scores, dtype=np.float64).ravel()
        return arr > thr

    def get_diagnostics(self) -> Dict[str, Union[float, int, str, bool]]:
        """Return operational diagnostics of the rolling SPOT calibrator."""
        gamma = self.last_gpd_fit_.gamma if self.last_gpd_fit_ is not None else float("nan")
        sigma = self.last_gpd_fit_.sigma if self.last_gpd_fit_ is not None else float("nan")
        converged = self.last_gpd_fit_.fit_converged if self.last_gpd_fit_ is not None else False
        return {
            "risk_level": self.risk_level,
            "window_size": self.window_size,
            "buffer_len": len(self.buffer_),
            "total_points": self.total_points_,
            "gated_points": self.gated_points_,
            "current_threshold": self.current_threshold_,
            "gpd_gamma": gamma,
            "gpd_sigma": sigma,
            "fit_converged": converged,
            "gated": self.gated,
            "gate_mode": self.gate_mode,
        }


def compute_rolling_spot_thresholds(
    test_scores: np.ndarray,
    train_scores: Optional[np.ndarray] = None,
    risk_level: float = 1e-3,
    init_percentile: float = 98.0,
    window_size: int = 1000,
    step: int = 100,
    gated: bool = False,
    gate_mode: str = "filter",
    max_consecutive_gated: Optional[int] = 100,
) -> np.ndarray:
    """Convenience function to compute dynamic rolling SPOT thresholds on a score stream."""
    calibrator = RollingSPOTCalibrator(
        risk_level=risk_level,
        init_percentile=init_percentile,
        window_size=window_size,
        step=step,
        gated=gated,
        gate_mode=gate_mode,
        max_consecutive_gated=max_consecutive_gated,
    )
    if train_scores is not None and len(train_scores) > 0:
        calibrator.fit(train_scores)
    return calibrator.process_stream(test_scores)


def compute_streaming_quantile_thresholds(
    test_scores: np.ndarray,
    train_scores: Optional[np.ndarray] = None,
    q: float = 1e-3,
    window_size: int = 1000,
    step: int = 1,
    gated: bool = False,
) -> np.ndarray:
    """Convenience function to compute streaming quantile thresholds on a score stream."""
    calibrator = StreamingQuantileCalibrator(
        q=q,
        window_size=window_size,
        step=step,
        gated=gated,
    )
    if train_scores is not None and len(train_scores) > 0:
        calibrator.fit(train_scores)
    return calibrator.process_stream(test_scores)
