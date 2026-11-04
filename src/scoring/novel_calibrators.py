"""Novel drift-resilient thresholding calibrators for time-series anomaly detection.

Implements three strictly causal thresholding architectures:
1. ``HalpernPrimalDualCalibrator``:
   Halpern-anchored primal-dual stochastic approximation at the boundary
   exponent p = 1/2 (LastIterateConvergence, arXiv:1905.10899) with
   second-moment preconditioning and excursion-strand quarantine.
2. ``StoppingBandCalibrator``:
   Two-barrier stopping-band resolution on a nested collar [u_inner, u_outer].
   Traces excursions entering u_inner and commits returning excursions to the
   nominal buffer while quarantining stopped strands that breach u_outer.
3. ``PMIConformalCalibrator``:
   Pointwise Mutual Information (PMI) causal context-normalized nonconformity
   paired with Selective Conformal Uncertainty (SConU) bulk gating and
   Hoeffding-Bentkus finite-sample risk control.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.scoring.evt_calibrator import EVTCalibrator


def _clean_finite(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def compute_subblock_drift_stats(cal: np.ndarray, n_blocks: int = 4) -> tuple[float, float]:
    """Measure inter-subblock quantile variation and trend across calibration sub-windows."""
    clean = _clean_finite(cal)
    if len(clean) < 40:
        return 0.0, 0.0
    k = min(n_blocks, max(2, len(clean) // 60))
    blocks = np.array_split(clean, k)
    q95s = np.array([float(np.quantile(b, 0.95)) for b in blocks if len(b) >= 15])
    if len(q95s) < 2:
        return 0.0, 0.0
    drift_std = float(np.std(q95s, ddof=1))
    trend = float(q95s[-1] - q95s[0])
    return drift_std, trend


def compute_calibration_anchor(
    cal: np.ndarray,
    risk_level: float = 1e-3,
    init_percentile: float = 98.0,
) -> dict[str, Any]:
    """Compute drift-aware base EVT anchor and terminal regime dispersion statistics."""
    clean = _clean_finite(cal)
    if len(clean) == 0:
        return {
            "th_full": 1.0,
            "anchor": 1.0,
            "med0": 0.0,
            "q98_0": 1.0,
            "spread0": 1.0,
            "drift_std": 0.0,
            "trend": 0.0,
            "tail_cal": np.ones(1, dtype=np.float64),
        }
    if float(np.ptp(clean)) < 1e-12:
        base_val = float(clean[0]) + 1.0
        return {
            "th_full": base_val,
            "anchor": base_val,
            "med0": float(clean[0]),
            "q98_0": base_val,
            "spread0": 1.0,
            "drift_std": 0.0,
            "trend": 0.0,
            "tail_cal": clean.copy(),
        }

    evt_full = EVTCalibrator(
        risk_level=risk_level,
        init_percentile=init_percentile,
        adaptive_kurtosis=False,
    ).fit(clean)
    th_full = float(evt_full.threshold_)

    tail_cal = clean[int(len(clean) * 0.4) :] if len(clean) >= 120 else clean
    evt_tail = EVTCalibrator(
        risk_level=risk_level,
        init_percentile=init_percentile,
        adaptive_kurtosis=False,
    ).fit(tail_cal)
    th_tail = float(evt_tail.threshold_)

    drift_std, trend = compute_subblock_drift_stats(clean)
    med0 = float(np.median(tail_cal))
    q98_0 = float(np.quantile(tail_cal, init_percentile / 100.0))
    spread0 = max(q98_0 - med0, float(np.std(tail_cal)), 1e-5)

    if trend < -0.5 * spread0:
        anchor = 0.65 * th_tail + 0.35 * th_full
    else:
        q_emp = float(np.quantile(clean, 1.0 - risk_level))
        anchor = max(th_full, q_emp) + 0.35 * drift_std + max(0.0, 0.25 * trend)

    return {
        "th_full": th_full,
        "anchor": float(anchor),
        "med0": med0,
        "q98_0": q98_0,
        "spread0": spread0,
        "drift_std": drift_std,
        "trend": trend,
        "tail_cal": tail_cal.copy(),
    }


class HalpernPrimalDualCalibrator:
    """Halpern-anchored primal-dual online threshold calibrator.

    Combines boundary-exponent (p = 1/2) Halpern anchoring:
        delta_{t+1} = (1 - beta_t) * delta_t + alpha_t * g_t / sqrt(v_t)
    where alpha_t = alpha_0 / sqrt(tau_t + gamma) and beta_t = gamma / (tau_t + gamma),
    with second-moment gradient conditioning and stopping-collar anomaly quarantine.
    """

    def __init__(
        self,
        risk_level: float = 1e-3,
        init_percentile: float = 98.0,
        gamma: float = 6.0,
        step_scale: float = 0.20,
        max_tau: float = 500.0,
        collar_commit_len: int = 12,
        fallback_threshold: float = 1.0,
    ) -> None:
        if not (0.0 < risk_level < 1.0):
            raise ValueError(f"risk_level must be in (0, 1), got {risk_level}")
        if not (50.0 <= init_percentile < 100.0):
            raise ValueError(f"init_percentile must be in [50, 100), got {init_percentile}")
        if gamma < 2.0:
            raise ValueError(f"Halpern anchor gamma must be >= 2.0, got {gamma}")

        self.risk_level = float(risk_level)
        self.init_percentile = float(init_percentile)
        self.gamma = float(gamma)
        self.step_scale = float(step_scale)
        self.max_tau = float(max_tau)
        self.collar_commit_len = int(collar_commit_len)
        self.fallback_threshold = float(fallback_threshold)

        self.anchor_: float = self.fallback_threshold
        self.med0_: float = 0.0
        self.q98_0_: float = self.fallback_threshold
        self.spread0_: float = 1.0
        self.p_u_: float = 1.0 - (self.init_percentile / 100.0)

        self.delta_: float = 0.0
        self.v_: float = self.p_u_ * (1.0 - self.p_u_)
        self.tau_: float = 1.0
        self.strand_: list[float] = []
        self.in_anom_: bool = False
        self.current_threshold_: float = self.fallback_threshold
        self.total_points_: int = 0
        self.quarantined_points_: int = 0

    def _reset_online_state(self) -> None:
        self.delta_ = 0.0
        self.v_ = self.p_u_ * (1.0 - self.p_u_)
        self.tau_ = 1.0
        self.strand_ = []
        self.in_anom_ = False
        self.current_threshold_ = self.anchor_
        self.total_points_ = 0
        self.quarantined_points_ = 0

    def fit(self, cal_scores: np.ndarray) -> "HalpernPrimalDualCalibrator":
        """Calibrate base anchor and dispersion scale on held-out calibration scores."""
        stats = compute_calibration_anchor(
            cal_scores,
            risk_level=self.risk_level,
            init_percentile=self.init_percentile,
        )
        if len(_clean_finite(cal_scores)) == 0:
            self.anchor_ = self.fallback_threshold
            self.med0_ = 0.0
            self.q98_0_ = self.fallback_threshold
            self.spread0_ = 1.0
        else:
            self.anchor_ = max(stats["anchor"], self.fallback_threshold if stats["med0"] == 0.0 and stats["th_full"] <= 1.0 else -math.inf)
            self.med0_ = stats["med0"]
            self.q98_0_ = stats["q98_0"]
            self.spread0_ = stats["spread0"]
        self._reset_online_state()
        return self

    def _apply_halpern_step(self, g: float) -> None:
        alpha0 = self.step_scale * self.spread0_
        self.v_ = 0.98 * self.v_ + 0.02 * (g * g)
        beta_t = self.gamma / (self.tau_ + self.gamma)
        alpha_t = alpha0 / math.sqrt(self.tau_ + self.gamma)
        self.delta_ = (1.0 - beta_t) * self.delta_ + alpha_t * (g / (math.sqrt(self.v_) + 1e-6))
        self.tau_ = min(self.tau_ + 1.0, self.max_tau)
        self.current_threshold_ = max(
            self.med0_ + 0.5 * self.spread0_,
            self.anchor_ + self.delta_,
        )

    def update(self, score: float) -> float:
        """Process a single scalar score causally and return the threshold active at this step."""
        s = float(score)
        th = self.current_threshold_
        if not math.isfinite(s):
            return th

        self.total_points_ += 1
        u_t = self.q98_0_ + self.delta_
        if s > th:
            self.strand_.clear()
            self.in_anom_ = True
            self.quarantined_points_ += 1
        elif s > u_t:
            if not self.in_anom_:
                self.strand_.append(s)
                if len(self.strand_) >= self.collar_commit_len:
                    for _ in self.strand_:
                        self._apply_halpern_step(1.0 - self.p_u_)
                    self.strand_.clear()
        else:
            if self.strand_ and not self.in_anom_:
                for _ in self.strand_:
                    self._apply_halpern_step(1.0 - self.p_u_)
            self.strand_.clear()
            self.in_anom_ = False
            self._apply_halpern_step(-self.p_u_)
        return th

    def process_stream(self, scores: np.ndarray) -> np.ndarray:
        """Process a stream statefully and return causal per-step thresholds."""
        arr = np.asarray(scores, dtype=np.float64).ravel()
        out = np.empty(len(arr), dtype=np.float64)
        for i, s in enumerate(arr):
            out[i] = self.update(float(s))
        return out

    def predict_thresholds(self, scores: np.ndarray) -> np.ndarray:
        """Reset online state to calibration anchor and evaluate a stream causally."""
        self._reset_online_state()
        return self.process_stream(scores)

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "anchor": self.anchor_,
            "current_threshold": self.current_threshold_,
            "delta": self.delta_,
            "tau": self.tau_,
            "total_points": self.total_points_,
            "quarantined_points": self.quarantined_points_,
        }


class StoppingBandCalibrator:
    """Two-barrier stopping-band calibrator on a nested collar [u_inner, u_outer].

    Follows score trajectories entering the inner contour u_inner:
    - Returning excursions (trajectories that drop back below u_inner without
      crossing u_outer) are committed to the nominal buffer to track regime drift.
    - Stopped crossings (trajectories that breach u_outer) have their entire
      excursion strand quarantined from the calibration buffer.
    """

    def __init__(
        self,
        risk_level: float = 1e-3,
        init_percentile: float = 98.0,
        buffer_size: int = 500,
        refit_step: int = 25,
        collar_commit_len: int = 16,
        anchor_blend: float = 0.45,
        fallback_threshold: float = 1.0,
    ) -> None:
        if not (0.0 < risk_level < 1.0):
            raise ValueError(f"risk_level must be in (0, 1), got {risk_level}")
        if not (50.0 <= init_percentile < 100.0):
            raise ValueError(f"init_percentile must be in [50, 100), got {init_percentile}")

        self.risk_level = float(risk_level)
        self.init_percentile = float(init_percentile)
        self.buffer_size = int(buffer_size)
        self.refit_step = int(refit_step)
        self.collar_commit_len = int(collar_commit_len)
        self.anchor_blend = float(anchor_blend)
        self.fallback_threshold = float(fallback_threshold)

        self.anchor_: float = self.fallback_threshold
        self.med0_: float = 0.0
        self.q98_0_: float = self.fallback_threshold
        self.spread0_: float = 1.0
        self.gap0_: float = 0.2
        self.init_buf_: list[float] = []

        self.u_in_: float = self.fallback_threshold
        self.u_out_: float = self.fallback_threshold
        self.current_threshold_: float = self.fallback_threshold
        self.buf_: list[float] = []
        self.strand_: list[float] = []
        self.crossed_: bool = False
        self.steps_since_refit_: int = 0
        self.total_points_: int = 0
        self.quarantined_strands_: int = 0

    def _reset_online_state(self) -> None:
        self.u_in_ = self.q98_0_
        self.u_out_ = self.anchor_
        self.current_threshold_ = self.u_out_
        self.buf_ = list(self.init_buf_)
        self.strand_ = []
        self.crossed_ = False
        self.steps_since_refit_ = 0
        self.total_points_ = 0
        self.quarantined_strands_ = 0

    def fit(self, cal_scores: np.ndarray) -> "StoppingBandCalibrator":
        """Fit initial stopping band [u_in, u_out] and collar clearance on calibration scores."""
        stats = compute_calibration_anchor(
            cal_scores,
            risk_level=self.risk_level,
            init_percentile=self.init_percentile,
        )
        tail_cal = stats["tail_cal"]
        self.anchor_ = stats["anchor"]
        self.med0_ = stats["med0"]
        self.q98_0_ = stats["q98_0"]
        self.spread0_ = stats["spread0"]
        self.gap0_ = max(self.anchor_ - self.q98_0_, 0.15 * self.spread0_)
        self.init_buf_ = [float(x) for x in tail_cal[-min(len(tail_cal), self.buffer_size) :]]
        self._reset_online_state()
        return self

    def _refit_band(self) -> None:
        arr = np.asarray(self.buf_, dtype=np.float64)
        q_in = float(np.quantile(arr, self.init_percentile / 100.0))
        med_b = float(np.median(arr))
        sp_b = max(q_in - med_b, 0.5 * self.spread0_)
        self.u_in_ = q_in
        scale_mult = 0.5 * (sp_b / self.spread0_ + 1.0)
        u_out_cand = self.u_in_ + self.gap0_ * scale_mult
        w = self.anchor_blend
        self.u_out_ = max(
            self.med0_ + 0.5 * self.spread0_,
            (1.0 - w) * u_out_cand + w * self.anchor_,
        )
        self.current_threshold_ = self.u_out_
        self.steps_since_refit_ = 0

    def update(self, score: float) -> float:
        """Process a single score through the stopping collar and return active u_outer."""
        s = float(score)
        th = self.u_out_
        if not math.isfinite(s):
            return th

        self.total_points_ += 1
        if s <= self.u_in_:
            if self.strand_ and not self.crossed_:
                self.buf_.extend(self.strand_)
                self.steps_since_refit_ += len(self.strand_)
            self.strand_.clear()
            self.crossed_ = False
            self.buf_.append(s)
            self.steps_since_refit_ += 1
        elif s <= self.u_out_:
            if not self.crossed_:
                self.strand_.append(s)
                if len(self.strand_) >= self.collar_commit_len:
                    self.buf_.extend(self.strand_)
                    self.steps_since_refit_ += len(self.strand_)
                    self.strand_.clear()
        else:
            if not self.crossed_:
                self.quarantined_strands_ += 1
            self.crossed_ = True
            self.strand_.clear()

        if len(self.buf_) > self.buffer_size:
            self.buf_ = self.buf_[-self.buffer_size :]
        if self.steps_since_refit_ >= self.refit_step and len(self.buf_) >= 80:
            self._refit_band()
        return th

    def process_stream(self, scores: np.ndarray) -> np.ndarray:
        arr = np.asarray(scores, dtype=np.float64).ravel()
        out = np.empty(len(arr), dtype=np.float64)
        for i, s in enumerate(arr):
            out[i] = self.update(float(s))
        return out

    def predict_thresholds(self, scores: np.ndarray) -> np.ndarray:
        self._reset_online_state()
        return self.process_stream(scores)

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "u_inner": self.u_in_,
            "u_outer": self.u_out_,
            "current_threshold": self.u_out_,
            "anchor": self.anchor_,
            "buffer_len": len(self.buf_),
            "total_points": self.total_points_,
            "quarantined_strands": self.quarantined_strands_,
        }


class PMIConformalCalibrator:
    """Pointwise Mutual Information (PMI) Context-Normalized Conformal Calibrator.

    Evaluates causal log-ratio surprise relative to a slow regime baseline mu_t:
        s_tilde_t = log((s_t + c_0) / (mu_t + c_0))
    during both calibration and inference. Combines EVT tail modeling on the
    context-normalized nonconformity scores with a Hoeffding-Bentkus finite-sample
    conformal quantile and SConU bulk-exchangeability gating.
    """

    def __init__(
        self,
        risk_level: float = 1e-3,
        init_percentile: float = 98.0,
        context_lr: float = 0.008,
        bulk_quantile: float = 0.92,
        anchor_blend: float = 0.35,
        collar_commit_len: int = 16,
        fallback_threshold: float = 1.0,
    ) -> None:
        if not (0.0 < risk_level < 1.0):
            raise ValueError(f"risk_level must be in (0, 1), got {risk_level}")
        if not (50.0 <= init_percentile < 100.0):
            raise ValueError(f"init_percentile must be in [50, 100), got {init_percentile}")

        self.risk_level = float(risk_level)
        self.init_percentile = float(init_percentile)
        self.context_lr = float(context_lr)
        self.bulk_quantile = float(bulk_quantile)
        self.anchor_blend = float(anchor_blend)
        self.collar_commit_len = int(collar_commit_len)
        self.fallback_threshold = float(fallback_threshold)

        self.anchor_: float = self.fallback_threshold
        self.med0_: float = 0.0
        self.spread0_: float = 1.0
        self.c0_: float = 1.0
        self.mu_ref_: float = 0.0
        self.pmi_crit_: float = 0.5
        self.pmi_bulk_: float = 0.1

        self.mu_t_: float = 0.0
        self.strand_: list[float] = []
        self.in_anom_: bool = False
        self.current_threshold_: float = self.fallback_threshold
        self.total_points_: int = 0

    def _compute_active_threshold(self) -> float:
        th_pmi = (self.mu_t_ + self.c0_) * math.exp(self.pmi_crit_) - self.c0_
        w = self.anchor_blend
        return max(
            self.med0_ + 0.5 * self.spread0_,
            (1.0 - w) * th_pmi + w * self.anchor_,
        )

    def _reset_online_state(self) -> None:
        self.mu_t_ = self.mu_ref_
        self.strand_ = []
        self.in_anom_ = False
        self.total_points_ = 0
        self.current_threshold_ = self._compute_active_threshold()

    def fit(self, cal_scores: np.ndarray) -> "PMIConformalCalibrator":
        """Run causal context normalization over calibration scores and fit HB/EVT quantile."""
        clean = _clean_finite(cal_scores)
        stats = compute_calibration_anchor(
            clean,
            risk_level=self.risk_level,
            init_percentile=self.init_percentile,
        )
        # Because mu_t already tracks downward calibration trends chronologically,
        # keep the static anchor at the full-sample upper envelope + subblock drift allowance.
        q_emp = float(np.quantile(clean, 1.0 - self.risk_level)) if len(clean) > 0 else self.fallback_threshold
        self.anchor_ = max(stats["th_full"], q_emp) + 0.35 * stats["drift_std"]
        self.med0_ = stats["med0"]
        self.spread0_ = stats["spread0"]
        self.c0_ = max(abs(self.med0_), self.spread0_, 1.0)

        if len(clean) < 10 or float(np.ptp(clean)) < 1e-12:
            self.mu_ref_ = self.med0_
            self.pmi_crit_ = math.log(max(self.anchor_ + self.c0_, 1e-6) / max(self.med0_ + self.c0_, 1e-6))
            self.pmi_bulk_ = 0.5 * self.pmi_crit_
            self._reset_online_state()
            return self

        eta = self.context_lr
        mu_t = float(np.median(clean[: min(len(clean), 64)]))
        pmi_cal = np.empty(len(clean), dtype=np.float64)
        for i, s in enumerate(clean):
            pmi_cal[i] = math.log(max(float(s) + self.c0_, 1e-6) / max(mu_t + self.c0_, 1e-6))
            mu_t = (1.0 - eta) * mu_t + eta * float(s)

        warm = min(len(clean) // 4, 64)
        pmi_valid = pmi_cal[warm:] if len(clean) - warm >= 50 else pmi_cal
        evt_pmi = EVTCalibrator(
            risk_level=self.risk_level,
            init_percentile=self.init_percentile,
            adaptive_kurtosis=False,
        ).fit(pmi_valid)
        n = len(pmi_valid)
        k_hb = min(n - 1, max(0, int(math.ceil((n + 1) * (1.0 - self.risk_level))) - 1))
        pmi_emp = float(np.sort(pmi_valid)[k_hb])
        self.pmi_crit_ = max(float(evt_pmi.threshold_), pmi_emp)
        self.pmi_bulk_ = float(np.quantile(pmi_valid, self.bulk_quantile))
        self.mu_ref_ = mu_t
        self._reset_online_state()
        return self

    def update(self, score: float) -> float:
        """Evaluate causal threshold at current step and update bulk context baseline."""
        s = float(score)
        th = self._compute_active_threshold()
        self.current_threshold_ = th
        if not math.isfinite(s):
            return th

        self.total_points_ += 1
        eta = self.context_lr
        u_bulk = (self.mu_t_ + self.c0_) * math.exp(self.pmi_bulk_) - self.c0_
        if s > th:
            self.strand_.clear()
            self.in_anom_ = True
        elif s > u_bulk:
            if not self.in_anom_:
                self.strand_.append(s)
                if len(self.strand_) >= self.collar_commit_len:
                    for x in self.strand_:
                        self.mu_t_ = (1.0 - eta) * self.mu_t_ + eta * x
                    self.strand_.clear()
        else:
            if self.strand_ and not self.in_anom_:
                for x in self.strand_:
                    self.mu_t_ = (1.0 - eta) * self.mu_t_ + eta * x
            self.strand_.clear()
            self.in_anom_ = False
            self.mu_t_ = (1.0 - eta) * self.mu_t_ + eta * s
            self.mu_t_ = 0.997 * self.mu_t_ + 0.003 * self.mu_ref_

        self.current_threshold_ = self._compute_active_threshold()
        return th

    def process_stream(self, scores: np.ndarray) -> np.ndarray:
        arr = np.asarray(scores, dtype=np.float64).ravel()
        out = np.empty(len(arr), dtype=np.float64)
        for i, s in enumerate(arr):
            out[i] = self.update(float(s))
        return out

    def predict_thresholds(self, scores: np.ndarray) -> np.ndarray:
        self._reset_online_state()
        return self.process_stream(scores)

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "anchor": self.anchor_,
            "mu_t": self.mu_t_,
            "mu_ref": self.mu_ref_,
            "pmi_crit": self.pmi_crit_,
            "current_threshold": self.current_threshold_,
            "total_points": self.total_points_,
        }
