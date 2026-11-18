"""Shared evaluation, calibration, and drift-monitoring primitives for TNNLS revision experiments.

Provides:
  - PID-scoped atomic CSV/JSON persistence (`atomic_write_csv`, `atomic_write_json`).
  - Analytical & Monte Carlo helpers for Proposition 1 (hijacker-coordinate threshold masking
    vs. exact coordinate-scale invariance of per-dial tripwires and quorums).
  - Sokal integrated autocorrelation (`tau_sokal`), effective sample size (`effective_sample_size`),
    and Moving-Block Bootstrap (`mbb_quorum_thresholds`) / Non-Overlapping Block Quantile
    calibrators for dependent nominal count series.
  - Effective embedding rank (`compute_rankme`) and `FrozenLatentDriftMonitor` for long-horizon
    progressive degradation and two-timescale hybrid detection.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats

from scripts.reviewer_analyses.calibration_quantile_reliability import tau_sokal


def atomic_write_csv(df: pd.DataFrame, target: Path) -> None:
    """Write DataFrame to `target` atomically via PID-scoped temporary file and `os.fsync`."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        df.to_csv(fh, index=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)


def atomic_write_json(payload: dict[str, Any], target: Path) -> None:
    """Write JSON dictionary to `target` atomically via PID-scoped temporary file and `os.fsync`."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)


def effective_sample_size(series: np.ndarray) -> tuple[float, float]:
    """Return `(tau_int, n_eff)` using Sokal's automatic window estimator (`c=5`)."""
    arr = np.asarray(series, dtype=np.float64)
    if arr.size <= 8 or float(np.std(arr)) < 1e-12:
        return 1.0, float(max(arr.size, 1))
    tau = float(max(1.0, tau_sokal(arr)))
    return tau, float(arr.size / tau)


def mbb_quorum_thresholds(
    cal_counts: np.ndarray,
    floor: float = 3.5,
    q: float = 0.995,
    safety: float = 1.12,
    block_len: int | None = None,
    n_boot: int = 400,
    seed: int = 42,
) -> dict[str, float]:
    """Compute deployed, Moving-Block Bootstrap (MBB), and block-subsampled quorum thresholds.

    Parameters
    ----------
    cal_counts : np.ndarray
        1D nominal calibration switch-count series of length `N_cal`.
    floor : float
        Minimum quorum floor (`3.5` for 128-dial Lite).
    q : float
        Target nominal quantile (`0.995`).
    safety : float
        Multiplicative quorum safety factor (`1.12`).
    block_len : int | None
        Moving-block length; defaults to `clip(ceil(2 * tau_int), 16, max(16, N_cal // 4))`.
    n_boot : int
        Number of bootstrap resamples.
    seed : int
        Deterministic RNG seed.
    """
    arr = np.asarray(cal_counts, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {
            "deployed": floor,
            "mbb_median": floor,
            "mbb_mean": floor,
            "mbb_p90": floor,
            "block_subsampled": floor,
            "block_len": 1.0,
        }

    deployed_q = float(np.percentile(arr, 100.0 * q))
    deployed = max(safety * deployed_q, floor)

    tau, _ = effective_sample_size(arr)
    if block_len is None:
        b_len = int(np.clip(np.ceil(2.0 * tau), 16, max(16, n // 4)))
    else:
        b_len = int(np.clip(block_len, 1, max(1, n)))
    b_len = min(b_len, n)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / b_len))
    max_start = max(1, n - b_len + 1)
    starts = rng.integers(0, max_start, size=(n_boot, n_blocks))
    offsets = np.arange(b_len, dtype=np.int64)[None, None, :]
    idx = (starts[:, :, None] + offsets).reshape(n_boot, n_blocks * b_len)[:, :n]
    boot_samples = arr[idx]
    boot_q = np.percentile(boot_samples, 100.0 * q, axis=1)

    # Non-overlapping block-max / thinned quantile control
    stride = max(1, int(np.round(tau)))
    thinned = arr[::stride]
    thinned_q = float(np.percentile(thinned, 100.0 * q)) if thinned.size >= 10 else deployed_q

    return {
        "deployed": float(deployed),
        "mbb_median": float(max(safety * float(np.median(boot_q)), floor)),
        "mbb_mean": float(max(safety * float(np.mean(boot_q)), floor)),
        "mbb_p90": float(max(safety * float(np.percentile(boot_q, 90.0)), floor)),
        "block_subsampled": float(max(safety * thinned_q, floor)),
        "block_len": float(b_len),
    }


def compute_rankme(embeddings: np.ndarray, eps: float = 1e-7) -> float:
    """Compute effective rank `RankMe(Z) = exp(-sum_k p_k log p_k)` of `[N, D]` embeddings."""
    z = np.asarray(embeddings, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 2:
        return 1.0
    z_centered = z - z.mean(axis=0, keepdims=True)
    svals = np.linalg.svd(z_centered, full_matrices=False, compute_uv=False)
    s_sum = float(np.sum(svals))
    if s_sum <= eps:
        return 1.0
    p = svals / s_sum + eps
    p = p / np.sum(p)
    return float(np.exp(-np.sum(p * np.log(p))))


def simulate_hijacker_masking_vs_quorum(
    dim: int = 32,
    n_hijack: int = 2,
    eta: float = 5.0,
    n_fault_coords: int = 10,
    fault_amp: float = 3.8,
    n_cal: int = 2000,
    dial_q: float = 0.985,
    quorum_q: float = 0.995,
    safety: float = 1.12,
    floor: float = 3.5,
    seed: int = 42,
) -> dict[str, float | bool]:
    """Verify Proposition 1: coordinate-scale invariance of quorum vs. scalar threshold masking."""
    rng = np.random.default_rng(seed)
    u_base = rng.standard_normal((n_cal, dim))

    scales = np.ones(dim, dtype=np.float64)
    scales[:n_hijack] = np.sqrt(1.0 + eta**2)
    u_drifted = u_base * scales[None, :]

    # 1. Global quadratic scalar detector
    s_cal_base = np.sum(u_base**2, axis=1)
    s_cal_drift = np.sum(u_drifted**2, axis=1)
    tau_glob_base = float(np.quantile(s_cal_base, quorum_q))
    tau_glob_drift = float(np.quantile(s_cal_drift, quorum_q))

    # 2. Coordinate-level tripwires & quorum (with tie-break floor -> 0 for exact homogeneity test)
    tau_coord_base = np.quantile(np.abs(u_base), dial_q, axis=0)
    tau_coord_drift = np.quantile(np.abs(u_drifted), dial_q, axis=0)
    trips_cal_base = (np.abs(u_base) > tau_coord_base[None, :]).sum(axis=1)
    trips_cal_drift = (np.abs(u_drifted) > tau_coord_drift[None, :]).sum(axis=1)
    k_quorum_base = max(safety * float(np.quantile(trips_cal_base, quorum_q)), floor)
    k_quorum_drift = max(safety * float(np.quantile(trips_cal_drift, quorum_q)), floor)

    # 3. Live fault vector on `n_fault_coords` non-hijacker coordinates while hijackers sit at 0
    u_fault = np.zeros(dim, dtype=np.float64)
    u_fault[n_hijack : n_hijack + n_fault_coords] = fault_amp
    s_fault = float(np.sum(u_fault**2))
    n_trips_fault = int(np.sum(np.abs(u_fault) > tau_coord_drift))

    return {
        "tau_glob_base": tau_glob_base,
        "tau_glob_drift": tau_glob_drift,
        "s_fault": s_fault,
        "scalar_detects_without_hijack": bool(s_fault > tau_glob_base),
        "scalar_masked_under_hijack": bool(s_fault <= tau_glob_drift),
        "k_quorum_base": float(k_quorum_base),
        "k_quorum_drift": float(k_quorum_drift),
        "quorum_exact_invariance": bool(np.array_equal(trips_cal_base, trips_cal_drift)),
        "quorum_trips_on_fault": float(n_trips_fault),
        "quorum_detects_under_hijack": bool(n_trips_fault > k_quorum_drift),
    }


@dataclass
class FrozenDriftDecision:
    """Stream-aligned scores and binary alarms from `FrozenLatentDriftMonitor`."""

    cal_scores: np.ndarray
    hold_scores: np.ndarray
    test_scores: np.ndarray
    threshold: float
    hold_pred: np.ndarray
    test_pred: np.ndarray


class FrozenLatentDriftMonitor:
    """Complementary long-horizon reference monitor for slow progressive ramps (`L >= C`).

    Anchors pre-terminal-LayerNorm or context summaries on training nominal windows (`X_fit`),
    tracks a causal EWMA trajectory (`alpha = 0.02`, ~50-step memory), scores Mahalanobis
    displacement from `X_fit`, and calibrates its alarm threshold strictly on `X_cal`.
    Requires zero changes to `LaminarJEPALiteModel` weights or loss.
    """

    def __init__(self, ewma_alpha: float = 0.02, reg: float = 1e-2, safety: float = 1.15, q: float = 0.995):
        self.ewma_alpha = float(ewma_alpha)
        self.reg = float(reg)
        self.safety = float(safety)
        self.q = float(q)
        self.mean_: np.ndarray | None = None
        self.prec_: np.ndarray | None = None

    @staticmethod
    @torch.no_grad()
    def extract_pre_ln_context_summaries(
        model: torch.nn.Module, windows: np.ndarray, context_len: int = 256, batch_size: int = 512
    ) -> np.ndarray:
        """Extract unnormalized fused encoder features (`fused` before terminal `F.layer_norm`) + window channel means."""
        model.eval()
        enc = getattr(model, "context_encoder", model)
        device = next(enc.parameters()).device
        chunks: list[np.ndarray] = []

        with torch.inference_mode():
            for i in range(0, len(windows), batch_size):
                ctx_np = np.ascontiguousarray(windows[i : i + batch_size, :context_len], dtype=np.float32)
                ctx = torch.from_numpy(ctx_np).to(device)
                # Combine latent summary z_ctx with standardized window mean/RMS summary so slow DC shift is preserved
                z_ctx = enc(ctx).detach().cpu().numpy().astype(np.float32)
                win_mean = ctx_np.mean(axis=1)
                win_std = ctx_np.std(axis=1)
                chunks.append(np.concatenate([z_ctx, win_mean, win_std], axis=1))
        return np.concatenate(chunks, axis=0)

    def fit_reference(self, fit_features: np.ndarray) -> None:
        """Fit reference mean and regularized inverse covariance on `X_fit` features."""
        x = np.asarray(fit_features, dtype=np.float64)
        self.mean_ = x.mean(axis=0)
        xc = x - self.mean_[None, :]
        cov = (xc.T @ xc) / max(len(x) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0.0)
        inv_eigs = 1.0 / (eigvals + self.reg * max(float(np.median(eigvals)), 1.0))
        self.prec_ = (eigvecs * inv_eigs[None, :]) @ eigvecs.T

    def _causal_ewma_mahalanobis(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.prec_ is None:
            raise RuntimeError("Call fit_reference() before scoring.")
        x = np.asarray(features, dtype=np.float64) - self.mean_[None, :]
        ewma = np.empty_like(x)
        alpha = self.ewma_alpha
        ewma[0] = x[0]
        for t in range(1, len(x)):
            ewma[t] = alpha * x[t] + (1.0 - alpha) * ewma[t - 1]
        quad = np.einsum("ti,ij,tj->t", ewma, self.prec_, ewma)
        return np.sqrt(np.maximum(quad, 0.0)).astype(np.float32)

    def calibrate_and_score(
        self,
        val_features: np.ndarray,
        test_features: np.ndarray,
        n_cal_w: int,
        suspect_size: int = 64,
    ) -> FrozenDriftDecision:
        """Calibrate causal threshold on `val[:n_cal_w]` and score isolated hold + test windows."""
        v_sc = self._causal_ewma_mahalanobis(val_features)
        t_sc = self._causal_ewma_mahalanobis(test_features)
        cal_sc = v_sc[:n_cal_w]
        hold_sc = v_sc[n_cal_w + suspect_size :]
        med = float(np.median(cal_sc))
        mad = float(np.median(np.abs(cal_sc - med))) + 1e-6
        th = max(self.safety * float(np.quantile(cal_sc, self.q)), med + 6.0 * mad)
        return FrozenDriftDecision(
            cal_scores=cal_sc,
            hold_scores=hold_sc,
            test_scores=t_sc,
            threshold=th,
            hold_pred=(hold_sc > th).astype(int),
            test_pred=(t_sc > th).astype(int),
        )


from src.scoring.laminar_lite_panel import build_position_drift_dials


@dataclass
class QuarantinedRecalibrationController:
    """Two-stage quarantined recalibration gate separating drift detection from update permission."""

    tau_dials: np.ndarray
    k_quorum: float
    dial_q: float = 0.985
    quorum_q: float = 0.995
    safety: float = 1.12
    floor: float = 3.5
    ks_thresh: float = 0.50
    max_buffer_windows: int = 400
    state: str = "NORMAL"

    def __post_init__(self) -> None:
        self.tau_dials = np.asarray(self.tau_dials, dtype=np.float32).copy()
        self.k_quorum = float(self.k_quorum)
        self._buffer: list[np.ndarray] = []
        self._rollback_snapshot: tuple[np.ndarray, float] | None = None

    @property
    def buffer_len(self) -> int:
        return int(sum(len(chunk) for chunk in self._buffer))

    def process_batch(
        self, dials_batch: np.ndarray, ks_distance: float
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Score `dials_batch` against active thresholds; quarantine shift samples without adapting."""
        d_arr = np.asarray(dials_batch, dtype=np.float32)
        trips = (d_arr > self.tau_dials.reshape(1, -1)).sum(axis=1).astype(np.float32)
        alarms = (trips > self.k_quorum).astype(np.int64)
        if float(ks_distance) >= self.ks_thresh or self.state == "QUARANTINE_ALERT":
            self.state = "QUARANTINE_ALERT"
            remaining = self.max_buffer_windows - self.buffer_len
            if remaining > 0:
                self._buffer.append(d_arr[:remaining].copy())
        return alarms, trips, self.state

    def confirm_mode_transition(self, mode_confirmed: bool = True) -> bool:
        """Apply buffered thresholds only upon external confirmation; discard otherwise."""
        if not mode_confirmed or self.buffer_len < 16:
            self._buffer.clear()
            self.state = "NORMAL"
            return False
        buf = np.concatenate(self._buffer, axis=0)
        self._rollback_snapshot = (self.tau_dials.copy(), float(self.k_quorum))
        new_tau = (np.percentile(buf, 100.0 * self.dial_q, axis=0) + 1e-4).astype(np.float32)
        new_trips = (buf > new_tau.reshape(1, -1)).sum(axis=1)
        new_k = max(float(np.percentile(new_trips, 100.0 * self.quorum_q)) * self.safety, self.floor)
        self.tau_dials = new_tau
        self.k_quorum = float(new_k)
        self._buffer.clear()
        self.state = "NORMAL"
        return True

    def rollback(self) -> bool:
        """Restore the pre-update primary reference thresholds if a mode transition is revoked."""
        self._buffer.clear()
        self.state = "NORMAL"
        if self._rollback_snapshot is None:
            return False
        prev_tau, prev_k = self._rollback_snapshot
        self.tau_dials = prev_tau.copy()
        self.k_quorum = float(prev_k)
        return True


@torch.no_grad()
def loco_channel_attribution(
    model: torch.nn.Module,
    panel_tripwires: np.ndarray,
    x_win: np.ndarray,
    context_len: int = 256,
    w_white: np.ndarray | None = None,
    rms_raw_cal: np.ndarray | None = None,
) -> np.ndarray:
    """Leave-one-channel-out (LOCO) attribution via one batched forward pass over `[K+1, W, K]`."""
    x_arr = np.asarray(x_win, dtype=np.float32)
    if x_arr.ndim != 2 or x_arr.shape[1] < 2:
        return np.zeros(x_arr.shape[1] if x_arr.ndim == 2 else 1, dtype=np.float32)
    k_ch = int(x_arr.shape[1])
    batch_np = np.repeat(x_arr[None, :, :], k_ch + 1, axis=0)
    idx = np.arange(k_ch)
    batch_np[idx + 1, :, idx] = 0.0  # Impute standardized training mean (0.0) per left-out channel

    raw_res, white_res = model.extract_raw_and_whitened_residuals(
        batch_np, context_len=context_len, batch_size=len(batch_np)
    )
    if w_white is not None and rms_raw_cal is not None:
        ew = w_white * white_res + (1.0 - w_white) * (raw_res / rms_raw_cal)
    else:
        ew = white_res
    dials = build_position_drift_dials(ew)
    tau = np.asarray(panel_tripwires, dtype=np.float32).reshape(1, -1)
    trips = (dials > tau).sum(axis=1).astype(np.float32)
    return np.maximum(0.0, trips[0] - trips[1:])


def _damped_zca_sqrt_np(cov: np.ndarray, reg: float = 1e-3, eigen_basis: bool = False) -> np.ndarray:
    """Compute damped precision square root in float64 (symmetric ZCA or eigenbasis PCA)."""
    sym = 0.5 * (cov + cov.T)
    evals, evecs = np.linalg.eigh(sym)
    pos = np.maximum(evals, 0.0)
    scales = np.sqrt(pos) / (pos + reg)
    if eigen_basis:
        return evecs * scales[None, :]
    return (evecs * scales[None, :]) @ evecs.T


def fit_horizon_whitening_operator(
    fit_raw: np.ndarray,
    mode: str = "full_zca",
    reg: float = 1e-3,
) -> dict[str, Any]:
    """Fit multi-horizon residual whitening operator on `fit_raw` of shape `(N_fit, M, D)`.

    Supported `mode` values:
      - `"full_zca"`: Dense `(MD) x (MD)` symmetric ZCA square root (cross-horizon + cross-coord).
      - `"shared_kronecker_zca"`: Single shared `D x D` ZCA (`I_M (x) Omega_D^{1/2}`), zero
        cross-horizon mixing and exact commutation with horizon mean/fluctuation.
      - `"block_diag_zca"`: Independent `D x D` ZCA per horizon `m` (`(+)_{m=1}^M Omega_m^{1/2}`).
      - `"pre_dial_zca"`: Independent `D x D` ZCA per raw horizon `m` plus separate `D x D` ZCA
        on the unwhitened horizon-mean vector `mean_m(e_m - mu_m)`.
      - `"pca_eigen"`: Non-symmetric `(MD) x (MD)` eigenbasis whitening `U diag(...)` control.
    """
    r64 = np.asarray(fit_raw, dtype=np.float64)
    if r64.ndim != 3:
        raise ValueError(f"Expected 3D fit_raw (N, M, D), got {r64.shape}.")
    n_fit, m_cnt, dim = r64.shape
    mu = r64.mean(axis=0, keepdims=True)
    cent = r64 - mu

    if mode in ("full_zca", "pca_eigen"):
        flat = cent.reshape(n_fit, m_cnt * dim)
        cov = (flat.T @ flat) / max(n_fit - 1, 1)
        w_mat = _damped_zca_sqrt_np(cov, reg=reg, eigen_basis=(mode == "pca_eigen"))
        return {"mode": mode, "mu": mu, "w_mat": w_mat}

    if mode == "shared_kronecker_zca":
        pooled = cent.reshape(n_fit * m_cnt, dim)
        cov_d = (pooled.T @ pooled) / max(len(pooled) - 1, 1)
        w_d = _damped_zca_sqrt_np(cov_d, reg=reg, eigen_basis=False)
        return {"mode": mode, "mu": mu, "w_d": w_d}

    if mode == "block_diag_zca":
        w_blocks = []
        for m in range(m_cnt):
            cm = cent[:, m, :]
            cov_m = (cm.T @ cm) / max(n_fit - 1, 1)
            w_blocks.append(_damped_zca_sqrt_np(cov_m, reg=reg, eigen_basis=False))
        return {"mode": mode, "mu": mu, "w_blocks": np.stack(w_blocks, axis=0)}

    if mode == "pre_dial_zca":
        w_blocks = []
        for m in range(m_cnt):
            cm = cent[:, m, :]
            cov_m = (cm.T @ cm) / max(n_fit - 1, 1)
            w_blocks.append(_damped_zca_sqrt_np(cov_m, reg=reg, eigen_basis=False))
        c_bar = cent.mean(axis=1)
        cov_bar = (c_bar.T @ c_bar) / max(n_fit - 1, 1)
        w_bar = _damped_zca_sqrt_np(cov_bar, reg=reg, eigen_basis=False)
        return {"mode": mode, "mu": mu, "w_blocks": np.stack(w_blocks, axis=0), "w_bar": w_bar}

    raise ValueError(f"Unsupported whitening mode: {mode!r}")


def apply_horizon_whitening_operator(raw: np.ndarray, op: dict[str, Any]) -> np.ndarray:
    """Apply fitted whitening operator from `fit_horizon_whitening_operator` to `(N, M, D)`."""
    r64 = np.asarray(raw, dtype=np.float64)
    n_win, m_cnt, dim = r64.shape
    cent = r64 - op["mu"]
    mode = str(op["mode"])
    if mode in ("full_zca", "pca_eigen"):
        return (cent.reshape(n_win, m_cnt * dim) @ op["w_mat"]).reshape(n_win, m_cnt, dim).astype(np.float32)
    if mode == "shared_kronecker_zca":
        return np.einsum("nmd,dk->nmk", cent, op["w_d"]).astype(np.float32)
    if mode in ("block_diag_zca", "pre_dial_zca"):
        return np.einsum("nmd,mdk->nmk", cent, op["w_blocks"]).astype(np.float32)
    raise ValueError(f"Unsupported whitening mode: {mode!r}")


