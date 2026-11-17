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
