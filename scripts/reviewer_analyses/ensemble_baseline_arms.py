"""Window-level statistics for detectors that compete with the 224-bit panel on the same latent residuals.

Every function here is calibrated only on the nominal calibration windows ``[:n_cal]`` of the
validation block and returns per-window statistics for the validation and test streams. Timeline
mapping and threshold calibration are done by the caller so that every arm shares the panel's
exact calibration / 64-step gap / held-out split.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.signal import lfilter
from scipy.stats import norm
from sklearn.covariance import LedoitWolf

D = 32
# Perspective groups of build_224_perspective_dials: positions (4 horizons x 32), drift, fluctuation,
# endpoint difference (right-edge acceleration).
GROUPS = {
    "P": np.arange(0, 4 * D),
    "D": np.arange(4 * D, 5 * D),
    "F": np.arange(5 * D, 6 * D),
    "E": np.arange(6 * D, 7 * D),
}
P4 = np.arange(3 * D, 4 * D)  # terminal-horizon (m=4) position dials


def robust_z(va: np.ndarray, te: np.ndarray, n_cal: int) -> tuple[np.ndarray, np.ndarray]:
    """One-sided robust z-scores (median/MAD on calibration windows), identical to the panel."""
    med = np.median(va[:n_cal], axis=0, keepdims=True)
    mad = 1.4826 * np.median(np.abs(va[:n_cal] - med), axis=0, keepdims=True) + 1e-4
    return np.maximum(0.0, (va - med) / mad), np.maximum(0.0, (te - med) / mad)


def conformal_pvalues(cal: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Per-dial split-conformal p-values p_j = (1 + #{cal_j >= x_j}) / (n_cal + 1)."""
    n = len(cal)
    srt = np.sort(cal, axis=0)
    ge = np.empty(x.shape, dtype=np.float64)
    for j in range(x.shape[1]):
        ge[:, j] = n - np.searchsorted(srt[:, j], x[:, j], side="left")
    return (1.0 + ge) / (n + 1.0)


def bh_min_adjusted(p: np.ndarray) -> np.ndarray:
    """Smallest Benjamini-Hochberg adjusted p-value per row (any rejection at alpha iff <= alpha)."""
    m = p.shape[1]
    ps = np.sort(p, axis=1) * m / np.arange(1, m + 1)
    return np.minimum(np.minimum.accumulate(ps[:, ::-1], axis=1)[:, -1], 1.0)


def pvalue_statistics(va_dials: np.ndarray, te_dials: np.ndarray, n_cal: int) -> dict[str, tuple]:
    """Bonferroni, BH, Fisher and Stouffer statistics of per-dial conformal p-values."""
    cal = va_dials[:n_cal]
    m = va_dials.shape[1]
    p_hi = 1.0 - 0.5 / (n_cal + 1.0)  # keep Phi^{-1}(1-p) finite for p = 1
    out: dict[str, list] = {"bonf": [], "bh": [], "fisher": [], "stouffer": [], "min_p": []}
    for x in (va_dials, te_dials):
        p = conformal_pvalues(cal, x)
        out["bonf"].append(-np.log10(np.minimum(p.min(axis=1) * m, 1.0)))
        out["bh"].append(-np.log10(bh_min_adjusted(p)))
        out["fisher"].append(-2.0 * np.log(p).sum(axis=1))
        out["stouffer"].append(norm.isf(np.minimum(p, p_hi)).sum(axis=1) / np.sqrt(m))
        out["min_p"].append(p.min(axis=1))
    return {k: tuple(v) for k, v in out.items()}


def _cov_cholesky(cal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calibration mean and Cholesky factor of the Ledoit-Wolf shrunk covariance."""
    mu = cal.mean(axis=0)
    cov = LedoitWolf(assume_centered=False).fit(cal).covariance_
    cov = cov + 1e-6 * np.trace(cov) / len(cov) * np.eye(len(cov))
    return mu, np.linalg.cholesky(cov)


def _quad(x_centered: np.ndarray, chol: np.ndarray) -> np.ndarray:
    return np.sum(solve_triangular(chol, x_centered.T, lower=True) ** 2, axis=0)


def hotelling_t2(va_x: np.ndarray, te_x: np.ndarray, n_cal: int) -> tuple[np.ndarray, np.ndarray]:
    """Hotelling T^2 with calibration mean and Ledoit-Wolf covariance (128 signed coordinates)."""
    mu, chol = _cov_cholesky(va_x[:n_cal])
    return _quad(va_x - mu, chol), _quad(te_x - mu, chol)


def mewma(va_x: np.ndarray, te_x: np.ndarray, n_cal: int, lam: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """MEWMA statistic Z_t' Sigma_Z^{-1} Z_t, Sigma_Z = lam/(2-lam) Sigma; restarts at each block start."""
    mu, chol = _cov_cholesky(va_x[:n_cal])
    chol_z = chol * np.sqrt(lam / (2.0 - lam))

    def run(x: np.ndarray) -> np.ndarray:
        z = lfilter([lam], [1.0, -(1.0 - lam)], x - mu, axis=0)
        return _quad(z, chol_z)

    return run(va_x), run(te_x)


def _cusum_paths(x: np.ndarray, k: float) -> np.ndarray:
    """Two-sided tabular CUSUM per coordinate, returns max(S+, S-) path (N, d)."""
    sp = np.zeros(x.shape[1])
    sm = np.zeros(x.shape[1])
    out = np.empty_like(x)
    for t in range(len(x)):
        sp = np.maximum(0.0, sp + x[t] - k)
        sm = np.maximum(0.0, sm - x[t] - k)
        out[t] = np.maximum(sp, sm)
    return out


def cusum_statistics(
    va_x: np.ndarray, te_x: np.ndarray, n_cal: int, k: float = 0.5, alpha: float = 0.005
) -> dict[str, tuple]:
    """Per-coordinate CUSUM on calibration-standardized coordinates.

    ``max``: max_j C_j (threshold calibrated by the caller).
    ``bonf``: max_j C_j / h_j with h_j the calibration (1 - alpha/d) quantile of coordinate j's own
    CUSUM path; alarm iff any coordinate exceeds its own limit (statistic >= 1).
    """
    mu = va_x[:n_cal].mean(axis=0)
    sd = np.maximum(va_x[:n_cal].std(axis=0), 1e-6)
    pv, pt = _cusum_paths((va_x - mu) / sd, k), _cusum_paths((te_x - mu) / sd, k)
    h = np.maximum(np.quantile(pv[:n_cal], 1.0 - alpha / va_x.shape[1], axis=0), 1e-6)
    return {"max": (pv.max(axis=1), pt.max(axis=1)), "bonf": ((pv / h).max(axis=1), (pt / h).max(axis=1))}


def effective_rank(x: np.ndarray) -> tuple[float, float]:
    """Entropy effective rank and participation ratio of the correlation spectrum of columns of x."""
    sd = x.std(axis=0)
    keep = sd > 1e-9
    if keep.sum() < 2:
        return float(keep.sum()), float(keep.sum())
    corr = np.nan_to_num(np.corrcoef(x[:, keep], rowvar=False))
    lam = np.clip(np.linalg.eigvalsh(corr), 0.0, None)
    p = lam / lam.sum()
    nz = p[p > 0]
    return float(np.exp(-(nz * np.log(nz)).sum())), float(lam.sum() ** 2 / (lam**2).sum())
