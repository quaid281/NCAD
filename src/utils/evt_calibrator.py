"""Extreme Value Theory (EVT) / Peaks-Over-Threshold (SPOT) Calibrator.

Under the Pickands-Balkema-de Haan theorem, the excesses of normal residual scores
above a sufficiently high threshold asymptotically follow a Generalized Pareto
Distribution (GPD). This module fits GPD parameters (gamma, sigma) to compute
distribution-aware, mathematically grounded anomaly thresholds and calibrated
outlier probabilities without requiring labeled anomaly data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import optimize


@dataclass
class GPDFitResult:
    gamma: float  # shape parameter (xi)
    sigma: float  # scale parameter
    threshold_init: float  # initial baseline threshold (t)
    n_total: int  # total sample size (N)
    n_excess: int  # number of excesses (N_t)
    fit_converged: bool
    method: str


@dataclass
class EVTThresholdResult:
    threshold: float
    risk_level: float
    gpd_fit: GPDFitResult
    method: str
    is_fallback: bool
    plateau_adjusted: bool = False

    def to_dict(self) -> dict:
        return {
            "threshold": float(self.threshold),
            "risk_level": float(self.risk_level),
            "gamma": float(self.gpd_fit.gamma),
            "sigma": float(self.gpd_fit.sigma),
            "threshold_init": float(self.gpd_fit.threshold_init),
            "n_total": int(self.gpd_fit.n_total),
            "n_excess": int(self.gpd_fit.n_excess),
            "fit_converged": bool(self.gpd_fit.fit_converged),
            "method": self.method,
            "is_fallback": bool(self.is_fallback),
            "plateau_adjusted": bool(self.plateau_adjusted),
        }


def _grimshaw_gpd_fit(excesses: np.ndarray) -> Optional[Tuple[float, float]]:
    """Grimshaw's algorithm for Maximum Likelihood Estimation of GPD.

    Solves the 1D equation u(theta) * v(theta) - 1 = 0, where theta = gamma / sigma.
    """
    y = np.sort(np.asarray(excesses, dtype=np.float64))
    n = len(y)
    if n < 5:
        return None

    y_max = y[-1]
    y_mean = np.mean(y)
    if y_max <= 1e-12 or y_mean <= 1e-12:
        return None

    def u_func(theta: float) -> float:
        return np.mean(1.0 / (1.0 + theta * y))

    def v_func(theta: float) -> float:
        return 1.0 + np.mean(np.log1p(theta * y))

    def objective(theta: float) -> float:
        return u_func(theta) * v_func(theta) - 1.0

    # Search interval for theta: theta > -1 / y_max
    # We test roots in two intervals: [-1/y_max + eps, 0) and (0, 2*(y_mean - y[0]) / (y[0]*y_mean)]
    eps = 1e-6
    low_bound = -1.0 / y_max + eps
    high_bound = 2.0 / (y_mean + 1e-8)

    candidates = []

    # Check negative branch
    try:
        if low_bound < -eps:
            val_low = objective(low_bound)
            val_zero_minus = objective(-eps)
            if val_low * val_zero_minus <= 0:
                sol = optimize.brentq(objective, low_bound, -eps, maxiter=100)
                candidates.append(float(sol))
    except Exception:
        pass

    # Check positive branch
    try:
        if high_bound > eps:
            val_zero_plus = objective(eps)
            val_high = objective(high_bound)
            if val_zero_plus * val_high <= 0:
                sol = optimize.brentq(objective, eps, high_bound, maxiter=100)
                candidates.append(float(sol))
    except Exception:
        pass

    if not candidates:
        # Fallback grid search
        grid = np.linspace(low_bound + eps, high_bound, 50)
        signs = np.sign([objective(th) for th in grid if 1.0 + th * y_max > 0])
        valid_thetas = [th for th in grid if 1.0 + th * y_max > 0]
        for i in range(len(signs) - 1):
            if signs[i] * signs[i + 1] <= 0 and abs(valid_thetas[i]) > eps and abs(valid_thetas[i+1]) > eps:
                try:
                    sol = optimize.brentq(objective, valid_thetas[i], valid_thetas[i+1], maxiter=100)
                    candidates.append(float(sol))
                except Exception:
                    pass

    best_loglik = -np.inf
    best_params = None

    for theta in candidates:
        if 1.0 + theta * y_max <= 0:
            continue
        gamma = float(np.mean(np.log1p(theta * y)))
        sigma = float(gamma / theta) if abs(theta) > 1e-12 else float(y_mean)
        if sigma <= 0:
            continue
        # Compute log-likelihood
        loglik = -n * np.log(sigma) - (1.0 / gamma + 1.0) * np.sum(np.log1p(theta * y))
        if loglik > best_loglik:
            best_loglik = loglik
            best_params = (gamma, sigma)

    return best_params


def _mle_gpd_fit(excesses: np.ndarray) -> Tuple[float, float]:
    """Direct numerical MLE fallback for Generalized Pareto Distribution."""
    y = np.asarray(excesses, dtype=np.float64)
    n = len(y)
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))

    # Initial guess via Method of Moments
    ratio = (y_mean / (y_std + 1e-8)) ** 2
    gamma_init = float(np.clip(0.5 * (1.0 - ratio), -0.5, 0.5))
    sigma_init = float(max(0.5 * y_mean * (1.0 + ratio), 1e-4))

    def neg_loglik(params: np.ndarray) -> float:
        gamma, sigma = params
        if sigma <= 1e-8:
            return 1e10
        term = 1.0 + gamma * y / sigma
        if np.any(term <= 0):
            return 1e10
        if abs(gamma) < 1e-6:
            return float(n * np.log(sigma) + np.sum(y) / sigma)
        return float(n * np.log(sigma) + (1.0 / gamma + 1.0) * np.sum(np.log(term)))

    res = optimize.minimize(
        neg_loglik,
        x0=np.array([gamma_init, sigma_init]),
        bounds=[(-0.9, 2.0), (1e-6, None)],
        method="L-BFGS-B",
    )

    if res.success:
        return float(res.x[0]), float(res.x[1])

    # Default to exponential fit (gamma = 0, sigma = mean)
    return 0.0, float(max(y_mean, 1e-4))


def fit_gpd(
    scores: np.ndarray,
    init_percentile: float = 98.0,
    min_excesses: int = 15,
) -> GPDFitResult:
    """Fit Generalized Pareto Distribution to tail excesses of scores."""
    clean_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    clean_scores = clean_scores[np.isfinite(clean_scores)]
    n_total = len(clean_scores)

    if n_total < 20:
        return GPDFitResult(
            gamma=0.0,
            sigma=1.0,
            threshold_init=float(np.max(clean_scores)) if n_total > 0 else 1.0,
            n_total=n_total,
            n_excess=0,
            fit_converged=False,
            method="insufficient_samples",
        )

    t_init = float(np.percentile(clean_scores, init_percentile))
    excesses = clean_scores[clean_scores > t_init] - t_init

    # If too few excesses, lower percentile to get enough tail samples
    if len(excesses) < min_excesses and init_percentile > 80.0:
        for p in (95.0, 90.0, 85.0):
            t_init = float(np.percentile(clean_scores, p))
            excesses = clean_scores[clean_scores > t_init] - t_init
            if len(excesses) >= min_excesses:
                break

    n_excess = len(excesses)
    if n_excess < 5 or np.all(excesses <= 1e-8):
        return GPDFitResult(
            gamma=0.0,
            sigma=float(max(np.std(clean_scores), 1e-4)),
            threshold_init=t_init,
            n_total=n_total,
            n_excess=n_excess,
            fit_converged=False,
            method="degenerate_excesses",
        )

    # 1. Try Grimshaw MLE
    grimshaw_res = _grimshaw_gpd_fit(excesses)
    if grimshaw_res is not None:
        gamma, sigma = grimshaw_res
        return GPDFitResult(
            gamma=gamma,
            sigma=sigma,
            threshold_init=t_init,
            n_total=n_total,
            n_excess=n_excess,
            fit_converged=True,
            method="grimshaw_mle",
        )

    # 2. Try Numerical Optimization MLE
    gamma, sigma = _mle_gpd_fit(excesses)
    return GPDFitResult(
        gamma=gamma,
        sigma=sigma,
        threshold_init=t_init,
        n_total=n_total,
        n_excess=n_excess,
        fit_converged=True,
        method="numerical_mle",
    )


class EVTCalibrator:
    """Extreme Value Theory (EVT / SPOT) Anomaly Calibrator.

    Fits Generalized Pareto Distribution on normal scores to estimate
    exact tail quantiles for risk level q and maps scores to p-values.
    """

    def __init__(
        self,
        risk_level: float = 1e-3,
        init_percentile: float = 98.0,
        min_excesses: int = 15,
        degenerate_epsilon: float = 1e-6,
    ):
        self.risk_level = risk_level
        self.init_percentile = init_percentile
        self.min_excesses = min_excesses
        self.degenerate_epsilon = degenerate_epsilon
        self.gpd_fit_: Optional[GPDFitResult] = None
        self.threshold_: float = 0.0

    def fit(self, scores: np.ndarray) -> "EVTCalibrator":
        """Fit EVT calibrator to normal calibration scores."""
        clean_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        clean_scores = clean_scores[np.isfinite(clean_scores)]

        # Check for plateau / saturation
        if len(clean_scores) > 20:
            max_s = float(np.max(clean_scores))
            plateau_eps = max(1e-6, 1e-5 * max(abs(max_s), 1.0))
            non_plateau = clean_scores[clean_scores < max_s - plateau_eps]
            if len(non_plateau) >= 20 and (len(clean_scores) - len(non_plateau)) / len(clean_scores) >= 0.005:
                clean_scores = non_plateau

        self.gpd_fit_ = fit_gpd(
            clean_scores,
            init_percentile=self.init_percentile,
            min_excesses=self.min_excesses,
        )

        res = self.compute_threshold(clean_scores, risk_level=self.risk_level)
        self.threshold_ = res.threshold
        return self

    def compute_threshold(
        self,
        scores: Optional[np.ndarray] = None,
        risk_level: Optional[float] = None,
    ) -> EVTThresholdResult:
        """Compute extreme quantile threshold for risk level q."""
        q = risk_level if risk_level is not None else self.risk_level
        if self.gpd_fit_ is None:
            if scores is None:
                raise RuntimeError("EVTCalibrator must be fitted before computing threshold.")
            self.fit(scores)

        fit = self.gpd_fit_
        t = fit.threshold_init
        n = max(fit.n_total, 1)
        n_t = max(fit.n_excess, 1)
        gamma = fit.gamma
        sigma = max(fit.sigma, 1e-8)

        if not fit.fit_converged or n_t < 3:
            # Fallback to empirical high percentile
            if scores is not None and len(scores) > 0:
                p_val = max(0.0, min(100.0, (1.0 - q) * 100.0))
                thresh = float(np.percentile(scores, p_val))
            else:
                thresh = float(t + sigma * 3.0)
            return EVTThresholdResult(
                threshold=max(thresh, self.degenerate_epsilon),
                risk_level=q,
                gpd_fit=fit,
                method="empirical_fallback",
                is_fallback=True,
            )

        # GPD Extreme Quantile Formula:
        # z_q = t + (sigma / gamma) * [ ( (q * N) / N_t )^(-gamma) - 1 ]
        ratio = (q * n) / n_t
        ratio = max(ratio, 1e-12)

        if abs(gamma) < 1e-5:
            # Exponential limit as gamma -> 0
            excess_q = -sigma * np.log(ratio)
        else:
            term = np.power(ratio, -gamma) - 1.0
            excess_q = (sigma / gamma) * term

        threshold = float(t + excess_q)

        # Safety bound: threshold cannot be lower than baseline t or negative
        if threshold < t or not np.isfinite(threshold):
            threshold = float(t + sigma * np.log(max(n_t / (q * n), 1.1)))

        threshold = max(threshold, self.degenerate_epsilon)

        return EVTThresholdResult(
            threshold=threshold,
            risk_level=q,
            gpd_fit=fit,
            method=fit.method,
            is_fallback=False,
        )

    def predict_tail_probability(self, scores: np.ndarray) -> np.ndarray:
        """Calculate tail probability P(X > s) under fitted GPD."""
        if self.gpd_fit_ is None:
            raise RuntimeError("EVTCalibrator must be fitted before predicting probabilities.")

        s = np.asarray(scores, dtype=np.float64)
        fit = self.gpd_fit_
        t = fit.threshold_init
        n = max(fit.n_total, 1)
        n_t = max(fit.n_excess, 1)
        gamma = fit.gamma
        sigma = max(fit.sigma, 1e-8)

        p_tail = np.ones_like(s, dtype=np.float64)
        above_mask = s > t

        if np.any(above_mask):
            y = s[above_mask] - t
            if abs(gamma) < 1e-5:
                surv = np.exp(-y / sigma)
            else:
                arg = np.maximum(1.0 + gamma * y / sigma, 1e-12)
                surv = np.power(arg, -1.0 / gamma)
            p_tail[above_mask] = (n_t / n) * surv

        # Below threshold: scale smoothly from 1.0 down to n_t / n
        below_mask = ~above_mask
        if np.any(below_mask):
            min_s = float(np.min(s))
            span = max(t - min_s, 1e-8)
            frac = np.clip((s[below_mask] - min_s) / span, 0.0, 1.0)
            p_tail[below_mask] = 1.0 - frac * (1.0 - n_t / n)

        return np.clip(p_tail, 1e-15, 1.0).astype(np.float32)

    def predict_anomaly_probability(self, scores: np.ndarray) -> np.ndarray:
        """Return calibrated anomaly confidence P(anomaly | s) = 1 - P(X > s)."""
        p_tail = self.predict_tail_probability(scores)
        return (1.0 - p_tail).astype(np.float32)
