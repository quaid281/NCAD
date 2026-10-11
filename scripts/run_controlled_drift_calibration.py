#!/usr/bin/env python3
"""Threshold-calibration reliability under controlled distribution shift
(Reviewer Major Concern 2 / Essential Experiment 2).

Uses dumped held-out nominal score arrays. For each (stream, seed, model,
mapping) cell:

Calibrators (all fit on `cal` unless noted):
  * spot            -- POT/GPD threshold at u=q98, design risk q=1e-3
  * spot_decluster  -- runs-declustered (r=5) GPD fit on cluster maxima,
                       N_u replaced by the cluster count (dependence-aware)
  * empirical       -- direct (1-q) empirical quantile of `cal`
  * rolling_emp     -- drift-aware rolling (1-q) quantile over a trailing
                       2000-point window of the *evaluation* stream (simulates
                       online recalibration; warmup excluded)
  * rolling_spot    -- step-adaptive SPOT: GPD refit every 500 eval steps on the
                       trailing 2000 points

Drift conditions applied to the `hold` eval block (sigma = SD of `cal`):
  * stationary (control)
  * mean_shift +0.5sigma / +1.0sigma / ramp 0->sigma
  * var_scale x1.5 / x2.0 (about the block mean)
  * acf rho=0.6 / rho=0.9 (AR(1)-smoothed innovations, marginal moments kept)
  * regime_change (+1sigma step at the block midpoint)

Reports achieved FPR, overshoot vs. q, and the extremal-index effective
cluster count on each eval block.

Output: reports/controlled_drift_calibration.csv
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scoring.evt_calibrator import fit_gpd  # noqa: E402

DUMP = ROOT / "reports" / "score_dumps"
OUT = ROOT / "reports" / "controlled_drift_calibration.csv"
Q = 1e-3
U_Q = 0.98
ROLL_W = 2000
ROLL_STEP = 500


def gpd_threshold(x: np.ndarray, q: float, u: float,
                  n_u_override=None, excesses_override=None) -> float:
    exc = excesses_override if excesses_override is not None else x[x > u] - u
    n_u = n_u_override if n_u_override is not None else len(exc)
    n = len(x)
    if len(exc) < 5 or n_u < 1:
        return float(np.quantile(x, 1.0 - q))
    fit = fit_gpd(exc)
    sigma, xi = fit.sigma, fit.gamma
    if abs(xi) < 1e-6:
        return float(u - sigma * math.log(q * n / n_u))
    return float(u + (sigma / xi) * ((q * n / n_u) ** (-xi) - 1.0))


def runs_decluster(idx: np.ndarray, r: int = 5):
    if len(idx) == 0:
        return []
    clusters = [[idx[0]]]
    for i in idx[1:]:
        if i - clusters[-1][-1] <= r:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    return clusters


def extremal_index(x: np.ndarray, u: float, r: int = 5) -> tuple:
    idx = np.where(x > u)[0]
    n_u = len(idx)
    n_cl = len(runs_decluster(idx, r))
    theta = n_cl / n_u if n_u else float("nan")
    return n_u, n_cl, theta


def spot_declustered(cal: np.ndarray, q: float, u: float) -> float:
    idx = np.where(cal > u)[0]
    clusters = runs_decluster(idx)
    if not clusters:
        return float(np.quantile(cal, 1.0 - q))
    cmax = np.array([cal[c].max() for c in clusters])
    return gpd_threshold(cal, q, u, n_u_override=len(clusters),
                         excesses_override=cmax - u)


def rolling_empirical(ev: np.ndarray, q: float, w: int = ROLL_W) -> np.ndarray:
    """Trailing-window empirical (1-q) quantile; NaN during warmup."""
    n = len(ev)
    thr = np.full(n, np.nan)
    for i in range(w, n):
        thr[i] = np.quantile(ev[i - w:i], 1.0 - q)
    return thr


def rolling_spot(ev: np.ndarray, q: float, u_q: float = U_Q,
                 w: int = ROLL_W, step: int = ROLL_STEP) -> np.ndarray:
    """Step-adaptive SPOT: refit GPD every `step` points on the trailing window."""
    n = len(ev)
    thr = np.full(n, np.nan)
    cur = np.nan
    for i in range(w, n):
        if (i - w) % step == 0 or not np.isfinite(cur):
            blk = ev[i - w:i]
            cur = gpd_threshold(blk, q, float(np.quantile(blk, u_q)))
        thr[i] = cur
    return thr


def drift_variants(ev: np.ndarray, sigma: float) -> dict:
    mu = ev.mean()
    z = (ev - mu) / max(ev.std(), 1e-12)
    out = {"stationary": ev}
    out["mean_shift_+0.5s"] = ev + 0.5 * sigma
    out["mean_shift_+1.0s"] = ev + 1.0 * sigma
    out["mean_ramp"] = ev + np.linspace(0, sigma, len(ev))
    out["var_scale_1.5"] = (ev - mu) * 1.5 + mu
    out["var_scale_2.0"] = (ev - mu) * 2.0 + mu
    for rho in (0.6, 0.9):
        y = np.empty_like(z)
        y[0] = z[0]
        for t in range(1, len(z)):
            y[t] = rho * y[t - 1] + math.sqrt(1 - rho ** 2) * z[t]
        out[f"acf_{rho}"] = y * ev.std() + mu
    rc = ev.copy()
    rc[len(rc) // 2:] += sigma
    out["regime_change_+1s"] = rc
    return out


def achieved_fpr(ev: np.ndarray, thr) -> float:
    if np.isscalar(thr):
        return float((ev > thr).mean())
    m = np.isfinite(thr)
    return float((ev[m] > thr[m]).mean()) if m.any() else float("nan")


def main():
    rows = []
    for fn in sorted(DUMP.glob("*.npz")):
        ds, ch, seed, model, mapping = fn.stem.split("__")
        d = np.load(fn)
        cal, hold = d["cal"], d["hold"]
        if len(cal) < 200 or len(hold) < 200:
            continue
        u = float(np.quantile(cal, U_Q))
        sigma = float(cal.std())
        static_thr = {
            "spot": gpd_threshold(cal, Q, u),
            "spot_decluster": spot_declustered(cal, Q, u),
            "empirical": float(np.quantile(cal, 1.0 - Q)),
        }
        for cond_name, ev in drift_variants(hold, sigma).items():
            thr_map = dict(static_thr)
            thr_map["rolling_emp"] = rolling_empirical(ev, Q)
            thr_map["rolling_spot"] = rolling_spot(ev, Q)
            u_ev = float(np.quantile(cal, U_Q))  # exceedances counted vs cal u
            n_u, n_cl, theta = extremal_index(ev, u_ev)
            for cal_name, thr in thr_map.items():
                fpr = achieved_fpr(ev, thr)
                rows.append(dict(
                    dataset=ds, channel=ch, seed=int(seed), model=model,
                    mapping=mapping, condition=cond_name, calibrator=cal_name,
                    achieved_fpr=fpr, overshoot=fpr / Q,
                    n_exceed_ev=n_u, n_clusters_ev=n_cl, extremal_theta=theta,
                    n_eff_ev=theta * n_u if n_u else float("nan"),
                ))
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print("wrote", OUT, len(out), "rows")


if __name__ == "__main__":
    main()
