#!/usr/bin/env python3
"""Statistical dependence of the 224 circuit-breaker dials on nominal data (19 streams x 3 seeds).

For every cached run the panel is calibrated exactly as deployed on the first half of the nominal
validation windows; dependence is then measured on the held-out nominal windows (second half after
the S-step gap), at window level before any timeline aggregation:

* dial correlation structure (Spearman on calibration z-scores), split into same-coordinate /
  same-perspective / other pairs;
* effective number of dials, N_eff = (sum lambda)^2 / sum lambda^2 of the correlation spectrum;
* simultaneous-trip counts N_224 on held-out nominal windows versus an independence (Poisson-binomial)
  model with the same per-dial trip rates: variance ratio and tail exceedance P(N_224 > 5);
* agreement between the 128-bit switches and the first 128 dials of the 224-bit panel.

Usage: python -m scripts.reviewer_analyses.breaker_dependence [--out CSV]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scripts.circuit_breaker_19stream.common import ROOT, S, SEEDS, load_cached, load_streams_19
from src.scoring.circuit_breaker_panel import MultiPerspectiveCircuitBreakerPanel, build_224_perspective_dials

D, GROUPS = 32, 7
COORD = np.tile(np.arange(D), GROUPS)
GROUP = np.repeat(np.arange(GROUPS), D)
SAME_COORD = (COORD[:, None] == COORD[None, :])
SAME_GROUP = (GROUP[:, None] == GROUP[None, :])
OFF_DIAG = ~np.eye(GROUPS * D, dtype=bool)


def spearman_matrix(x: np.ndarray) -> np.ndarray:
    """Spearman correlation of columns; constant columns get zero correlation."""
    r = np.apply_along_axis(rankdata, 0, x)
    r -= r.mean(axis=0)
    sd = r.std(axis=0)
    ok = sd > 0
    corr = np.zeros((x.shape[1], x.shape[1]))
    rz = r[:, ok] / sd[ok]
    corr[np.ix_(ok, ok)] = (rz.T @ rz) / len(rz)
    return corr


def poisson_binomial_tail(p: np.ndarray, k: float) -> float:
    """P(sum Bernoulli(p_j) > k) by exact convolution."""
    dist = np.zeros(len(p) + 1)
    dist[0] = 1.0
    for pj in p:
        dist[1:] = dist[1:] * (1 - pj) + dist[:-1] * pj
        dist[0] *= 1 - pj
    return float(dist[int(np.floor(k)) + 1 :].sum())


def analyse_run(va_ew: np.ndarray, panel: MultiPerspectiveCircuitBreakerPanel) -> dict:
    n_cal = len(va_ew) // 2
    dials = build_224_perspective_dials(va_ew)
    med = np.median(dials[:n_cal], axis=0)
    mad = 1.4826 * np.median(np.abs(dials[:n_cal] - med), axis=0) + 1e-4
    z_cal = (dials[:n_cal] - med) / mad
    corr = spearman_matrix(z_cal)
    lam = np.clip(np.linalg.eigvalsh(corr), 0.0, None)
    n_eff = float(lam.sum() ** 2 / max((lam**2).sum(), 1e-12))

    abs_c = np.abs(corr)
    same_c = SAME_COORD & OFF_DIAG
    same_g = SAME_GROUP & ~SAME_COORD
    other = ~SAME_GROUP & ~SAME_COORD

    bits = panel.calibrate_and_trip_windows(va_ew, va_ew[:1], n_cal)
    hold = bits["bits224_va"][n_cal + S :]
    n_trip = hold.sum(axis=1)
    p_j = hold.mean(axis=0)
    var_indep = float((p_j * (1 - p_j)).sum())
    agree_128 = float((bits["bits128_va"] == bits["bits224_va"][:, :128]).mean())
    return dict(
        n_eff=n_eff,
        n_eff_frac=n_eff / (GROUPS * D),
        mean_abs_r_same_coord=float(abs_c[same_c].mean()),
        mean_abs_r_same_group=float(abs_c[same_g].mean()),
        mean_abs_r_other=float(abs_c[other].mean()),
        frac_pairs_abs_r_gt_0p5=float((abs_c[OFF_DIAG] > 0.5).mean()),
        hold_mean_trips=float(n_trip.mean()),
        hold_var_ratio=float(n_trip.var() / max(var_indep, 1e-12)),
        hold_p_gt5_obs=float((n_trip > 5).mean()),
        hold_p_gt5_indep=poisson_binomial_tail(p_j, 5),
        agree_128_vs_224pos=agree_128,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports" / "breaker_dependence_19stream.csv"))
    args = ap.parse_args()

    panel = MultiPerspectiveCircuitBreakerPanel()
    rows = []
    for ds, ch in load_streams_19():
        for seed in SEEDS:
            rows.append(dict(stream=f"{ds}:{ch}", seed=seed, **analyse_run(load_cached(ds, ch, seed)["va_ew"], panel)))
        print(f"done {ds}:{ch}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    per_stream = df.drop(columns="seed").groupby("stream").mean()
    summary = per_stream.agg(["mean", "median", "min", "max"]).T
    print("\n=== BREAKER DEPENDENCE (stream-level, n=19) ===")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
