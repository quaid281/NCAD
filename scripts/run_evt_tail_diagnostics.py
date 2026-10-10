#!/usr/bin/env python3
"""GPD tail-fit diagnostics for the EVT/SPOT calibrator (Reviewer Major Concern 4).

The manuscript reports that SPOT systematically overshoots its nominal design
risk on held-out telemetry. This script separates the *possible* causes the
reviewer asked us to distinguish:

- tail-estimation error: fitted GPD shape (xi) / scale (sigma), number of
  exceedances above u, and a Kolmogorov-Smirnov goodness-of-fit statistic of
  the excesses against the fitted GPD;
- temporal dependence: number of contiguous exceedance *clusters* above u vs
  the raw exceedance count (clustering ratio), plus realized FPR after
  declustering-aware interpretation;
- distribution drift: realized held-out FPR under the SPOT threshold vs the
  nominal design risk q = 1e-3, and vs an empirical 99.5% rule fit on the
  same calibration scores.

Scope: a diverse subset of non-GHL primary-benchmark streams (varied channel
counts K and nominal-block sizes), two models (TS-JEPA = latent residual,
TimesNet = reconstruction baseline), three seeds. The full 33-stream scan is
unnecessary for this diagnostic because the question is per-stream tail-fit
behavior, not aggregate detection quality.

Output: reports/evt_tail_diagnostics.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_multiseed_heldout_benchmark import (  # noqa: E402
    C, S, W,
    aggregate_window_scores,
    load_stream,
    train_model,
)
from src.data.data_loader import DataLoader  # noqa: E402
from src.scoring.evt_calibrator import EVTCalibrator, fit_gpd  # noqa: E402

STREAMS = [
    ("Daphnet", "S01R01E1"),
    ("Daphnet", "S03R01E0"),
    ("SMAP", "A-7"),
    ("SMAP", "D-3"),
    ("MSL", "C-1"),
    ("MSL", "M-1"),
    ("swan", "sf"),
    ("metro", "traffic-volume"),
    ("Genesis", "default"),
]
MODELS = ["TS-JEPA", "TimesNet"]
SEEDS = [42, 123, 456]
Q = 1e-3


def exceedance_clusters(x: np.ndarray, u: float) -> tuple[int, float]:
    """Count contiguous runs above u; return (n_clusters, mean_cluster_len)."""
    above = x > u
    if not above.any():
        return 0, 0.0
    starts = np.diff(np.concatenate([[0], above.astype(int)])) == 1
    n_clusters = int(starts.sum())
    return n_clusters, float(above.sum()) / max(n_clusters, 1)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for ds, ch in STREAMS:
        train, test, y = load_stream(ds, ch)
        n_val = int(len(train) * 0.3)
        tr, va = train[:-n_val], train[-n_val:]
        mu, sd = tr.mean(0), tr.std(0)
        sc = np.where(sd < 1e-8, 1.0, np.maximum(sd, 0.01))
        tr_s, va_s = (tr - mu) / sc, (va - mu) / sc
        tr_w = DataLoader.create_windows(tr_s, W, step=16, copy=False)
        va_w = DataLoader.create_windows(va_s, W, step=1, copy=False)
        k = train.shape[1]
        print(f"== {ds}:{ch} K={k} cal-block={len(va)}", flush=True)
        for seed in SEEDS:
            for name in MODELS:
                torch.manual_seed(seed)
                np.random.seed(seed)
                score = train_model(name, tr_w, k, device, epochs=10)
                vws = score(va_w)
                vp, vm = aggregate_window_scores(
                    vws, len(va), C, S, step=1, reducer="mean", mapping_method="smear")
                v = vp[vm]
                half = len(v) // 2
                cal, hold = v[:half], v[half + S:]
                if len(cal) < 100 or len(hold) < 100:
                    print(f"   skip {name} seed={seed}: cal={len(cal)} hold={len(hold)}")
                    continue

                gpd = fit_gpd(cal, init_percentile=98.0)
                u = gpd.threshold_init
                excesses = cal[cal > u] - u
                # KS goodness-of-fit of excesses vs fitted GPD
                if gpd.fit_converged and len(excesses) >= 10:
                    ks = sstats.kstest(
                        excesses,
                        lambda t: sstats.genpareto.cdf(t, c=gpd.gamma, scale=gpd.sigma))
                    ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
                else:
                    ks_stat, ks_p = float("nan"), float("nan")

                n_cl, mean_cl = exceedance_clusters(cal, u)
                cal_spot = EVTCalibrator(risk_level=Q, init_percentile=98.0,
                                         adaptive_kurtosis=False)
                cal_spot.fit(cal)
                th_spot = float(cal_spot.threshold_)
                th_p995 = float(np.percentile(cal, 99.5))

                rows.append(dict(
                    dataset=ds, channel=ch, seed=seed, model=name,
                    n_cal=len(cal), n_hold=len(hold),
                    u_p98=u, n_exceed=gpd.n_excess, gpd_xi=gpd.gamma,
                    gpd_sigma=gpd.sigma, gpd_method=gpd.method,
                    fit_converged=bool(gpd.fit_converged),
                    ks_stat=ks_stat, ks_p=ks_p,
                    n_exceed_clusters=n_cl, mean_cluster_len=mean_cl,
                    cluster_ratio=n_cl / max(gpd.n_excess, 1),
                    spot_threshold=th_spot,
                    spot_hold_fpr=float((hold > th_spot).mean()),
                    p995_hold_fpr=float((hold > th_p995).mean()),
                    u_hold_fpr=float((hold > u).mean()),
                    design_risk=Q,
                ))
                print(f"   {name} s={seed}: xi={gpd.gamma:.3f} n_exc={gpd.n_excess} "
                      f"KS_p={ks_p:.3f} clust={n_cl}/{gpd.n_excess} "
                      f"holdFPR={rows[-1]['spot_hold_fpr']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "reports" / "evt_tail_diagnostics.csv",
                                  index=False)
    out = ROOT / "reports" / "evt_tail_diagnostics.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
