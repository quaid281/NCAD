#!/usr/bin/env python3
"""Residual-distribution diagnostics over dumped score arrays (Reviewer MC1).

For every (stream, seed, model, mapping) npz produced by
run_multiseed_heldout_benchmark.py --dump-scores, compute:

  * tail shape on held-out nominal scores: excess kurtosis, q90/q95/q99/q99.9,
    tail ratio (q99.9 - q95)/(q99 - q95)
  * dependence: lag-1/5/64 autocorrelation
  * extremal clustering: runs-declustering (r=5) above u=q98 -> n_exceedances,
    n_clusters, extremal index theta, effective independent count N_eff=theta*N_u
  * calibration: SPOT(u=0.98, q=1e-3) threshold fit on `cal`, achieved FPR on
    `hold`, plus moving-block bootstrap 95% CI
  * separation (test): PR-AUC and ROC-AUC of test_scores vs test_labels

This distinguishes residual *tail conditioning* from anomaly-vs-normal
*score separation* -- the two mechanisms the reviewer asked us to separate.

Output: reports/residual_diagnostics.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_multiseed_heldout_benchmark import block_bootstrap_ci  # noqa: E402
from src.scoring.evt_calibrator import EVTCalibrator  # noqa: E402

DUMP = ROOT / "reports" / "score_dumps"
OUT = ROOT / "reports" / "residual_diagnostics.csv"


def acf(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag + 1:
        return float("nan")
    a, b = x[:-lag], x[lag:]
    if a.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


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


def diagnose(fn: Path) -> dict | None:
    d = np.load(fn)
    cal, hold = d["cal"], d["hold"]
    ts, tl = d["test_scores"], d["test_labels"]
    ds, ch, seed, model, mapping = fn.stem.split("__")

    nom = np.concatenate([cal, hold])
    u = float(np.quantile(cal, 0.98))
    exc_idx = np.where(cal > u)[0]
    clusters = runs_decluster(exc_idx)
    n_u = len(exc_idx)
    n_cl = len(clusters)
    theta = n_cl / n_u if n_u else float("nan")

    q90, q95, q99, q999 = np.percentile(nom, [90, 95, 99, 99.9])
    tail_ratio = float((q999 - q95) / (q99 - q95 + 1e-12))

    cal_evt = EVTCalibrator(risk_level=1e-3, init_percentile=98.0,
                            adaptive_kurtosis=False)
    cal_evt.fit(cal)
    th = float(cal_evt.threshold_)
    ind = (hold > th).astype(float)
    lo, hi = block_bootstrap_ci(ind)

    if tl.sum() > 0:
        pr_auc = float(average_precision_score(tl, ts))
        roc_auc = float(roc_auc_score(tl, ts))
    else:
        pr_auc = roc_auc = float("nan")

    return dict(
        dataset=ds, channel=ch, seed=int(seed), model=model, mapping=mapping,
        n_cal=len(cal), n_hold=len(hold),
        kurtosis=float(stats.kurtosis(nom, fisher=True)),
        q90=q90, q95=q95, q99=q99, q999=q999, tail_ratio=tail_ratio,
        acf1=acf(nom, 1), acf5=acf(nom, 5), acf64=acf(nom, 64),
        n_exceed_u=n_u, n_clusters=n_cl, extremal_theta=theta,
        n_eff=theta * n_u if n_u else float("nan"),
        spot_threshold=th, hold_fpr=float(ind.mean()),
        hold_fpr_lo=lo, hold_fpr_hi=hi,
        test_pr_auc=pr_auc, test_roc_auc=roc_auc,
    )


def main():
    rows = []
    files = sorted(DUMP.glob("*.npz"))
    print(f"{len(files)} dump files")
    for fn in files:
        try:
            rows.append(diagnose(fn))
        except Exception as e:
            print("FAIL", fn.name, type(e).__name__, e)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print("wrote", OUT, len(out), "rows")


if __name__ == "__main__":
    main()
