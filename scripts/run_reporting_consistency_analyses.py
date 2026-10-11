#!/usr/bin/env python3
"""Round-4 reporting-consistency analyses.

1. Detection delay on the *intersection* of detected events: for each stream,
   at a common nominal budget (q*=0.05 trailing), restrict delay statistics to
   events detected by BOTH TS-JEPA and the comparator -> removes the
   detected-set confound the referee flagged.
2. Trivial all-positive F1 (2*pi/(1+pi)) per stream + whether each model's
   Point-F1 exceeds it (Supplement table material).
3. Stream-level link between residual tail conditioning and calibration
   outcome: realized test FPR (SPOT threshold) vs calibration-block excess
   kurtosis and a calibration->evaluation drift statistic (KS distance between
   cal and test nominal scores). Spearman correlations across stream-seed cells.

Outputs:
  reports/delay_intersection.csv
  reports/trivial_f1_comparison.csv
  reports/tail_calibration_link.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_multiseed_heldout_benchmark import event_stats  # noqa: E402
from src.scoring.evt_calibrator import EVTCalibrator, compute_excess_kurtosis
from src.scoring.event_fusion import _extract_segments  # noqa: E402

DUMP = ROOT / "reports" / "score_dumps"
Q_STAR = 5e-2
MAIN = "TS-JEPA"


def event_delays(tl, pred):
    """Per-event detection delays (or None for missed events)."""
    segs = _extract_segments(tl)
    out = []
    for gs, ge in segs:
        w = np.where(pred[gs:ge] > 0)[0]
        out.append(float(w[0]) if len(w) else np.nan)
    return np.array(out)


def detected_mask(tl, pred):
    segs = _extract_segments(tl)
    m = np.zeros(len(segs), bool)
    for i, (gs, ge) in enumerate(segs):
        m[i] = pred[gs:ge].sum() > 0
    return m


def main():
    files = sorted(DUMP.glob("*__trailing.npz"))
    cache = {}
    for fn in files:
        ds, ch, seed, model, mapping = fn.stem.split("__")
        d = np.load(fn)
        ts = np.where(np.isfinite(d["test_scores"]), d["test_scores"], np.inf)
        th = float(np.quantile(d["cal"], 1.0 - Q_STAR))
        pred = (ts > th).astype(int)
        cache[(ds, ch, int(seed), model)] = dict(
            labels=d["test_labels"], pred=pred,
            delays=event_delays(d["test_labels"], pred),
            det=detected_mask(d["test_labels"], pred),
            cal=d["cal"], ts=ts)

    # ---- 1. delay on event intersection ------------------------------------
    rows = []
    models = sorted({k[3] for k in cache})
    for (ds, ch, seed, model), a in cache.items():
        b = cache.get((ds, ch, seed, MAIN))
        if model == MAIN or b is None or len(a["delays"]) == 0:
            continue
        inter = a["det"] & b["det"] & np.isfinite(a["delays"]) \
            & np.isfinite(b["delays"])
        if inter.sum() < 1:
            continue
        rows.append(dict(dataset=ds, channel=ch, seed=seed, model=model,
                         n_intersect=int(inter.sum()),
                         delay_main=float(np.nanmean(b["delays"][inter])),
                         delay_other=float(np.nanmean(a["delays"][inter])),
                         delay_own_all=float(np.nanmean(a["delays"])),
                         delay_main_all=float(np.nanmean(b["delays"]))))
    pd.DataFrame(rows).to_csv(ROOT / "reports" / "delay_intersection.csv",
                              index=False)

    # ---- 2. trivial F1 ------------------------------------------------------
    trows = []
    for (ds, ch, seed, model), a in cache.items():
        tl = a["labels"]
        pi = float(tl.mean())
        triv = 2 * pi / (1 + pi) if pi > 0 else 0.0
        pred = a["pred"]
        tp = int(((pred == 1) & (tl == 1)).sum())
        fp = int(((pred == 1) & (tl == 0)).sum())
        fn = int(((pred == 0) & (tl == 1)).sum())
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-12)
        trows.append(dict(dataset=ds, channel=ch, seed=seed, model=model,
                          anomaly_rate=pi, trivial_f1=triv, model_f1=f1,
                          beats_trivial=f1 > triv))
    pd.DataFrame(trows).to_csv(
        ROOT / "reports" / "trivial_f1_comparison.csv", index=False)

    # ---- 3. tail conditioning vs calibration outcome -----------------------
    lrows = []
    for (ds, ch, seed, model), a in cache.items():
        cal, ts = a["cal"], a["ts"]
        evt = EVTCalibrator(risk_level=1e-3, init_percentile=98.0,
                            adaptive_kurtosis=False)
        evt.fit(cal)
        th = float(evt.threshold_)
        tl = a["labels"]
        nom = tl == 0
        realized = float(((ts > th) & nom).sum() / max(nom.sum(), 1))
        ks = float(stats.ks_2samp(cal, ts[nom]).statistic) \
            if nom.sum() > 10 else np.nan
        lrows.append(dict(dataset=ds, channel=ch, seed=seed, model=model,
                          cal_kurtosis=float(compute_excess_kurtosis(cal)),
                          ks_drift=ks, spot_threshold=th,
                          realized_fpr=realized))
    link = pd.DataFrame(lrows)
    for x in ["cal_kurtosis", "ks_drift"]:
        rho, p = stats.spearmanr(link[x], link.realized_fpr,
                                 nan_policy="omit")
        print(f"Spearman({x}, realized FPR) = {rho:.3f} (p={p:.2e})")
    link.to_csv(ROOT / "reports" / "tail_calibration_link.csv", index=False)
    print("wrote reports/{delay_intersection,trivial_f1_comparison,"
          "tail_calibration_link}.csv")


if __name__ == "__main__":
    main()
