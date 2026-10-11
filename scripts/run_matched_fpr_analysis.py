#!/usr/bin/env python3
"""Detection quality at matched false-positive operating points (Reviewer Exp 3).

For every dumped (stream, seed, model, mapping) cell:
  * calibrate empirical-quantile thresholds on the `cal` block at target nominal
    FPR q* in {1e-3, 5e-3, 1e-2, 5e-2} -- plus the standard SPOT threshold as
    reference -- so all models are compared at a *common* nominal alarm budget;
  * report achieved hold-block FPR, and on test: point precision/recall/F1,
    event recall, mean detection delay, recall within 64 steps, and false
    alarms per 1,000 nominal timesteps;
  * attribution sensitivity (smear mapping): re-score under three conventions
    for pre-onset alarms: "strict" (alarm before onset = FP), "credit"
    (pre-onset alarm within S steps counts as event detection, removed from
    FP count), "exclude" (pre-onset points dropped from both FP and nominal
    denominators).

Output: reports/matched_fpr_evaluation.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_multiseed_heldout_benchmark import event_stats  # noqa: E402
from src.scoring.evt_calibrator import EVTCalibrator  # noqa: E402
from src.scoring.event_fusion import _extract_segments  # noqa: E402

DUMP = ROOT / "reports" / "score_dumps"
OUT = ROOT / "reports" / "matched_fpr_evaluation.csv"
Q_GRID = [1e-3, 5e-3, 1e-2, 5e-2]
S = 64


def recall_at_budget(y: np.ndarray, pred: np.ndarray, budget: int) -> float:
    segs = _extract_segments(y)
    if not segs:
        return float("nan")
    hit = 0
    for gs, ge in segs:
        w = np.where(pred[gs:ge] > 0)[0]
        if len(w) and w[0] <= budget:
            hit += 1
    return hit / len(segs)


def preonset_points(y: np.ndarray, s: int) -> np.ndarray:
    """Boolean mask of nominal points within s steps before an onset."""
    mask = np.zeros(len(y), dtype=bool)
    for gs, _ in _extract_segments(y):
        mask[max(0, gs - s):gs] = True
    return mask & (y == 0)


def test_metrics(ts: np.ndarray, tl: np.ndarray, th: float) -> dict:
    pred = (ts > th).astype(int)
    tp = int(((pred == 1) & (tl == 1)).sum())
    fp = int(((pred == 1) & (tl == 0)).sum())
    fn = int(((pred == 0) & (tl == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    er, dl = event_stats(tl, pred)
    nominal = int((tl == 0).sum())
    return dict(
        point_precision=prec, point_recall=rec, point_f1=f1,
        event_recall=er, mean_delay=dl,
        recall_at_64=recall_at_budget(tl, pred, 64),
        far_per_1k=fp / max(nominal, 1) * 1000.0,
        fp=fp, nominal_points=nominal,
    )


def attribution_sensitivity(ts: np.ndarray, tl: np.ndarray, th: float) -> dict:
    """Re-score under the three pre-onset conventions."""
    pred = (ts > th).astype(int)
    pre = preonset_points(tl, S)
    out = {}
    # strict: as-is
    nominal = (tl == 0).sum()
    out["strict_fpr"] = float(((pred == 1) & (tl == 0)).sum() / max(nominal, 1))
    # credit: pre-onset alarms removed from FP; events with a hit in the
    # [onset-S, onset) window count as detected at delay 0.
    pred_c = pred.copy()
    pred_c[pre] = 0
    segs = _extract_segments(tl)
    hit = 0
    for gs, ge in segs:
        if pred_c[gs:ge].sum() > 0 or pred[max(0, gs - S):gs].sum() > 0:
            hit += 1
    out["credit_event_recall"] = hit / len(segs) if segs else float("nan")
    fp_c = ((pred_c == 1) & (tl == 0)).sum()
    out["credit_fpr"] = float(fp_c / max(nominal, 1))
    # exclude: pre-onset points dropped from denominators
    keep = ~pre
    fp_e = ((pred[keep] == 1) & (tl[keep] == 0)).sum()
    nom_e = (tl[keep] == 0).sum()
    out["exclude_fpr"] = float(fp_e / max(nom_e, 1))
    return out


def main():
    rows = []
    for fn in sorted(DUMP.glob("*.npz")):
        ds, ch, seed, model, mapping = fn.stem.split("__")
        d = np.load(fn)
        cal, hold, ts, tl = d["cal"], d["hold"], d["test_scores"], d["test_labels"]
        ths = {f"q*={q:g}": float(np.quantile(cal, 1.0 - q)) for q in Q_GRID}
        evt = EVTCalibrator(risk_level=1e-3, init_percentile=98.0,
                            adaptive_kurtosis=False)
        evt.fit(cal)
        ths["SPOT(q=1e-3)"] = float(evt.threshold_)
        for cal_name, th in ths.items():
            m = test_metrics(ts, tl, th)
            row = dict(dataset=ds, channel=ch, seed=int(seed), model=model,
                       mapping=mapping, calibrator=cal_name, threshold=th,
                       hold_fpr=float((hold > th).mean()), **m)
            if mapping == "smear":
                row.update(attribution_sensitivity(ts, tl, th))
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print("wrote", OUT, len(out), "rows")


if __name__ == "__main__":
    main()
