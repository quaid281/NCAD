#!/usr/bin/env python3
"""Operating-characteristic analysis (Round-4 reviewer MC 3.1 / Q1).

Reframes the primary comparison as recall vs. realized false-positive rate:

  * For each dumped (stream, seed, model, mapping) cell, thresholds are chosen
    on the CAL block at a fine grid of nominal budgets q in [1e-4, 0.3].
    Metrics are then evaluated on the TEST block: realized FPR, point
    precision/recall/F1, event recall, mean detection delay, FAR/1k.
    This makes the design-vs-realized FPR gap explicit (drift).
  * Partial-AUC: mean event recall over nominal budgets <= 0.10 (area under the
    recall-vs-nominal-budget curve, normalized to [0,1]).
  * Aggregation: seeds averaged within stream; per-model mean curve over the
    shared nominal-budget grid; dataset-clustered bootstrap CIs for pairwise
    partial-AUC and budget-point differences (resampling datasets, not
    streams, to respect pseudo-replication within a testbed).
  * GHL-excluded sensitivity aggregate for Q7.

Outputs:
  reports/operating_characteristics.csv   per-cell metrics at each grid point
  reports/oc_partial_auc.csv              per-cell partial-AUC summary
  reports/oc_model_curves.csv             per-model aggregated curve + CI
  reports/oc_pairwise_tests.csv           clustered-bootstrap pairwise deltas
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
from src.scoring.event_fusion import _extract_segments  # noqa: E402

DUMP = ROOT / "reports" / "score_dumps"
Q_GRID = np.array([1e-4, 3e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1,
                   2e-1, 3e-1])
PAUC_BUDGET = 0.10
N_BOOT = 2000
RNG = np.random.default_rng(2024)
GHL = "GHL"


def cell_metrics(cal, ts, tl, th):
    pred = (ts > th).astype(int)
    nominal = max(int((tl == 0).sum()), 1)
    fp = int(((pred == 1) & (tl == 0)).sum())
    tp = int(((pred == 1) & (tl == 1)).sum())
    fn = int(((pred == 0) & (tl == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    er, dl = event_stats(tl, pred)
    return dict(
        design_fpr=None,  # filled by caller
        realized_fpr=fp / nominal,
        hold_fpr=None,
        point_precision=prec, point_recall=rec,
        point_f1=2 * prec * rec / max(prec + rec, 1e-12),
        event_recall=er, mean_delay=dl,
        far_per_1k=fp / nominal * 1000.0,
    )


def stream_mean(df):
    """Average seed-level rows within a stream."""
    return (df.groupby(["dataset", "channel", "model", "mapping", "q"],
                       as_index=False)
              .mean(numeric_only=True))


def cluster_boot_diff(df, metric, model_a, model_b, mapping,
                      cohort=None, n_boot=N_BOOT):
    """Dataset-clustered bootstrap on the paired stream-mean difference."""
    a = df[(df.model == model_a) & (df.mapping == mapping)]
    b = df[(df.model == model_b) & (df.mapping == mapping)]
    key = ["dataset", "channel"]
    a = a.set_index(key)[metric]
    b = b.set_index(key)[metric]
    diff = (a - b).dropna()
    if cohort == "nonghl":
        diff = diff[~diff.index.get_level_values(0).str.contains(GHL)]
    if len(diff) < 3:
        return dict(n=len(diff), delta=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    datasets = diff.index.get_level_values(0)
    uniq = np.unique(datasets)
    obs = float(diff.mean())
    boots = np.empty(n_boot)
    arr = diff.to_numpy()
    ds_arr = datasets.to_numpy()
    for i in range(n_boot):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([arr[ds_arr == d] for d in pick])
        boots[i] = sel.mean()
    return dict(n=len(diff), delta=obs,
                ci_lo=float(np.quantile(boots, 0.025)),
                ci_hi=float(np.quantile(boots, 0.975)),
                boot_p=float(2 * min((boots <= 0).mean(),
                                     (boots >= 0).mean())))


def main():
    rows = []
    files = sorted(DUMP.glob("*.npz"))
    for k, fn in enumerate(files):
        ds, ch, seed, model, mapping = fn.stem.split("__")
        d = np.load(fn)
        cal, ts, tl = d["cal"], d["test_scores"], d["test_labels"]
        if not np.isfinite(ts).all():
            ts = np.where(np.isfinite(ts), ts, np.inf)
        for q in Q_GRID:
            th = float(np.quantile(cal, 1.0 - q))
            m = cell_metrics(cal, ts, tl, th)
            rows.append(dict(dataset=ds, channel=ch, seed=int(seed),
                             model=model, mapping=mapping, q=q,
                             threshold=th, **m))
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(files)}")
    long = pd.DataFrame(rows)
    long["design_fpr"] = long.q
    long.to_csv(ROOT / "reports" / "operating_characteristics.csv",
                index=False)

    # Per-cell partial AUC over budgets <= PAUC_BUDGET
    pauc_rows = []
    for (ds, ch, seed, model, mapping), g in long.groupby(
            ["dataset", "channel", "seed", "model", "mapping"]):
        g = g[g.q <= PAUC_BUDGET].sort_values("q")
        pauc = float(np.trapezoid(g.event_recall, g.q) / PAUC_BUDGET) \
            if len(g) else np.nan
        pauc_rows.append(dict(dataset=ds, channel=ch, seed=seed, model=model,
                              mapping=mapping, partial_auc=pauc,
                              event_recall_q05=float(
                                  g[np.isclose(g.q, 5e-2)].event_recall.iloc[0])
                              if (g.q == 5e-2).any() else np.nan))
    pauc = pd.DataFrame(pauc_rows)
    pauc.to_csv(ROOT / "reports" / "oc_partial_auc.csv", index=False)

    # Stream-mean aggregated curve per model (non-GHL + all-cohort variants)
    sm = stream_mean(long)
    curves = []
    for (model, mapping), g in sm.groupby(["model", "mapping"]):
        for cohort_name, sub in (("all", g),
                                 ("nonghl", g[~g.dataset.str.contains(GHL)])):
            piv = sub.groupby("q").event_recall.mean()
            rf = sub.groupby("q").realized_fpr.mean()
            for q in piv.index:
                curves.append(dict(model=model, mapping=mapping,
                                   cohort=cohort_name, q=q,
                                   event_recall=piv[q],
                                   realized_fpr=rf[q]))
    pd.DataFrame(curves).to_csv(
        ROOT / "reports" / "oc_model_curves.csv", index=False)

    # Pairwise clustered-bootstrap tests vs TS-JEPA, both cohorts
    tests = []
    sm_p = (pauc.groupby(["dataset", "channel", "model", "mapping"],
                         as_index=False).partial_auc.mean())
    sm_e = (pauc.groupby(["dataset", "channel", "model", "mapping"],
                         as_index=False).event_recall_q05.mean())
    for mapping in ["trailing", "smear"]:
        for cohort in ["all", "nonghl"]:
            for model in sorted(sm_p.model.unique()):
                if model == "TS-JEPA":
                    continue
                for metric, df_ in (("partial_auc", sm_p),
                                    ("event_recall_q05", sm_e)):
                    r = cluster_boot_diff(df_, metric, "TS-JEPA", model,
                                          mapping, cohort)
                    tests.append(dict(mapping=mapping, cohort=cohort,
                                      comparison=f"TS-JEPA - {model}",
                                      metric=metric, **r))
    pd.DataFrame(tests).to_csv(ROOT / "reports" / "oc_pairwise_tests.csv",
                               index=False)
    print("wrote reports/{operating_characteristics,oc_partial_auc,"
          "oc_model_curves,oc_pairwise_tests}.csv")


if __name__ == "__main__":
    main()
