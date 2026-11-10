#!/usr/bin/env python3
"""Is the 224-bit panel just an ensemble of threshold detectors?  Dial-group redundancy and
alternative aggregations / classical SPC on the SAME cached latent residuals (19 streams x 3 seeds).

All arms share the panel's split: calibration on the first half of the nominal validation timeline,
a 64-step gap, and held-out nominal evaluation on the rest. No test labels are used for calibration.

Arm families
  PANEL   deployed 224-bit breaker (balanced gates), reproduced exactly as in Table II.
  A_*     dial-group subsets with the single-quorum machinery of the perspective ladder
          (``eval_subset_panel``: guarded per-dial q98.5 tripwires + quorum q99.5 x 1.12 with floor).
  B_*     alternative aggregations over the same 224 dials (max-statistic, top-k voting,
          conformal Bonferroni / BH / Fisher / Stouffer).
  C_*     classical SPC on the whitened residual coordinates (Hotelling T^2, MEWMA, CUSUM).
  R_*     1D scalar-collapse reference (SPOT on the backbone score).

Calibrated arms map a window statistic to the timeline (trailing: tau = t + S - 1; smear: mean over
covering windows) and threshold it at a calibration-half quantile or SPOT. Fixed-rule arms (top-k with
k fixed, Bonferroni, BH, Bonferroni-CUSUM) alarm per window; under smear a time step alarms when at
least half of its covering windows alarm.

Usage: python -m scripts.reviewer_analyses.ensemble_and_redundancy_baselines
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import average_precision_score

from scripts.circuit_breaker_19stream.common import C, ROOT, S, SEEDS, fit_spot, load_cached, load_streams_19
from scripts.circuit_breaker_19stream.run_perspective_ladder import eval_subset_panel
from scripts.reviewer_analyses.ensemble_baseline_arms import (
    GROUPS,
    P4,
    cusum_statistics,
    effective_rank,
    hotelling_t2,
    mewma,
    pvalue_statistics,
    robust_z,
)
from scripts.run_multiseed_heldout_benchmark import event_stats
from src.scoring.circuit_breaker_panel import (
    MultiPerspectiveCircuitBreakerPanel,
    aggregate_vector_stream,
    build_224_perspective_dials,
)
from src.scoring.event_fusion import aggregate_window_scores, compute_metrics

MAPPINGS = ("trailing", "smear")
ALPHA = 0.01
REF = "PANEL_224_full"
METRICS = ["f1", "precision", "recall", "pr_auc", "er", "delay", "hold_fpr", "test_fpr", "alarm_rate"]
SUBSETS = [c for r in range(1, 5) for c in itertools.combinations("PDFE", r)]


def subset_name(groups) -> str:
    return "A_" + "+".join(groups)


# ----------------------------------------------------------------------------------------------
# Timeline mapping and decisions
# ----------------------------------------------------------------------------------------------
def map_stat(stat: np.ndarray, n_points: int, mapping: str) -> tuple[np.ndarray, np.ndarray]:
    vp, mask = aggregate_vector_stream(np.asarray(stat, np.float64)[:, None], n_points, C, S, mapping)
    return vp[mask, 0].astype(np.float64), mask


def decide(stat_va, stat_te, rule, len_va, len_test, mapping):
    """Return (held-out preds, test preds, test score, test valid mask, threshold)."""
    v, _ = map_stat(stat_va, len_va, mapping)
    t, tm = map_stat(stat_te, len_test, mapping)
    half = len(v) // 2
    if rule in ("q995", "spot"):
        th = float(np.percentile(v[:half], 99.5)) if rule == "q995" else fit_spot(v[:half])
        return (v[half + S :] > th).astype(int), (t > th).astype(int), t, tm, th
    bv, _ = map_stat((np.asarray(stat_va) >= rule).astype(float), len_va, mapping)
    bt, _ = map_stat((np.asarray(stat_te) >= rule).astype(float), len_test, mapping)
    return (bv[half + S :] >= 0.5).astype(int), (bt >= 0.5).astype(int), t, tm, float(rule)


def score_row(tl, hp, tp, score) -> dict:
    mt = compute_metrics(tl, tp, scores=tp.astype(float), valid_mask=np.ones_like(tl, bool), use_pa=False,
                         include_events=False)
    er, dl = event_stats(tl, tp)
    return dict(
        f1=float(mt["f1"]), precision=float(mt["precision"]), recall=float(mt["recall"]),
        pr_auc=float(average_precision_score(tl, score)) if 0 < tl.sum() < len(tl) else np.nan,
        er=float(er) if er == er else np.nan, delay=float(dl) if dl == dl else np.nan,
        hold_fpr=float(hp.mean()) if len(hp) else np.nan,
        test_fpr=float(mt["fp"] / max((tl == 0).sum(), 1)), alarm_rate=float(tp.mean()),
    )


# ----------------------------------------------------------------------------------------------
# Per stream x seed evaluation
# ----------------------------------------------------------------------------------------------
def window_arms(d: dict, bits: dict, va_224, te_224, n_cal) -> list[tuple[str, str, np.ndarray, np.ndarray, object]]:
    """(arm, family, stat_va, stat_te, rule) for every non-panel arm built on window statistics."""
    z_va, z_te = robust_z(va_224, te_224, n_cal)
    tau = np.percentile(z_va[:n_cal], 98.5, axis=0, keepdims=True) + 1e-4
    cnt_va, cnt_te = bits["bits224_va"].sum(axis=1), bits["bits224_te"].sum(axis=1)
    pv = pvalue_statistics(va_224, te_224, n_cal)
    x_va = d["va_ew"].reshape(len(d["va_ew"]), -1)
    x_te = d["te_ew"].reshape(len(d["te_ew"]), -1)
    t2_va, t2_te = hotelling_t2(x_va, x_te, n_cal)
    mw_va, mw_te = mewma(x_va, x_te, n_cal, lam=0.2)
    cu = cusum_statistics(d["va_ew"][:, 3, :], d["te_ew"][:, 3, :], n_cal, k=0.5, alpha=0.005)
    m = va_224.shape[1]
    arms = [
        ("B1_maxz_q995", "B", z_va.max(1), z_te.max(1), "q995"),
        ("B1_maxz_spot", "B", z_va.max(1), z_te.max(1), "spot"),
        ("B1_max_exceed_ratio_q995", "B", (z_va / tau).max(1), (z_te / tau).max(1), "q995"),
    ]
    arms += [(f"B2_top{k}_fixed", "B", cnt_va, cnt_te, float(k)) for k in (1, 3, 5, 10)]
    arms += [
        ("B2_topk_q995", "B", cnt_va, cnt_te, "q995"),
        ("B3_conformal_bonferroni", "B", *pv["bonf"], float(-np.log10(ALPHA))),
        ("B3_conformal_bh", "B", *pv["bh"], float(-np.log10(ALPHA))),
        ("B4_fisher_q995", "B", *pv["fisher"], "q995"),
        ("B4_stouffer_q995", "B", *pv["stouffer"], "q995"),
        ("C1_hotelling_T2_spot", "C", t2_va, t2_te, "spot"),
        ("C1_hotelling_T2_q995", "C", t2_va, t2_te, "q995"),
        ("C1_sumsq_whitened_spot", "C", (x_va**2).sum(1), (x_te**2).sum(1), "spot"),
        ("C2_mewma_q995", "C", mw_va, mw_te, "q995"),
        ("C2_mewma_spot", "C", mw_va, mw_te, "spot"),
        ("C3_cusum_max_q995", "C", *cu["max"], "q995"),
        ("C4_cusum_bonferroni_any", "C", *cu["bonf"], 1.0),
        ("R_1D_scalar_spot", "R", d["va_s_base"], d["te_s_base"], "spot"),
    ]
    assert m == 224
    return arms


def redundancy_rows(stream, seed, bits, va_224, n_cal, group_preds) -> list[dict]:
    """Window-level redundancy diagnostics between perspective groups."""
    rows = []
    for split, b in (("cal", bits["bits224_va"][:n_cal]), ("test", bits["bits224_te"])):
        cnt = np.stack([b[:, idx].sum(axis=1) for idx in GROUPS.values()], axis=1)
        rho = spearmanr(cnt).statistic if cnt.std(axis=0).min() > 0 else np.full((4, 4), np.nan)
        for (i, gi), (j, gj) in itertools.combinations(enumerate(GROUPS), 2):
            rows.append(dict(stream=stream, seed=seed, metric=f"spearman_tripcount_{split}", pair=f"{gi}-{gj}",
                             value=float(rho[i, j]) if np.ndim(rho) == 2 else float(rho)))
    z_cal, _ = robust_z(va_224, va_224[:1], n_cal)
    for name, idx in [("all224", np.arange(224)), *GROUPS.items()]:
        er, pr = effective_rank(z_cal[:n_cal, idx])
        rows.append(dict(stream=stream, seed=seed, metric="erank_cal_z", pair=name, value=er))
        rows.append(dict(stream=stream, seed=seed, metric="participation_ratio_cal_z", pair=name, value=pr))
    for gi, gj in itertools.combinations(GROUPS, 2):
        a, b = group_preds[gi].astype(bool), group_preds[gj].astype(bool)
        union = (a | b).sum()
        rows.append(dict(stream=stream, seed=seed, metric="jaccard_test_alarms_trailing", pair=f"{gi}-{gj}",
                         value=float((a & b).sum() / union) if union else np.nan))
    return rows


def evaluate_run(ds, ch, seed, panel) -> tuple[list[dict], list[dict]]:
    d = load_cached(ds, ch, seed)
    stream = f"{ds}:{ch}"
    va_ew, te_ew, y = d["va_ew"], d["te_ew"], d["y"]
    len_va, len_test = d["len_va"], d["len_test"]
    n_cal = len(va_ew) // 2
    va_224, te_224 = build_224_perspective_dials(va_ew), build_224_perspective_dials(te_ew)
    bits = panel.calibrate_and_trip_windows(va_ew, te_ew, n_cal)
    arms = window_arms(d, bits, va_224, te_224, n_cal)
    rows, group_preds = [], {}

    def add(arm, family, mapping, hp, tp, score, tl, th=np.nan):
        rows.append(dict(stream=stream, dataset=ds, seed=seed, mapping=mapping, arm=arm, family=family,
                         threshold=th, **score_row(tl, hp, tp, score)))

    for mapping in MAPPINGS:
        dec = panel.evaluate_stream_decisions(bits, len_va, len_test, mapping)
        tl = y[dec.valid_mask]
        add(REF, "PANEL", mapping, dec.hold_pred_balanced, dec.test_pred_balanced, dec.test_switch_counts, tl,
            dec.k_gate_balanced)
        for groups in SUBSETS + [("P4",)]:
            idx = P4 if groups == ("P4",) else np.concatenate([GROUPS[g] for g in groups])
            hp, tp, cnt = eval_subset_panel(va_224[:, idx], te_224[:, idx], n_cal, len_va, len_test, mapping)
            add(subset_name(groups), "A", mapping, hp, tp, cnt, tl)
            if mapping == "trailing" and len(groups) == 1 and groups[0] in GROUPS:
                group_preds[groups[0]] = tp
        for arm, family, s_va, s_te, rule in arms:
            hp, tp, score, tm, th = decide(s_va, s_te, rule, len_va, len_test, mapping)
            if not np.array_equal(tm, dec.valid_mask):
                raise RuntimeError(f"valid-mask mismatch for {arm} on {stream}/{seed}/{mapping}")
            add(arm, family, mapping, hp, tp, score, tl, th)
    return rows, redundancy_rows(stream, seed, bits, va_224, n_cal, group_preds)


# ----------------------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------------------
def paired(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> tuple[float, float, float, float]:
    """Mean difference a-b, two-sided Wilcoxon p, and 95% stream-bootstrap CI."""
    ok = np.isfinite(a) & np.isfinite(b)
    diff = a[ok] - b[ok]
    if len(diff) == 0:
        return np.nan, np.nan, np.nan, np.nan
    p = 1.0 if np.allclose(diff, 0.0) else float(wilcoxon(a[ok], b[ok]).pvalue)
    boot = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(axis=1)
    return float(diff.mean()), p, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    ps = df.groupby(["stream", "dataset", "mapping", "arm", "family"], as_index=False)[METRICS].mean()
    rng = np.random.default_rng(0)
    out = []
    for (mapping, arm, family), g in ps.groupby(["mapping", "arm", "family"], sort=False):
        ref = ps[(ps["mapping"] == mapping) & (ps["arm"] == REF)].set_index("stream")
        g = g.set_index("stream").loc[ref.index]
        row = dict(mapping=mapping, arm=arm, family=family, n_streams=len(g))
        row.update({m: float(g[m].mean()) for m in METRICS})
        macro = g.groupby("dataset")[["f1", "er", "hold_fpr", "pr_auc"]].mean().mean()
        row.update({f"macro_{k}": float(v) for k, v in macro.items()})
        for m in ("f1", "hold_fpr", "er", "pr_auc"):
            dm, p, lo, hi = paired(g[m].to_numpy(float), ref[m].to_numpy(float), rng)
            row.update({f"d_{m}_vs_panel": dm, f"wilcoxon_p_{m}": p, f"ci_lo_{m}": lo, f"ci_hi_{m}": hi})
        out.append(row)
    return pd.DataFrame(out)


def group_contributions(df: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-group-out deltas and Shapley values of the 4 perspective groups (stream means)."""
    ps = df.groupby(["stream", "mapping", "arm"], as_index=False)[METRICS].mean()
    rng = np.random.default_rng(1)
    rows = []
    for mapping in MAPPINGS:
        sub = ps[ps["mapping"] == mapping].pivot(index="stream", columns="arm")
        full = subset_name("PDFE")
        for metric in ("f1", "er", "hold_fpr", "pr_auc"):
            val = {frozenset(s): float(sub[(metric, subset_name(s))].mean()) for s in SUBSETS}
            val[frozenset()] = 0.0 if metric != "pr_auc" else np.nan
            for g in "PDFE":
                others = [x for x in "PDFE" if x != g]
                shap = 0.0
                for r in range(0, 4):
                    for s in itertools.combinations(others, r):
                        w = math.factorial(r) * math.factorial(3 - r) / 24.0
                        shap += w * (val[frozenset(s) | {g}] - val[frozenset(s)])
                loo_name = subset_name(tuple(others))
                dm, p, lo, hi = paired(sub[(metric, full)].to_numpy(float), sub[(metric, loo_name)].to_numpy(float), rng)
                rows.append(dict(mapping=mapping, metric=metric, group=g, full_value=val[frozenset("PDFE")],
                                 without_group_value=val[frozenset(others)], loo_delta=dm, loo_wilcoxon_p=p,
                                 loo_ci_lo=lo, loo_ci_hi=hi, alone_value=val[frozenset(g)], shapley=shap))
    return pd.DataFrame(rows)


def atomic_csv(df: pd.DataFrame, path: str) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", newline="") as fh:
        df.to_csv(fh, index=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports" / "ensemble_spc_baselines_19stream.csv"))
    args = ap.parse_args()
    stem = args.out[:-4]
    panel = MultiPerspectiveCircuitBreakerPanel(context_size=C, suspect_size=S)
    rows, red = [], []
    for ds, ch in load_streams_19():
        t0 = time.time()
        for seed in SEEDS:
            r, rr = evaluate_run(ds, ch, seed, panel)
            rows += r
            red += rr
        print(f"{ds}:{ch} done in {time.time() - t0:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    atomic_csv(df, args.out)
    summ = summarize(df)
    atomic_csv(summ, f"{stem}_summary.csv")
    atomic_csv(group_contributions(df), f"{stem}_group_contrib.csv")
    red_df = pd.DataFrame(red)
    atomic_csv(red_df, f"{stem}_redundancy.csv")

    pd.set_option("display.width", 250)
    cols = ["arm", "f1", "precision", "recall", "er", "hold_fpr", "delay", "pr_auc", "wilcoxon_p_f1", "ci_lo_f1",
            "ci_hi_f1", "wilcoxon_p_hold_fpr"]
    for mapping in MAPPINGS:
        print(f"\n=== {mapping.upper()} ===")
        print(summ[summ["mapping"] == mapping][cols].round(4).to_string(index=False))
    print("\n=== REDUNDANCY (mean over stream x seed) ===")
    print(red_df.groupby(["metric", "pair"])["value"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
