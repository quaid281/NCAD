"""Compare detector banks, split-conformal variants, and chronological cross-fitting.

Thresholds use only nominal calibration windows. Held-out and test alert rates
are reported rather than forced to match, since regime drift affects each detector
differently. Also evaluates 2-fold chronological cross-fitting across the two
disjoint nominal validation halves (separated by the 64-step isolation gap).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.circuit_breaker_19stream.common import C, ROOT, S, SEEDS, load_cached, load_streams_19
from scripts.circuit_breaker_19stream.run_perspective_ladder import eval_subset_panel
from scripts.reviewer_analyses.ensemble_and_redundancy_baselines import (
    REF,
    atomic_csv,
    map_stat,
    score_row,
    summarize,
    window_arms,
)
from scripts.reviewer_analyses.ensemble_baseline_arms import conformal_pvalues, hotelling_t2, robust_z
from src.scoring.circuit_breaker_panel import MultiPerspectiveCircuitBreakerPanel, build_224_perspective_dials

CONTROLS = {
    "B1_maxz_q995",
    "B2_topk_q995",
    "B4_fisher_q995",
    "B4_stouffer_q995",
    "C1_hotelling_T2_q995",
    "C2_mewma_q995",
    "C3_cusum_max_q995",
}


def threshold_for_calibration_rate(calibration: np.ndarray, rate: float) -> float:
    if not len(calibration) or not 0.0 <= rate <= 1.0:
        raise ValueError("Nominal calibration statistic and target rate must be valid")
    return float(np.quantile(calibration, 1.0 - rate))


def split_conformal_threshold(calibration: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal order statistic ceil((n + 1) * (1 - alpha))."""
    n = len(calibration)
    rank = min(n, max(1, math.ceil((n + 1) * (1.0 - alpha))))
    return float(np.partition(calibration, rank - 1)[rank - 1])


def panel_calibration_rate(bits: dict[str, np.ndarray], k_pref: float, n_cal: int) -> float:
    pref = bits["bits224_va"][:n_cal].sum(axis=1)
    sub = bits["bits128_va"][:n_cal].sum(axis=1)
    edge = bits["fast_edge_va"][:n_cal].sum(axis=1)
    k_sub = max(float(np.percentile(sub, 99.5)) * 1.12, 3.5)
    k_edge = max(float(np.percentile(edge, 99.9)) * 1.25, 2.5)
    return float(np.mean((pref > k_pref) | (sub > k_sub) | (edge > k_edge)))


def evaluate_crossfit_run(ds: str, ch: str, seed: int, panel: MultiPerspectiveCircuitBreakerPanel) -> list[dict]:
    """2-fold chronological cross-fitting across nominal validation halves with S-step gap."""
    d = load_cached(ds, ch, seed)
    va, te, y = d["va_ew"], d["te_ew"], d["y"]
    n_va = len(va)
    n_cal = n_va // 2
    if n_va <= n_cal + S + 32:
        return []
    # Fold 1: calibrate on [0, n_cal), evaluate nominal on [n_cal + S, n_va)
    bits_f1 = panel.calibrate_and_trip_windows(va, te, n_cal)
    dec_f1 = panel.evaluate_stream_decisions(bits_f1, d["len_va"], d["len_test"], "trailing")
    tl = y[dec_f1.valid_mask]
    # Fold 2: reverse nominal order so [n_cal + S, n_va) is calibration and [0, n_cal) is held-out
    va_f2 = np.concatenate([va[n_cal + S :], va[n_cal : n_cal + S], va[:n_cal]], axis=0)
    n_cal_f2 = len(va) - (n_cal + S)
    bits_f2 = panel.calibrate_and_trip_windows(va_f2, te, n_cal_f2)
    cnt224_va2 = bits_f2["bits224_va"].sum(axis=1)
    cnt128_va2 = bits_f2["bits128_va"].sum(axis=1)
    edge_va2 = bits_f2["fast_edge_va"].sum(axis=1)
    k224_2 = max(float(np.percentile(cnt224_va2[:n_cal_f2], 99.5)) * 1.12, 5.0)
    k128_2 = max(float(np.percentile(cnt128_va2[:n_cal_f2], 99.5)) * 1.12, 3.5)
    kedge_2 = max(float(np.percentile(edge_va2[:n_cal_f2], 99.9)) * 1.25, 2.5)
    hold_f2 = (
        (cnt224_va2[n_cal_f2 + S :] > k224_2)
        | (cnt128_va2[n_cal_f2 + S :] > k128_2)
        | (edge_va2[n_cal_f2 + S :] > kedge_2)
    ).astype(int)
    te_cnt224_2 = bits_f2["bits224_te"].sum(axis=1)
    te_cnt128_2 = bits_f2["bits128_te"].sum(axis=1)
    te_edge_2 = bits_f2["fast_edge_te"].sum(axis=1)
    tp_f2 = ((te_cnt224_2 > k224_2) | (te_cnt128_2 > k128_2) | (te_edge_2 > kedge_2)).astype(int)
    tp_f1 = dec_f1.test_pred_balanced.astype(int)
    sc_avg = 0.5 * (dec_f1.test_switch_counts + te_cnt224_2)
    hold_pool = np.concatenate([dec_f1.hold_pred_balanced.astype(int), hold_f2])
    base = dict(stream=f"{ds}:{ch}", dataset=ds, seed=seed, n_cal_f1=n_cal, n_cal_f2=n_cal_f2)
    return [
        dict(**base, mode="Fold1_Forward_Standard", **score_row(tl, dec_f1.hold_pred_balanced, tp_f1, dec_f1.test_switch_counts)),
        dict(**base, mode="Fold2_Reverse_Swap", **score_row(tl, hold_f2, tp_f2, te_cnt224_2)),
        dict(**base, mode="CrossFit_Consensus_AND", **score_row(tl, hold_pool, tp_f1 & tp_f2, sc_avg)),
        dict(**base, mode="CrossFit_Union_OR", **score_row(tl, hold_pool, tp_f1 | tp_f2, sc_avg)),
    ]


def evaluate_run(ds: str, ch: str, seed: int, panel: MultiPerspectiveCircuitBreakerPanel) -> list[dict]:
    d = load_cached(ds, ch, seed)
    va, te, y = d["va_ew"], d["te_ew"], d["y"]
    n_cal = len(va) // 2
    va_224, te_224 = build_224_perspective_dials(va), build_224_perspective_dials(te)
    bits = panel.calibrate_and_trip_windows(va, te, n_cal)
    dec = panel.evaluate_stream_decisions(bits, d["len_va"], d["len_test"], "trailing")
    cal_rate = panel_calibration_rate(bits, dec.k_gate_balanced, n_cal)
    tl = y[dec.valid_mask]
    shared = dict(stream=f"{ds}:{ch}", dataset=ds, seed=seed, mapping="trailing", n_cal=n_cal,
                  target_cal_rate=cal_rate)
    rows = [
        dict(**shared, arm=REF, family="PANEL", threshold=dec.k_gate_balanced,
             actual_cal_rate=cal_rate, **score_row(tl, dec.hold_pred_balanced, dec.test_pred_balanced,
                                                   dec.test_switch_counts))
    ]
    bh224_pair: tuple[np.ndarray, np.ndarray] | None = None
    for arm, family, s_va, s_te, _ in window_arms(d, bits, va_224, te_224, n_cal):
        if arm == "B3_conformal_bh":
            bh224_pair = (s_va, s_te)
        if arm not in CONTROLS:
            continue
        v, _ = map_stat(s_va, d["len_va"], "trailing")
        t, mask = map_stat(s_te, d["len_test"], "trailing")
        if not np.array_equal(mask, dec.valid_mask) or len(v) != len(va):
            raise RuntimeError(f"Timeline mismatch for {arm} on {ds}:{ch} seed {seed}")
        threshold = threshold_for_calibration_rate(v[:n_cal], cal_rate)
        rows.append(dict(**shared, arm=arm, family=family, threshold=threshold,
                         actual_cal_rate=float(np.mean(v[:n_cal] > threshold)),
                         **score_row(tl, v[n_cal + S :] > threshold, t > threshold, t)))

    # 1. Marginal per-coordinate calibration (no quorum; union across 128 coordinates)
    abs_va = np.abs(va.reshape(len(va), -1))
    abs_te = np.abs(te.reshape(len(te), -1))
    z128_va, z128_te = robust_z(abs_va, abs_te, n_cal)
    q985_128 = np.percentile(z128_va[:n_cal], 98.5, axis=0, keepdims=True) + 1e-4
    u985_va, _ = map_stat((z128_va > q985_128).any(axis=1).astype(float), d["len_va"], "trailing")
    u985_te, _ = map_stat((z128_te > q985_128).any(axis=1).astype(float), d["len_test"], "trailing")
    max128_va, _ = map_stat(z128_va.max(axis=1), d["len_va"], "trailing")
    max128_te, _ = map_stat(z128_te.max(axis=1), d["len_test"], "trailing")
    rows.append(dict(**shared, arm="M1_marginal_coord_q985_union", family="M", threshold=0.985,
                     actual_cal_rate=float(np.mean(u985_va[:n_cal] >= 0.5)),
                     **score_row(tl, u985_va[n_cal + S :] >= 0.5, u985_te >= 0.5, max128_te)))
    marg_q = float(np.clip(1.0 - max(cal_rate, 1.0 / max(n_cal, 1)), 0.5, 1.0))
    qm_128 = np.quantile(z128_va[:n_cal], marg_q, axis=0, keepdims=True) + 1e-6
    um_va, _ = map_stat((z128_va > qm_128).any(axis=1).astype(float), d["len_va"], "trailing")
    um_te, _ = map_stat((z128_te > qm_128).any(axis=1).astype(float), d["len_test"], "trailing")
    rows.append(dict(**shared, arm="M2_marginal_coord_matched_union", family="M", threshold=marg_q,
                     actual_cal_rate=float(np.mean(um_va[:n_cal] >= 0.5)),
                     **score_row(tl, um_va[n_cal + S :] >= 0.5, um_te >= 0.5, max128_te)))

    # 1b. System-level matched 128-coord continuous max-z & raw top-k coordinate voting (no drift/guards)
    sys_max128_thr = threshold_for_calibration_rate(max128_va[:n_cal], cal_rate)
    rows.append(dict(**shared, arm="S1_coord128_maxz_system_matched", family="S", threshold=sys_max128_thr,
                     actual_cal_rate=float(np.mean(max128_va[:n_cal] > sys_max128_thr)),
                     **score_row(tl, max128_va[n_cal + S :] > sys_max128_thr, max128_te > sys_max128_thr, max128_te)))
    cnt128_raw_va, _ = map_stat((z128_va > q985_128).sum(axis=1).astype(float), d["len_va"], "trailing")
    cnt128_raw_te, _ = map_stat((z128_te > q985_128).sum(axis=1).astype(float), d["len_test"], "trailing")
    sys_topk128_thr = threshold_for_calibration_rate(cnt128_raw_va[:n_cal], cal_rate)
    rows.append(dict(**shared, arm="S2_coord128_topk_system_matched", family="S", threshold=sys_topk128_thr,
                     actual_cal_rate=float(np.mean(cnt128_raw_va[:n_cal] > sys_topk128_thr)),
                     **score_row(tl, cnt128_raw_va[n_cal + S :] > sys_topk128_thr, cnt128_raw_te > sys_topk128_thr, cnt128_raw_te)))

    # 1c. Per-coordinate & 224-dial FWER / FDR and discrete conformal union
    p_va = conformal_pvalues(abs_va[:n_cal], abs_va)
    p_te = conformal_pvalues(abs_va[:n_cal], abs_te)
    s_coord_va = -np.log10(np.maximum(p_va.min(axis=1), 1e-12))
    s_coord_te = -np.log10(np.maximum(p_te.min(axis=1), 1e-12))
    sidak_thr = -math.log10(1.0 - (1.0 - 0.005) ** (1.0 / abs_va.shape[1]))
    rows.append(dict(**shared, arm="D1_per_coord_conformal_sidak005", family="D", threshold=sidak_thr,
                     actual_cal_rate=float(np.mean(s_coord_va[:n_cal] >= sidak_thr)),
                     **score_row(tl, s_coord_va[n_cal + S :] >= sidak_thr, s_coord_te >= sidak_thr, s_coord_te)))
    assert bh224_pair is not None
    bh_thr = -math.log10(0.005)
    bh_va, _ = map_stat(bh224_pair[0], d["len_va"], "trailing")
    bh_te, _ = map_stat(bh224_pair[1], d["len_test"], "trailing")
    rows.append(dict(**shared, arm="D1b_224dial_conformal_bh005", family="D", threshold=bh_thr,
                     actual_cal_rate=float(np.mean(bh_va[:n_cal] >= bh_thr)),
                     **score_row(tl, bh_va[n_cal + S :] >= bh_thr, bh_te >= bh_thr, bh_te)))
    coord_m_thr = threshold_for_calibration_rate(s_coord_va[:n_cal], cal_rate)
    rows.append(dict(**shared, arm="D1_per_coord_conformal_matched", family="D", threshold=coord_m_thr,
                     actual_cal_rate=float(np.mean(s_coord_va[:n_cal] > coord_m_thr)),
                     **score_row(tl, s_coord_va[n_cal + S :] > coord_m_thr, s_coord_te > coord_m_thr, s_coord_te)))

    # 2. Multivariate split-conformal (whitened L2 and Ledoit-Wolf Mahalanobis)
    l2_va = np.sum(abs_va ** 2, axis=1)
    l2_te = np.sum(abs_te ** 2, axis=1)
    l2_c005 = split_conformal_threshold(l2_va[:n_cal], 0.005)
    l2_cm = threshold_for_calibration_rate(l2_va[:n_cal], cal_rate)
    rows.append(dict(**shared, arm="D2_mv_conformal_l2_a005", family="D", threshold=l2_c005,
                     actual_cal_rate=float(np.mean(l2_va[:n_cal] > l2_c005)),
                     **score_row(tl, l2_va[n_cal + S :] > l2_c005, l2_te > l2_c005, l2_te)))
    rows.append(dict(**shared, arm="D2_mv_conformal_l2_matched", family="D", threshold=l2_cm,
                     actual_cal_rate=float(np.mean(l2_va[:n_cal] > l2_cm)),
                     **score_row(tl, l2_va[n_cal + S :] > l2_cm, l2_te > l2_cm, l2_te)))
    t2_va, t2_te = hotelling_t2( va.reshape(len(va), -1), te.reshape(len(te), -1), n_cal )
    t2_c005 = split_conformal_threshold(t2_va[:n_cal], 0.005)
    rows.append(dict(**shared, arm="D3_mv_conformal_mahal_a005", family="D", threshold=t2_c005,
                     actual_cal_rate=float(np.mean(t2_va[:n_cal] > t2_c005)),
                     **score_row(tl, t2_va[n_cal + S :] > t2_c005, t2_te > t2_c005, t2_te)))

    # 3. Reduced 160-dial (P+D) & simpler 224-dial single-quorum (no horizon-onset relay)
    hp_160, tp_160, cnt_160 = eval_subset_panel(va_224[:, :160], te_224[:, :160], n_cal, d["len_va"], d["len_test"], "trailing")
    hp_224nr, tp_224nr, cnt_224nr = eval_subset_panel(va_224, te_224, n_cal, d["len_va"], d["len_test"], "trailing")
    rows.append(dict(**shared, arm="A_160dial_PD_single_quorum", family="A", threshold=np.nan,
                     actual_cal_rate=np.nan, **score_row(tl, hp_160, tp_160, cnt_160)))
    rows.append(dict(**shared, arm="A_224dial_no_relay_single_quorum", family="A", threshold=np.nan,
                     actual_cal_rate=np.nan, **score_row(tl, hp_224nr, tp_224nr, cnt_224nr)))
    # 4. Continuous PR-AUC score weighting ablation (Critique 3C: weighted vs unweighted)
    for mp in ("trailing", "smear"):
        for smode in ("weighted", "unweighted_224_edge", "unweighted_224"):
            dec_s = panel.evaluate_stream_decisions(bits, d["len_va"], d["len_test"], mp, score_mode=smode)
            tl_s = y[dec_s.valid_mask]
            rows.append(dict(
                stream=f"{ds}:{ch}", dataset=ds, seed=seed, mapping=mp, n_cal=n_cal,
                target_cal_rate=cal_rate, arm=f"SCORE_{smode}", family="SCORE",
                threshold=dec_s.k_gate_balanced, actual_cal_rate=cal_rate,
                **score_row(tl_s, dec_s.hold_pred_balanced, dec_s.test_pred_balanced, dec_s.test_switch_counts),
            ))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports" / "matched_alert_rate_conformal_19stream.csv"))
    ap.add_argument("--crossfit-out", default=str(ROOT / "reports" / "chronological_crossfit_19stream.csv"))
    ap.add_argument("--max-streams", type=int, default=0)
    args = ap.parse_args()
    out_path = Path(args.out)
    streams = load_streams_19()
    if args.max_streams:
        streams = streams[: args.max_streams]
    panel = MultiPerspectiveCircuitBreakerPanel(context_size=C, suspect_size=S)
    rows, cf_rows = [], []
    for ds, ch in streams:
        for seed in SEEDS:
            rows.extend(evaluate_run(ds, ch, seed, panel))
            cf_rows.extend(evaluate_crossfit_run(ds, ch, seed, panel))
        print(f"{ds}:{ch} matched-conformal & cross-fit controls done", flush=True)
    df = pd.DataFrame(rows)
    df_tr = df[df["mapping"] == "trailing"].copy()
    summ = summarize(df_tr[df_tr["family"] != "SCORE"])
    actual = df_tr.groupby(["stream", "arm"], as_index=False)[["target_cal_rate", "actual_cal_rate"]].mean()
    cal = actual.groupby("arm")[["target_cal_rate", "actual_cal_rate"]].mean().reset_index()
    summ = summ.merge(cal, on="arm", validate="many_to_one")
    atomic_csv(df, str(out_path))
    atomic_csv(summ, str(out_path.with_name(out_path.stem + "_summary.csv")))
    print("\n=== CONFORMAL, DETECTOR-BANK & REDUCED-PANEL SUMMARY ===")
    print(summ[["arm", "f1", "pr_auc", "er", "hold_fpr", "test_fpr", "target_cal_rate", "actual_cal_rate"]].to_string(index=False))

    sc_df = df[df["family"] == "SCORE"].groupby(["stream", "mapping", "arm"], as_index=False)[["pr_auc", "f1"]].mean()
    print("\n=== CONTINUOUS PR-AUC WEIGHTING ABLATION (19 STREAMS x 3 SEEDS) ===")
    print(sc_df.groupby(["mapping", "arm"])[["pr_auc", "f1"]].mean().to_string())

    cf_df = pd.DataFrame(cf_rows)
    cf_stream = cf_df.groupby(["stream", "mode"], as_index=False)[["f1", "er", "hold_fpr", "test_fpr", "pr_auc"]].mean()
    cf_summ = cf_stream.groupby("mode")[["f1", "er", "hold_fpr", "test_fpr", "pr_auc"]].agg(["mean", "median"])
    atomic_csv(cf_df, str(args.crossfit_out))
    print("\n=== 2-FOLD CHRONOLOGICAL CROSS-FITTING SUMMARY (19 STREAMS) ===")
    print(cf_summ.to_string())


if __name__ == "__main__":
    main()
