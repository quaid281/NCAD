#!/usr/bin/env python3
"""Perspective expansion ladder (1 -> 32 -> 128 -> 160 -> 192 -> 224 dials) on cached 19-stream residuals.

Reproduces Table II(a). Note: the cache stores whitened residuals only, so the calibration-stabilized
blend of Eq. (3) is not applied in this cohort. Subset rungs use one quorum counter with floor
max(2, 3.5 * n_dials / 128).

Usage: python -m scripts.circuit_breaker_19stream.run_perspective_ladder [--out CSV]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from scripts.circuit_breaker_19stream.common import C, ROOT, S, SEEDS, fit_spot, load_cached, load_streams_19
from scripts.run_multiseed_heldout_benchmark import event_stats
from src.scoring.circuit_breaker_panel import (
    MultiPerspectiveCircuitBreakerPanel,
    aggregate_prefix_224_stream,
    aggregate_vector_stream,
    build_224_perspective_dials,
)
from src.scoring.event_fusion import aggregate_window_scores, compute_metrics


def eval_subset_panel(va_dials, te_dials, n_cal_w, len_va, len_test, mapping):
    """Calibrate guarded per-dial switches on a dial subset and apply a single nominal quorum."""
    med = np.median(va_dials[:n_cal_w], axis=0, keepdims=True)
    mad = 1.4826 * np.median(np.abs(va_dials[:n_cal_w] - med), axis=0, keepdims=True) + 1e-4
    z_va = np.maximum(0.0, (va_dials - med) / mad)
    z_te = np.maximum(0.0, (te_dials - med) / mad)
    q1 = max(1, n_cal_w // 2)
    drift = np.maximum(
        1.0,
        (np.percentile(z_va[q1:n_cal_w], 98.0, axis=0, keepdims=True) + 0.5)
        / (np.percentile(z_va[:q1], 98.0, axis=0, keepdims=True) + 0.5),
    )
    s_cal = med + mad
    scale = np.maximum(1.0, s_cal / (1.5 * np.median(s_cal, keepdims=True) + 1e-4))
    guard = np.where(drift > 1.35, drift, 1.0) * np.where(scale > 1.25, scale, 1.0)
    tau = (np.percentile(z_va[:n_cal_w], 98.5, axis=0, keepdims=True) + 1e-4) * guard
    vp, vm = aggregate_vector_stream((z_va > tau).astype(np.float32), len_va, C, S, mapping)
    tp, tm = aggregate_vector_stream((z_te > tau).astype(np.float32), len_test, C, S, mapping)
    v_cnt, t_cnt = vp[vm].sum(axis=1), tp[tm].sum(axis=1)
    half = len(v_cnt) // 2
    floor_k = max(2.0, 3.5 * (va_dials.shape[1] / 128.0))
    k_bal = max(float(np.percentile(v_cnt[:half], 99.5)) * 1.12, floor_k)
    return (v_cnt[half + S :] > k_bal).astype(int), (t_cnt > k_bal).astype(int), t_cnt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports" / "perspective_ladder_19stream.csv"))
    args = ap.parse_args()

    panel = MultiPerspectiveCircuitBreakerPanel(context_size=C, suspect_size=S)
    rows = []
    for ds, ch in load_streams_19():
        for seed in SEEDS:
            d = load_cached(ds, ch, seed)
            va_ew, te_ew, y = d["va_ew"], d["te_ew"], d["y"]
            len_va, len_test = d["len_va"], d["len_test"]
            n_cal_w = len(va_ew) // 2
            va_224, te_224 = build_224_perspective_dials(va_ew), build_224_perspective_dials(te_ew)
            bits_full = panel.calibrate_and_trip_windows(va_ew, te_ew, n_cal_w)

            for mapping in ("trailing", "smear"):
                dec = panel.evaluate_stream_decisions(bits_full, len_va, len_test, mapping)
                tl = y[dec.valid_mask]
                pr_full = float(average_precision_score(tl, dec.test_switch_counts))

                # 1D scalar collapse on the same backbone, masked exactly like the panel counters.
                _, vm = aggregate_prefix_224_stream(bits_full["bits224_va"], len_va, C, S, mapping)
                vp_b, _ = aggregate_window_scores(d["va_s_base"], len_va, C, S, step=1, reducer="mean", mapping_method=mapping)
                tp_b, _ = aggregate_window_scores(d["te_s_base"], len_test, C, S, step=1, reducer="mean", mapping_method=mapping)
                v_b, t_b = vp_b[vm], tp_b[dec.valid_mask]
                half = len(v_b) // 2
                th = fit_spot(v_b[:half])
                variants = [("0_1D_Scalar_Collapse", (v_b[half + S :] > th).astype(int), (t_b > th).astype(int),
                             float(average_precision_score(tl, t_b)))]
                for name, sl in (("1_32Bit_SingleHorizon", slice(96, 128)), ("2_128Bit_SubHorizons", slice(0, 128)),
                                 ("3_160Bit_Plus_MeanDrift", slice(0, 160)), ("4_192Bit_Plus_Fluctuation", slice(0, 192))):
                    hp, tp, cnt = eval_subset_panel(va_224[:, sl], te_224[:, sl], n_cal_w, len_va, len_test, mapping)
                    variants.append((name, hp, tp, float(average_precision_score(tl, cnt))))
                variants += [("5_224Bit_Strict", dec.hold_pred_strict, dec.test_pred_strict, pr_full),
                             ("6_224Bit_Breaker_Full", dec.hold_pred_balanced, dec.test_pred_balanced, pr_full)]

                for vname, hp, tp, prv in variants:
                    mt = compute_metrics(tl, tp, scores=tp.astype(float), valid_mask=np.ones_like(tl, bool), use_pa=False)
                    er, dl = event_stats(tl, tp)
                    rows.append(dict(stream=f"{ds}:{ch}", seed=seed, mapping=mapping, variant=vname,
                                     f1=float(mt["f1"]), pr_auc=prv,
                                     er=float(er) if er == er else np.nan, delay=float(dl) if dl == dl else np.nan,
                                     hold_fpr=float(hp.mean())))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    ps = df.groupby(["stream", "mapping", "variant"]).mean(numeric_only=True).reset_index()
    for mapping in ("trailing", "smear"):
        print(f"\n=== PERSPECTIVE LADDER ({mapping.upper()}) ===")
        print(ps[ps["mapping"] == mapping].groupby("variant")[["f1", "pr_auc", "er", "delay", "hold_fpr"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
