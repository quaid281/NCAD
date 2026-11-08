#!/usr/bin/env python3
"""Leave-one-out component ablations of the 224-bit panel on cached 19-stream residuals (Table II(b)).

Note: the cache stores whitened residuals only, so the calibration-stabilized blend of Eq. (3) is not
applied in this cohort.

Usage: python -m scripts.circuit_breaker_19stream.run_component_ablations [--out CSV]
"""

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


def run_ablation(out_csv: str):
    panel = MultiPerspectiveCircuitBreakerPanel(context_size=C, suspect_size=S)
    rows = []

    for ds, ch in load_streams_19():
        for seed in SEEDS:
            data = load_cached(ds, ch, seed)
            len_va = data["len_va"]
            len_test = data["len_test"]
            y = data["y"]
            va_ew_raw = data["va_ew"]
            te_ew_raw = data["te_ew"]
            n_cal_w = len(va_ew_raw) // 2
            # Standard panel bits (no Eq. (3) blend: raw residuals are not cached)
            bits_full = panel.calibrate_and_trip_windows(va_ew_raw, te_ew_raw, n_cal_w)

            # Ablation: w/o Hijack Guards (guard_128 = 1, guard_224 = 1)
            va_128 = np.abs(va_ew_raw).reshape(len(va_ew_raw), -1)
            te_128 = np.abs(te_ew_raw).reshape(len(te_ew_raw), -1)
            va_224 = build_224_perspective_dials(va_ew_raw)
            te_224 = build_224_perspective_dials(te_ew_raw)

            med_128 = np.median(va_128[:n_cal_w], axis=0, keepdims=True)
            mad_128 = 1.4826 * np.median(np.abs(va_128[:n_cal_w] - med_128), axis=0, keepdims=True) + 1e-4
            z128_va = np.maximum(0.0, (va_128 - med_128) / mad_128)
            z128_te = np.maximum(0.0, (te_128 - med_128) / mad_128)

            med_224 = np.median(va_224[:n_cal_w], axis=0, keepdims=True)
            mad_224 = 1.4826 * np.median(np.abs(va_224[:n_cal_w] - med_224), axis=0, keepdims=True) + 1e-4
            z224_va = np.maximum(0.0, (va_224 - med_224) / mad_224)
            z224_te = np.maximum(0.0, (te_224 - med_224) / mad_224)

            tau_128_noguard = np.percentile(z128_va[:n_cal_w], 98.5, axis=0, keepdims=True) + 1e-4
            tau_224_noguard = np.percentile(z224_va[:n_cal_w], 98.5, axis=0, keepdims=True) + 1e-4
            bits128_va_ng = (z128_va > tau_128_noguard).astype(np.float32)
            bits128_te_ng = (z128_te > tau_128_noguard).astype(np.float32)
            bits224_va_ng = (z224_va > tau_224_noguard).astype(np.float32)
            bits224_te_ng = (z224_te > tau_224_noguard).astype(np.float32)

            z7_va = z224_va.reshape(-1, 7, 32)
            z7_te = z224_te.reshape(-1, 7, 32)
            tau_edge_ng = np.percentile(z7_va[:n_cal_w], 99.3, axis=0, keepdims=True) + 1e-4
            b_edge_va_ng = z7_va > tau_edge_ng
            b_edge_te_ng = z7_te > tau_edge_ng
            fast_edge_va_ng = (b_edge_va_ng[:, 3, :] & (b_edge_va_ng[:, 6, :] | b_edge_va_ng[:, 5, :] | b_edge_va_ng[:, 2, :])).astype(np.float32)
            fast_edge_te_ng = (b_edge_te_ng[:, 3, :] & (b_edge_te_ng[:, 6, :] | b_edge_te_ng[:, 5, :] | b_edge_te_ng[:, 2, :])).astype(np.float32)

            base_va = data["va_s_base"]
            base_te = data["te_s_base"]

            for mapping in ("trailing", "smear"):
                # 1. Full 224-Bit Breaker & 224-Bit Strict
                dec = panel.evaluate_stream_decisions(bits_full, len_va, len_test, mapping)
                tl = y[dec.valid_mask]
                pr_224 = float(average_precision_score(tl, dec.test_switch_counts))

                # 2. w/o 32 Right-Edge Onset Relays (224-Bit + 128-Bit Quorum Only)
                vp_pref, vm = aggregate_prefix_224_stream(bits_full["bits224_va"], len_va, C, S, mapping)
                tp_pref, tm = aggregate_prefix_224_stream(bits_full["bits224_te"], len_test, C, S, mapping)
                vp_b128, _ = aggregate_vector_stream(bits_full["bits128_va"], len_va, C, S, mapping)
                tp_b128, _ = aggregate_vector_stream(bits_full["bits128_te"], len_test, C, S, mapping)
                vpref, tpref = vp_pref[vm], tp_pref[tm]
                vb128, tb128 = vp_b128[vm], tp_b128[tm]
                half = len(vpref) // 2

                k_128_bal = max(float(np.percentile(vb128[:half].sum(axis=1), 99.5)) * 1.12, 3.5)
                k_pref_bal = max(float(np.percentile(vpref[:half].sum(axis=1), 99.5)) * 1.12, 5.0)
                hold_no_relay = np.maximum(
                    (vb128[half + S:].sum(axis=1) > k_128_bal).astype(int),
                    (vpref[half + S:].sum(axis=1) > k_pref_bal).astype(int),
                )
                test_no_relay = np.maximum(
                    (tb128.sum(axis=1) > k_128_bal).astype(int),
                    (tpref.sum(axis=1) > k_pref_bal).astype(int),
                )
                pr_no_relay = float(average_precision_score(tl, tpref.sum(axis=1) + tb128.sum(axis=1)))

                # 3. 128-Bit Sub-Block Panel Only (w/o 96 Reynolds drift/fluctuation/accel dials & w/o relays)
                hold_128_only = (vb128[half + S:].sum(axis=1) > k_128_bal).astype(int)
                test_128_only = (tb128.sum(axis=1) > k_128_bal).astype(int)
                pr_128_only = float(average_precision_score(tl, tb128.sum(axis=1)))

                # 4. w/o Prefix-Horizon Routing (Uniform 64-Step Smear across all 224 dials)
                vp_unif224, _ = aggregate_vector_stream(bits_full["bits224_va"], len_va, C, S, mapping)
                tp_unif224, _ = aggregate_vector_stream(bits_full["bits224_te"], len_test, C, S, mapping)
                vp_edge, _ = aggregate_vector_stream(bits_full["fast_edge_va"], len_va, C, S, mapping)
                tp_edge, _ = aggregate_vector_stream(bits_full["fast_edge_te"], len_test, C, S, mapping)
                vunif, tunif = vp_unif224[vm], tp_unif224[tm]
                vedge, tedge = vp_edge[vm], tp_edge[tm]
                k_unif_bal = max(float(np.percentile(vunif[:half].sum(axis=1), 99.5)) * 1.12, 5.0)
                k_edge = max(float(np.percentile(vedge[:half].sum(axis=1), 99.90)) * 1.25, 2.5)
                hold_unif = np.maximum(
                    np.maximum((vb128[half + S:].sum(axis=1) > k_128_bal).astype(int), (vunif[half + S:].sum(axis=1) > k_unif_bal).astype(int)),
                    (vedge[half + S:].sum(axis=1) > k_edge).astype(int),
                )
                test_unif = np.maximum(
                    np.maximum((tb128.sum(axis=1) > k_128_bal).astype(int), (tunif.sum(axis=1) > k_unif_bal).astype(int)),
                    (tedge.sum(axis=1) > k_edge).astype(int),
                )
                pr_unif = float(average_precision_score(tl, tunif.sum(axis=1) + tb128.sum(axis=1) + 2.0 * tedge.sum(axis=1)))

                # 5. w/o Per-Dial Hijack Guards (gamma_j = 1)
                bits_ng = {
                    "bits128_va": bits128_va_ng,
                    "bits128_te": bits128_te_ng,
                    "bits224_va": bits224_va_ng,
                    "bits224_te": bits224_te_ng,
                    "fast_edge_va": fast_edge_va_ng,
                    "fast_edge_te": fast_edge_te_ng,
                }
                dec_ng = panel.evaluate_stream_decisions(bits_ng, len_va, len_test, mapping)
                pr_ng = float(average_precision_score(tl, dec_ng.test_switch_counts))

                # 6. 1D Scalar Collapse Baseline
                vp_base, _ = aggregate_window_scores(base_va, len_va, C, S, step=1, reducer="mean", mapping_method=mapping)
                tp_base, _ = aggregate_window_scores(base_te, len_test, C, S, step=1, reducer="mean", mapping_method=mapping)
                v_base, test_base = vp_base[vm], tp_base[tm]
                th_base = fit_spot(v_base[:half])
                hold_base = (v_base[half + S:] > th_base).astype(int)
                pred_base = (test_base > th_base).astype(int)
                pr_base = float(average_precision_score(tl, test_base))

                variants = [
                    ("1_224Bit_Breaker_Full", dec.hold_pred_balanced, dec.test_pred_balanced, pr_224),
                    ("2_224Bit_Strict", dec.hold_pred_strict, dec.test_pred_strict, pr_224),
                    ("3_wo_Onset_Relays", hold_no_relay, test_no_relay, pr_no_relay),
                    ("4_wo_Prefix_Routing", hold_unif, test_unif, pr_unif),
                    ("5_wo_Hijack_Guards", dec_ng.hold_pred_balanced, dec_ng.test_pred_balanced, pr_ng),
                    ("6_128Bit_SubBlock_Only", hold_128_only, test_128_only, pr_128_only),
                    ("7_1D_Scalar_Collapse", hold_base, pred_base, pr_base),
                ]
                for vname, hp, tp, prv in variants:
                    mt = compute_metrics(tl, tp, scores=tp.astype(float), valid_mask=np.ones_like(tl, bool), use_pa=False)
                    er, dl = event_stats(tl, tp)
                    rows.append(dict(
                        stream=f"{ds}:{ch}", seed=seed, mapping=mapping, variant=vname,
                        f1=float(mt["f1"]), pr_auc=prv,
                        er=float(er) if er == er else np.nan,
                        delay=float(dl) if dl == dl else np.nan,
                        hold_fpr=float(hp.mean()),
                        test_fpr=float(mt["fp"] / max((tl == 0).sum(), 1)),
                    ))

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    ps = df.groupby(["stream", "mapping", "variant"]).mean(numeric_only=True).reset_index()
    for mapping in ("trailing", "smear"):
        sub = ps[ps["mapping"] == mapping]
        print(f"\n=== 19-STREAM 224-BIT COMPONENT ABLATION ({mapping.upper()}) ===")
        print(sub.groupby("variant")[["f1", "pr_auc", "er", "delay", "hold_fpr", "test_fpr"]].mean().round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports" / "ablation_224bit_components_19stream.csv"))
    run_ablation(ap.parse_args().out)

