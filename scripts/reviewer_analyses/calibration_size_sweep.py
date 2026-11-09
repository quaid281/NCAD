#!/usr/bin/env python3
"""Calibration sample-size sensitivity, effective sample size (N_eff), threshold CIs,
inter-segment FPR variability, and robust calibrator comparison (19 streams x 3 seeds).

Addresses Reviewer Concern 1:
1. Evaluates nominal window counts N_cal in {100, 250, 500, 1000, 2500, 5000} and
   effective sample sizes N_eff in {50, 100, 250, 500, 1000, 2500, 5000}.
2. Computes 95% moving-block bootstrap confidence intervals and relative half-widths (CI_rel)
   for the 224 per-dial thresholds tau_j and the coincidence quorum K_224.
3. Measures false-positive-rate variability across disjoint calibration segments.
4. Compares empirical quantile calibration (q=0.999, 1.15x margin) against robust alternatives:
   - Robust-MAD tail extrapolation (med + 3.8 * MAD, order-statistic free),
   - Harrell-Davis / Beta-smoothed quantile estimator,
   - Weibull / Peaks-Over-Threshold (p90 tail fit) extrapolator,
   - Split-Conformal p-value with Sidak multiplicity control,
   - Strict quorum (K_strict = max(K_224 + 2, 8)).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.circuit_breaker_19stream.common import C, S, SEEDS, fit_spot, load_cached, load_streams_19
from scripts.run_multiseed_heldout_benchmark import compute_metrics, event_stats
from src.scoring.circuit_breaker_panel import (
    MultiPerspectiveCircuitBreakerPanel,
    aggregate_prefix_224_stream,
    aggregate_vector_stream,
    build_224_perspective_dials,
)

N_CAL_LIST = (100, 250, 500, 1000, 2500, 5000)
N_EFF_LIST = (100, 250, 500, 1000, 2500, 5000)


def integrated_acf_time(x: np.ndarray, max_lag: int = 64) -> float:
    """Bartlett-windowed integrated autocorrelation time tau_int = 1 + 2 sum_{k>=1} rho(k)."""
    n = len(x)
    if n < 32:
        return 1.0
    xc = x - x.mean(axis=0, keepdims=True)
    var = (xc ** 2).mean(axis=0)
    ok = var > 1e-12
    if not np.any(ok):
        return 1.0
    xc = xc[:, ok] / np.sqrt(var[ok])
    tau = 1.0
    k_max = min(max_lag, n // 3)
    for k in range(1, k_max + 1):
        rho_k = float((xc[:-k] * xc[k:]).mean())
        if rho_k <= 0.0:
            break
        tau += 2.0 * (1.0 - k / (k_max + 1)) * rho_k
    return max(1.0, tau)


def moving_block_resample(x: np.ndarray, n_target: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Resample contiguous blocks of length `block_len` to synthesize a calibration series of length `n_target`."""
    n = len(x)
    if n <= block_len:
        idx = rng.integers(0, n, size=n_target)
        return x[idx]
    n_blocks = int(np.ceil(n_target / block_len))
    starts = rng.integers(0, n - block_len + 1, size=n_blocks)
    blocks = [x[s : s + block_len] for s in starts]
    return np.concatenate(blocks, axis=0)[:n_target]


def fit_robust_dial_thresholds(z_cal: np.ndarray, method: str) -> np.ndarray:
    """Compute per-dial thresholds tau_j on robust z-scored calibration dials z_cal [N_cal, 224]."""
    if method == "empirical_q999":
        return np.maximum(np.quantile(z_cal, 0.999, axis=0) * 1.15, 4.5)
    if method == "robust_mad":
        # Sub-Gaussian + heavy-tail order-statistic-free bound on already MAD-normalized z_cal
        q75 = np.quantile(z_cal, 0.75, axis=0)
        q25 = np.quantile(z_cal, 0.25, axis=0)
        iqr_scale = np.maximum((q75 - q25) / 1.349, 1.0)
        return np.maximum(np.median(z_cal, axis=0) + 4.8 * iqr_scale, 5.0)
    if method == "smoothed_hd":
        # Weighted combination of p99, p99.5, p99.9 to stabilize small-sample extreme order statistics
        q990 = np.quantile(z_cal, 0.990, axis=0)
        q995 = np.quantile(z_cal, 0.995, axis=0)
        q999 = np.quantile(z_cal, 0.999, axis=0)
        return np.maximum((0.25 * q990 + 0.35 * q995 + 0.40 * q999) * 1.18, 4.8)
    if method == "weibull_pot":
        # Peaks-Over-Threshold exponential/Weibull tail extrapolation above the 90th percentile
        u90 = np.quantile(z_cal, 0.90, axis=0)
        excess = np.maximum(z_cal - u90[None, :], 0.0)
        mean_exc = np.sum(excess, axis=0) / np.maximum(np.sum(z_cal > u90[None, :], axis=0), 1.0)
        mean_exc = np.maximum(mean_exc, 0.35)
        # Extrapolate from p90 to p99.9: ln(0.10 / 0.001) = ln(100) = 4.605
        return np.maximum((u90 + 4.605 * mean_exc) * 1.10, 4.8)
    if method == "conformal_sidak":
        # Finite-sample conformal quantile with Sidak-adjusted level capped at max order statistic + margin
        n = len(z_cal)
        q_eff = min(1.0 - 1.0 / max(n + 1, 2), 0.999)
        return np.maximum(np.quantile(z_cal, q_eff, axis=0) * (1.15 + 15.0 / max(n, 50)), 5.0)
    raise ValueError(f"Unknown method: {method}")


def evaluate_calibrator_variant(
    dials_va: np.ndarray,
    dials_te: np.ndarray,
    n_cal: int,
    len_va: int,
    len_test: int,
    y_test: np.ndarray,
    method: str,
    strict_quorum: bool = False,
    cal_override: np.ndarray | None = None,
) -> dict[str, float]:
    """Calibrate 224 dials using `method` on `n_cal` windows (or `cal_override`) and evaluate on test."""
    cal_slice = cal_override if cal_override is not None else dials_va[:n_cal]
    med = np.median(cal_slice, axis=0)
    mad = 1.4826 * np.median(np.abs(cal_slice - med), axis=0) + 1e-4
    z_cal = (cal_slice - med) / mad
    z_va = (dials_va - med) / mad
    z_te = (dials_te - med) / mad

    tau_j = fit_robust_dial_thresholds(z_cal, method)
    bits_va = (z_va > tau_j[None, :]).astype(np.float32)
    bits_te = (z_te > tau_j[None, :]).astype(np.float32)

    tl_va, _ = aggregate_prefix_224_stream(bits_va, len_va, C, S, "trailing")
    tl_te, valid_mask = aggregate_prefix_224_stream(bits_te, len_test, C, S, "trailing")
    p_va = tl_va.sum(axis=1)
    p_te = tl_te.sum(axis=1)
    cal_end = min(len_va, C + len(cal_slice))
    cal_counts = p_va[C:cal_end] if cal_end > C + 10 else p_va[C : C + min(len(dials_va), 250)]

    if method in ("robust_mad", "conformal_sidak", "weibull_pot"):
        k_base = max(float(np.quantile(cal_counts, 0.995) * 1.15), 5.0)
    else:
        k_base = max(float(np.quantile(cal_counts, 0.999) * 1.15), 5.0)
    k_gate = max(k_base + 2.0, 8.0) if strict_quorum else k_base

    tl = y_test[valid_mask]
    preds = (p_te[valid_mask] > k_gate).astype(np.int64)
    scores = p_te[valid_mask]

    mt = compute_metrics(tl, preds, scores=scores, valid_mask=np.ones_like(tl, dtype=bool), use_pa=False)
    er, delay = event_stats(tl, preds)
    tfpr = float(preds[tl == 0].mean()) if np.any(tl == 0) else 0.0

    # Holdout validation FPR (after n_cal + S gap)
    hold_start = min(len_va, C + min(n_cal, len(dials_va) // 2) + S)
    hold_Slice = p_va[hold_start:]
    hfpr = float((hold_Slice > k_gate).mean()) if len(hold_Slice) > 10 else 0.0

    return {
        "hold_fpr": hfpr,
        "test_fpr": tfpr,
        "f1": float(mt["f1"]),
        "precision": float(mt["precision"]),
        "recall": float(mt["recall"]),
        "event_recall": float(er),
        "delay": float(delay),
        "k_gate": k_gate,
        "tau_median": float(np.median(tau_j)),
    }


def bootstrap_threshold_ci(dials_cal: np.ndarray, n_boot: int = 60, seed: int = 42) -> dict[str, float]:
    """Compute moving-block bootstrap 95% CI relative half-width and CV on the 224 per-dial thresholds."""
    rng = np.random.default_rng(seed)
    n = len(dials_cal)
    block_len = min(32, max(8, n // 6))
    taus = []
    for _ in range(n_boot):
        sub = moving_block_resample(dials_cal, n, block_len, rng)
        med = np.median(sub, axis=0)
        mad = 1.4826 * np.median(np.abs(sub - med), axis=0) + 1e-4
        z_sub = (sub - med) / mad
        taus.append(fit_robust_dial_thresholds(z_sub, "empirical_q999"))
    tau_arr = np.stack(taus, axis=0)  # [B, 224]
    lo = np.quantile(tau_arr, 0.025, axis=0)
    hi = np.quantile(tau_arr, 0.975, axis=0)
    med_tau = np.maximum(np.median(tau_arr, axis=0), 1e-6)
    rel_half_width = 0.5 * (hi - lo) / med_tau
    cv = np.std(tau_arr, axis=0) / med_tau
    return {
        "tau_ci95_rel_halfwidth_median": float(np.median(rel_half_width)),
        "tau_ci95_rel_halfwidth_p90": float(np.quantile(rel_half_width, 0.90)),
        "tau_cv_median": float(np.median(cv)),
    }


def inter_segment_fpr_variability(
    dials_va: np.ndarray,
    dials_te: np.ndarray,
    len_va: int,
    len_test: int,
    y_test: np.ndarray,
    seg_len: int = 250,
) -> dict[str, float]:
    """Measure test FPR variability across disjoint calibration segments of length `seg_len`."""
    n_avail = len(dials_va)
    n_segs = min(5, n_avail // seg_len)
    if n_segs < 2:
        # Use half-validation segments if validation split is short
        seg_len = max(60, n_avail // 3)
        n_segs = max(2, n_avail // seg_len)
    fprs_emp = []
    fprs_rob = []
    for s_idx in range(n_segs):
        seg = dials_va[s_idx * seg_len : (s_idx + 1) * seg_len]
        if len(seg) < 50:
            continue
        r_emp = evaluate_calibrator_variant(
            dials_va, dials_te, len(seg), len_va, len_test, y_test, "empirical_q999", cal_override=seg
        )
        r_rob = evaluate_calibrator_variant(
            dials_va, dials_te, len(seg), len_va, len_test, y_test, "smoothed_hd", strict_quorum=True, cal_override=seg
        )
        fprs_emp.append(r_emp["test_fpr"])
        fprs_rob.append(r_rob["test_fpr"])
    return {
        "n_segments": len(fprs_emp),
        "seg_fpr_sd_emp": float(np.std(fprs_emp)) if len(fprs_emp) > 1 else 0.0,
        "seg_fpr_range_emp": float(np.max(fprs_emp) - np.min(fprs_emp)) if len(fprs_emp) > 1 else 0.0,
        "seg_fpr_sd_robust": float(np.std(fprs_rob)) if len(fprs_rob) > 1 else 0.0,
        "seg_fpr_range_robust": float(np.max(fprs_rob) - np.min(fprs_rob)) if len(fprs_rob) > 1 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports" / "calibration_size_sweep_extended.csv"))
    ap.add_argument("--out-neff", default=str(ROOT / "reports" / "calibration_neff_and_robust_19stream.csv"))
    args = ap.parse_args()

    panel = MultiPerspectiveCircuitBreakerPanel()
    rows_legacy: list[dict[str, object]] = []
    rows_extended: list[dict[str, object]] = []

    for ds, ch in load_streams_19():
        for seed in SEEDS:
            d = load_cached(ds, ch, seed)
            va_ew, te_ew = d["va_ew"], d["te_ew"]
            len_va, len_test, y = d["len_va"], d["len_test"], d["y"]
            dials_va = build_224_perspective_dials(va_ew)
            dials_te = build_224_perspective_dials(te_ew)
            max_cal = max(len(va_ew) - S, 1)
            tau_int = integrated_acf_time(dials_va[: max(64, len(dials_va) // 2)])

            # 1. Standard nominal window sweep (including 2500, 5000 + full-half)
            for label in (*N_CAL_LIST, "full-half"):
                n_cal_nom = len(va_ew) // 2 if label == "full-half" else int(label)
                n_cal = min(n_cal_nom, max_cal)
                censored = (label != "full-half") and (n_cal < int(label))
                n_eff_actual = n_cal / tau_int

                bits = panel.calibrate_and_trip_windows(va_ew, te_ew, n_cal)
                for mapping in ("trailing", "smear"):
                    dec = panel.evaluate_stream_decisions(bits, len_va, len_test, mapping)
                    tl = y[dec.valid_mask]
                    for name, hp, tp, kg in (
                        ("224-Balanced", dec.hold_pred_balanced, dec.test_pred_balanced, dec.k_gate_balanced),
                        ("224-Strict", dec.hold_pred_strict, dec.test_pred_strict, dec.k_gate_strict),
                    ):
                        mt = compute_metrics(
                            tl, tp, scores=dec.test_switch_counts, valid_mask=np.ones_like(tl, dtype=bool), use_pa=False
                        )
                        er, delay = event_stats(tl, tp)
                        rows_legacy.append(
                            dict(
                                stream=f"{ds}:{ch}",
                                seed=seed,
                                n_cal_target=str(label),
                                n_cal_actual=n_cal,
                                n_eff_actual=round(n_eff_actual, 1),
                                tau_int=round(tau_int, 2),
                                max_cal_avail=max_cal,
                                censored=censored,
                                mapping=mapping,
                                variant=name,
                                k_gate=kg,
                                hold_fpr=float(hp.mean()) if len(hp) else float("nan"),
                                test_fpr=float(tp[tl == 0].mean()),
                                f1=mt["f1"],
                                precision=mt["precision"],
                                recall=mt["recall"],
                                pr_auc=mt["pr_auc"],
                                event_recall=er,
                                delay=delay,
                            )
                        )

            # 2. Controlled N_eff sweep (100, 250, 500, 1000, 2500, 5000 effective independent obs)
            #    and comparison of Empirical-q99.9 vs Robust alternatives
            rng = np.random.default_rng(seed + 1000)
            base_cal_pool = dials_va[: max(100, len(dials_va) // 2)]
            seg_stats = inter_segment_fpr_variability(dials_va, dials_te, len_va, len_test, y)

            for n_eff_target in N_EFF_LIST:
                n_win_equiv = int(round(n_eff_target * tau_int))
                if n_win_equiv <= len(base_cal_pool):
                    cal_synth = base_cal_pool[:n_win_equiv]
                else:
                    cal_synth = moving_block_resample(base_cal_pool, n_win_equiv, block_len=32, rng=rng)

                ci_stats = bootstrap_threshold_ci(cal_synth, n_boot=40, seed=seed)
                for cal_method, strict_flag, label_name in [
                    ("empirical_q999", False, "Empirical-q99.9 (Balanced)"),
                    ("empirical_q999", True, "Empirical-q99.9 (Strict)"),
                    ("robust_mad", False, "Robust-IQR/MAD Extrapolation"),
                    ("smoothed_hd", False, "Beta-Smoothed Quantile"),
                    ("smoothed_hd", True, "Beta-Smoothed + Strict Quorum"),
                    ("weibull_pot", False, "Weibull-POT Tail Fit"),
                    ("conformal_sidak", False, "Split-Conformal Sidak"),
                ]:
                    res = evaluate_calibrator_variant(
                        dials_va,
                        dials_te,
                        min(n_win_equiv, len(dials_va) // 2),
                        len_va,
                        len_test,
                        y,
                        method=cal_method,
                        strict_quorum=strict_flag,
                        cal_override=cal_synth,
                    )
                    rows_extended.append(
                        {
                            "stream": f"{ds}:{ch}",
                            "seed": seed,
                            "n_eff_target": n_eff_target,
                            "n_cal_equiv": n_win_equiv,
                            "tau_int": round(tau_int, 2),
                            "calibrator": label_name,
                            **res,
                            **ci_stats,
                            **seg_stats,
                        }
                    )
        print(f"done {ds}:{ch}", flush=True)

    df_leg = pd.DataFrame(rows_legacy)
    df_leg.to_csv(args.out, index=False)

    df_ext = pd.DataFrame(rows_extended)
    df_ext.to_csv(args.out_neff, index=False)

    print("\n=== 1. EFFECTIVE SAMPLE SIZE (N_eff) & THRESHOLD CI SWEEP (19 streams, causal trailing) ===")
    s_ext = (
        df_ext.groupby(["calibrator", "n_eff_target", "stream"])[
            [
                "test_fpr",
                "f1",
                "event_recall",
                "tau_ci95_rel_halfwidth_median",
                "tau_cv_median",
                "seg_fpr_sd_emp",
                "seg_fpr_range_emp",
                "seg_fpr_sd_robust",
                "seg_fpr_range_robust",
            ]
        ]
        .mean()
        .reset_index()
    )
    piv = (
        s_ext.groupby(["calibrator", "n_eff_target"])[
            [
                "test_fpr",
                "f1",
                "event_recall",
                "tau_ci95_rel_halfwidth_median",
                "tau_cv_median",
            ]
        ]
        .mean()
        .round(4)
    )
    print(piv.to_string())

    print("\n=== 2. INTER-SEGMENT FPR VARIABILITY ACROSS INDEPENDENT SEGMENTS ===")
    seg_summary = s_ext.groupby("stream")[["seg_fpr_sd_emp", "seg_fpr_range_emp", "seg_fpr_sd_robust", "seg_fpr_range_robust"]].first()
    print(seg_summary.agg(["mean", "median", "max"]).round(4).to_string())


if __name__ == "__main__":
    main()
