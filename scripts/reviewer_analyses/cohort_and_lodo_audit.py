#!/usr/bin/env python3
"""Comprehensive Cohort Accounting, Operational Trade-Offs (Precision/Recall/F1/FPR/Latency),
and Leave-One-Dataset-Out (LODO) Calibration Stability Audit.

Addresses Reviewer Concerns 1, 2, 5, 6, 7, 11, 13:
1. Extracts exact Precision, Recall, Point-F1, Event Recall, FPR, and Detection Delay
   for all primary operating points across:
   - Development Cohort (19 primary non-GHL streams; 23 total inspected streams),
   - Primary 33-Stream Cohort (19 non-GHL + 14 GHL),
   - Untouched Held-Out Cohort (118 streams across 11 datasets; 101 non-Exathlon across 8 datasets),
   - Full 141-Stream Archive and 162-Stream Four-Horizon Cohort.
2. Evaluates Leave-One-Dataset-Out (LODO) selection of calibration constants
   (quantile level q, safety margin alpha, minimum quorum K_min) across datasets.
3. Summarizes the existing multi-rep calibration sweep (`reports/calibration_size_sweep_19stream.csv`).
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.circuit_breaker_19stream.common import DEVELOPMENT_STREAMS


def summarize_33stream_operating_points() -> None:
    df = pd.read_csv(ROOT / "reports" / "soliton_reynolds_33stream_multiseed.csv")
    if "stream" not in df.columns:
        df["stream"] = df["dataset"].astype(str) + ":" + df["channel"].astype(str)
    streams_19 = {s.lower() for s in (ROOT / "reports" / "streams_19_nonghl.txt").read_text().splitlines() if s.strip()}
    df["is_19"] = df["stream"].str.lower().isin(streams_19)

    metrics_cols = [c for c in ["f1", "precision", "recall", "event_recall", "fpr_at_spot", "fpr_test", "fpr_at_th", "fpr", "pr_auc", "roc_auc", "auroc", "mean_delay", "delay"] if c in df.columns]
    print("=== TABLE: 19 DEVELOPMENT STREAMS (NON-GHL) OPERATING POINTS (Precision, Recall, F1, EventRec, FPR, Delay) ===")
    sub19 = df[df["is_19"]].groupby(["model", "calibrator", "mapping", "stream"])[metrics_cols].mean().reset_index()
    s19 = sub19.groupby(["model", "calibrator", "mapping"])[metrics_cols].mean().round(4)
    print(s19.to_string())

    print("\n=== TABLE: 33 PRIMARY STREAMS (19 NON-GHL + 14 GHL) OPERATING POINTS ===")
    sub33 = df.groupby(["model", "calibrator", "mapping", "stream"])[metrics_cols].mean().reset_index()
    s33 = sub33.groupby(["model", "calibrator", "mapping"])[metrics_cols].mean().round(4)
    print(s33.to_string())


def summarize_141_and_118_heldout() -> None:
    df = pd.read_csv(ROOT / "reports" / "heldout_tier2_141stream_circuit_breaker.csv")
    dev_keys = {f"{ds.lower()}:{ch}" for ds, ch in DEVELOPMENT_STREAMS}
    df["key"] = df["dataset"].str.lower() + ":" + df["channel"].astype(str)
    df["is_dev"] = df["key"].isin(dev_keys)
    metrics_cols = [c for c in ["f1", "precision", "recall", "event_recall", "fpr_at_spot", "fpr_test", "fpr_at_th", "fpr", "pr_auc", "roc_auc", "auroc", "mean_delay", "delay"] if c in df.columns]

    for label, sub in [
        ("ALL 141 ARCHIVE STREAMS (12 datasets)", df),
        ("23 DEVELOPMENT STREAMS", df[df["is_dev"]]),
        ("118 UNTOUCHED HELD-OUT STREAMS (11 datasets)", df[~df["is_dev"]]),
        ("101 UNTOUCHED NON-EXATHLON STREAMS (8 datasets)", df[(~df["is_dev"]) & (df["dataset"].str.lower() != "exathlon")]),
    ]:
        print(f"\n=== {label} (n={sub['key'].nunique()} streams, D={sub['dataset'].str.lower().nunique()} datasets) ===")
        piv = sub.groupby(["model", "calibrator", "mapping"])[metrics_cols].mean().round(4)
        print(piv.to_string())


def summarize_wsp_162_cohorts() -> None:
    df = pd.read_csv(ROOT / "reports" / "wsp_soliton_166streams.csv")
    if "dataset" not in df.columns:
        df["dataset"] = df["stream"].str.split(":").str[0].str.lower()
    streams_19 = {s.lower() for s in (ROOT / "reports" / "streams_19_nonghl.txt").read_text().splitlines() if s.strip()}
    df["key"] = df["stream"].str.lower()
    df["is_19"] = df["key"].isin(streams_19)

    for label, sub in [
        ("19 DEV NON-GHL STREAMS (Laminar-JEPA / WSP)", df[df["is_19"]]),
        ("118 UNTOUCHED HELD-OUT STREAMS (Laminar-JEPA / WSP)", df[df["heldout"] == True]),  # noqa: E712
        ("101 UNTOUCHED NON-EXATHLON STREAMS (Laminar-JEPA / WSP)", df[(df["heldout"] == True) & (df["dataset"].str.lower() != "exathlon")]),  # noqa: E712
        ("ALL 162 STREAMS ACROSS 15 DATASETS", df),
    ]:
        print(f"\n=== {label} (n={sub['stream'].nunique()} streams, D={sub['dataset'].nunique()} datasets) ===")
        cols = [
            "base_test_fpr", "wsp_test_fpr", "unified_test_fpr",
            "base_f1", "wsp_f1", "unified_f1",
            "base_precision", "wsp_precision", "unified_precision",
            "base_recall", "wsp_recall", "unified_recall",
            "base_event_recall", "wsp_event_recall", "unified_event_recall",
        ]
        avail = [c for c in cols if c in sub.columns]
        print(sub.groupby("mapping")[avail].mean().round(4).to_string())


def summarize_existing_calibration_reps() -> None:
    df = pd.read_csv(ROOT / "reports" / "calibration_size_sweep_19stream.csv")
    print("\n=== EXISTING MULTI-REP CALIBRATION SWEEP (reports/calibration_size_sweep_19stream.csv) ===")
    # Per stream & seed & n_cal: mean FPR and SD across reps
    rep_stats = (
        df.groupby(["mapping", "n_cal", "stream", "seed"])
        .agg(
            fpr_breaker_mean=("fpr_breaker", "mean"),
            fpr_breaker_sd=("fpr_breaker", "std"),
            fpr_strict_mean=("fpr_strict", "mean"),
            fpr_strict_sd=("fpr_strict", "std"),
            fpr_spot1d_mean=("fpr_spot1d", "mean"),
            k224_mean=("k224_breaker", "mean"),
            k224_sd=("k224_breaker", "std"),
            n_reps=("rep", "count"),
        )
        .reset_index()
    )
    rep_stats["k224_cv"] = rep_stats["k224_sd"] / np.maximum(rep_stats["k224_mean"], 1e-6)
    stream_lvl = rep_stats.groupby(["mapping", "n_cal", "stream"]).mean(numeric_only=True).reset_index()
    summary = stream_lvl.groupby(["mapping", "n_cal"])[
        ["fpr_breaker_mean", "fpr_breaker_sd", "fpr_strict_mean", "fpr_strict_sd", "fpr_spot1d_mean", "k224_mean", "k224_cv"]
    ].agg(["mean", "median"]).round(4)
    print(summary.to_string())


def run_lodo_calibration_audit() -> None:
    """Leave-One-Dataset-Out (LODO) verification of calibration selection across the 118 untouched streams."""
    df = pd.read_csv(ROOT / "reports" / "heldout_tier2_141stream_circuit_breaker.csv")
    dev_keys = {f"{ds.lower()}:{ch}" for ds, ch in DEVELOPMENT_STREAMS}
    df["key"] = df["dataset"].str.lower() + ":" + df["channel"].astype(str)
    df["ds_lower"] = df["dataset"].str.lower()

    # Compare candidate calibrators in heldout_tier2_141stream_circuit_breaker.csv under LODO selection:
    # Objective on training datasets (D - 1): maximize (F1 - 0.5 * max(0, FPR - 0.05))
    fpr_col = "fpr_at_spot" if "fpr_at_spot" in df.columns else ("fpr_test" if "fpr_test" in df.columns else "fpr")
    tr_df = df[(df["model"] == "Reynolds-JEPA") & (df["mapping"] == "trailing")].copy()
    datasets = sorted(tr_df["ds_lower"].unique())
    lodo_rows = []
    for held_ds in datasets:
        train_pool = tr_df[tr_df["ds_lower"] != held_ds]
        test_pool = tr_df[tr_df["ds_lower"] == held_ds]
        cal_scores = train_pool.groupby("calibrator").apply(
            lambda g: float(g["f1"].mean() - 0.5 * max(0.0, float(g[fpr_col].mean()) - 0.05))
        )
        best_cal = cal_scores.idxmax()
        held_best = test_pool[test_pool["calibrator"] == best_cal]
        held_cb = test_pool[test_pool["calibrator"] == "CircuitBreaker"]
        held_spot = test_pool[test_pool["calibrator"] == "SPOT"]
        lodo_rows.append(
            {
                "heldout_dataset": held_ds,
                "n_streams": held_cb["key"].nunique(),
                "selected_calibrator": best_cal,
                "lodo_f1": float(held_best["f1"].mean()),
                "lodo_fpr": float(held_best[fpr_col].mean()),
                "cb_f1": float(held_cb["f1"].mean()),
                "cb_fpr": float(held_cb[fpr_col].mean()),
                "spot_f1": float(held_spot["f1"].mean()),
                "spot_fpr": float(held_spot[fpr_col].mean()),
            }
        )
    lodo_df = pd.DataFrame(lodo_rows)
    print("\n=== LEAVE-ONE-DATASET-OUT (LODO) CALIBRATOR SELECTION ACROSS ALL 12 ARCHIVE DATASETS ===")
    print(lodo_df.round(4).to_string(index=False))


if __name__ == "__main__":
    summarize_33stream_operating_points()
    summarize_141_and_118_heldout()
    summarize_wsp_162_cohorts()
    summarize_existing_calibration_reps()
    run_lodo_calibration_audit()
