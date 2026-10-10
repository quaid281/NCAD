#!/usr/bin/env python3
"""Per-Channel Deviation Reference Detector for Fault-Type Evaluation.

Companion to run_fault_type_gate_evaluation.py. TS-JEPA under every gating mode
missed nearly all single-channel injected faults; this script supplies the
trivial reference detector showing that the same injected faults ARE detectable
in principle by direct observation-space channel monitoring.

Detector: for each channel k, score_t = |x_{t,k} - median(train_k)| / MAD(train_k)
aggregated by max over channels. Threshold calibrated on the same held-out
nominal validation split at the same empirical quantile used by SPOT (p99.9).

Output: reports/fault_type_reference_detector.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_fault_type_gate_evaluation import inject_faults, event_recall, EVENT_LEN


def channel_deviation_scores(arr: np.ndarray, med: np.ndarray, mad: np.ndarray) -> np.ndarray:
    z = np.abs(arr - med) / np.maximum(mad, 1e-3)
    return z.max(axis=1)


def rolling_std(arr: np.ndarray, win: int = 32) -> np.ndarray:
    """Per-channel rolling standard deviation, same length as arr (trailing)."""
    csum = np.cumsum(arr, axis=0)
    csum2 = np.cumsum(arr * arr, axis=0)
    m = np.zeros_like(arr)
    m[win:] = csum[win:] - csum[:-win]
    m[:win] = csum[:win]
    cnt = np.minimum(np.arange(1, len(arr) + 1), win)[:, None]
    mean = m / cnt
    s2 = np.zeros_like(arr)
    s2[win:] = csum2[win:] - csum2[:-win]
    s2[:win] = csum2[:win]
    var = np.maximum(s2 / cnt - mean ** 2, 0.0)
    return np.sqrt(var)


def channel_lowvar_scores(arr: np.ndarray, rstd_med: np.ndarray, rstd_mad: np.ndarray) -> np.ndarray:
    """Variance-collapse score: max over channels of how far the rolling std
    sits BELOW its nominal level (flatline/stuck/dropout signature)."""
    rs = rolling_std(arr)
    z = (rstd_med - rs) / np.maximum(rstd_mad, 1e-3)
    return z.max(axis=1)


def evaluate(dataset: str, channel: str) -> List[dict]:
    ds_dir = ROOT / "mTSBench_data" / dataset
    if channel == "default":
        train_p = ds_dir / f"{dataset}_train.csv"
        if not train_p.exists():
            train_p = ds_dir / f"{dataset}.csv"
        test_p = ds_dir / f"{dataset}_test.csv"
    else:
        train_p = ds_dir / f"{dataset}_{channel}_train.csv"
        if not train_p.exists():
            train_p = ds_dir / f"{channel}_train.csv"
        test_p = ds_dir / f"{dataset}_{channel}_test.csv"
        if not test_p.exists():
            test_p = ds_dir / f"{channel}_test.csv"

    train_df = pd.read_csv(train_p)
    test_df = pd.read_csv(test_p)
    numeric_cols = [c for c in train_df.columns if c not in ["timestamp", "is_anomaly", "label"]]
    lbl_col = [c for c in test_df.columns if c in ["is_anomaly", "label", "labels"]]
    base_labels = test_df[lbl_col[0]].to_numpy().astype(int) if lbl_col else np.zeros(len(test_df), dtype=int)

    train_vals = train_df[numeric_cols].to_numpy(np.float32)
    test_vals = test_df[numeric_cols].to_numpy(np.float32)
    mean = train_vals.mean(axis=0, keepdims=True)
    std = np.maximum(train_vals.std(axis=0, keepdims=True), 1e-2)
    train_scaled = (train_vals - mean) / std
    test_scaled = (test_vals - mean) / std

    med = np.median(train_scaled, axis=0)
    mad = np.median(np.abs(train_scaled - med), axis=0)

    rng = np.random.default_rng(42)
    fault_pack = inject_faults(test_scaled, base_labels, rng)
    if not fault_pack:
        return []

    # Calibrate on last 20% of nominal training data (held-out validation)
    n_cal = int(len(train_scaled) * 0.2)
    cal_arr = train_scaled[-n_cal:]
    thresh = {
        "channel_dev_maxz": float(np.quantile(channel_deviation_scores(cal_arr, med, mad), 0.999)),
        "channel_lowvar": float(np.quantile(
            channel_lowvar_scores(
                cal_arr,
                np.median(rolling_std(train_scaled), axis=0),
                np.median(np.abs(rolling_std(train_scaled)
                                 - np.median(rolling_std(train_scaled), axis=0)), axis=0),
            ), 0.999)),
    }

    rstd_med = np.median(rolling_std(train_scaled), axis=0)
    rstd_mad = np.median(np.abs(rolling_std(train_scaled) - rstd_med), axis=0)

    detectors = {
        "channel_dev_maxz": lambda a: channel_deviation_scores(a, med, mad),
        "channel_lowvar": lambda a: channel_lowvar_scores(a, rstd_med, rstd_mad),
    }

    rows = []
    for det_name, det_fn in detectors.items():
        for fault, (arr, labels, segs, ch) in fault_pack.items():
            sc = det_fn(arr)
            preds = (sc > thresh[det_name]).astype(int)
            nominal_mask = labels == 0
            fpr = float(((preds > 0) & nominal_mask).sum() / max(nominal_mask.sum(), 1))
            fault_mask = np.zeros(len(labels), dtype=bool)
            for s, e in segs:
                fault_mask[s:e] = True
            pt_rec = float(((preds > 0) & fault_mask).sum() / max(fault_mask.sum(), 1))
            ev_rec = event_recall(labels, preds, segs)
            rows.append({
                "dataset": dataset, "channel": channel, "detector": det_name,
                "fault_type": fault, "faulted_channel": ch, "n_events": len(segs),
                "point_recall": pt_rec, "event_recall": ev_rec,
                "nominal_fpr": fpr, "threshold": thresh[det_name],
            })
    return rows


def main() -> None:
    streams = [
        ("SMD", "machine-1-2"),
        ("MSL", "M-1"),
        ("SMAP", "P-3"),
        ("swan", "sf"),
        ("GECCO", "water_quality"),
        ("metro", "traffic-volume"),
    ]
    out_csv = ROOT / "reports" / "fault_type_reference_detector.csv"
    rows: List[dict] = []
    for ds, ch in streams:
        try:
            rows.extend(evaluate(ds, ch))
            print(f"{ds}:{ch} done")
        except Exception as e:
            print(f"FAILED {ds}:{ch}: {e}")
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("\n=== Reference detector summary ===")
    print(df.groupby(["detector", "fault_type"])[["point_recall", "event_recall", "nominal_fpr"]].mean().to_string())


if __name__ == "__main__":
    main()
