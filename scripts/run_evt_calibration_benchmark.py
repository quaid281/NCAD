#!/usr/bin/env python3
"""Run Comprehensive EVT Threshold Calibration & Kurtosis Adaptation Benchmark.

Resolves Reviewer Major Concern 3:
1. Compares thresholding calibration procedures:
   - Non-EVT: Max-of-Validation, 99.5th Empirical Percentile, Gaussian 3-Sigma (mu + 3*sigma)
   - Standard Static SPOT (Unadapted EVT: q=1e-3, u=0.98)
   - Rolling Dynamic SPOT (online sliding window buffer)
   - Kurtosis-Adaptive EVT (Internal GPD adaptation and multiplicative rule: tau * (1 + beta * tanh(gamma_2)))
2. Evaluates sensitivity to:
   - Initial tail quantile u in {0.95, 0.98, 0.99}
   - Design risk level q in {1e-2, 1e-3, 1e-4}
   - Kurtosis adaptation coefficient beta in {0.0, 0.083, 0.167, 0.25, 0.5}
   - Standard moment kurtosis vs. robust quantile kurtosis (Moors 1988)
3. Evaluates on:
   - Held-out nominal validation split (measuring empirical False Positive Rate vs. design risk q)
   - Test sequence (measuring Point-F1, Precision, Recall, False Alarms)
   Across 5 benchmark streams: SMAP:P-3, SMD:machine-1-2, MSL:M-1, Daphnet:S01R01E1, GECCO:water_quality.

Output:
- reports/evt_calibration_benchmark.csv
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CSMConfig
from src.data.data_loader import DataLoader
from src.engine.trainer import build_ts_jepa_model, split_train_validation as split_train_val
from src.scoring.event_fusion import (
    aggregate_window_scores,
    calibrate_evt_threshold,
    compute_metrics,
    event_level_filter,
    moving_average,
    positive_robust_z,
    robust_stats,
)
from src.scoring.evt_calibrator import EVTCalibrator, compute_excess_kurtosis, fit_gpd


def compute_robust_moors_kurtosis(scores: np.ndarray) -> float:
    """Moors (1988) robust quantile-based kurtosis centered around zero for normal."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    s = s[np.isfinite(s)]
    if len(s) < 20:
        return 0.0
    # Octiles E1 through E7 (12.5%, 25%, 37.5%, 50%, 62.5%, 75%, 87.5%)
    octiles = np.percentile(s, [12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5])
    e1, e2, e3, e4, e5, e6, e7 = octiles
    denom = max(e6 - e2, 1e-8)  # IQR
    # Moors formula: T = ( (E7 - E5) + (E3 - E1) ) / (E6 - E2)
    # For standard normal, T approx 1.233. Centering: T - 1.233
    t_moors = ((e7 - e5) + (e3 - e1)) / denom
    return float(np.clip(t_moors - 1.233, -2.0, 50.0))


def run_evt_benchmark_on_stream(
    dataset_name: str,
    channel_name: str,
    device: torch.device,
    epochs: int = 15,
) -> List[Dict]:
    """Train TS-JEPA on stream, extract validation & test residuals, evaluate all thresholding schemes."""
    print(f"\n=== Evaluating EVT Calibration on Stream {dataset_name}:{channel_name} ===")
    ds_dir = ROOT / "mTSBench_data" / dataset_name
    if channel_name == "default":
        train_path = ds_dir / f"{dataset_name}_train.csv"
        if not train_path.exists():
            train_path = ds_dir / f"{dataset_name}.csv"
        test_path = ds_dir / f"{dataset_name}_test.csv"
    else:
        train_path = ds_dir / f"{dataset_name}_{channel_name}_train.csv"
        if not train_path.exists():
            train_path = ds_dir / f"{channel_name}_train.csv"
        test_path = ds_dir / f"{dataset_name}_{channel_name}_test.csv"
        if not test_path.exists():
            test_path = ds_dir / f"{channel_name}_test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    numeric_cols = [c for c in train_df.columns if c not in ["timestamp", "is_anomaly", "label"]]
    lbl_col = [c for c in test_df.columns if c in ["is_anomaly", "label", "labels"]]
    test_labels = test_df[lbl_col[0]].to_numpy().astype(int) if lbl_col else np.zeros(len(test_df), dtype=int)

    train_vals = train_df[numeric_cols].to_numpy(dtype=np.float32)
    test_vals = test_df[numeric_cols].to_numpy(dtype=np.float32)

    context_size = 256
    suspect_size = 64
    window_size = context_size + suspect_size

    # Split training chronologically into train (80%) and held-out nominal validation (20%)
    val_size = int(len(train_vals) * 0.20)
    train_split = train_vals[:-val_size]
    val_split = train_vals[-val_size:]

    # Standardize
    train_mean = np.mean(train_split, axis=0)
    train_std = np.std(train_split, axis=0)
    scale = np.where(train_std < 1e-8, 1.0, np.maximum(train_std, 0.01))
    train_scaled = (train_split - train_mean) / scale
    val_scaled = (val_split - train_mean) / scale
    test_scaled = (test_vals - train_mean) / scale

    # Windows
    tr_windows = DataLoader.create_windows(train_scaled, window_size, step=16, copy=False)
    val_windows = DataLoader.create_windows(val_scaled, window_size, step=1, copy=False)
    test_windows = DataLoader.create_windows(test_scaled, window_size, step=1, copy=False)

    cfg = CSMConfig(
        model_type="ts_jepa",
        context_size=context_size,
        suspect_size=suspect_size,
        latent_dim=32,
    )
    input_dim = train_vals.shape[1]
    model = build_ts_jepa_model(cfg, input_dim=input_dim, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Train TS-JEPA on 80% train split
    model.train()
    for ep in range(epochs):
        perm = np.random.permutation(len(tr_windows))
        for b in range(0, len(perm), 16):
            batch = torch.from_numpy(np.ascontiguousarray(tr_windows[perm[b:b+16]])).float().to(device)
            ctx = batch[:, :context_size]
            tgt = batch[:, context_size:]
            loss, _ = model.compute_objective(ctx, tgt, cfg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.update_target_encoder()

    # Score extraction helper
    def extract_scores(w_arr):
        model.eval()
        res = []
        with torch.no_grad():
            for i in range(0, len(w_arr), 128):
                chunk = torch.from_numpy(np.ascontiguousarray(w_arr[i:i+128])).float().to(device)
                disc = model.compute_predictive_discrepancy(chunk[:, :context_size], chunk[:, context_size:])
                res.append(disc.cpu().numpy())
        return np.concatenate(res, axis=0)

    val_win_scores = extract_scores(val_windows)
    test_win_scores = extract_scores(test_windows)

    # Map to point-level using standard dense suspect smearing
    val_pt_scores, val_mask = aggregate_window_scores(val_win_scores, len(val_split), context_size, suspect_size, step=1)
    test_pt_scores, test_mask = aggregate_window_scores(test_win_scores, len(test_vals), context_size, suspect_size, step=1)

    val_nominal_scores = val_pt_scores[val_mask]
    test_sc = test_pt_scores[test_mask]
    test_lbl = test_labels[test_mask]

    nominal_pts_val = len(val_nominal_scores)
    nominal_pts_test = np.sum(test_lbl == 0)

    # Diagnostic tail metrics on nominal validation scores
    moment_kurt = compute_excess_kurtosis(val_nominal_scores)
    moors_kurt = compute_robust_moors_kurtosis(val_nominal_scores)

    records = []

    # 1. Non-EVT Baselines
    non_evt_methods = {
        "Non-EVT: Max-of-Validation": float(np.max(val_nominal_scores)),
        "Non-EVT: 99.5th Percentile": float(np.percentile(val_nominal_scores, 99.5)),
        "Non-EVT: Gaussian 3-Sigma (mu+3sigma)": float(np.mean(val_nominal_scores) + 3.0 * np.std(val_nominal_scores)),
    }
    for m_name, thresh in non_evt_methods.items():
        val_fp = int(np.sum(val_nominal_scores > thresh))
        val_fpr = val_fp / max(nominal_pts_val, 1)

        preds = (test_sc > thresh).astype(int)
        m = compute_metrics(test_lbl, preds, scores=test_sc, valid_mask=np.ones_like(test_lbl, dtype=bool), use_pa=False)
        test_fpr = m["fp"] / max(nominal_pts_test, 1)

        records.append({
            "dataset": dataset_name,
            "channel": channel_name,
            "category": "Non-EVT Baselines",
            "method": m_name,
            "design_risk_q": np.nan,
            "init_quantile_u": np.nan,
            "beta_coef": np.nan,
            "threshold": thresh,
            "val_kurtosis_moment": moment_kurt,
            "val_kurtosis_moors": moors_kurt,
            "val_fp": val_fp,
            "val_empirical_fpr": val_fpr,
            "test_fp": m["fp"],
            "test_empirical_fpr": test_fpr,
            "test_point_f1": m.get("f1", 0.0),
            "test_precision": m.get("precision", 0.0),
            "test_recall": m.get("recall", 0.0),
        })

    # 2. Standard Static SPOT (Unadapted EVT) vs. Quantile u and Risk q Sweeps
    for u in [0.95, 0.98, 0.99]:
        for q in [1e-2, 1e-3, 1e-4]:
            calib = EVTCalibrator(risk_level=q, init_percentile=u * 100.0, adaptive_kurtosis=False)
            calib.fit(val_nominal_scores)
            thresh = calib.threshold_

            val_fp = int(np.sum(val_nominal_scores > thresh))
            val_fpr = val_fp / max(nominal_pts_val, 1)

            preds = (test_sc > thresh).astype(int)
            m = compute_metrics(test_lbl, preds, scores=test_sc, valid_mask=np.ones_like(test_lbl, dtype=bool), use_pa=False)
            test_fpr = m["fp"] / max(nominal_pts_test, 1)

            records.append({
                "dataset": dataset_name,
                "channel": channel_name,
                "category": "Standard Static SPOT (Unadapted)",
                "method": f"Static SPOT (u={u}, q={q:g})",
                "design_risk_q": q,
                "init_quantile_u": u,
                "beta_coef": 0.0,
                "threshold": thresh,
                "val_kurtosis_moment": moment_kurt,
                "val_kurtosis_moors": moors_kurt,
                "val_fp": val_fp,
                "val_empirical_fpr": val_fpr,
                "test_fp": m["fp"],
                "test_empirical_fpr": test_fpr,
                "test_point_f1": m.get("f1", 0.0),
                "test_precision": m.get("precision", 0.0),
                "test_recall": m.get("recall", 0.0),
            })

    # 3. Kurtosis-Adaptive EVT (Multiplicative beta sweep with base u=0.98, q=1e-3)
    base_u = 0.98
    base_q = 1e-3
    calib_base = EVTCalibrator(risk_level=base_q, init_percentile=base_u * 100.0, adaptive_kurtosis=False)
    calib_base.fit(val_nominal_scores)
    base_thresh = calib_base.threshold_

    for beta in [0.0, 0.083, 0.167, 0.25, 0.50]:
        # tau_eff = tau * (1 + beta * tanh(gamma_2))
        corr_factor = 1.0 + beta * math.tanh(moment_kurt)
        thresh = float(base_thresh * corr_factor)

        val_fp = int(np.sum(val_nominal_scores > thresh))
        val_fpr = val_fp / max(nominal_pts_val, 1)

        preds = (test_sc > thresh).astype(int)
        m = compute_metrics(test_lbl, preds, scores=test_sc, valid_mask=np.ones_like(test_lbl, dtype=bool), use_pa=False)
        test_fpr = m["fp"] / max(nominal_pts_test, 1)

        records.append({
            "dataset": dataset_name,
            "channel": channel_name,
            "category": "Kurtosis-Adaptive Formula: tau * (1 + beta*tanh(gamma_2))",
            "method": f"Kurtosis-Adaptive (beta={beta:g})",
            "design_risk_q": base_q,
            "init_quantile_u": base_u,
            "beta_coef": beta,
            "threshold": thresh,
            "val_kurtosis_moment": moment_kurt,
            "val_kurtosis_moors": moors_kurt,
            "val_fp": val_fp,
            "val_empirical_fpr": val_fpr,
            "test_fp": m["fp"],
            "test_empirical_fpr": test_fpr,
            "test_point_f1": m.get("f1", 0.0),
            "test_precision": m.get("precision", 0.0),
            "test_recall": m.get("recall", 0.0),
        })

    # 4. Internal GPD Kurtosis-Adaptive Params (q_decay and p_boost)
    calib_internal = EVTCalibrator(risk_level=base_q, init_percentile=base_u * 100.0, adaptive_kurtosis=True)
    calib_internal.fit(val_nominal_scores)
    thresh_int = calib_internal.threshold_

    val_fp = int(np.sum(val_nominal_scores > thresh_int))
    val_fpr = val_fp / max(nominal_pts_val, 1)

    preds = (test_sc > thresh_int).astype(int)
    m = compute_metrics(test_lbl, preds, scores=test_sc, valid_mask=np.ones_like(test_lbl, dtype=bool), use_pa=False)
    test_fpr = m["fp"] / max(nominal_pts_test, 1)

    records.append({
        "dataset": dataset_name,
        "channel": channel_name,
        "category": "Internal GPD Adaptive Params (q_decay & p_boost)",
        "method": "Internal GPD Adaptive EVT",
        "design_risk_q": base_q,
        "init_quantile_u": base_u,
        "beta_coef": np.nan,
        "threshold": thresh_int,
        "val_kurtosis_moment": moment_kurt,
        "val_kurtosis_moors": moors_kurt,
        "val_fp": val_fp,
        "val_empirical_fpr": val_fpr,
        "test_fp": m["fp"],
        "test_empirical_fpr": test_fpr,
        "test_point_f1": m.get("f1", 0.0),
        "test_precision": m.get("precision", 0.0),
        "test_recall": m.get("recall", 0.0),
    })

    # 5. Robust Moors Quantile Kurtosis Adaptation
    # tau_eff_robust = tau * (1 + 0.167 * tanh(moors_kurt))
    thresh_moors = float(base_thresh * (1.0 + 0.167 * math.tanh(moors_kurt)))
    val_fp = int(np.sum(val_nominal_scores > thresh_moors))
    val_fpr = val_fp / max(nominal_pts_val, 1)

    preds = (test_sc > thresh_moors).astype(int)
    m = compute_metrics(test_lbl, preds, scores=test_sc, valid_mask=np.ones_like(test_lbl, dtype=bool), use_pa=False)
    test_fpr = m["fp"] / max(nominal_pts_test, 1)

    records.append({
        "dataset": dataset_name,
        "channel": channel_name,
        "category": "Robust Moors Quantile Kurtosis Adaptation",
        "method": "Moors Robust Kurtosis EVT (beta=0.167)",
        "design_risk_q": base_q,
        "init_quantile_u": base_u,
        "beta_coef": 0.167,
        "threshold": thresh_moors,
        "val_kurtosis_moment": moment_kurt,
        "val_kurtosis_moors": moors_kurt,
        "val_fp": val_fp,
        "val_empirical_fpr": val_fpr,
        "test_fp": m["fp"],
        "test_empirical_fpr": test_fpr,
        "test_point_f1": m.get("f1", 0.0),
        "test_precision": m.get("precision", 0.0),
        "test_recall": m.get("recall", 0.0),
    })

    # 6. Online Rolling Dynamic SPOT
    # Emulates streaming buffer where threshold dynamically adjusts with a rolling memory of 1000 steps
    window_buf_len = 1000
    rolling_preds = np.zeros(len(test_sc), dtype=int)
    # Warm up buffer with the tail of validation scores
    roll_buf = list(val_nominal_scores[-window_buf_len:])
    roll_calib = EVTCalibrator(risk_level=base_q, init_percentile=base_u * 100.0, adaptive_kurtosis=False)
    roll_calib.fit(np.array(roll_buf))
    cur_t = roll_calib.threshold_

    for i in range(len(test_sc)):
        val_i = test_sc[i]
        if val_i > cur_t:
            rolling_preds[i] = 1
        else:
            rolling_preds[i] = 0
            # Only update buffer with nominal scores
            roll_buf.append(val_i)
            if len(roll_buf) > window_buf_len:
                roll_buf.pop(0)
            if (i + 1) % 100 == 0:
                roll_calib.fit(np.array(roll_buf))
                cur_t = roll_calib.threshold_

    m_roll = compute_metrics(test_lbl, rolling_preds, scores=test_sc, valid_mask=np.ones_like(test_lbl, dtype=bool), use_pa=False)
    records.append({
        "dataset": dataset_name,
        "channel": channel_name,
        "category": "Rolling Dynamic SPOT",
        "method": "Rolling SPOT (buffer=1000, step=100)",
        "design_risk_q": base_q,
        "init_quantile_u": base_u,
        "beta_coef": np.nan,
        "threshold": np.nan,
        "val_kurtosis_moment": moment_kurt,
        "val_kurtosis_moors": moors_kurt,
        "val_fp": np.nan,
        "val_empirical_fpr": np.nan,
        "test_fp": m_roll["fp"],
        "test_empirical_fpr": m_roll["fp"] / max(nominal_pts_test, 1),
        "test_point_f1": m_roll.get("f1", 0.0),
        "test_precision": m_roll.get("precision", 0.0),
        "test_recall": m_roll.get("recall", 0.0),
    })

    return records


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running EVT Calibration & Kurtosis Adaptation Benchmark on device: {device}")

    streams = [
        ("SMAP", "P-3"),
        ("SMD", "machine-1-2"),
        ("MSL", "M-1"),
        ("Daphnet", "S01R01E1"),
        ("GECCO", "water_quality"),
    ]

    all_records = []
    out_csv = ROOT / "reports" / "evt_calibration_benchmark.csv"

    for ds, ch in streams:
        recs = run_evt_benchmark_on_stream(ds, ch, device=device, epochs=15)
        all_records.extend(recs)
        pd.DataFrame(all_records).to_csv(out_csv, index=False)
        print(f"Saved intermediate records to {out_csv}")

    df = pd.DataFrame(all_records)
    print("\n" + "="*80)
    print("EVT CALIBRATION & KURTOSIS ADAPTATION BENCHMARK SUMMARY")
    print("="*80)
    summary = df.groupby(["category", "method"])[["val_empirical_fpr", "test_empirical_fpr", "test_point_f1", "test_precision", "test_recall"]].mean().round(4)
    print(summary.to_string())


if __name__ == "__main__":
    main()
