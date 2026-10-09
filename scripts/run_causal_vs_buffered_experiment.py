#!/usr/bin/env python3
"""Run Causal vs. Buffered Detection Experiment.

Resolves Reviewer 3.2:
1. Setting A (Strictly Causal): Decision at time T uses strictly observations up to T (zero lookahead).
2. Setting B (Buffered): Target window buffering S=64 with explicit latency and delay accounting.

Outputs:
- Point-F1, Event-F1, Event-Recall
- Mean Detection Delay (timesteps from onset)
- Event Recall at Delay Budgets (tau <= 5, 10, 64 steps)
- False Alarms per 1,000 steps (FAR_1k)
- Empirical False Positive Rate (FPR)
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
    compute_event_metrics,
    event_level_filter,
    moving_average,
    positive_robust_z,
    robust_stats,
    _extract_segments,
)
from src.models.baselines import TimesNet, TranAD


def compute_detection_delay_stats(labels: np.ndarray, predictions: np.ndarray):
    """Compute detection delay and event recall at budgets."""
    gt_segs = _extract_segments(labels)
    if len(gt_segs) == 0:
        return {
            "mean_delay": 0.0,
            "median_delay": 0.0,
            "recall_at_5": 1.0,
            "recall_at_10": 1.0,
            "recall_at_64": 1.0,
            "detected_events": 0,
            "total_events": 0,
        }

    delays = []
    r_5, r_10, r_64 = 0, 0, 0
    detected_count = 0

    for gs, ge in gt_segs:
        # Check alarms occurring from onset onwards
        alarms_in_event = np.where(predictions[gs:ge] > 0)[0]
        if len(alarms_in_event) > 0:
            first_alarm = alarms_in_event[0] # delay relative to gs
            delays.append(first_alarm)
            detected_count += 1
            if first_alarm <= 5:
                r_5 += 1
            if first_alarm <= 10:
                r_10 += 1
            if first_alarm <= 64:
                r_64 += 1
        else:
            # Check if alarm occurred slightly after ge (within 64 steps)
            after_alarms = np.where(predictions[ge:min(ge+64, len(predictions))] > 0)[0]
            if len(after_alarms) > 0:
                first_alarm = (ge - gs) + after_alarms[0]
                delays.append(first_alarm)
                detected_count += 1
                if first_alarm <= 64:
                    r_64 += 1

    total_events = len(gt_segs)
    mean_delay = float(np.mean(delays)) if len(delays) > 0 else float("nan")
    median_delay = float(np.median(delays)) if len(delays) > 0 else float("nan")

    return {
        "mean_delay": mean_delay,
        "median_delay": median_delay,
        "recall_at_5": r_5 / total_events,
        "recall_at_10": r_10 / total_events,
        "recall_at_64": r_64 / total_events,
        "detected_events": detected_count,
        "total_events": total_events,
    }


def evaluate_stream(
    dataset: str,
    channel: str,
    device: torch.device,
    epochs: int = 15,
):
    ds_dir = ROOT / "mTSBench_data" / dataset
    if channel == "default":
        train_path = ds_dir / f"{dataset}_train.csv"
        if not train_path.exists():
            train_path = ds_dir / f"{dataset}.csv"
        test_path = ds_dir / f"{dataset}_test.csv"
    else:
        train_path = ds_dir / f"{dataset}_{channel}_train.csv"
        if not train_path.exists():
            train_path = ds_dir / f"{channel}_train.csv"
        test_path = ds_dir / f"{dataset}_{channel}_test.csv"
        if not test_path.exists():
            test_path = ds_dir / f"{channel}_test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    numeric_cols = [c for c in train_df.columns if c not in ["timestamp", "is_anomaly", "label"]]
    lbl_col = [c for c in test_df.columns if c in ["is_anomaly", "label", "labels"]]
    test_labels = test_df[lbl_col[0]].to_numpy().astype(int) if lbl_col else np.zeros(len(test_df), dtype=int)

    train_vals = train_df[numeric_cols].to_numpy(dtype=np.float32)
    test_vals = test_df[numeric_cols].to_numpy(dtype=np.float32)

    train_mean = np.mean(train_vals, axis=0)
    train_std = np.std(train_vals, axis=0)
    scale = np.where(train_std < 1e-8, 1.0, np.maximum(train_std, 0.01))
    train_scaled = (train_vals - train_mean) / scale
    test_scaled = (test_vals - train_mean) / scale

    context_size = 256
    suspect_size = 64
    window_size = context_size + suspect_size

    train_windows = DataLoader.create_windows(train_scaled, window_size, step=10, copy=False)
    test_windows = DataLoader.create_windows(test_scaled, window_size, step=1, copy=False)
    training_data, val_data = split_train_val(train_windows, val_split=0.2, seed=42, window_size=window_size, step=10)

    input_dim = len(numeric_cols)
    models_to_test = ["TS-JEPA", "TimesNet", "TranAD"]
    results = []

    for model_name in models_to_test:
        print(f"  Training {model_name} on {dataset}:{channel}...")
        if model_name == "TS-JEPA":
            cfg = CSMConfig(model_type="ts_jepa", context_size=context_size, suspect_size=suspect_size, latent_dim=32)
            model = build_ts_jepa_model(cfg, input_dim=input_dim, device=device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for ep in range(epochs):
                model.train()
                perm = np.random.permutation(len(training_data))
                for b in range(0, len(perm), 16):
                    batch = torch.from_numpy(np.ascontiguousarray(training_data[perm[b:b+16]])).float().to(device)
                    ctx = batch[:, :context_size]
                    tgt = batch[:, context_size:]
                    loss, _ = model.compute_objective(ctx, tgt, cfg)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    model.update_target_encoder()

            def get_scores(w_arr):
                model.eval()
                res = []
                with torch.no_grad():
                    for i in range(0, len(w_arr), 128):
                        chunk = torch.from_numpy(np.ascontiguousarray(w_arr[i:i+128])).float().to(device)
                        disc = model.compute_predictive_discrepancy(chunk[:, :context_size], chunk[:, context_size:])
                        res.append(disc.cpu().numpy())
                return np.concatenate(res, axis=0)

        elif model_name == "TimesNet":
            model = TimesNet(c_in=input_dim, d_model=48, d_ff=96, e_layers=2).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for ep in range(epochs):
                model.train()
                perm = np.random.permutation(len(training_data))
                for b in range(0, len(perm), 16):
                    batch = torch.from_numpy(np.ascontiguousarray(training_data[perm[b:b+16]])).float().to(device)
                    out = model(batch)
                    loss = nn.functional.mse_loss(out, batch)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

            def get_scores(w_arr):
                model.eval()
                res = []
                with torch.no_grad():
                    for i in range(0, len(w_arr), 128):
                        chunk = torch.from_numpy(np.ascontiguousarray(w_arr[i:i+128])).float().to(device)
                        sc = model.compute_anomaly_scores(chunk)
                        res.append(sc[:, context_size:].mean(dim=-1).cpu().numpy())
                return np.concatenate(res, axis=0)

        elif model_name == "TranAD":
            model = TranAD(c_in=input_dim, d_model=48, n_heads=4, e_layers=2, d_layers=2, d_ff=96).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for ep in range(epochs):
                model.train()
                perm = np.random.permutation(len(training_data))
                for b in range(0, len(perm), 16):
                    batch = torch.from_numpy(np.ascontiguousarray(training_data[perm[b:b+16]])).float().to(device)
                    rec1, rec2 = model(batch)
                    l1, l2 = model.adversarial_loss(rec1, rec2, batch, epoch=ep+1)
                    loss = l1 + l2
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

            def get_scores(w_arr):
                model.eval()
                res = []
                with torch.no_grad():
                    for i in range(0, len(w_arr), 128):
                        chunk = torch.from_numpy(np.ascontiguousarray(w_arr[i:i+128])).float().to(device)
                        sc = model.compute_anomaly_scores(chunk)
                        res.append(sc[:, context_size:].mean(dim=-1).cpu().numpy())
                return np.concatenate(res, axis=0)

        # Compute raw window scores
        train_dense = DataLoader.create_windows(train_scaled, window_size, step=1, copy=False)
        train_win_sc = get_scores(train_dense)
        test_win_sc = get_scores(test_windows)
        stats = robust_stats(train_win_sc)
        train_z = positive_robust_z(train_win_sc, stats)
        test_z = positive_robust_z(test_win_sc, stats)

        # Now evaluate both settings: Causal (Trailing, Lookahead=0) vs. Buffered (Smear, Lookahead=64)
        for mode, mapping, lookahead_steps in [
            ("Strictly Causal (0-Lookahead)", "trailing", 0),
            ("Buffered Window (64-Lookahead)", "smear", 64),
        ]:
            pt_sc, val_m = aggregate_window_scores(
                test_z,
                n_points=len(test_df),
                context_size=context_size,
                suspect_size=suspect_size,
                step=1,
                reducer="mean",
                mapping_method=mapping,
            )
            tr_pt, tr_m = aggregate_window_scores(
                train_z,
                n_points=len(train_df),
                context_size=context_size,
                suspect_size=suspect_size,
                step=1,
                reducer="mean",
                mapping_method=mapping,
            )
            pt_sc = moving_average(pt_sc, 5)
            tr_pt = moving_average(tr_pt, 5)
            evt_res = calibrate_evt_threshold(tr_pt[tr_m], risk_level=1e-3, init_percentile=98.0, adaptive_kurtosis=True)
            preds = event_level_filter(pt_sc, evt_res.threshold, val_m, min_run=2, extreme_factor=1.75) * val_m

            # Compute standard metrics
            m_pt = compute_metrics(test_labels, preds, scores=pt_sc, valid_mask=val_m, use_pa=False)
            ev_m = compute_event_metrics(test_labels, preds)
            delay_stats = compute_detection_delay_stats(test_labels, preds)

            nominal_pts = np.sum(test_labels == 0)
            far_1k = (m_pt["fp"] / max(nominal_pts, 1)) * 1000.0
            fpr = (m_pt["fp"] / max(nominal_pts, 1))

            results.append({
                "dataset": dataset,
                "channel": channel,
                "model": model_name,
                "evaluation_setting": mode,
                "lookahead_delay_steps": lookahead_steps,
                "point_f1": m_pt.get("f1", 0.0),
                "point_precision": m_pt.get("precision", 0.0),
                "point_recall": m_pt.get("recall", 0.0),
                "event_recall": ev_m["event_recall"],
                "mean_delay_steps": delay_stats["mean_delay"],
                "recall_at_5": delay_stats["recall_at_5"],
                "recall_at_10": delay_stats["recall_at_10"],
                "recall_at_64": delay_stats["recall_at_64"],
                "false_positives": m_pt["fp"],
                "far_per_1k": far_1k,
                "empirical_fpr": fpr,
            })

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Causal vs. Buffered Evaluation on device: {device}")

    streams = [
        ("SMAP", "P-3"),
        ("SMD", "machine-1-2"),
        ("MSL", "M-1"),
        ("Daphnet", "S01R01E1"),
        ("GECCO", "water_quality"),
    ]

    all_results = []
    out_csv = ROOT / "reports" / "causal_vs_buffered_evaluation.csv"

    for ds, ch in streams:
        print(f"\n=== Evaluating Stream {ds}:{ch} ===")
        res = evaluate_stream(ds, ch, device=device, epochs=15)
        all_results.extend(res)
        pd.DataFrame(all_results).to_csv(out_csv, index=False)
        print(f"Saved intermediate results to {out_csv}")

    df = pd.DataFrame(all_results)
    print("\n" + "="*80)
    print("SUMMARY BY MODEL AND EVALUATION SETTING")
    print("="*80)
    summary = df.groupby(["model", "evaluation_setting"]).agg({
        "point_f1": "mean",
        "point_precision": "mean",
        "point_recall": "mean",
        "event_recall": "mean",
        "mean_delay_steps": "mean",
        "recall_at_10": "mean",
        "recall_at_64": "mean",
        "false_positives": "mean",
        "far_per_1k": "mean",
        "empirical_fpr": "mean",
    }).round(4)
    print(summary.to_string())


if __name__ == "__main__":
    main()
