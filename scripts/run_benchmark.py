"""Unified Multi-Dataset Benchmark Suite for TS-JEPA.

Supports benchmarking on:
- Daphnet (Biomedical Parkinsonian gait)
- Exathlon (Cloud distributed streaming logs / JVM leaks)
- SMAP / MSL (Spacecraft telemetry)
- room-occupancy (IoT environmental)
- OPPORTUNITY (77-sensor body activity)
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import (
    HybridTCNEncoder,
    RelationalGATEncoder,
    RelationalGAT_JEPAModel,
    TSJEPAModel,
    jepa_vicreg_loss,
)
from src.models.train_model import split_train_validation as split_train_val
from src.data.data_loader import DataLoader
from src.utils.event_fusion import (
    aggregate_window_scores,
    calibrate_evt_threshold,
    compute_metrics,
    event_level_filter,
    local_deviation_scores,
    moving_average,
    positive_robust_z,
    robust_stats,
)


DEFAULT_DATASET_CHANNELS = {
    "Daphnet": ["S01R01E1", "S02R01E0", "S02R02E0", "S03R01E0", "S03R01E1", "S03R02E0"],
    "Exathlon": ["10_2_1000000_67", "10_3_1000000_75", "10_4_1000000_79", "1_2_100000_68", "1_4_1000000_80", "1_5_1000000_86"],
    "SMAP": ["A-1", "A-2", "A-7", "D-3", "P-3", "E-1"],
    "room-occupancy": ["default", "1"],
    "OPPORTUNITY": ["S1-ADL2", "S1-ADL3", "S1-ADL4", "S1-ADL5", "S2-ADL1", "S2-ADL2"],
}


def evaluate_channel(
    dataset_name: str,
    chan_name: str,
    train_path: Path,
    test_path: Path,
    encoder_type: str = "tcn",
    use_mahalanobis: bool = False,
    context_size: int = 256,
    suspect_size: int = 64,
    latent_dim: int = 32,
    epochs: int = 15,
    batch_size: int = 32,
    risk_level: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> dict:
    window_size = context_size + suspect_size
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    numeric_cols = [c for c in train_df.columns if c not in ["timestamp", "is_anomaly"]]
    test_labels = test_df["is_anomaly"].to_numpy().astype(int) if "is_anomaly" in test_df.columns else None

    train_mean = train_df[numeric_cols].mean().to_numpy()
    train_std = train_df[numeric_cols].std().to_numpy()
    scale = np.maximum(train_std, 0.01)
    train_scaled = (train_df[numeric_cols].to_numpy() - train_mean) / scale
    test_scaled = (test_df[numeric_cols].to_numpy() - train_mean) / scale

    train_windows = DataLoader.create_windows(train_scaled, window_size, step=10)
    test_windows = DataLoader.create_windows(test_scaled, window_size, step=1)

    training_data, val_data = split_train_val(train_windows, val_split=0.1, seed=42)
    input_dim = len(numeric_cols)
    t0 = time.time()

    if encoder_type == "tcn":
        base_enc = HybridTCNEncoder(input_dim=input_dim, latent_dim=latent_dim, filters=48, tcn_layers=3, dropout=0.20)
        model = TSJEPAModel(context_encoder=base_enc, latent_dim=latent_dim, predictor_hidden_dim=64, ema_decay=0.995).to(device)
    elif encoder_type == "gat":
        model = RelationalGAT_JEPAModel(input_dim=input_dim, latent_dim=latent_dim, filters=48, tcn_layers=3, gat_layers=2, dropout=0.20).to(device)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(len(training_data))
        for b in range(0, len(perm), batch_size):
            clean = training_data[perm[b : b + batch_size]]
            ctx = torch.from_numpy(clean[:, :context_size]).float().to(device)
            tgt = torch.from_numpy(clean[:, context_size:]).float().to(device)

            if encoder_type == "tcn":
                z_ctx, z_tgt_true, z_tgt_pred = model(ctx, tgt)
                loss = jepa_vicreg_loss(z_tgt_pred, z_tgt_true, z_context=z_ctx)
            else:
                loss = model.compute_loss(ctx, tgt)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.update_target_encoder()

    if use_mahalanobis:
        ctx_all = torch.from_numpy(train_windows[:, :context_size]).float().to(device)
        tgt_all = torch.from_numpy(train_windows[:, context_size:]).float().to(device)
        model.fit_mahalanobis_covariance(ctx_all, tgt_all)

    def compute_discrepancy(arr):
        model.eval()
        res = []
        with torch.no_grad():
            for i in range(0, len(arr), 2048):
                ctx = torch.from_numpy(arr[i : i + 2048, :context_size]).float().to(device)
                tgt = torch.from_numpy(arr[i : i + 2048, context_size:]).float().to(device)
                disc = model.compute_predictive_discrepancy(ctx, tgt, use_mahalanobis=use_mahalanobis).cpu().numpy()
                res.append(disc)
        return np.concatenate(res, axis=0)

    train_disc = compute_discrepancy(train_windows)
    test_disc = compute_discrepancy(test_windows)

    disc_stats = robust_stats(train_disc)
    test_local = local_deviation_scores(test_windows, context_size)
    train_local = local_deviation_scores(train_windows, context_size)
    local_stats = robust_stats(train_local)

    disc_z = positive_robust_z(test_disc, disc_stats, clip=20.0)
    loc_z = positive_robust_z(test_local, local_stats, clip=20.0)
    win_scores = 0.7 * disc_z + 0.3 * loc_z

    train_disc_z = positive_robust_z(train_disc, disc_stats, clip=20.0)
    train_loc_z = positive_robust_z(train_local, local_stats, clip=20.0)
    train_win_scores = 0.7 * train_disc_z + 0.3 * train_loc_z

    # Aggregate window scores to points
    pt_scores, valid_mask = aggregate_window_scores(
        win_scores,
        n_points=len(test_df),
        context_size=context_size,
        suspect_size=suspect_size,
        step=1,
        reducer="mean",
        mapping_method="middle",
    )
    test_scores = moving_average(pt_scores, 12)
    valid_scores = test_scores[valid_mask]

    train_pt_scores, train_valid_mask = aggregate_window_scores(
        train_win_scores,
        n_points=(len(train_windows) - 1) * 10 + window_size,
        context_size=context_size,
        suspect_size=suspect_size,
        step=10,
        reducer="mean",
        mapping_method="middle",
    )
    train_smoothed = moving_average(train_pt_scores, 12)
    train_valid_scores = train_smoothed[train_valid_mask]

    # EVT Calibration
    evt_res = calibrate_evt_threshold(train_valid_scores, risk_level=risk_level, init_percentile=98.0)
    evt_th = evt_res.threshold

    preds_evt = event_level_filter(test_scores, evt_th, valid_mask, min_run=2, extreme_factor=1.75) * valid_mask.astype(np.float32)
    m_pa = compute_metrics(test_labels, preds_evt, valid_mask=valid_mask, use_pa=True)
    m_pt = compute_metrics(test_labels, preds_evt, valid_mask=valid_mask, use_pa=False)

    best_pa_f1 = 0.0
    candidates = np.percentile(valid_scores, np.linspace(0.0, 100.0, 150))
    for th in candidates:
        p = event_level_filter(test_scores, th, valid_mask, min_run=2, extreme_factor=1.75) * valid_mask.astype(np.float32)
        m = compute_metrics(test_labels, p, valid_mask=valid_mask, use_pa=True)
        if m.get("f1", 0.0) > best_pa_f1:
            best_pa_f1 = m["f1"]

    elapsed = time.time() - t0

    return {
        "dataset": dataset_name,
        "channel": chan_name,
        "encoder": encoder_type,
        "scoring": "mahalanobis" if use_mahalanobis else "euclidean",
        "elapsed_sec": round(elapsed, 2),
        "threshold": float(evt_th),
        "evt_converged": bool(evt_res.gpd_fit.fit_converged),
        "pa_f1": m_pa.get("f1", 0.0),
        "pa_precision": m_pa.get("precision", 0.0),
        "pa_recall": m_pa.get("recall", 0.0),
        "point_f1": m_pt.get("f1", 0.0),
        "point_precision": m_pt.get("precision", 0.0),
        "point_recall": m_pt.get("recall", 0.0),
        "oracle_pa_f1": float(best_pa_f1),
    }


def main():
    parser = argparse.ArgumentParser(description="Unified TS-JEPA Multi-Dataset Benchmark")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "Daphnet", "Exathlon", "SMAP", "room-occupancy", "OPPORTUNITY"])
    parser.add_argument("--encoder", type=str, default="tcn", choices=["tcn", "gat"])
    parser.add_argument("--scoring", type=str, default="euclidean", choices=["euclidean", "mahalanobis"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else args.device if args.device != "auto" else "cpu")
    datasets_to_run = list(DEFAULT_DATASET_CHANNELS.keys()) if args.dataset == "all" else [args.dataset]
    use_mahalanobis = (args.scoring == "mahalanobis")

    print("=" * 90)
    print(f"TS-JEPA UNIFIED BENCHMARK SUITE")
    print(f"Datasets: {datasets_to_run} | Encoder: {args.encoder} | Scoring: {args.scoring}")
    print(f"Device: {device} | Epochs: {args.epochs}")
    print("=" * 90)

    all_results = []

    for ds_name in datasets_to_run:
        channels = DEFAULT_DATASET_CHANNELS[ds_name]
        ds_dir = ROOT / "mTSBench_data" / ds_name
        print(f"\n=======================================================")
        print(f"Evaluating {ds_name} ({len(channels)} channels)")
        print(f"=======================================================")

        for chan in channels:
            print(f"  Channel: {chan:16s} ... ", end="", flush=True)
            if chan == "default":
                train_p = ds_dir / f"{ds_name}_train.csv"
                test_p = ds_dir / f"{ds_name}_test.csv"
            else:
                train_p = ds_dir / f"{ds_name}_{chan}_train.csv"
                test_p = ds_dir / f"{ds_name}_{chan}_test.csv"

            if not train_p.exists() or not test_p.exists():
                print(f"Skipping (files missing)")
                continue

            res = evaluate_channel(
                dataset_name=ds_name,
                chan_name=chan,
                train_path=train_p,
                test_path=test_p,
                encoder_type=args.encoder,
                use_mahalanobis=use_mahalanobis,
                epochs=args.epochs,
                device=device,
            )
            pa_f1 = res.get("pa_f1", 0.0)
            pt_f1 = res.get("point_f1", 0.0)
            oracle = res.get("oracle_pa_f1", 0.0)
            print(f"Done in {res['elapsed_sec']}s | PA-F1: {pa_f1:.4f} | Point-F1: {pt_f1:.4f} | Oracle: {oracle:.4f}")
            all_results.append(res)

    results_df = pd.DataFrame(all_results)
    out_csv = ROOT / "reports" / f"ts_jepa_benchmark_{args.dataset}_{args.encoder}_{args.scoring}.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved benchmark results to {out_csv}")

    print("\n" + "=" * 90)
    print("DATASET MACRO SUMMARY:")
    print("=" * 90)
    summary = results_df.groupby("dataset").agg(
        Channels=("channel", "count"),
        Mean_PA_F1=("pa_f1", "mean"),
        Mean_Point_F1=("point_f1", "mean"),
        Mean_Oracle_F1=("oracle_pa_f1", "mean"),
    )
    print(summary.to_string())
    print("-" * 90)
    print(f"Overall Grand Macro PA-F1: {results_df['pa_f1'].mean():.4f} | Oracle: {results_df['oracle_pa_f1'].mean():.4f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
