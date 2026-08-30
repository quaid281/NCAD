"""Grand Slam Benchmark: TS-JEPA (TCN Backbone) with Mahalanobis Covariance-Whitened Discrepancy on Daphnet."""

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

from src.models.tcn_encoder import HybridTCNEncoder, contrastive_loss
from src.models.ts_jepa import TSJEPAModel, jepa_vicreg_loss
from src.models.successor_memory import CounterfactualSuccessorMemory, SuccessorMemoryConfig
from src.models.train_model import split_train_validation as split_train_val
from src.data.data_loader import DataLoader
from src.models.anomaly_injector import AnomalyInjectionConfig, ContextualAnomalyInjector
from src.utils.event_fusion import (
    aggregate_window_scores,
    calibrate_evt_threshold,
    compute_metrics,
    event_level_filter,
    fuse_evidence_scores,
    local_deviation_scores,
    moving_average,
    positive_robust_z,
    robust_stats,
)


def train_and_eval_channel(
    chan_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mode: str,
    context_size: int = 256,
    suspect_size: int = 64,
    latent_dim: int = 32,
    epochs: int = 15,
    batch_size: int = 32,
    risk_level: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> dict:
    window_size = context_size + suspect_size
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

    if mode == "NCAD_TCN":
        model = HybridTCNEncoder(input_dim=input_dim, latent_dim=latent_dim, filters=48, tcn_layers=3, dropout=0.20).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        injector = ContextualAnomalyInjector(
            AnomalyInjectionConfig(injection_ratio=0.5, min_anomaly_len=16, max_anomaly_len=64),
            seed=42,
        )

        for epoch in range(1, epochs + 1):
            model.train()
            perm = np.random.permutation(len(training_data))
            for b in range(0, len(perm), batch_size):
                clean = training_data[perm[b : b + batch_size]]
                mod, lbl = injector.inject_batch(clean, context_size)
                loss = contrastive_loss(
                    model(torch.from_numpy(mod).float().to(device)),
                    model(torch.from_numpy(clean[:, :context_size]).float().to(device)),
                    torch.from_numpy(lbl).float().to(device),
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        def encode(arr):
            model.eval()
            res = []
            with torch.no_grad():
                for i in range(0, len(arr), 2048):
                    t = torch.from_numpy(arr[i : i + 2048]).float().to(device)
                    res.append(model(t).cpu().numpy())
            return np.concatenate(res, axis=0) if res else np.empty((0, latent_dim))

        train_ctx_emb = encode(train_windows[:, :context_size])
        train_suc_emb = encode(train_windows[:, context_size:])
        memory = CounterfactualSuccessorMemory(SuccessorMemoryConfig(n_neighbors=8, max_memory_windows=4000, seed=42))
        memory.fit(train_ctx_emb, train_suc_emb)

        train_local = local_deviation_scores(train_windows, context_size)
        succ_stats = robust_stats(memory.calibration_successor_scores)
        local_stats = robust_stats(train_local[memory.sample_indices])

        test_ctx_emb = encode(test_windows[:, :context_size])
        test_suc_emb = encode(test_windows[:, context_size:])
        query = memory.query(test_ctx_emb, test_suc_emb)
        test_local = local_deviation_scores(test_windows, context_size)

        succ_z = positive_robust_z(query.successor_scores, succ_stats, clip=20.0)
        loc_z = positive_robust_z(test_local, local_stats, clip=20.0)
        ctx_ratio = np.minimum(query.context_distances / max(float(memory.context_threshold), 1e-6), 3.0)

        win_scores = fuse_evidence_scores(successor_z=succ_z, local_z=loc_z, context_ratio=ctx_ratio)

        train_succ_z = positive_robust_z(memory.calibration_successor_scores, succ_stats, clip=20.0)
        train_loc_z = positive_robust_z(train_local[memory.sample_indices], local_stats, clip=20.0)
        train_ctx_ratio = np.minimum(memory.calibration_context_distances / max(float(memory.context_threshold), 1e-6), 3.0)
        train_win_scores = fuse_evidence_scores(successor_z=train_succ_z, local_z=train_loc_z, context_ratio=train_ctx_ratio)

    elif mode in ("TS_JEPA_Euclidean", "TS_JEPA_Mahalanobis"):
        base_enc = HybridTCNEncoder(input_dim=input_dim, latent_dim=latent_dim, filters=48, tcn_layers=3, dropout=0.20)
        model = TSJEPAModel(context_encoder=base_enc, latent_dim=latent_dim, predictor_hidden_dim=64, ema_decay=0.995).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        for epoch in range(1, epochs + 1):
            model.train()
            perm = np.random.permutation(len(training_data))
            for b in range(0, len(perm), batch_size):
                clean = training_data[perm[b : b + batch_size]]
                ctx = torch.from_numpy(clean[:, :context_size]).float().to(device)
                tgt = torch.from_numpy(clean[:, context_size:]).float().to(device)

                z_ctx, z_tgt_true, z_tgt_pred = model(ctx, tgt)
                loss = jepa_vicreg_loss(z_tgt_pred, z_tgt_true, z_context=z_ctx)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                model.update_target_encoder()

        use_mahal = (mode == "TS_JEPA_Mahalanobis")
        if use_mahal:
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
                    disc = model.compute_predictive_discrepancy(ctx, tgt, use_mahalanobis=use_mahal).cpu().numpy()
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
    else:
        raise ValueError(f"Unknown mode: {mode}")

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

    # EVT Calibration on training normal distribution
    evt_res = calibrate_evt_threshold(train_valid_scores, risk_level=risk_level, init_percentile=98.0)
    evt_th = evt_res.threshold

    preds_evt = event_level_filter(test_scores, evt_th, valid_mask, min_run=2, extreme_factor=1.75) * valid_mask.astype(np.float32)
    m_pa = compute_metrics(test_labels, preds_evt, valid_mask=valid_mask, use_pa=True)
    m_pt = compute_metrics(test_labels, preds_evt, valid_mask=valid_mask, use_pa=False)

    # Compute Oracle Upper Bound
    best_pa_f1 = 0.0
    candidates = np.percentile(valid_scores, np.linspace(0.0, 100.0, 150))
    for th in candidates:
        p = event_level_filter(test_scores, th, valid_mask, min_run=2, extreme_factor=1.75) * valid_mask.astype(np.float32)
        m = compute_metrics(test_labels, p, valid_mask=valid_mask, use_pa=True)
        if m.get("f1", 0.0) > best_pa_f1:
            best_pa_f1 = m["f1"]

    elapsed = time.time() - t0

    return {
        "channel": chan_name,
        "mode": mode,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Daphnet")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else args.device if args.device != "auto" else "cpu")
    dataset_dir = ROOT / "mTSBench_data" / args.dataset

    train_files = sorted(glob.glob(str(dataset_dir / f"{args.dataset}*_train.csv")))
    channels = [Path(f).name.replace(f"{args.dataset}_", "").replace("_train.csv", "") for f in train_files]

    bench_channels = ["S01R01E1", "S02R01E0", "S02R02E0", "S03R01E0", "S03R01E1", "S03R02E0"]
    channels = [c for c in bench_channels if c in channels]

    print("=" * 85)
    print(f"GRAND SLAM BENCHMARK: Baseline NCAD vs TS-JEPA (Euclidean) vs TS-JEPA (Mahalanobis)")
    print(f"Dataset: {args.dataset} | Device: {device} | Channels: {channels} | Epochs: {args.epochs}")
    print("=" * 85)

    all_results = []
    modes = ["NCAD_TCN", "TS_JEPA_Euclidean", "TS_JEPA_Mahalanobis"]

    for chan in channels:
        print(f"\n--- Channel: {chan} ---")
        train_path = dataset_dir / f"{args.dataset}_{chan}_train.csv"
        test_path = dataset_dir / f"{args.dataset}_{chan}_test.csv"
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        for mode in modes:
            print(f"  Running {mode:22s} ... ", end="", flush=True)
            res = train_and_eval_channel(
                chan_name=chan,
                train_df=train_df,
                test_df=test_df,
                mode=mode,
                epochs=args.epochs,
                device=device,
            )
            pa_f1 = res.get("pa_f1", 0.0)
            pt_f1 = res.get("point_f1", 0.0)
            oracle_f1 = res.get("oracle_pa_f1", 0.0)
            print(f"Done in {res['elapsed_sec']}s | PA-F1: {pa_f1:.4f} | Point-F1: {pt_f1:.4f} | Oracle: {oracle_f1:.4f}")
            all_results.append(res)

    results_df = pd.DataFrame(all_results)
    out_csv = ROOT / "reports" / "daphnet_grand_slam_comparison.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved raw results to {out_csv}")

    # Summary Pivot Table
    pivot_pa = results_df.pivot(index="channel", columns="mode", values="pa_f1")
    print("\n" + "=" * 85)
    print("GRAND SLAM CONSOLIDATED PA-F1 COMPARISON TABLE (Daphnet):")
    print("=" * 85)
    print(pivot_pa.to_string())

    print("\nMacro Averages across all channels:")
    for mode in modes:
        avg_pa = results_df[results_df["mode"] == mode]["pa_f1"].mean()
        avg_pt = results_df[results_df["mode"] == mode]["point_f1"].mean()
        avg_oracle = results_df[results_df["mode"] == mode]["oracle_pa_f1"].mean()
        print(f"  {mode:22s} -> Mean PA-F1: {avg_pa:.4f} | Mean Point-F1: {avg_pt:.4f} | Mean Oracle PA-F1: {avg_oracle:.4f}")
    print("=" * 85)


if __name__ == "__main__":
    main()
