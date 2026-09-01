"""Systematic Ablation Sweeps for TS-JEPA.

Runs 4 key architectural ablation experiments:
1. VICReg Loss Components: Invariance only vs Invariance+Variance vs Full VICReg (Invariance+Variance+Covariance).
2. Latent Dimension Capacity: D in {8, 16, 32, 64}.
3. EMA Momentum Decay: m in {0.0 (No EMA), 0.90, 0.99, 0.995, 0.999}.
4. Anomaly Discrepancy Metric: Euclidean vs Mahalanobis Covariance-Whitening.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.data_loader import DataLoader
from src.models import (
    HybridTCNEncoder,
    TSJEPAModel,
    jepa_vicreg_loss,
)
from src.models.legacy.train_model import split_train_validation as split_train_val
from src.scoring.event_fusion import (
    aggregate_window_scores,
    calibrate_evt_threshold,
    compute_metrics,
    event_level_filter,
    local_deviation_scores,
    moving_average,
    positive_robust_z,
    robust_stats,
)


def run_single_ablation_trial(
    train_windows: np.ndarray,
    test_windows: np.ndarray,
    test_labels: np.ndarray,
    input_dim: int,
    test_df_len: int,
    context_size: int = 256,
    suspect_size: int = 64,
    latent_dim: int = 32,
    filters: int = 48,
    tcn_layers: int = 3,
    ema_decay: float = 0.995,
    sim_weight: float = 1.0,
    var_weight: float = 1.0,
    cov_weight: float = 0.05,
    use_mahalanobis: bool = False,
    epochs: int = 15,
    batch_size: int = 32,
    risk_level: float = 1e-3,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    window_size = context_size + suspect_size
    training_data, val_data = split_train_val(train_windows, val_split=0.1, seed=seed)

    base_enc = HybridTCNEncoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        filters=filters,
        tcn_layers=tcn_layers,
        dropout=0.20,
    )
    model = TSJEPAModel(
        context_encoder=base_enc,
        latent_dim=latent_dim,
        predictor_hidden_dim=64,
        ema_decay=ema_decay,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(len(training_data))
        for b in range(0, len(perm), batch_size):
            clean = training_data[perm[b : b + batch_size]]
            ctx = torch.from_numpy(clean[:, :context_size]).float().to(device)
            tgt = torch.from_numpy(clean[:, context_size:]).float().to(device)

            z_ctx, z_tgt_true, z_tgt_pred = model(ctx, tgt)
            loss = jepa_vicreg_loss(
                z_target_pred=z_tgt_pred,
                z_target_true=z_tgt_true,
                z_context=z_ctx,
                sim_weight=sim_weight,
                var_weight=var_weight,
                cov_weight=cov_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.update_target_encoder()

    elapsed = time.time() - t0

    # Diagnostic checks on representation health
    model.eval()
    with torch.no_grad():
        sample_ctx = torch.from_numpy(train_windows[:500, :context_size]).float().to(device)
        sample_z = model.context_encoder(sample_ctx)
        latent_std = torch.std(sample_z, dim=0).mean().item()
        
        z_cent = sample_z - sample_z.mean(dim=0, keepdim=True)
        cov_m = (z_cent.T @ z_cent) / max(len(sample_z) - 1, 1)
        off_diag_cov = (cov_m - torch.diag(torch.diag(cov_m))).abs().mean().item()

    if use_mahalanobis:
        ctx_all = torch.from_numpy(train_windows[:, :context_size]).float().to(device)
        tgt_all = torch.from_numpy(train_windows[:, context_size:]).float().to(device)
        model.fit_mahalanobis_covariance(ctx_all, tgt_all)

    def compute_discrepancy(arr: np.ndarray) -> np.ndarray:
        model.eval()
        res = []
        with torch.no_grad():
            for i in range(0, len(arr), 2048):
                ctx = torch.from_numpy(arr[i : i + 2048, :context_size]).float().to(device)
                tgt = torch.from_numpy(arr[i : i + 2048, context_size:]).float().to(device)
                disc = model.compute_predictive_discrepancy(ctx, tgt, use_mahalanobis=use_mahalanobis).cpu().numpy()
                res.append(disc)
        return np.concatenate(res, axis=0) if res else np.empty(0)

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

    pt_scores, valid_mask = aggregate_window_scores(
        win_scores,
        n_points=test_df_len,
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

    return {
        "elapsed_sec": round(elapsed, 2),
        "latent_std": round(latent_std, 4),
        "off_diag_cov": round(off_diag_cov, 4),
        "pa_f1": m_pa.get("f1", 0.0),
        "pa_precision": m_pa.get("precision", 0.0),
        "pa_recall": m_pa.get("recall", 0.0),
        "point_f1": m_pt.get("f1", 0.0),
        "oracle_pa_f1": float(best_pa_f1),
    }


def main():
    parser = argparse.ArgumentParser(description="TS-JEPA Systematic Ablation Suite")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else args.device if args.device != "auto" else "cpu")
    
    # Representative benchmark channels for ablation
    benchmark_channels = [
        ("Daphnet", "S01R01E1"),
        ("Daphnet", "S02R02E0"),
        ("Exathlon", "10_4_1000000_79"),
        ("SMAP", "P-3"),
    ]

    print("=" * 90)
    print("TS-JEPA SYSTEMATIC ABLATION SUITE")
    print(f"Device: {device} | Channels: {benchmark_channels} | Epochs: {args.epochs}")
    print("=" * 90)

    # Preload channel data
    data_cache = {}
    context_size = 256
    suspect_size = 64
    window_size = context_size + suspect_size

    for ds_name, chan in benchmark_channels:
        ds_dir = ROOT / "mTSBench_data" / ds_name
        train_p = ds_dir / f"{ds_name}_{chan}_train.csv"
        test_p = ds_dir / f"{ds_name}_{chan}_test.csv"
        train_df = pd.read_csv(train_p)
        test_df = pd.read_csv(test_p)
        numeric_cols = [c for c in train_df.columns if c not in ["timestamp", "is_anomaly"]]
        test_labels = test_df["is_anomaly"].to_numpy().astype(int)

        train_mean = train_df[numeric_cols].mean().to_numpy()
        train_std = train_df[numeric_cols].std().to_numpy()
        scale = np.maximum(train_std, 0.01)
        train_scaled = (train_df[numeric_cols].to_numpy() - train_mean) / scale
        test_scaled = (test_df[numeric_cols].to_numpy() - train_mean) / scale

        train_windows = DataLoader.create_windows(train_scaled, window_size, step=10)
        test_windows = DataLoader.create_windows(test_scaled, window_size, step=1)

        data_cache[f"{ds_name}_{chan}"] = {
            "train_windows": train_windows,
            "test_windows": test_windows,
            "test_labels": test_labels,
            "input_dim": len(numeric_cols),
            "test_df_len": len(test_df),
        }

    all_ablation_records = []

    # =========================================================================
    # ABLATION 1: VICReg Loss Components
    # =========================================================================
    print("\n" + "#" * 60)
    print("1. ABLATION: VICReg Loss Formulations")
    print("#" * 60)

    vicreg_configs = [
        {"name": "Invariance Only (MSE)", "sim": 1.0, "var": 0.0, "cov": 0.0},
        {"name": "Invariance + Variance Hinge", "sim": 1.0, "var": 1.0, "cov": 0.0},
        {"name": "Full VICReg (Invariance + Var + Cov)", "sim": 1.0, "var": 1.0, "cov": 0.05},
    ]

    for cfg in vicreg_configs:
        print(f"\nConfiguration: {cfg['name']}")
        for key, data in data_cache.items():
            print(f"  Channel: {key:26s} ... ", end="", flush=True)
            res = run_single_ablation_trial(
                train_windows=data["train_windows"],
                test_windows=data["test_windows"],
                test_labels=data["test_labels"],
                input_dim=data["input_dim"],
                test_df_len=data["test_df_len"],
                sim_weight=cfg["sim"],
                var_weight=cfg["var"],
                cov_weight=cfg["cov"],
                epochs=args.epochs,
                device=device,
            )
            print(f"PA-F1: {res['pa_f1']:.4f} | Latent Std: {res['latent_std']:.3f} | Cov OffDiag: {res['off_diag_cov']:.3f}")
            all_ablation_records.append({
                "ablation_study": "1_loss_components",
                "variant": cfg["name"],
                "channel": key,
                **res,
            })

    # =========================================================================
    # ABLATION 2: Latent Dimension Scaling (D)
    # =========================================================================
    print("\n" + "#" * 60)
    print("2. ABLATION: Latent Dimension Scaling (D)")
    print("#" * 60)

    dim_configs = [8, 16, 32, 64]
    for d in dim_configs:
        print(f"\nConfiguration: Latent Dim D = {d}")
        for key, data in data_cache.items():
            print(f"  Channel: {key:26s} ... ", end="", flush=True)
            res = run_single_ablation_trial(
                train_windows=data["train_windows"],
                test_windows=data["test_windows"],
                test_labels=data["test_labels"],
                input_dim=data["input_dim"],
                test_df_len=data["test_df_len"],
                latent_dim=d,
                epochs=args.epochs,
                device=device,
            )
            print(f"PA-F1: {res['pa_f1']:.4f} | Latent Std: {res['latent_std']:.3f}")
            all_ablation_records.append({
                "ablation_study": "2_latent_dim",
                "variant": f"D = {d}",
                "channel": key,
                **res,
            })

    # =========================================================================
    # ABLATION 3: EMA Momentum Decay Rate (m)
    # =========================================================================
    print("\n" + "#" * 60)
    print("3. ABLATION: Target Encoder EMA Momentum Decay (m)")
    print("#" * 60)

    ema_configs = [
        {"name": "No EMA (m=0.0)", "decay": 0.0},
        {"name": "m = 0.90", "decay": 0.90},
        {"name": "m = 0.99", "decay": 0.99},
        {"name": "m = 0.995 (Default)", "decay": 0.995},
        {"name": "m = 0.999", "decay": 0.999},
    ]

    for cfg in ema_configs:
        print(f"\nConfiguration: {cfg['name']}")
        for key, data in data_cache.items():
            print(f"  Channel: {key:26s} ... ", end="", flush=True)
            res = run_single_ablation_trial(
                train_windows=data["train_windows"],
                test_windows=data["test_windows"],
                test_labels=data["test_labels"],
                input_dim=data["input_dim"],
                test_df_len=data["test_df_len"],
                ema_decay=cfg["decay"],
                epochs=args.epochs,
                device=device,
            )
            print(f"PA-F1: {res['pa_f1']:.4f}")
            all_ablation_records.append({
                "ablation_study": "3_ema_momentum",
                "variant": cfg["name"],
                "channel": key,
                **res,
            })

    # Save results
    out_dir = ROOT / "reports" / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_ablations = pd.DataFrame(all_ablation_records)
    out_csv = out_dir / "systematic_ablation_results.csv"
    df_ablations.to_csv(out_csv, index=False)
    print(f"\nSaved raw ablation results to {out_csv}")

    print("\n" + "=" * 90)
    print("CONSOLIDATED ABLATION SUMMARY (Mean PA-F1 & Representation Health):")
    print("=" * 90)
    summary = df_ablations.groupby(["ablation_study", "variant"]).agg(
        Mean_PA_F1=("pa_f1", "mean"),
        Mean_Point_F1=("point_f1", "mean"),
        Mean_Latent_Std=("latent_std", "mean"),
        Mean_OffDiag_Cov=("off_diag_cov", "mean"),
        Mean_Oracle_F1=("oracle_pa_f1", "mean"),
    )
    print(summary.to_string())
    print("=" * 90)


if __name__ == "__main__":
    main()
