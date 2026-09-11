"""Ablation Runner for Potential-Flow JEPA (PF-JEPA).

Compares:
1. ts_jepa_base         : Point-JEPA (Deterministic latent regression)
2. ts_jepa_flow         : Standard Flow-JEPA (Unconstrained OT-CFM vector field)
3. pf_jepa_midpoint     : Potential-Flow JEPA (Conservative velocity v = -nabla Phi, scoring via D_M)
4. pf_jepa_curvature    : Potential-Flow JEPA with Energy Laplacian Curvature Scoring (D_M + Delta Phi)
5. pf_jepa_regime       : Potential-Flow JEPA with Harmonic Grassmannian Subspace Regimes + Curvature

Evaluated under exact 50-epoch training, Cosine Annealing, and EVT tail calibration.
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
    FlowTSJEPA,
    HybridTCNEncoder,
    PotentialFlowJEPA,
    PotentialFlowJEPAModel,
    TSJEPAModel,
    flow_matching_vicreg_loss,
    jepa_vicreg_loss,
)
from src.models.legacy.train_model import split_train_validation as split_train_val
from src.scoring.event_fusion import (
    aggregate_window_scores,
    calibrate_evt_threshold,
    compute_metrics,
    event_level_filter,
    moving_average,
)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_eval_variant(
    variant: str,
    dataset_name: str,
    chan_name: str,
    train_path: Path,
    test_path: Path,
    seed: int = 42,
    context_size: int = 256,
    suspect_size: int = 64,
    epochs: int = 50,
    batch_size: int = 32,
    latent_dim: int = 32,
    risk_level: float = 1e-3,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> dict:
    set_seed(seed)
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

    training_data, _ = split_train_val(train_windows, val_split=0.1, seed=seed)
    input_dim = len(numeric_cols)
    total_steps = epochs * max(1, len(training_data) // batch_size)
    global_step = 0
    t0 = time.time()

    # =========================================================================
    # Variant 1: Point-JEPA (Deterministic Baseline)
    # =========================================================================
    if variant == "ts_jepa_base":
        base_enc = HybridTCNEncoder(input_dim=input_dim, latent_dim=latent_dim, filters=48, tcn_layers=6, dropout=0.20)
        model = TSJEPAModel(context_encoder=base_enc, latent_dim=latent_dim, predictor_hidden_dim=64, ema_decay=0.996).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        for epoch in range(1, epochs + 1):
            model.train()
            perm = np.random.permutation(len(training_data))
            for b in range(0, len(perm), batch_size):
                global_step += 1
                batch_arr = training_data[perm[b : b + batch_size]]
                ctx = torch.from_numpy(batch_arr[:, :context_size]).float().to(device)
                tgt = torch.from_numpy(batch_arr[:, context_size:]).float().to(device)
                z_ctx, z_tgt_true, z_tgt_pred = model(ctx, tgt)
                loss = jepa_vicreg_loss(z_tgt_pred, z_tgt_true, z_context=z_ctx, cov_weight=0.5)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                ema_val = 0.996 + (0.9995 - 0.996) * 0.5 * (1.0 - np.cos(np.pi * global_step / total_steps))
                model.update_target_encoder(decay=ema_val)
            scheduler.step()

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
                    disc = model.compute_predictive_discrepancy(ctx, tgt, use_mahalanobis=True).cpu().numpy()
                    res.append(disc)
            return np.concatenate(res, axis=0)

    # =========================================================================
    # Variant 2: Standard Flow-JEPA (OT-CFM Vector Field)
    # =========================================================================
    elif variant == "ts_jepa_flow":
        base_enc = HybridTCNEncoder(input_dim=input_dim, latent_dim=latent_dim, filters=48, tcn_layers=6, dropout=0.20)
        model = FlowTSJEPA(
            context_encoder=base_enc,
            latent_dim=latent_dim,
            predictor_hidden_dim=64,
            predictor_layers=3,
            ema_decay=0.996,
        ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        for epoch in range(1, epochs + 1):
            model.train()
            perm = np.random.permutation(len(training_data))
            for b in range(0, len(perm), batch_size):
                global_step += 1
                batch_arr = training_data[perm[b : b + batch_size]]
                ctx = torch.from_numpy(batch_arr[:, :context_size]).float().to(device)
                tgt = torch.from_numpy(batch_arr[:, context_size:]).float().to(device)
                z_ctx, z_tgt_true, v_pred, v_target = model(ctx, tgt)
                loss, _ = flow_matching_vicreg_loss(
                    v_pred=v_pred,
                    v_target=v_target,
                    z_ctx=z_ctx,
                    z_tgt_true=z_tgt_true,
                    cov_weight=0.5,
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                ema_val = 0.996 + (0.9995 - 0.996) * 0.5 * (1.0 - np.cos(np.pi * global_step / total_steps))
                model.update_target_encoder(decay=ema_val)
            scheduler.step()

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
                    disc = model.compute_predictive_discrepancy(ctx, tgt, use_mahalanobis=True).cpu().numpy()
                    res.append(disc)
            return np.concatenate(res, axis=0)

    # =========================================================================
    # Variants 3, 4, 5: Potential-Flow JEPA Variants
    # =========================================================================
    elif variant in ["pf_jepa_midpoint", "pf_jepa_curvature", "pf_jepa_regime"]:
        use_regimes = (variant == "pf_jepa_regime")
        include_curv = (variant in ["pf_jepa_curvature", "pf_jepa_regime"])

        base_enc = HybridTCNEncoder(input_dim=input_dim, latent_dim=latent_dim, filters=48, tcn_layers=6, dropout=0.20)
        model = PotentialFlowJEPA(
            context_encoder=base_enc,
            latent_dim=latent_dim,
            predictor_hidden_dim=64,
            predictor_layers=3,
            n_regimes=4 if use_regimes else 1,
            subspace_dim=8,
            use_regimes=use_regimes,
            ema_decay=0.996,
            dropout=0.10,
        ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        for epoch in range(1, epochs + 1):
            model.train()
            perm = np.random.permutation(len(training_data))
            for b in range(0, len(perm), batch_size):
                global_step += 1
                batch_arr = training_data[perm[b : b + batch_size]]
                ctx = torch.from_numpy(batch_arr[:, :context_size]).float().to(device)
                tgt = torch.from_numpy(batch_arr[:, context_size:]).float().to(device)
                loss, _ = model.compute_objective(ctx, tgt, cov_weight=0.5, grassmann_weight=0.05)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                ema_val = 0.996 + (0.9995 - 0.996) * 0.5 * (1.0 - np.cos(np.pi * global_step / total_steps))
                model.update_target_encoder(decay=ema_val)
            scheduler.step()

        ctx_all = torch.from_numpy(train_windows[:, :context_size]).float().to(device)
        tgt_all = torch.from_numpy(train_windows[:, context_size:]).float().to(device)
        model.fit_mahalanobis_covariance(ctx_all, tgt_all)

        def compute_discrepancy(arr):
            model.eval()
            res = []
            chunk_size = 512 if include_curv else 2048
            for i in range(0, len(arr), chunk_size):
                ctx = torch.from_numpy(arr[i : i + chunk_size, :context_size]).float().to(device)
                tgt = torch.from_numpy(arr[i : i + chunk_size, context_size:]).float().to(device)
                disc = model.compute_predictive_discrepancy(
                    ctx,
                    tgt,
                    use_mahalanobis=True,
                    include_curvature=include_curv,
                    curvature_weight=0.05,
                ).cpu().numpy()
                res.append(disc)
            return np.concatenate(res, axis=0)

    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Compute window scores
    test_win_scores = compute_discrepancy(test_windows)
    train_win_scores = compute_discrepancy(train_windows)

    # Point-level mapping and smoothing
    pt_scores, valid_mask = aggregate_window_scores(
        test_win_scores,
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
        n_points=len(train_df),
        context_size=context_size,
        suspect_size=suspect_size,
        step=1,
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

    # Oracle PA-F1 search
    best_pa_f1 = 0.0
    candidates = np.percentile(valid_scores, np.linspace(0.0, 100.0, 150))
    for th in candidates:
        p = event_level_filter(test_scores, th, valid_mask, min_run=2, extreme_factor=1.75) * valid_mask.astype(np.float32)
        m = compute_metrics(test_labels, p, valid_mask=valid_mask, use_pa=True)
        if m.get("f1", 0.0) > best_pa_f1:
            best_pa_f1 = m["f1"]

    elapsed = time.time() - t0

    result = {
        "dataset": dataset_name,
        "channel": chan_name,
        "variant": variant,
        "seed": seed,
        "epochs": epochs,
        "elapsed_sec": round(elapsed, 2),
        "threshold": float(evt_th),
        "pa_f1": m_pa.get("f1", 0.0),
        "pa_precision": m_pa.get("precision", 0.0),
        "pa_recall": m_pa.get("recall", 0.0),
        "point_f1": m_pt.get("f1", 0.0),
        "point_precision": m_pt.get("precision", 0.0),
        "point_recall": m_pt.get("recall", 0.0),
        "oracle_pa_f1": float(best_pa_f1),
    }
    print(
        f"[{dataset_name} | {chan_name} | {variant}] -> PA-F1: {result['pa_f1']:.4f}, "
        f"Point-F1: {result['point_f1']:.4f}, Oracle PA-F1: {result['oracle_pa_f1']:.4f} ({elapsed:.1f}s)"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Ablate Potential-Flow JEPA on benchmark channels")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_csv", type=str, default="results/ablations/potential_flow_ablation_summary.csv")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Running Potential-Flow JEPA ablations on device: {device}")

    # Channels to ablate
    test_channels = [
        ("Daphnet", "S01R01E1", ROOT / "mTSBench_data/Daphnet/Daphnet_S01R01E1_train.csv", ROOT / "mTSBench_data/Daphnet/Daphnet_S01R01E1_test.csv"),
        ("Daphnet", "S02R01E0", ROOT / "mTSBench_data/Daphnet/Daphnet_S02R01E0_train.csv", ROOT / "mTSBench_data/Daphnet/Daphnet_S02R01E0_test.csv"),
        ("OPPORTUNITY", "S1-ADL2", ROOT / "mTSBench_data/OPPORTUNITY/OPPORTUNITY_S1-ADL2_train.csv", ROOT / "mTSBench_data/OPPORTUNITY/OPPORTUNITY_S1-ADL2_test.csv"),
        ("room-occupancy", "1", ROOT / "mTSBench_data/room-occupancy/room-occupancy_1_train.csv", ROOT / "mTSBench_data/room-occupancy/room-occupancy_1_test.csv"),
    ]

    variants = [
        "ts_jepa_base",
        "ts_jepa_flow",
        "pf_jepa_midpoint",
        "pf_jepa_curvature",
        "pf_jepa_regime",
    ]

    all_results = []
    for ds_name, ch_name, train_p, test_p in test_channels:
        if not train_p.exists() or not test_p.exists():
            print(f"Skipping {ds_name} {ch_name} (data files not found).")
            continue
        print(f"\n==================== DATASET: {ds_name} | CHANNEL: {ch_name} ====================")
        for var in variants:
            res = train_and_eval_variant(
                variant=var,
                dataset_name=ds_name,
                chan_name=ch_name,
                train_path=train_p,
                test_path=test_p,
                epochs=args.epochs,
                device=device,
            )
            all_results.append(res)

    df_res = pd.DataFrame(all_results)
    out_path = ROOT / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_path, index=False)
    print(f"\nSaved ablation summary to {out_path}")

    # Pivot table display
    pivot_pa = df_res.pivot(index=["dataset", "channel"], columns="variant", values="pa_f1")
    print("\n--- Summary: Point-Adjusted F1 (PA-F1) by Variant ---")
    print(pivot_pa.to_string())

    pivot_pt = df_res.pivot(index=["dataset", "channel"], columns="variant", values="point_f1")
    print("\n--- Summary: Point-level F1 (Point-F1) by Variant ---")
    print(pivot_pt.to_string())


if __name__ == "__main__":
    main()
