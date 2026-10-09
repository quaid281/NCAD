#!/usr/bin/env python3
"""Run Operator-Entropy Mathematical Validation.

Resolves Reviewer 3.1A:
Directly measures latent rank, singular-value distributions, effective rank (RankMe),
representation covariance, and modal preservation across:
1. Vanilla TS-JEPA (Pure prediction loss, no variance/covariance/entropy)
2. VICReg TS-JEPA (Standard variance + covariance regularization on representations)
3. Representation-Entropy JEPA (Spectral entropy applied to the representation Gram matrix ZZ^T)
4. Operator-Entropy JEPA (Proposed: Spectral entropy applied to the transition operator K)

Outputs:
- Effective Rank of Transition Operator (exp(H(K)))
- Effective Rank of Latent Representation Space (exp(H(Z)))
- Condition Number of Representation Covariance (kappa(C_Z))
- Singular Value Decay / Uniformity
- Anomaly Detection Point-F1 and False Positive Rate (FPR) under EVT
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def compute_effective_rank(matrix: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute RankMe effective rank: exp(-sum(p_i * ln(p_i)))."""
    # matrix: (M, N)
    with torch.no_grad():
        U, S, V = torch.linalg.svd(matrix, full_matrices=False)
        S_sq = S ** 2
        sum_sq = torch.sum(S_sq)
        if sum_sq < eps:
            return 1.0
        p = S_sq / sum_sq
        p = p[p > eps]
        entropy = -torch.sum(p * torch.log(p))
        return float(torch.exp(entropy).item())


def compute_representation_diagnostics(z_tensor: torch.Tensor) -> dict:
    """Compute representation effective rank, condition number, and covariance rank."""
    with torch.no_grad():
        N, D = z_tensor.shape
        z_cent = z_tensor - torch.mean(z_tensor, dim=0, keepdim=True)
        cov = (z_cent.T @ z_cent) / max(N - 1, 1) # (D, D)
        eff_rank_z = compute_effective_rank(z_cent)
        
        # Eigenvalues of covariance
        eigvals = torch.linalg.eigvalsh(cov)
        eigvals = torch.clamp(eigvals, min=1e-8)
        cond_num = float((torch.max(eigvals) / torch.min(eigvals)).item())
        
        # Percentage of variance explained by top 3 components
        total_var = torch.sum(eigvals)
        top3_var = torch.sum(eigvals[-3:]) / max(total_var.item(), 1e-8)
        
        return {
            "eff_rank_z": eff_rank_z,
            "covariance_cond_num": cond_num,
            "top3_var_fraction": float(top3_var.item()),
        }


def run_operator_entropy_comparison(
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
    latent_dim = 32

    variants = [
        "Vanilla_TS-JEPA",
        "VICReg_TS-JEPA",
        "Representation-Entropy_JEPA",
        "Operator-Entropy_JEPA",
    ]

    records = []

    for variant in variants:
        print(f"  Training {variant} on {dataset}:{channel}...")
        
        # Configure model
        if variant == "Vanilla_TS-JEPA":
            cfg = CSMConfig(
                model_type="ts_jepa",
                context_size=context_size,
                suspect_size=suspect_size,
                latent_dim=latent_dim,
                vicreg_var_weight=0.0,
                vicreg_cov_weight=0.0,
            )
            model = build_ts_jepa_model(cfg, input_dim=input_dim, device=device)
        elif variant == "VICReg_TS-JEPA":
            cfg = CSMConfig(
                model_type="ts_jepa",
                context_size=context_size,
                suspect_size=suspect_size,
                latent_dim=latent_dim,
                vicreg_var_weight=1.0,
                vicreg_cov_weight=0.04,
            )
            model = build_ts_jepa_model(cfg, input_dim=input_dim, device=device)
        elif variant == "Representation-Entropy_JEPA":
            # Uses TS-JEPA with spectral entropy on Z
            cfg = CSMConfig(
                model_type="ts_jepa",
                context_size=context_size,
                suspect_size=suspect_size,
                latent_dim=latent_dim,
                vicreg_var_weight=0.0,
                vicreg_cov_weight=0.0,
            )
            model = build_ts_jepa_model(cfg, input_dim=input_dim, device=device)
        elif variant == "Operator-Entropy_JEPA":
            cfg = CSMConfig(
                model_type="operator_entropy_jepa",
                context_size=context_size,
                suspect_size=suspect_size,
                latent_dim=latent_dim,
                vicreg_var_weight=1.0,
                vicreg_cov_weight=0.04,
            )
            model = build_ts_jepa_model(cfg, input_dim=input_dim, device=device)

        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        for ep in range(epochs):
            model.train()
            perm = np.random.permutation(len(training_data))
            for b in range(0, len(perm), 16):
                batch = torch.from_numpy(np.ascontiguousarray(training_data[perm[b:b+16]])).float().to(device)
                ctx = batch[:, :context_size]
                tgt = batch[:, context_size:]

                if variant == "Representation-Entropy_JEPA":
                    # Custom loss: MSE pred + spectral entropy on representation Gram matrix
                    z_ctx = model.context_encoder(ctx)
                    z_pred = model.predictor(z_ctx)
                    z_tgt = model.target_encoder(tgt)
                    mse_loss = F.mse_loss(z_pred, z_tgt)
                    
                    # Spectral entropy on Z_ctx
                    Z = z_ctx - torch.mean(z_ctx, dim=0, keepdim=True)
                    cov_Z = (Z.T @ Z) / max(Z.shape[0] - 1, 1)
                    S_sq = torch.linalg.eigvalsh(cov_Z)
                    S_sq = torch.clamp(S_sq, min=1e-8)
                    p = S_sq / torch.sum(S_sq)
                    entropy_Z = -torch.sum(p * torch.log(p))
                    # Maximize entropy -> minimize negative entropy
                    loss = mse_loss - 0.1 * entropy_Z
                else:
                    loss, _ = model.compute_objective(ctx, tgt, cfg)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                model.update_target_encoder()

        # Evaluate Diagnostics on Validation/Test representations
        model.eval()
        with torch.no_grad():
            sample_batch = torch.from_numpy(np.ascontiguousarray(val_data[:min(500, len(val_data))])).float().to(device)
            z_val = model.context_encoder(sample_batch[:, :context_size])
            rep_diag = compute_representation_diagnostics(z_val)

            # Effective rank of transition operator / predictor
            if hasattr(model, "koopman_operator"):
                K_mat = model.koopman_operator.get_operator()
                eff_rank_op = compute_effective_rank(K_mat)
            elif hasattr(model, "K") and model.K is not None:
                eff_rank_op = compute_effective_rank(model.K)
            elif hasattr(model, "transition_matrix") and model.transition_matrix is not None:
                eff_rank_op = compute_effective_rank(model.transition_matrix)
            elif hasattr(model, "predictor") and model.predictor is not None:
                w = None
                if hasattr(model.predictor, "net"):
                    w = model.predictor.net[0].weight
                elif isinstance(model.predictor, nn.Sequential):
                    w = model.predictor[0].weight
                eff_rank_op = compute_effective_rank(w) if w is not None else float(config.latent_dim)
            else:
                eff_rank_op = float(config.latent_dim)

        # Anomaly scoring and EVT evaluation
        def get_scores(w_arr):
            model.eval()
            res = []
            with torch.no_grad():
                for i in range(0, len(w_arr), 128):
                    chunk = torch.from_numpy(np.ascontiguousarray(w_arr[i:i+128])).float().to(device)
                    disc = model.compute_predictive_discrepancy(chunk[:, :context_size], chunk[:, context_size:])
                    res.append(disc.cpu().numpy())
            return np.concatenate(res, axis=0)

        train_dense = DataLoader.create_windows(train_scaled, window_size, step=1, copy=False)
        train_win_sc = get_scores(train_dense)
        test_win_sc = get_scores(test_windows)
        stats = robust_stats(train_win_sc)
        train_z = positive_robust_z(train_win_sc, stats)
        test_z = positive_robust_z(test_win_sc, stats)

        pt_sc, val_m = aggregate_window_scores(
            test_z,
            n_points=len(test_df),
            context_size=context_size,
            suspect_size=suspect_size,
            step=1,
            reducer="mean",
            mapping_method="smear",
        )
        tr_pt, tr_m = aggregate_window_scores(
            train_z,
            n_points=len(train_df),
            context_size=context_size,
            suspect_size=suspect_size,
            step=1,
            reducer="mean",
            mapping_method="smear",
        )
        pt_sc = moving_average(pt_sc, 5)
        tr_pt = moving_average(tr_pt, 5)
        evt_res = calibrate_evt_threshold(tr_pt[tr_m], risk_level=1e-3, init_percentile=98.0, adaptive_kurtosis=True)
        preds = event_level_filter(pt_sc, evt_res.threshold, val_m, min_run=2, extreme_factor=1.75) * val_m

        m_pt = compute_metrics(test_labels, preds, scores=pt_sc, valid_mask=val_m, use_pa=False)
        nominal_pts = np.sum(test_labels == 0)
        fpr = m_pt["fp"] / max(nominal_pts, 1)

        records.append({
            "dataset": dataset,
            "channel": channel,
            "variant": variant,
            "eff_rank_operator": eff_rank_op,
            "eff_rank_representation": rep_diag["eff_rank_z"],
            "covariance_cond_num": rep_diag["covariance_cond_num"],
            "top3_var_fraction": rep_diag["top3_var_fraction"],
            "point_f1": m_pt.get("f1", 0.0),
            "point_precision": m_pt.get("precision", 0.0),
            "point_recall": m_pt.get("recall", 0.0),
            "false_positives": m_pt["fp"],
            "empirical_fpr": fpr,
        })

    return records


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Operator-Entropy Spectral Validation on device: {device}")

    streams = [
        ("SMAP", "P-3"),
        ("SMD", "machine-1-2"),
        ("MSL", "M-1"),
        ("Daphnet", "S01R01E1"),
        ("GECCO", "water_quality"),
    ]

    all_records = []
    out_csv = ROOT / "reports" / "operator_entropy_spectral_validation.csv"

    for ds, ch in streams:
        print(f"\n=== Evaluating Stream {ds}:{ch} ===")
        res = run_operator_entropy_comparison(ds, ch, device=device, epochs=15)
        all_records.extend(res)
        pd.DataFrame(all_records).to_csv(out_csv, index=False)
        print(f"Saved intermediate records to {out_csv}")

    df = pd.DataFrame(all_records)
    print("\n" + "="*80)
    print("OPERATOR-ENTROPY SPECTRAL DIAGNOSTICS SUMMARY")
    print("="*80)
    summary = df.groupby("variant").agg({
        "eff_rank_operator": "mean",
        "eff_rank_representation": "mean",
        "covariance_cond_num": "mean",
        "top3_var_fraction": "mean",
        "point_f1": "mean",
        "false_positives": "mean",
        "empirical_fpr": "mean",
    }).round(4)
    print(summary.to_string())


if __name__ == "__main__":
    main()
