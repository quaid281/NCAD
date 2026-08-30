"""Unified Rigorous SOTA Benchmarking Suite for Multivariate Time-Series Anomaly Detection.

Implements official training protocols, cosine annealing learning rate schedules,
multi-seed evaluation (mean +/- std), dual metric reporting (PA-F1 and unadjusted Point-F1),
and EVT tail calibration.

Models benchmarked:
1. TS-JEPA (Ours - Joint-Embedding Predictive Architecture with VICReg & Mahalanobis)
2. Anomaly Transformer (ICLR 2022 - Two-Phase Minimax Optimization)
3. TimesNet (ICLR 2023 - 2D Multi-Periodic Inception)
4. DCdetector (KDD 2023 - Dual Attention Multi-scale Contrastive)
5. TranAD (VLDB 2022 - Adversarial Two-Phase Transformer)
6. NCAD-TCN (2021 - Contextual Outlier Exposure Baseline)
"""

from __future__ import annotations

import argparse
import gc
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
    AnomalyInjectionConfig,
    AnomalyTransformer,
    ContextualAnomalyInjector,
    DCdetector,
    HybridTCNEncoder,
    PatchTSJEPA,
    RelationalGAT_JEPAModel,
    TimesNet,
    TranAD,
    TSJEPAModel,
    contrastive_loss,
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


def discover_channels(dataset_name: str, requested_channels: Optional[List[str]] = None) -> List[str]:
    """Dynamically discover channels for a dataset in mTSBench_data."""
    if requested_channels and requested_channels != ["all"]:
        return requested_channels

    ds_dir = ROOT / "mTSBench_data" / dataset_name
    if not ds_dir.exists():
        return []

    train_files = sorted(list(ds_dir.glob("*_train.csv")))
    discovered = []
    for tf in train_files:
        stem = tf.name.replace("_train.csv", "")
        if stem == dataset_name:
            discovered.append("default")
        elif stem.startswith(f"{dataset_name}_"):
            discovered.append(stem[len(dataset_name) + 1 :])
        else:
            discovered.append(stem)

    return discovered


DEFAULT_DATASET_CHANNELS = {
    "Daphnet": ["S01R01E1", "S02R01E0", "S02R02E0", "S03R01E0", "S03R01E1", "S03R02E0"],
    "Exathlon": ["10_2_1000000_67", "10_3_1000000_75", "10_4_1000000_79", "1_2_100000_68", "1_4_1000000_80", "1_5_1000000_86"],
    "SMAP": ["A-1", "A-2", "A-7", "D-3", "P-3", "E-1"],
    "MSL": ["C-1", "C-2", "D-14", "D-15", "D-16", "M-1"],
    "room-occupancy": ["default", "1"],
    "OPPORTUNITY": ["S1-ADL2", "S1-ADL3", "S1-ADL4", "S1-ADL5", "S2-ADL1", "S2-ADL2"],
}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_score_channel(
    model_name: str,
    dataset_name: str,
    chan_name: str,
    train_path: Path,
    test_path: Path,
    seed: int = 42,
    context_size: int = 256,
    suspect_size: int = 64,
    epochs: int = 50,
    batch_size: int = 32,
    risk_level: float = 1e-3,
    use_mahalanobis: bool = True,
    mapping_method: str = "middle",
    device: torch.device = torch.device("cpu"),
) -> dict:
    set_seed(seed)
    window_size = context_size + suspect_size
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 1. Schema and column consistency validation
    numeric_cols_train = [c for c in train_df.columns if c not in ["timestamp", "is_anomaly"]]
    numeric_cols_test = [c for c in test_df.columns if c not in ["timestamp", "is_anomaly"]]
    if numeric_cols_train != numeric_cols_test:
        raise ValueError(f"Train/Test feature column mismatch for {chan_name}: {numeric_cols_train} vs {numeric_cols_test}")

    numeric_cols = numeric_cols_train

    # 2. Timestamp monotonicity check if present
    if "timestamp" in train_df.columns:
        t_train = pd.to_datetime(train_df["timestamp"], errors="coerce") if not np.issubdtype(train_df["timestamp"].dtype, np.number) else train_df["timestamp"]
        if not t_train.is_monotonic_increasing:
            train_df = train_df.sort_values("timestamp").reset_index(drop=True)
    if "timestamp" in test_df.columns:
        t_test = pd.to_datetime(test_df["timestamp"], errors="coerce") if not np.issubdtype(test_df["timestamp"].dtype, np.number) else test_df["timestamp"]
        if not t_test.is_monotonic_increasing:
            test_df = test_df.sort_values("timestamp").reset_index(drop=True)

    # 3. Label extraction & validation
    test_labels = None
    if "is_anomaly" in test_df.columns:
        raw_labels = test_df["is_anomaly"].to_numpy()
        if not np.all(np.isin(raw_labels, [0, 1])):
            raise ValueError(f"Test labels for {chan_name} contain non-binary values: {np.unique(raw_labels)}")
        test_labels = raw_labels.astype(int)

    # 4. Finiteness validation & robust z-score normalization
    train_vals = train_df[numeric_cols].to_numpy(dtype=np.float32)
    test_vals = test_df[numeric_cols].to_numpy(dtype=np.float32)
    if not np.all(np.isfinite(train_vals)):
        raise ValueError(f"Training features for {chan_name} contain non-finite (NaN/Inf) values.")
    if not np.all(np.isfinite(test_vals)):
        raise ValueError(f"Test features for {chan_name} contain non-finite (NaN/Inf) values.")

    train_mean = np.mean(train_vals, axis=0)
    train_std = np.std(train_vals, axis=0)
    scale = np.where(train_std < 1e-8, 1.0, np.maximum(train_std, 0.01))
    train_scaled = (train_vals - train_mean) / scale
    test_scaled = (test_vals - train_mean) / scale

    train_windows = DataLoader.create_windows(train_scaled, window_size, step=10)
    test_windows = DataLoader.create_windows(test_scaled, window_size, step=1)

    training_data, _ = split_train_val(train_windows, val_split=0.1, seed=seed, window_size=window_size, step=10)
    input_dim = len(numeric_cols)
    t0 = time.time()

    # =========================================================================
    # 1. TS-JEPA (Ours)
    # =========================================================================
    if model_name == "ts_jepa":
        base_enc = HybridTCNEncoder(input_dim=input_dim, latent_dim=32, filters=48, tcn_layers=6, dropout=0.20)
        model = TSJEPAModel(context_encoder=base_enc, latent_dim=32, predictor_hidden_dim=64, ema_decay=0.996).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        total_steps = epochs * max(1, len(training_data) // batch_size)
        global_step = 0

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

                # Cosine EMA schedule from 0.996 to 0.9995
                ema_val = 0.996 + (0.9995 - 0.996) * 0.5 * (1.0 - np.cos(np.pi * global_step / total_steps))
                model.update_target_encoder(decay=ema_val)
            scheduler.step()

        if use_mahalanobis:
            ctx_all = torch.from_numpy(train_windows[:, :context_size]).float().to(device)
            tgt_all = torch.from_numpy(train_windows[:, context_size:]).float().to(device)
            model.fit_mahalanobis_covariance(ctx_all, tgt_all)

        def compute_discrepancy(arr):
            model.eval()
            res = []
            with torch.no_grad():
                for i in range(0, len(arr), 256):
                    ctx = torch.from_numpy(arr[i : i + 256, :context_size]).float().to(device)
                    tgt = torch.from_numpy(arr[i : i + 256, context_size:]).float().to(device)
                    disc = model.compute_predictive_discrepancy(ctx, tgt, use_mahalanobis=use_mahalanobis).cpu().numpy()
                    res.append(disc)
            return np.concatenate(res, axis=0)

    # =========================================================================
    # 2. Anomaly Transformer (ICLR 2022 - Official Two-Phase Minimax)
    # =========================================================================
    elif model_name == "anomaly_transformer":
        model = AnomalyTransformer(c_in=input_dim, d_model=48, n_heads=4, e_layers=2, d_ff=96, dropout=0.1).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        for epoch in range(1, epochs + 1):
            model.train()
            perm = np.random.permutation(len(training_data))
            for b in range(0, len(perm), batch_size):
                batch_arr = training_data[perm[b : b + batch_size]]
                x = torch.from_numpy(batch_arr).float().to(device)

                # Minimax Phase 1 (Prior update)
                loss_prior, _ = model.minimax_losses(x, lambda_weight=3.0)
                optimizer.zero_grad(set_to_none=True)
                loss_prior.backward()
                optimizer.step()

                # Minimax Phase 2 (Series update)
                _, loss_series = model.minimax_losses(x, lambda_weight=3.0)
                optimizer.zero_grad(set_to_none=True)
                loss_series.backward()
                optimizer.step()
            scheduler.step()

        def compute_discrepancy(arr):
            model.eval()
            res = []
            with torch.no_grad():
                for i in range(0, len(arr), 256):
                    x = torch.from_numpy(arr[i : i + 256]).float().to(device)
                    scores = model.compute_anomaly_scores(x)
                    res.append(scores[:, context_size:].mean(dim=-1).cpu().numpy())
            return np.concatenate(res, axis=0)

    # =========================================================================
    # 3. TimesNet (ICLR 2023 - 2D Inception)
    # =========================================================================
    elif model_name == "timesnet":
        model = TimesNet(c_in=input_dim, d_model=48, d_ff=48, e_layers=2, top_k=3, dropout=0.1).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        for epoch in range(1, epochs + 1):
            model.train()
            perm = np.random.permutation(len(training_data))
            for b in range(0, len(perm), batch_size):
                batch_arr = training_data[perm[b : b + batch_size]]
                x = torch.from_numpy(batch_arr).float().to(device)
                rec = model(x)
                loss = torch.mean((rec - x) ** 2)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            scheduler.step()

        def compute_discrepancy(arr):
            model.eval()
            res = []
            with torch.no_grad():
                for i in range(0, len(arr), 256):
                    x = torch.from_numpy(arr[i : i + 256]).float().to(device)
                    scores = model.compute_anomaly_scores(x)
                    res.append(scores[:, context_size:].mean(dim=-1).cpu().numpy())
            return np.concatenate(res, axis=0)

    # =========================================================================
    # 4. DCdetector (KDD 2023 - Multi-scale Contrastive)
    # =========================================================================
    elif model_name == "dcdetector":
        model = DCdetector(c_in=input_dim, patch_size1=8, patch_size2=16, d_model=48, n_heads=4, e_layers=2).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        for epoch in range(1, epochs + 1):
            model.train()
            perm = np.random.permutation(len(training_data))
            for b in range(0, len(perm), batch_size):
                batch_arr = training_data[perm[b : b + batch_size]]
                x = torch.from_numpy(batch_arr).float().to(device)
                z1, z2 = model(x)
                loss = model.contrastive_loss(z1, z2)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            scheduler.step()

        def compute_discrepancy(arr):
            model.eval()
            res = []
            with torch.no_grad():
                for i in range(0, len(arr), 256):
                    x = torch.from_numpy(arr[i : i + 256]).float().to(device)
                    scores = model.compute_anomaly_scores(x)
                    res.append(scores[:, context_size:].mean(dim=-1).cpu().numpy())
            return np.concatenate(res, axis=0)

    # =========================================================================
    # 5. TranAD (VLDB 2022 - Two-Phase Adversarial)
    # =========================================================================
    elif model_name == "tranad":
        model = TranAD(c_in=input_dim, d_model=48, n_heads=4, e_layers=2, d_layers=2, d_ff=96).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        for epoch in range(1, epochs + 1):
            model.train()
            perm = np.random.permutation(len(training_data))
            for b in range(0, len(perm), batch_size):
                batch_arr = training_data[perm[b : b + batch_size]]
                x = torch.from_numpy(batch_arr).float().to(device)
                rec1, rec2 = model(x)
                l1, l2 = model.adversarial_loss(rec1, rec2, x, epoch=epoch)
                loss = l1 + l2

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            scheduler.step()

        def compute_discrepancy(arr):
            model.eval()
            res = []
            with torch.no_grad():
                for i in range(0, len(arr), 256):
                    x = torch.from_numpy(arr[i : i + 256]).float().to(device)
                    scores = model.compute_anomaly_scores(x)
                    res.append(scores[:, context_size:].mean(dim=-1).cpu().numpy())
            return np.concatenate(res, axis=0)

    # =========================================================================
    # 6. NCAD-TCN (2021 - Contextual Outlier Exposure Baseline)
    # =========================================================================
    elif model_name == "ncad":
        model = HybridTCNEncoder(input_dim=input_dim, latent_dim=32, filters=48, tcn_layers=3, dropout=0.20).to(device)
        injector = ContextualAnomalyInjector(
            AnomalyInjectionConfig(injection_ratio=0.5, min_anomaly_len=16, max_anomaly_len=64),
            seed=seed,
        )
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        for epoch in range(1, epochs + 1):
            model.train()
            perm = np.random.permutation(len(training_data))
            for b in range(0, len(perm), batch_size):
                clean = training_data[perm[b : b + batch_size]]
                injected, lbl = injector.inject_batch(clean, context_size)
                z_full = model(torch.from_numpy(injected).float().to(device))
                z_ctx = model(torch.from_numpy(clean[:, :context_size]).float().to(device))
                loss = contrastive_loss(z_full, z_ctx, torch.from_numpy(lbl).float().to(device))

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            scheduler.step()

        def compute_discrepancy(arr):
            model.eval()
            res = []
            with torch.no_grad():
                for i in range(0, len(arr), 256):
                    w = torch.from_numpy(arr[i : i + 256]).float().to(device)
                    ctx = torch.from_numpy(arr[i : i + 256, :context_size]).float().to(device)
                    z_w = model(w)
                    z_c = model(ctx)
                    disc = torch.linalg.norm(z_w - z_c, dim=-1).cpu().numpy()
                    res.append(disc)
            return np.concatenate(res, axis=0)

    # =========================================================================
    # 7. Patch-Level Sequence JEPA (Idea 1)
    # =========================================================================
    elif model_name in ["patch_ts_jepa", "patch_jepa"]:
        patch_size = 16
        n_tgt_patches = suspect_size // patch_size
        model = PatchTSJEPA(
            input_dim=input_dim,
            patch_size=patch_size,
            d_model=48,
            n_heads=4,
            n_layers=2,
            d_ff=96,
            n_target_patches=n_tgt_patches,
            ema_decay=0.996,
            dropout=0.10,
        ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        total_steps = epochs * max(1, len(training_data) // batch_size)
        global_step = 0

        for epoch in range(1, epochs + 1):
            model.train()
            perm = np.random.permutation(len(training_data))
            for b in range(0, len(perm), batch_size):
                global_step += 1
                batch_arr = training_data[perm[b : b + batch_size]]
                ctx = torch.from_numpy(batch_arr[:, :context_size]).float().to(device)
                tgt = torch.from_numpy(batch_arr[:, context_size:]).float().to(device)
                h_ctx, h_tgt_true, h_tgt_pred = model(ctx, tgt)
                loss = model.compute_patch_loss(h_tgt_pred, h_tgt_true, h_ctx=h_ctx, cov_weight=0.5)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                ema_val = 0.996 + (0.9995 - 0.996) * 0.5 * (1.0 - np.cos(np.pi * global_step / total_steps))
                model.update_target_encoder(decay=ema_val)
            scheduler.step()

        if use_mahalanobis:
            ctx_all = torch.from_numpy(train_windows[:, :context_size]).float().to(device)
            tgt_all = torch.from_numpy(train_windows[:, context_size:]).float().to(device)
            model.fit_mahalanobis_covariance(ctx_all, tgt_all)

        def compute_discrepancy(arr):
            model.eval()
            res = []
            with torch.no_grad():
                for i in range(0, len(arr), 256):
                    ctx = torch.from_numpy(arr[i : i + 256, :context_size]).float().to(device)
                    tgt = torch.from_numpy(arr[i : i + 256, context_size:]).float().to(device)
                    disc = model.compute_predictive_discrepancy(ctx, tgt, use_mahalanobis=use_mahalanobis).cpu().numpy()
                    res.append(disc)
            return np.concatenate(res, axis=0)

    # =========================================================================
    # 8. Relational Graph Attention JEPA (Idea 3)
    # =========================================================================
    elif model_name in ["gat_jepa", "relational_gat_jepa"]:
        model = RelationalGAT_JEPAModel(
            input_dim=input_dim,
            latent_dim=32,
            filters=48,
            tcn_layers=3,
            gat_layers=2,
            gat_heads=4,
            dropout=0.20,
            ema_decay=0.996,
        ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        total_steps = epochs * max(1, len(training_data) // batch_size)
        global_step = 0

        for epoch in range(1, epochs + 1):
            model.train()
            perm = np.random.permutation(len(training_data))
            for b in range(0, len(perm), batch_size):
                global_step += 1
                batch_arr = training_data[perm[b : b + batch_size]]
                ctx = torch.from_numpy(batch_arr[:, :context_size]).float().to(device)
                tgt = torch.from_numpy(batch_arr[:, context_size:]).float().to(device)
                loss = model.compute_loss(ctx, tgt, cov_weight=0.5)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                ema_val = 0.996 + (0.9995 - 0.996) * 0.5 * (1.0 - np.cos(np.pi * global_step / total_steps))
                model.update_target_encoder(decay=ema_val)
            scheduler.step()

        if use_mahalanobis:
            ctx_all = torch.from_numpy(train_windows[:, :context_size]).float().to(device)
            tgt_all = torch.from_numpy(train_windows[:, context_size:]).float().to(device)
            model.fit_mahalanobis_covariance(ctx_all, tgt_all)

        def compute_discrepancy(arr):
            model.eval()
            res = []
            with torch.no_grad():
                for i in range(0, len(arr), 256):
                    ctx = torch.from_numpy(arr[i : i + 256, :context_size]).float().to(device)
                    tgt = torch.from_numpy(arr[i : i + 256, context_size:]).float().to(device)
                    disc = model.compute_predictive_discrepancy(ctx, tgt, use_mahalanobis=use_mahalanobis).cpu().numpy()
                    res.append(disc)
            return np.concatenate(res, axis=0)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # =========================================================================
    # Discrepancy Normalization & EVT Calibration
    # =========================================================================
    train_windows_dense = DataLoader.create_windows(train_scaled, window_size, step=1)
    train_disc_dense = compute_discrepancy(train_windows_dense)
    disc_stats = robust_stats(train_disc_dense)

    test_disc = compute_discrepancy(test_windows)

    # Fit EVT on unclipped robust normalized scores to avoid artificial mass at 20.0
    disc_z = positive_robust_z(test_disc, disc_stats, clip=None)
    train_disc_z = positive_robust_z(train_disc_dense, disc_stats, clip=None)

    win_scores = disc_z
    train_win_scores = train_disc_z

    # Aggregate window scores to points (step=1 dense)
    pt_scores, valid_mask = aggregate_window_scores(
        win_scores,
        n_points=len(test_df),
        context_size=context_size,
        suspect_size=suspect_size,
        step=1,
        reducer="mean",
        mapping_method=mapping_method,
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
        mapping_method=mapping_method,
    )
    train_smoothed = moving_average(train_pt_scores, 12)
    train_valid_scores = train_smoothed[train_valid_mask]

    # EVT Extreme Value Theory Tail Calibration
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

    # =========================================================================
    # Explicit CUDA Cleanup and Memory Release
    # =========================================================================
    del model, optimizer
    if "scheduler" in locals():
        del scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "dataset": dataset_name,
        "channel": chan_name,
        "model": model_name,
        "seed": seed,
        "epochs": epochs,
        "elapsed_sec": round(elapsed, 2),
        "threshold": float(evt_th),
        "point_f1": m_pt.get("f1", 0.0),
        "point_precision": m_pt.get("precision", 0.0),
        "point_recall": m_pt.get("recall", 0.0),
        "tp": m_pt.get("tp", 0),
        "fp": m_pt.get("fp", 0),
        "fn": m_pt.get("fn", 0),
        "tn": m_pt.get("tn", 0),
        "pa_f1": m_pa.get("f1", 0.0),
        "pa_precision": m_pa.get("precision", 0.0),
        "pa_recall": m_pa.get("recall", 0.0),
        "oracle_pa_f1": float(best_pa_f1),
    }


def main():
    parser = argparse.ArgumentParser(description="Unified Rigorous SOTA Benchmark Suite")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ts_jepa", "anomaly_transformer", "timesnet", "dcdetector", "tranad", "ncad"],
        help="Models: ts_jepa, patch_ts_jepa, anomaly_transformer, timesnet, dcdetector, tranad, ncad",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["all"],
        help="Datasets: all, Daphnet, Exathlon, SMAP, MSL, SMD, room-occupancy, OPPORTUNITY, etc.",
    )
    parser.add_argument("--channels", nargs="*", default=None, help="Specific channels or 'all'")
    parser.add_argument("--subset", action="store_true", help="Run only the 6-channel development subset per dataset")
    parser.add_argument("--causal", action="store_true", help="Use strict causal trailing window alignment (zero lookahead)")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size for GPU training (default 64)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default 50)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Random seeds to evaluate")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_csv", type=str, default="reports/sota_baseline_comparison.csv")
    args = parser.parse_args()

    mapping_method = "trailing" if args.causal else "middle"
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device == "auto" else args.device if args.device != "auto" else "cpu"
    )
    all_available = [p.name for p in (ROOT / "mTSBench_data").iterdir() if p.is_dir() and not p.name.startswith(".")]
    datasets_to_run = all_available if "all" in args.dataset else args.dataset
    out_p = ROOT / args.output_csv
    out_p.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("RIGOROUS SOTA BENCHMARK EVALUATION (2021-2024)")
    print(f"Models: {args.models}")
    print(f"Datasets: {datasets_to_run} | Seeds: {args.seeds} | Epochs: {args.epochs} | Batch Size: {args.batch_size} | Causal: {args.causal} | Device: {device}")
    print(f"Output File: {out_p}")
    print("=" * 100)

    all_results = []

    for seed in args.seeds:
        print(f"\n>>> Running Evaluation for Seed {seed} <<<")
        for ds_name in datasets_to_run:
            if args.subset and ds_name in DEFAULT_DATASET_CHANNELS:
                channels = DEFAULT_DATASET_CHANNELS[ds_name]
            else:
                channels = discover_channels(ds_name, args.channels)

            if not channels:
                continue

            ds_dir = ROOT / "mTSBench_data" / ds_name
            print(f"\n=======================================================")
            print(f"Evaluating {ds_name} ({len(channels)} channels) [Seed {seed}]")
            print(f"=======================================================")

            for chan in channels:
                if chan == "default":
                    train_p = ds_dir / f"{ds_name}_train.csv"
                    test_p = ds_dir / f"{ds_name}_test.csv"
                else:
                    train_p = ds_dir / f"{ds_name}_{chan}_train.csv"
                    test_p = ds_dir / f"{ds_name}_{chan}_test.csv"

                if not train_p.exists() or not test_p.exists():
                    print(f"  [SKIPPED] Missing {train_p.name}")
                    continue

                for model_name in args.models:
                    print(f"  [{model_name:19s}] Chan: {chan:16s} ... ", end="", flush=True)
                    try:
                        res = train_and_score_channel(
                            model_name=model_name,
                            dataset_name=ds_name,
                            chan_name=chan,
                            train_path=train_p,
                            test_path=test_p,
                            seed=seed,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            mapping_method=mapping_method,
                            device=device,
                        )
                        all_results.append(res)

                        # Incremental CSV saving after every single channel/model run
                        df_incremental = pd.DataFrame([res])
                        if not out_p.exists():
                            df_incremental.to_csv(out_p, index=False)
                        else:
                            df_incremental.to_csv(out_p, mode="a", header=False, index=False)

                        print(
                            f"Point-F1: {res['point_f1']:.4f} (P: {res['point_precision']:.4f}, R: {res['point_recall']:.4f}, TP: {res['tp']}, FP: {res['fp']}) | "
                            f"PA-F1: {res['pa_f1']:.4f} | Oracle: {res['oracle_pa_f1']:.4f} ({res['elapsed_sec']}s)"
                        )
                    except Exception as e:
                        print(f"FAILED: {e}")
                    finally:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()

    if all_results:
        df = pd.DataFrame(all_results)
        out_p = ROOT / args.output_csv
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        print(f"\n[Saved full results to {out_p}]")

        print("\n" + "=" * 90)
        print("PRIMARY REAL-WORLD METRICS: UNADJUSTED POINT-F1 (MEAN +/- STD)")
        print("=" * 90)
        pivot_pt = df.pivot_table(index="dataset", columns="model", values="point_f1", aggfunc=["mean", "std"])
        print(pivot_pt.round(4).to_markdown())

        print("\n" + "=" * 90)
        print("UNADJUSTED POINT PRECISION (MEAN +/- STD)")
        print("=" * 90)
        pivot_prec = df.pivot_table(index="dataset", columns="model", values="point_precision", aggfunc=["mean", "std"])
        print(pivot_prec.round(4).to_markdown())

        print("\n" + "=" * 90)
        print("UNADJUSTED POINT RECALL (MEAN +/- STD)")
        print("=" * 90)
        pivot_rec = df.pivot_table(index="dataset", columns="model", values="point_recall", aggfunc=["mean", "std"])
        print(pivot_rec.round(4).to_markdown())

        print("\n" + "=" * 90)
        print("SECONDARY METRICS: POINT-ADJUSTED F1 (PA-F1) FOR LITERATURE COMPATIBILITY")
        print("=" * 90)
        pivot_pa = df.pivot_table(index="dataset", columns="model", values="pa_f1", aggfunc=["mean", "std"])
        print(pivot_pa.round(4).to_markdown())


if __name__ == "__main__":
    main()
