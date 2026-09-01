"""Systematic Architectural Ablation Runner for TS-JEPA.

Compares:
1. ts_jepa_base        : Baseline TCN TS-JEPA (Global temporal+statistical pooling -> R^32)
2. ts_jepa_patch       : Patch-Level Sequence JEPA (Tokenized patches -> Sequence Transformer)
3. ts_jepa_multiscale  : Multi-Horizon Hierarchical JEPA (Simultaneous multi-horizon forecasting S in {16, 64})
4. ts_jepa_gat         : Relational Spatial-Temporal Graph Attention JEPA (Cross-channel topological modeling)

Evaluated with identical 50-epoch training, Cosine Annealing, and dense EVT calibration.
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
    MultiScaleTSJEPA,
    PatchFlowJEPA,
    PatchTSJEPA,
    RelationalGAT_JEPAModel,
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
    positive_robust_z,
    robust_stats,
)

DEFAULT_DATASET_CHANNELS = {
    "Daphnet": ["S01R01E1", "S02R01E0", "S02R02E0", "S03R01E0", "S03R01E1", "S03R02E0"],
    "SMAP": ["A-1", "A-2", "A-7", "D-3", "P-3", "E-1"],
    "Exathlon": ["10_2_1000000_67", "10_3_1000000_75", "10_4_1000000_79", "1_2_100000_68", "1_4_1000000_80", "1_5_1000000_86"],
    "room-occupancy": ["default", "1"],
    "OPPORTUNITY": ["S1-ADL2", "S1-ADL3", "S1-ADL4", "S1-ADL5", "S2-ADL1", "S2-ADL2"],
}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_score_ablation(
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
    risk_level: float = 1e-3,
    use_mahalanobis: bool = True,
    device: torch.device = torch.device("cpu"),
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
    # Variant 1: Base TCN TS-JEPA
    # =========================================================================
    if variant == "ts_jepa_base":
        base_enc = HybridTCNEncoder(input_dim=input_dim, latent_dim=32, filters=48, tcn_layers=6, dropout=0.20)
        model = TSJEPAModel(context_encoder=base_enc, latent_dim=32, predictor_hidden_dim=64, ema_decay=0.996).to(device)
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

    # =========================================================================
    # Variant 2: Patch-Level Sequence JEPA (Idea 1)
    # =========================================================================
    elif variant == "ts_jepa_patch":
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
                for i in range(0, len(arr), 1024):
                    ctx = torch.from_numpy(arr[i : i + 1024, :context_size]).float().to(device)
                    tgt = torch.from_numpy(arr[i : i + 1024, context_size:]).float().to(device)
                    disc = model.compute_predictive_discrepancy(ctx, tgt, use_mahalanobis=use_mahalanobis).cpu().numpy()
                    res.append(disc)
            return np.concatenate(res, axis=0)

    # =========================================================================
    # Variant 3: Multi-Horizon Hierarchical JEPA (Idea 2)
    # =========================================================================
    elif variant == "ts_jepa_multiscale":
        model = MultiScaleTSJEPA(
            input_dim=input_dim,
            latent_dim=32,
            horizons=(16, 64),
            filters=48,
            tcn_layers=6,
            dropout=0.20,
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
                z_ctx, z_tgt_true_dict, z_tgt_pred_dict = model(ctx, tgt)
                loss = model.compute_multiscale_loss(z_tgt_pred_dict, z_tgt_true_dict, z_ctx=z_ctx, cov_weight=0.5)

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
                for i in range(0, len(arr), 2048):
                    ctx = torch.from_numpy(arr[i : i + 2048, :context_size]).float().to(device)
                    tgt = torch.from_numpy(arr[i : i + 2048, context_size:]).float().to(device)
                    disc = model.compute_predictive_discrepancy(ctx, tgt, use_mahalanobis=use_mahalanobis).cpu().numpy()
                    res.append(disc)
            return np.concatenate(res, axis=0)

    # =========================================================================
    # Variant 4: Relational Graph Attention JEPA (Idea 3)
    # =========================================================================
    elif variant == "ts_jepa_gat":
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
                for i in range(0, len(arr), 2048):
                    ctx = torch.from_numpy(arr[i : i + 2048, :context_size]).float().to(device)
                    tgt = torch.from_numpy(arr[i : i + 2048, context_size:]).float().to(device)
                    disc = model.compute_predictive_discrepancy(ctx, tgt, use_mahalanobis=use_mahalanobis).cpu().numpy()
                    res.append(disc)
            return np.concatenate(res, axis=0)

    # =========================================================================
    # Variant 5: Conditional Flow Matching TS-JEPA (FlowTSJEPA)
    # =========================================================================
    elif variant == "ts_jepa_flow":
        base_enc = HybridTCNEncoder(input_dim=input_dim, latent_dim=32, filters=48, tcn_layers=6, dropout=0.20)
        model = FlowTSJEPA(
            context_encoder=base_enc,
            latent_dim=32,
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

    # =========================================================================
    # Variant 6: Patch Sequence Flow Matching JEPA (PatchFlowJEPA)
    # =========================================================================
    elif variant == "ts_jepa_patch_flow":
        patch_size = 16
        n_tgt_patches = suspect_size // patch_size
        model = PatchFlowJEPA(
            input_dim=input_dim,
            patch_size=patch_size,
            d_model=48,
            n_heads=4,
            n_layers=2,
            d_ff=96,
            n_target_patches=n_tgt_patches,
            predictor_layers=2,
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
                h_ctx, z_tgt_true, v_pred, v_target = model(ctx, tgt)
                loss, _ = flow_matching_vicreg_loss(
                    v_pred=v_pred.reshape(-1, 48),
                    v_target=v_target.reshape(-1, 48),
                    z_ctx=h_ctx.reshape(-1, 48),
                    z_tgt_true=z_tgt_true.reshape(-1, 48),
                    cov_weight=0.5,
                )

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
                for i in range(0, len(arr), 1024):
                    ctx = torch.from_numpy(arr[i : i + 1024, :context_size]).float().to(device)
                    tgt = torch.from_numpy(arr[i : i + 1024, context_size:]).float().to(device)
                    disc = model.compute_predictive_discrepancy(ctx, tgt, use_mahalanobis=use_mahalanobis).cpu().numpy()
                    res.append(disc)
            return np.concatenate(res, axis=0)

    else:
        raise ValueError(f"Unknown ablation variant: {variant}")

    # =========================================================================
    # Dense EVT Calibration & Evaluation
    # =========================================================================
    train_windows_dense = DataLoader.create_windows(train_scaled, window_size, step=1)
    train_disc_dense = compute_discrepancy(train_windows_dense)
    disc_stats = robust_stats(train_disc_dense)

    test_disc = compute_discrepancy(test_windows)
    disc_z = positive_robust_z(test_disc, disc_stats, clip=20.0)
    train_disc_z = positive_robust_z(train_disc_dense, disc_stats, clip=20.0)

    win_scores = disc_z
    train_win_scores = train_disc_z

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
        n_points=len(train_df),
        context_size=context_size,
        suspect_size=suspect_size,
        step=1,
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

    elapsed = time.time() - t0

    return {
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


def discover_dataset_series(
    dataset_name: str,
    data_root: Path = ROOT / "mTSBench_data",
    channels_filter: Optional[List[str]] = None,
    all_channels: bool = True,
    max_channels: Optional[int] = None,
) -> List[Tuple[str, Path, Path]]:
    """Discover all valid (train, test) series pairs in a dataset directory."""
    ds_dir = data_root / dataset_name
    if not ds_dir.exists():
        return []

    # If specific channels requested
    if channels_filter:
        pairs = []
        for chan in channels_filter:
            if chan == "default":
                train_p = ds_dir / f"{dataset_name}_train.csv"
                test_p = ds_dir / f"{dataset_name}_test.csv"
            else:
                train_p = ds_dir / f"{dataset_name}_{chan}_train.csv"
                test_p = ds_dir / f"{dataset_name}_{chan}_test.csv"
            if train_p.exists() and test_p.exists():
                pairs.append((chan, train_p, test_p))
        return pairs

    # If using curated preset subset
    if not all_channels and dataset_name in DEFAULT_DATASET_CHANNELS:
        pairs = []
        for chan in DEFAULT_DATASET_CHANNELS[dataset_name]:
            if chan == "default":
                train_p = ds_dir / f"{dataset_name}_train.csv"
                test_p = ds_dir / f"{dataset_name}_test.csv"
            else:
                train_p = ds_dir / f"{dataset_name}_{chan}_train.csv"
                test_p = ds_dir / f"{dataset_name}_{chan}_test.csv"
            if train_p.exists() and test_p.exists():
                pairs.append((chan, train_p, test_p))
        return pairs

    # Full automatic discovery of ALL series in the dataset directory
    pairs = []
    train_files = sorted(list(ds_dir.glob("*_train.csv")))
    for tf in train_files:
        test_f = tf.parent / tf.name.replace("_train.csv", "_test.csv")
        if test_f.exists():
            # Extract channel / series name
            prefix = f"{dataset_name}_"
            raw_name = tf.name.replace("_train.csv", "")
            if raw_name.startswith(prefix):
                chan_name = raw_name[len(prefix):]
            elif raw_name == dataset_name:
                chan_name = "default"
            else:
                chan_name = raw_name
            pairs.append((chan_name, tf, test_f))

    if max_channels is not None and max_channels > 0:
        pairs = pairs[:max_channels]

    return pairs


def get_all_available_datasets(data_root: Path = ROOT / "mTSBench_data") -> List[str]:
    """Get all dataset directory names."""
    return sorted([d.name for d in data_root.iterdir() if d.is_dir() and not d.name.startswith(".")])


def main():
    parser = argparse.ArgumentParser(description="TS-JEPA Architectural Ablation Runner")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["ts_jepa_base", "ts_jepa_patch", "ts_jepa_flow", "ts_jepa_patch_flow"],
        help="Ablation variants to evaluate",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["Daphnet"],
        help="Datasets to evaluate (or 'all' for all 19 datasets in mTSBench_data)",
    )
    parser.add_argument("--channels", nargs="*", default=None, help="Specific channels/series to evaluate")
    parser.add_argument(
        "--all_channels",
        action="store_true",
        default=True,
        help="Evaluate ALL channels in each dataset (defaults to True, discovering all 348 series)",
    )
    parser.add_argument(
        "--quick_subset",
        action="store_true",
        default=False,
        help="Evaluate only the curated 6-channel preset per dataset",
    )
    parser.add_argument(
        "--max_channels_per_dataset",
        type=int,
        default=None,
        help="Cap the number of channels evaluated per dataset",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs per channel")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Random seeds")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_csv", type=str, default="reports/ts_jepa_ablation_results.csv")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device == "auto" else args.device if args.device != "auto" else "cpu"
    )

    if "all" in args.dataset:
        datasets_to_run = get_all_available_datasets()
    else:
        datasets_to_run = args.dataset

    run_all_chans = args.all_channels and not args.quick_subset

    print("=" * 100)
    print("TS-JEPA ARCHITECTURAL ABLATION BENCHMARK")
    print(f"Variants: {args.variants}")
    print(f"Datasets: {datasets_to_run}")
    print(f"Mode: {'ALL CHANNELS (Comprehensive 348 Series)' if run_all_chans else 'Curated Subset'}")
    print(f"Seeds: {args.seeds} | Epochs: {args.epochs} | Device: {device}")
    print("=" * 100)

    all_results = []

    for seed in args.seeds:
        for ds_name in datasets_to_run:
            series_pairs = discover_dataset_series(
                dataset_name=ds_name,
                channels_filter=args.channels,
                all_channels=run_all_chans,
                max_channels=args.max_channels_per_dataset,
            )

            if not series_pairs:
                print(f"Skipping {ds_name} (no valid train/test series found)")
                continue

            print("\n=======================================================")
            print(f"Ablation on {ds_name} ({len(series_pairs)} series pairs) [Seed {seed}]")
            print("=======================================================")

            for chan, train_p, test_p in series_pairs:
                for var in args.variants:
                    print(f"  [{var:20s}] {ds_name:12s} - {chan:20s} ... ", end="", flush=True)
                    try:
                        res = train_and_score_ablation(
                            variant=var,
                            dataset_name=ds_name,
                            chan_name=chan,
                            train_path=train_p,
                            test_path=test_p,
                            seed=seed,
                            epochs=args.epochs,
                            device=device,
                        )
                        all_results.append(res)
                        print(
                            f"PA-F1: {res['pa_f1']:.4f} (P: {res['pa_precision']:.4f}, R: {res['pa_recall']:.4f}) | "
                            f"Point-F1: {res['point_f1']:.4f} | Oracle: {res['oracle_pa_f1']:.4f} ({res['elapsed_sec']}s)"
                        )
                    except Exception as e:
                        print(f"FAILED: {e}")

    if all_results:
        df = pd.DataFrame(all_results)
        out_p = ROOT / args.output_csv
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        print(f"\n[Saved full ablation results to {out_p}]")

        print("\n" + "=" * 90)
        print("SUMMARY ABLATION MACRO POINT-ADJUSTED F1 (PA-F1)")
        print("=" * 90)
        pivot_pa = df.pivot_table(index="dataset", columns="variant", values="pa_f1", aggfunc=["mean", "std"])
        print(pivot_pa.round(4).to_markdown())

        print("\n" + "=" * 90)
        print("SUMMARY ABLATION MACRO UNADJUSTED POINT-F1")
        print("=" * 90)
        pivot_pt = df.pivot_table(index="dataset", columns="variant", values="point_f1", aggfunc=["mean", "std"])
        print(pivot_pt.round(4).to_markdown())

        print("\n" + "=" * 90)
        print("SUMMARY ABLATION MACRO ORACLE CEILINGS")
        print("=" * 90)
        pivot_orc = df.pivot_table(index="dataset", columns="variant", values="oracle_pa_f1", aggfunc=["mean", "std"])
        print(pivot_orc.round(4).to_markdown())


if __name__ == "__main__":
    main()
