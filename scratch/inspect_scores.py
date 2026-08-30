"""Script to inspect score distributions and the threshold gap."""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import PatchFlowJEPA, FlowTSJEPA, flow_matching_vicreg_loss
from src.data.data_loader import DataLoader
from src.utils.event_fusion import (
    aggregate_window_scores,
    calibrate_evt_threshold,
    compute_metrics,
    event_level_filter,
    moving_average,
    positive_robust_z,
    robust_stats,
)
from src.models.train_model import split_train_validation as split_train_val

def inspect_score_distribution():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_dir = ROOT / "mTSBench_data" / "Daphnet"
    chan = "S02R01E0"
    train_path = ds_dir / f"Daphnet_{chan}_train.csv"
    test_path = ds_dir / f"Daphnet_{chan}_test.csv"

    context_size = 256
    suspect_size = 64
    window_size = context_size + suspect_size

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    numeric_cols = [c for c in train_df.columns if c not in ["timestamp", "is_anomaly"]]
    test_labels = test_df["is_anomaly"].to_numpy().astype(int)

    train_mean = train_df[numeric_cols].mean().to_numpy()
    train_std = train_df[numeric_cols].std().to_numpy()
    scale = np.maximum(train_std, 0.01)
    train_scaled = (train_df[numeric_cols].to_numpy() - train_mean) / scale
    test_scaled = (test_df[numeric_cols].to_numpy() - train_mean) / scale

    train_windows = DataLoader.create_windows(train_scaled, window_size, step=10)
    test_windows = DataLoader.create_windows(test_scaled, window_size, step=1)
    training_data, _ = split_train_val(train_windows, val_split=0.1, seed=42)

    patch_size = 16
    n_tgt_patches = suspect_size // patch_size
    model = PatchFlowJEPA(
        input_dim=len(numeric_cols),
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 30
    batch_size = 32
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

    def compute_discrepancy(arr):
        model.eval()
        res = []
        with torch.no_grad():
            for i in range(0, len(arr), 1024):
                ctx = torch.from_numpy(arr[i : i + 1024, :context_size]).float().to(device)
                tgt = torch.from_numpy(arr[i : i + 1024, context_size:]).float().to(device)
                _, win_disc = model.compute_patch_instantaneous_discrepancy(ctx, tgt, n_eval_times=3)
                res.append(win_disc.cpu().numpy())
        return np.concatenate(res, axis=0)

    train_windows_dense = DataLoader.create_windows(train_scaled, window_size, step=1)
    train_disc_dense = compute_discrepancy(train_windows_dense)
    disc_stats = robust_stats(train_disc_dense)
    print(f"Train Raw Discrepancy -> min: {np.min(train_disc_dense):.4f}, median: {disc_stats.median:.4f}, iqr: {disc_stats.iqr:.4f}, max: {np.max(train_disc_dense):.4f}")

    test_disc = compute_discrepancy(test_windows)
    print(f"Test Raw Discrepancy  -> min: {np.min(test_disc):.4f}, median: {np.median(test_disc):.4f}, max: {np.max(test_disc):.4f}")

    disc_z = positive_robust_z(test_disc, disc_stats, clip=20.0)
    train_disc_z = positive_robust_z(train_disc_dense, disc_stats, clip=20.0)

    pt_scores, valid_mask = aggregate_window_scores(
        disc_z,
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
        train_disc_z,
        n_points=len(train_df),
        context_size=context_size,
        suspect_size=suspect_size,
        step=1,
        reducer="mean",
        mapping_method="middle",
    )
    train_smoothed = moving_average(train_pt_scores, 12)
    train_valid_scores = train_smoothed[train_valid_mask]

    print(f"Train Valid Smoothed Scores -> min: {np.min(train_valid_scores):.4f}, 50%: {np.median(train_valid_scores):.4f}, 95%: {np.percentile(train_valid_scores, 95):.4f}, max: {np.max(train_valid_scores):.4f}")
    print(f"Test Valid Smoothed Scores  -> min: {np.min(valid_scores):.4f}, 50%: {np.median(valid_scores):.4f}, 95%: {np.percentile(valid_scores, 95):.4f}, max: {np.max(valid_scores):.4f}")

    best_pa_f1 = 0.0
    best_th = 0.0
    candidates = np.percentile(valid_scores, np.linspace(0.0, 100.0, 200))
    for th in candidates:
        p = event_level_filter(test_scores, th, valid_mask, min_run=2, extreme_factor=1.75) * valid_mask.astype(np.float32)
        m = compute_metrics(test_labels, p, valid_mask=valid_mask, use_pa=True)
        if m.get("f1", 0.0) > best_pa_f1:
            best_pa_f1 = m["f1"]
            best_th = th

    print(f"\nBest Oracle Threshold: {best_th:.4f} with PA-F1: {best_pa_f1:.4f}")

    # Now let's see what EVT produces on train_valid_scores vs raw train scores
    evt_res = calibrate_evt_threshold(train_valid_scores, risk_level=1e-3, init_percentile=98.0)
    print(f"EVT Threshold on train_valid_scores: {evt_res.threshold:.4f} (gamma: {evt_res.gpd_fit.gamma:.4f}, sigma: {evt_res.gpd_fit.sigma:.4f}, t_init: {evt_res.gpd_fit.threshold_init:.4f})")

    # Let's inspect test labels and predictions with different thresholds
    for factor in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0]:
        th = float(np.percentile(train_valid_scores, 100.0 - factor * 10.0))
        p = event_level_filter(test_scores, th, valid_mask, min_run=2, extreme_factor=1.75) * valid_mask.astype(np.float32)
        m = compute_metrics(test_labels, p, valid_mask=valid_mask, use_pa=True)
        print(f"Top {factor*10:4.1f}% Train Threshold ({th:.4f}) -> PA-F1: {m.get('f1', 0.0):.4f} (P: {m.get('precision', 0.0):.4f}, R: {m.get('recall', 0.0):.4f}, Pred Positives: {int(p.sum())})")

if __name__ == "__main__":
    inspect_score_distribution()
