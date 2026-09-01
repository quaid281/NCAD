"""Script to test and calibrate EVT tail for Flow-JEPA models."""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import PatchFlowJEPA, FlowTSJEPA, HybridTCNEncoder, flow_matching_vicreg_loss
from src.data.data_loader import DataLoader
from src.scoring.event_fusion import (
    aggregate_window_scores,
    calibrate_evt_threshold,
    compute_metrics,
    event_level_filter,
    moving_average,
    positive_robust_z,
    robust_stats,
)
from src.models.legacy.train_model import split_train_validation as split_train_val

def test_flow_evt_calibration():
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

    test_disc = compute_discrepancy(test_windows)
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

    # Evaluate Oracle Ceiling
    best_pa_f1 = 0.0
    candidates = np.percentile(valid_scores, np.linspace(0.0, 100.0, 200))
    for th in candidates:
        p = event_level_filter(test_scores, th, valid_mask, min_run=2, extreme_factor=1.75) * valid_mask.astype(np.float32)
        m = compute_metrics(test_labels, p, valid_mask=valid_mask, use_pa=True)
        if m.get("f1", 0.0) > best_pa_f1:
            best_pa_f1 = m["f1"]

    print(f"--- Oracle Ceiling: {best_pa_f1:.4f} ---")

    # Test different EVT parameters:
    print("\n--- Testing EVT Parameter Grid ---")
    for init_p in [90.0, 92.0, 95.0, 98.0]:
        for q in [1e-3, 2.5e-3, 5e-3, 1e-2]:
            evt_res = calibrate_evt_threshold(train_valid_scores, risk_level=q, init_percentile=init_p)
            evt_th = evt_res.threshold
            preds_evt = event_level_filter(test_scores, evt_th, valid_mask, min_run=2, extreme_factor=1.75) * valid_mask.astype(np.float32)
            m_pa = compute_metrics(test_labels, preds_evt, valid_mask=valid_mask, use_pa=True)
            print(f"init_percentile={init_p:4.1f}%, risk_q={q:.4f} -> Thresh: {evt_th:.4f} | PA-F1: {m_pa.get('f1', 0.0):.4f} (P: {m_pa.get('precision', 0.0):.4f}, R: {m_pa.get('recall', 0.0):.4f})")

    # Test Chi-squared Wilson-Hilferty transformation (cube root of z scores)
    print("\n--- Testing Wilson-Hilferty / Power Transform (x^(1/3)) ---")
    train_wh = np.cbrt(np.maximum(train_valid_scores, 0.0))
    test_wh = np.cbrt(np.maximum(test_scores, 0.0))
    for init_p in [90.0, 95.0, 98.0]:
        for q in [1e-3, 2.5e-3, 5e-3]:
            evt_res = calibrate_evt_threshold(train_wh, risk_level=q, init_percentile=init_p)
            evt_th = evt_res.threshold
            preds_evt = event_level_filter(test_wh, evt_th, valid_mask, min_run=2, extreme_factor=1.75) * valid_mask.astype(np.float32)
            m_pa = compute_metrics(test_labels, preds_evt, valid_mask=valid_mask, use_pa=True)
            print(f"[Wilson-Hilferty] init_p={init_p}%, q={q:.4f} -> Thresh: {evt_th:.4f} | PA-F1: {m_pa.get('f1', 0.0):.4f} (P: {m_pa.get('precision', 0.0):.4f}, R: {m_pa.get('recall', 0.0):.4f})")

if __name__ == "__main__":
    test_flow_evt_calibration()
