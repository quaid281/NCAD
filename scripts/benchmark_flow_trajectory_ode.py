"""Systematic Benchmark: Trajectory ODE Discrepancy vs Instantaneous Flow Discrepancy.

Evaluates:
1. Instantaneous Midpoint Vector Field (0-step integration)
2. Trajectory ODE - 1 step (Euler)
3. Trajectory ODE - 2 steps (Midpoint RK2)
4. Trajectory ODE - 4 steps (Midpoint RK2)
5. Trajectory ODE - 4 steps (Classic RK4)

Across multiple channels with EVT calibration, PA-F1, and Oracle ceilings.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import (
    FlowTSJEPA,
    HybridTCNEncoder,
    PatchFlowJEPA,
    flow_matching_vicreg_loss,
)
from src.data.data_loader import DataLoader
from src.models.train_model import split_train_validation as split_train_val
from src.utils.event_fusion import (
    aggregate_window_scores,
    calibrate_evt_threshold,
    compute_metrics,
    event_level_filter,
    moving_average,
    positive_robust_z,
    robust_stats,
)


def evaluate_channel_ode_modes(
    dataset_name: str,
    chan: str,
    epochs: int = 40,
    seed: int = 42,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> List[Dict]:
    ds_dir = ROOT / "mTSBench_data" / dataset_name
    train_path = ds_dir / f"{dataset_name}_{chan}_train.csv"
    test_path = ds_dir / f"{dataset_name}_{chan}_test.csv"

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
    training_data, _ = split_train_val(train_windows, val_split=0.1, seed=seed)

    input_dim = len(numeric_cols)
    patch_size = 16
    n_tgt_patches = suspect_size // patch_size

    # Build PatchFlowJEPA model
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    batch_size = 32
    total_steps = epochs * max(1, len(training_data) // batch_size)
    global_step = 0

    print(f"\nTraining PatchFlowJEPA on {dataset_name} [{chan}] ({epochs} epochs)...")
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

    # Define Discrepancy Solvers / Modes
    def get_discrepancy_fn(mode: str):
        model.eval()

        if mode == "instantaneous_midpoint":
            def disc(arr):
                res = []
                with torch.no_grad():
                    for i in range(0, len(arr), 1024):
                        ctx = torch.from_numpy(arr[i : i + 1024, :context_size]).float().to(device)
                        tgt = torch.from_numpy(arr[i : i + 1024, context_size:]).float().to(device)
                        h_ctx = model.context_encoder(ctx)
                        z_tgt = model.target_encoder(tgt)
                        t_mid = torch.full((len(ctx),), 0.5, device=device, dtype=torch.float32)
                        v_pred = model.flow_predictor(0.5 * z_tgt, t_mid, h_ctx)
                        diff = torch.linalg.norm(v_pred - z_tgt, dim=-1).mean(dim=-1)
                        res.append(diff.cpu().numpy())
                return np.concatenate(res, axis=0)
            return disc

        elif mode.startswith("trajectory_"):
            parts = mode.split("_")
            steps = int(parts[1])
            solver_name = parts[2]

            def disc(arr):
                res = []
                with torch.no_grad():
                    for i in range(0, len(arr), 1024):
                        ctx = torch.from_numpy(arr[i : i + 1024, :context_size]).float().to(device)
                        tgt = torch.from_numpy(arr[i : i + 1024, context_size:]).float().to(device)
                        z_tgt_obs = model.target_encoder(tgt)
                        z_init = torch.zeros(len(ctx), n_tgt_patches, 48, device=device)
                        z_gen = model.sample_target_patches(ctx, n_steps=steps, solver=solver_name, z_init=z_init)
                        diff = torch.linalg.norm(z_tgt_obs - z_gen, dim=-1).mean(dim=-1)
                        res.append(diff.cpu().numpy())
                return np.concatenate(res, axis=0)
            return disc

        else:
            raise ValueError(f"Unknown mode: {mode}")

    modes = [
        "instantaneous_midpoint",
        "trajectory_1_euler",
        "trajectory_2_midpoint",
        "trajectory_4_midpoint",
    ]

    train_windows_dense = DataLoader.create_windows(train_scaled, window_size, step=1)
    results = []

    for mode in modes:
        t_start = time.time()
        disc_fn = get_discrepancy_fn(mode)

        train_disc = disc_fn(train_windows_dense)
        disc_stats = robust_stats(train_disc)

        test_disc = disc_fn(test_windows)
        disc_z = positive_robust_z(test_disc, disc_stats, clip=20.0)
        train_disc_z = positive_robust_z(train_disc, disc_stats, clip=20.0)

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

        evt_res = calibrate_evt_threshold(train_valid_scores, risk_level=1e-3, init_percentile=98.0)
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

        infer_time = time.time() - t_start
        row = {
            "channel": chan,
            "mode": mode,
            "pa_f1": m_pa.get("f1", 0.0),
            "pa_precision": m_pa.get("precision", 0.0),
            "pa_recall": m_pa.get("recall", 0.0),
            "point_f1": m_pt.get("f1", 0.0),
            "oracle_pa_f1": float(best_pa_f1),
            "infer_sec": round(infer_time, 2),
        }
        results.append(row)
        print(
            f"  Mode: {mode:25s} | PA-F1: {row['pa_f1']:.4f} (P: {row['pa_precision']:.4f}, R: {row['pa_recall']:.4f}) | "
            f"Oracle: {row['oracle_pa_f1']:.4f} ({row['infer_sec']}s)"
        )

    return results


def main():
    all_results = []
    # Test on key Daphnet channels
    for chan in ["S02R01E0", "S01R01E1", "S03R02E0"]:
        res = evaluate_channel_ode_modes(dataset_name="Daphnet", chan=chan, epochs=40)
        all_results.extend(res)

    df = pd.DataFrame(all_results)
    out_p = ROOT / "reports" / "flow_trajectory_ode_comparison.csv"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_p, index=False)

    print("\n" + "=" * 90)
    print("TRAJECTORY ODE DISCREPANCY VS INSTANTANEOUS DISCREPANCY COMPARISON")
    print("=" * 90)
    pivot = df.pivot_table(index="channel", columns="mode", values=["pa_f1", "oracle_pa_f1", "infer_sec"])
    print(pivot.round(4).to_markdown())


if __name__ == "__main__":
    main()
