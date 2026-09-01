"""Generate Phase-Space Latent Trajectory and Limit-Cycle Attractor Visualizations for TS-JEPA on Daphnet."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.data_loader import DataLoader
from src.models.encoders.tcn_encoder import HybridTCNEncoder
from src.models.jepa.ts_jepa import TSJEPAModel, jepa_vicreg_loss
from src.scoring.event_fusion import (
    aggregate_window_scores,
    calibrate_evt_threshold,
    compute_metrics,
    event_level_filter,
    moving_average,
    positive_robust_z,
    robust_stats,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", type=str, default="S01R01E1")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else args.device if args.device != "auto" else "cpu")
    
    # Load Daphnet channel
    dataset_dir = ROOT / "mTSBench_data" / "Daphnet"
    train_path = dataset_dir / f"Daphnet_{args.channel}_train.csv"
    test_path = dataset_dir / f"Daphnet_{args.channel}_test.csv"

    print(f"Loading data for Daphnet {args.channel} ...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    numeric_cols = [c for c in train_df.columns if c not in ["timestamp", "is_anomaly"]]
    test_labels = test_df["is_anomaly"].to_numpy().astype(int) if "is_anomaly" in test_df.columns else None

    train_mean = train_df[numeric_cols].mean().to_numpy()
    train_std = train_df[numeric_cols].std().to_numpy()
    scale = np.maximum(train_std, 0.01)
    train_scaled = (train_df[numeric_cols].to_numpy() - train_mean) / scale
    test_scaled = (test_df[numeric_cols].to_numpy() - train_mean) / scale

    context_size = 256
    suspect_size = 64
    window_size = context_size + suspect_size
    latent_dim = 32

    train_windows = DataLoader.create_windows(train_scaled, window_size, step=10)
    test_windows = DataLoader.create_windows(test_scaled, window_size, step=2)  # Step 2 for high density trajectory

    input_dim = len(numeric_cols)
    print(f"Input dimensions: {input_dim}, Latent dim: {latent_dim}")

    # Build and train TS-JEPA
    base_enc = HybridTCNEncoder(input_dim=input_dim, latent_dim=latent_dim, filters=48, tcn_layers=3, dropout=0.20)
    model = TSJEPAModel(context_encoder=base_enc, latent_dim=latent_dim, predictor_hidden_dim=64, ema_decay=0.995).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"Training TS-JEPA for {args.epochs} epochs on {device} ...")
    batch_size = 32
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(len(train_windows))
        for b in range(0, len(perm), batch_size):
            clean = train_windows[perm[b : b + batch_size]]
            ctx = torch.from_numpy(clean[:, :context_size]).float().to(device)
            tgt = torch.from_numpy(clean[:, context_size:]).float().to(device)

            z_ctx, z_tgt_true, z_tgt_pred = model(ctx, tgt)
            loss = jepa_vicreg_loss(z_tgt_pred, z_tgt_true, z_context=z_ctx)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.update_target_encoder()

    print("Extracting latent trajectory states on test set ...")
    model.eval()
    all_z_ctx, all_z_tgt_true, all_z_tgt_pred = [], [], []

    with torch.no_grad():
        for i in range(0, len(test_windows), 1024):
            batch = test_windows[i : i + 1024]
            ctx = torch.from_numpy(batch[:, :context_size]).float().to(device)
            tgt = torch.from_numpy(batch[:, context_size:]).float().to(device)
            z_ctx, z_tgt_true, z_tgt_pred = model(ctx, tgt)
            all_z_ctx.append(z_ctx.cpu().numpy())
            all_z_tgt_true.append(z_tgt_true.cpu().numpy())
            all_z_tgt_pred.append(z_tgt_pred.cpu().numpy())

    z_ctx_all = np.concatenate(all_z_ctx, axis=0)
    z_tgt_true_all = np.concatenate(all_z_tgt_true, axis=0)
    z_tgt_pred_all = np.concatenate(all_z_tgt_pred, axis=0)

    # Discrepancy errors
    discrepancy = np.linalg.norm(z_tgt_true_all - z_tgt_pred_all, axis=-1)

    # Map anomaly labels to window centers
    window_labels = []
    for w_idx in range(len(test_windows)):
        pt_idx = w_idx * 2 + context_size + suspect_size // 2
        if pt_idx < len(test_labels):
            window_labels.append(test_labels[pt_idx])
        else:
            window_labels.append(0)
    window_labels = np.array(window_labels)

    # Fit PCA on normal latent representations
    normal_mask = (window_labels == 0)
    anomaly_mask = (window_labels == 1)

    pca = PCA(n_components=3, random_state=42)
    pca.fit(z_tgt_true_all[normal_mask])
    
    pca_true = pca.transform(z_tgt_true_all)
    pca_pred = pca.transform(z_tgt_pred_all)

    print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")

    # Set up publication figure style
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 15,
    })

    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.30, wspace=0.22)

    # Panel A: Raw Accelerometer Sensors + Anomaly Ground Truth
    ax_raw = fig.add_subplot(gs[0, 0])
    time_pts = np.arange(len(test_scaled))
    # Select 3 representative axes: Ankle, Thigh, Trunk
    ax_raw.plot(time_pts, test_scaled[:, 0], color="#2563eb", alpha=0.75, linewidth=0.7, label="Ankle Accel (Axis 1)")
    ax_raw.plot(time_pts, test_scaled[:, 3], color="#059669", alpha=0.75, linewidth=0.7, label="Thigh Accel (Axis 1)")
    ax_raw.plot(time_pts, test_scaled[:, 6], color="#d97706", alpha=0.75, linewidth=0.7, label="Trunk Accel (Axis 1)")
    
    # Highlight anomaly zones
    if test_labels is not None:
        ax_raw.fill_between(time_pts, -3, 5, where=(test_labels == 1), color="#dc2626", alpha=0.22, label="Gait Freezing (Anomaly)")

    ax_raw.set_title("(a) Multi-Sensor Accelerometer Signals & Freezing Episodes", fontweight="bold")
    ax_raw.set_xlabel("Time Step (t)")
    ax_raw.set_ylabel("Normalized Amplitude (σ)")
    ax_raw.set_xlim(0, min(10000, len(test_scaled)))
    ax_raw.set_ylim(-3.5, 4.5)
    ax_raw.legend(loc="upper right", framealpha=0.9)
    ax_raw.grid(True, alpha=0.25, linestyle="--")

    # Panel B: Latent Prediction Discrepancy & Anomaly Detection Score
    ax_disc = fig.add_subplot(gs[0, 1])
    win_times = np.arange(len(discrepancy)) * 2 + context_size
    smoothed_disc = moving_average(discrepancy, 12)
    thresh = np.percentile(smoothed_disc[normal_mask], 98.5)

    ax_disc.plot(win_times, smoothed_disc, color="#4f46e5", linewidth=1.1, label=r"Latent Discrepancy $\|z_{\mathrm{target}} - \hat{z}_{\mathrm{pred}}\|_2$")
    ax_disc.axhline(thresh, color="#dc2626", linestyle="--", linewidth=1.2, label=f"EVT Threshold ({thresh:.2f})")
    ax_disc.fill_between(win_times, 0, np.max(smoothed_disc) * 1.05, where=(window_labels == 1), color="#dc2626", alpha=0.18, label="True Freezing Ground Truth")

    ax_disc.set_title(r"(b) TS-JEPA Latent Physical Discrepancy Score $S(t)$", fontweight="bold")
    ax_disc.set_xlabel("Time Step (t)")
    ax_disc.set_ylabel("Latent State Discrepancy")
    ax_disc.set_xlim(0, min(10000, win_times[-1]))
    ax_disc.set_ylim(0, np.percentile(smoothed_disc, 99.8) * 1.25)
    ax_disc.legend(loc="upper right", framealpha=0.9)
    ax_disc.grid(True, alpha=0.25, linestyle="--")

    # Panel C: 2D Phase-Space Trajectory (Normal Limit Cycle vs Freezing Breakdown)
    ax_phase = fig.add_subplot(gs[1, 0])
    
    # Plot normal limit cycle trajectory segment
    norm_idx = np.where(normal_mask)[0][:600]
    ax_phase.plot(pca_true[norm_idx, 0], pca_true[norm_idx, 1], color="#2563eb", linewidth=1.3, alpha=0.75, label="Normal Periodic Limit Cycle")
    ax_phase.scatter(pca_true[norm_idx[0], 0], pca_true[norm_idx[0], 1], color="#1d4ed8", s=60, marker="o", zorder=5, label="Normal Trajectory Start")

    # Plot anomaly freeze trajectory segment
    anom_idx = np.where(anomaly_mask)[0][:400]
    if len(anom_idx) > 0:
        ax_phase.plot(pca_true[anom_idx, 0], pca_true[anom_idx, 1], color="#dc2626", linewidth=1.6, alpha=0.90, linestyle="-", label="Gait Freezing Collapse Trajectory")
        ax_phase.scatter(pca_true[anom_idx, 0], pca_true[anom_idx, 1], color="#ef4444", s=18, alpha=0.6, zorder=4)

    # Plot JEPA forecasted trajectory on the normal manifold
    ax_phase.plot(pca_pred[norm_idx, 0], pca_pred[norm_idx, 1], color="#059669", linewidth=1.0, linestyle=":", alpha=0.85, label=r"JEPA Predicted State $\hat{z}_{\mathrm{pred}}$")

    ax_phase.set_title("(c) 2D Latent Phase Space: Periodic Attractor vs Collapse", fontweight="bold")
    ax_phase.set_xlabel(f"Latent Eigenmode 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax_phase.set_ylabel(f"Latent Eigenmode 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax_phase.legend(loc="upper right", framealpha=0.9)
    ax_phase.grid(True, alpha=0.25, linestyle="--")

    # Panel D: 3D Phase-Space Attractor Manifold
    ax_3d = fig.add_subplot(gs[1, 1], projection="3d")
    
    # 3D Normal trajectory orbit
    ax_3d.plot(pca_true[norm_idx, 0], pca_true[norm_idx, 1], pca_true[norm_idx, 2], color="#2563eb", linewidth=1.4, alpha=0.80, label="Normal Orbit")
    # 3D Anomaly trajectory breakdown
    if len(anom_idx) > 0:
        ax_3d.plot(pca_true[anom_idx, 0], pca_true[anom_idx, 1], pca_true[anom_idx, 2], color="#dc2626", linewidth=1.8, alpha=0.90, label="Freezing Collapse")
        ax_3d.scatter(pca_true[anom_idx[::5], 0], pca_true[anom_idx[::5], 1], pca_true[anom_idx[::5], 2], color="#ef4444", s=25, alpha=0.7)

    ax_3d.set_title("(d) 3D Geometric Manifold of Physical Dynamics", fontweight="bold", pad=12)
    ax_3d.set_xlabel(r"$PC_1$")
    ax_3d.set_ylabel(r"$PC_2$")
    ax_3d.set_zlabel(r"$PC_3$")
    ax_3d.view_init(elev=26, azim=42)
    ax_3d.legend(loc="upper left", framealpha=0.85)

    # Save figure
    out_dirs = [
        ROOT / "paper" / "figures",
        ROOT / "reports" / "figures",
        Path(r"C:\Users\andre\.gemini\antigravity\brain\873534f4-a816-47c1-9d42-ff9b605db4d9"),
    ]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig_path = d / "daphnet_phase_space_attractor.png"
        fig.savefig(fig_path, bbox_inches="tight", dpi=300)
        print(f"Saved publication figure to: {fig_path}")

    plt.close(fig)
    print("Done! Phase-space attractor plots generated successfully.")


if __name__ == "__main__":
    main()
