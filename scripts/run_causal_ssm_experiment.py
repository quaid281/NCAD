"""Validation Experiment: Causal State-Space Flow-JEPA (CausalSSMFlowJEPA).

Integrates:
1. Spatial-Temporal Causal Relational GAT (dynamic sensor adjacency)
2. Selective State-Space (SSM) sequence context modeling
3. Continuous Optimal Transport Conditional Flow Matching (OT-CFM)
4. Split Conformal Prediction Calibrator for finite-sample false alarm control P(FA) <= alpha
5. Zero-Shot Counterfactual Root-Cause Channel Attribution

Usage:
    python scripts/run_causal_ssm_experiment.py --dataset SMAP --channel A-7 --epochs 25 --device auto
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.jepa.causal_ssm_flow_jepa import CausalSSMFlowJEPA
from src.scoring.conformal_calibrator import SplitConformalCalibrator
from src.scoring.event_fusion import compute_metrics
from src.scoring.evt_calibrator import EVTCalibrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CausalSSMExperiment")


def create_sliding_windows(
    data: np.ndarray,
    window_size: int,
    step_size: int = 1,
) -> np.ndarray:
    """Create 3D sliding windows: (num_windows, window_size, channels)."""
    n_samples, n_channels = data.shape
    if n_samples < window_size:
        raise ValueError(f"Data length {n_samples} is smaller than window size {window_size}")
    indices = np.arange(0, n_samples - window_size + 1, step_size)
    windows = np.zeros((len(indices), window_size, n_channels), dtype=np.float32)
    for i, idx in enumerate(indices):
        windows[i] = data[idx : idx + window_size]
    return windows


def run_causal_ssm_experiment(
    dataset_name: str = "SMAP",
    channel_name: str = "A-7",
    epochs: int = 25,
    batch_size: int = 64,
    context_size: int = 128,
    suspect_size: int = 32,
    latent_dim: int = 32,
    hidden_dim: int = 64,
    alpha: float = 0.01,
    learning_rate: float = 1e-3,
    device_str: str = "auto",
):
    start_time = time.time()
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    logger.info("=" * 80)
    logger.info(f"STARTING CAUSAL SSM FLOW-JEPA EXPERIMENT")
    logger.info(f"Dataset: {dataset_name} | Channel: {channel_name} | Device: {device} | Epochs: {epochs}")
    logger.info("=" * 80)

    # 1. Load Data
    candidate_dirs = [
        PROJECT_ROOT / "mTSBench_data" / dataset_name,
        PROJECT_ROOT / "data" / dataset_name,
    ]

    train_path = None
    test_path = None

    for ds_dir in candidate_dirs:
        if not ds_dir.exists():
            continue
        if channel_name == "default":
            candidates = [
                (ds_dir / f"{dataset_name}_train.csv", ds_dir / f"{dataset_name}_test.csv"),
                (ds_dir / "train.csv", ds_dir / "test.csv"),
                (ds_dir / f"{dataset_name}_default_train.csv", ds_dir / f"{dataset_name}_default_test.csv"),
            ]
        else:
            candidates = [
                (ds_dir / f"{dataset_name}_{channel_name}_train.csv", ds_dir / f"{dataset_name}_{channel_name}_test.csv"),
                (ds_dir / f"{channel_name}_train.csv", ds_dir / f"{channel_name}_test.csv"),
            ]
        for tr_c, te_c in candidates:
            if tr_c.exists() and te_c.exists():
                train_path = tr_c
                test_path = te_c
                break
        if train_path is not None:
            break

    if train_path is None or test_path is None:
        raise FileNotFoundError(f"Could not find dataset files for {dataset_name} channel {channel_name} in candidate dirs: {candidate_dirs}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    numeric_cols = [c for c in train_df.columns if c not in ["timestamp", "is_anomaly"]]
    test_labels = (
        test_df["is_anomaly"].values.astype(np.int32)
        if "is_anomaly" in test_df.columns
        else np.zeros(len(test_df), dtype=np.int32)
    )

    train_vals = train_df[numeric_cols].values.astype(np.float32)
    test_vals = test_df[numeric_cols].values.astype(np.float32)

    # Robust standardization using training statistics
    mean = np.nanmean(train_vals, axis=0, keepdims=True)
    std = np.nanstd(train_vals, axis=0, keepdims=True)
    std = np.where(std < 1e-4, 1.0, std)

    train_data = np.nan_to_num((train_vals - mean) / std)
    test_data = np.nan_to_num((test_vals - mean) / std)

    if train_data.ndim == 1:
        train_data = train_data[:, None]
        test_data = test_data[:, None]

    n_train, n_channels = train_data.shape
    n_test = len(test_data)
    n_anomalies = int(np.sum(test_labels))
    logger.info(
        f"Data Loaded: Train Shape=({n_train}, {n_channels}) | "
        f"Test Shape=({n_test}, {n_channels}) | Anomaly Timesteps={n_anomalies} ({100 * n_anomalies / n_test:.2f}%)"
    )

    # 2. Windowing
    total_window = context_size + suspect_size
    train_windows = create_sliding_windows(train_data, total_window, step_size=2)
    train_ctx = train_windows[:, :context_size, :]
    train_tgt = train_windows[:, context_size:, :]

    test_windows = create_sliding_windows(test_data, total_window, step_size=1)
    test_ctx = test_windows[:, :context_size, :]
    test_tgt = test_windows[:, context_size:, :]

    # Split train into train & validation calibration sets
    n_total_train = len(train_windows)
    n_cal = max(100, int(0.20 * n_total_train))
    n_fit = n_total_train - n_cal

    fit_ctx, cal_ctx = train_ctx[:n_fit], train_ctx[n_fit:]
    fit_tgt, cal_tgt = train_tgt[:n_fit], train_tgt[n_fit:]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(fit_ctx), torch.from_numpy(fit_tgt)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    # 3. Model Instantiation
    model = CausalSSMFlowJEPA(
        in_channels=n_channels,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        node_dim=max(16, latent_dim),
        ssm_layers=2,
        gat_layers=2,
        num_heads=min(4, max(1, n_channels)),
        flow_layers=3,
        dropout=0.10,
        ema_decay=0.996,
        vicreg_weight=0.10,
        graph_sparsity_weight=1e-4,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 4. Training Loop
    logger.info("Training CausalSSMFlowJEPA with Optimal Transport Flow Matching...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_cfm_loss = 0.0
        epoch_vic_loss = 0.0
        epoch_graph_loss = 0.0
        n_batches = 0

        for x_c, x_t in train_loader:
            x_c, x_t = x_c.to(device), x_t.to(device)
            optimizer.zero_grad()

            loss, diag = model(x_c, x_t, return_diagnostics=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.update_target_encoder()

            epoch_cfm_loss += diag["cfm_loss"]
            epoch_vic_loss += diag["vic_loss"]
            epoch_graph_loss += diag["graph_loss"]
            n_batches += 1

        scheduler.step()
        if epoch % 5 == 0 or epoch == epochs:
            logger.info(
                f"Epoch [{epoch:02d}/{epochs:02d}] - "
                f"CFM Loss: {epoch_cfm_loss / n_batches:.4f} | "
                f"VICReg Loss: {epoch_vic_loss / n_batches:.4f} | "
                f"Graph Loss: {epoch_graph_loss / n_batches:.4f}"
            )

    # 5. Split Conformal Calibration over Nominal Holdout Residuals
    model.eval()
    logger.info("Calibrating Split Conformal False-Alarm Bounds...")
    with torch.no_grad():
        cal_ctx_t = torch.from_numpy(cal_ctx).to(device)
        cal_tgt_t = torch.from_numpy(cal_tgt).to(device)
        cal_scores = model.compute_anomaly_score(cal_ctx_t, cal_tgt_t).cpu().numpy()

    conformal_calibrator = SplitConformalCalibrator(alpha=alpha)
    conf_res = conformal_calibrator.calibrate(cal_scores)
    logger.info(f"Conformal Calibration: alpha={alpha:.4f} -> Threshold q_(1-alpha) = {conf_res.threshold:.4f}")

    # Also fit standard EVT Extreme Value Theory Calibrator for comparison
    evt_calibrator = EVTCalibrator(init_percentile=98.0, risk_level=alpha)
    evt_calibrator.fit(cal_scores)
    evt_res = evt_calibrator.compute_threshold(cal_scores, risk_level=alpha)
    logger.info(f"EVT Calibration: Risk={alpha:.4f} -> Threshold = {evt_res.threshold:.4f} (Method: {evt_res.method})")

    # 6. Test Scoring & Counterfactual Attribution
    logger.info("Evaluating on Test Sequence...")
    test_scores_list = []
    test_p_values_list = []
    top_causes_list = []

    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_ctx), torch.from_numpy(test_tgt)),
        batch_size=batch_size,
        shuffle=False,
    )

    with torch.no_grad():
        for x_c, x_t in test_loader:
            x_c, x_t = x_c.to(device), x_t.to(device)
            scores = model.compute_anomaly_score(x_c, x_t).cpu().numpy()
            test_scores_list.extend(scores)

            # Counterfactual channel ranking if multivariate
            if n_channels > 1:
                _, top_c = model.counterfactual_root_cause_attribution(x_c, x_t)
                top_causes_list.extend(top_c.cpu().numpy())

    raw_window_scores = np.array(test_scores_list)

    # Align window scores to point-level test timeline
    point_scores = np.zeros(n_test, dtype=np.float32)
    counts = np.zeros(n_test, dtype=np.int32)
    for i in range(len(raw_window_scores)):
        # Target interval corresponds to [i + context_size, i + total_window]
        st = i + context_size
        en = min(n_test, i + total_window)
        point_scores[st:en] += raw_window_scores[i]
        counts[st:en] += 1
    counts[counts == 0] = 1
    point_scores = point_scores / counts

    # Compute metrics under EVT and Conformal Thresholds
    evt_metrics = compute_metrics(test_labels, point_scores, evt_res.threshold)
    conformal_metrics = compute_metrics(test_labels, point_scores, conf_res.threshold)

    # Compute empirical false alarm rate on normal timesteps
    nominal_mask = test_labels == 0
    anom_mask = test_labels == 1
    conf_fa_rate = float(np.mean(point_scores[nominal_mask] > conf_res.threshold))
    conf_recall = float(np.mean(point_scores[anom_mask] > conf_res.threshold)) if np.any(anom_mask) else 0.0

    # 7. Extract Learned Causal Adjacency Topology
    with torch.no_grad():
        sample_ctx = torch.from_numpy(train_ctx[:4]).to(device)
        causal_graph = model.get_causal_graph(sample_ctx).cpu().numpy()
        mean_adj = np.mean(causal_graph, axis=0)

    elapsed = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"EXPERIMENT RESULTS: {dataset_name} (Channel: {channel_name}) in {elapsed:.2f}s")
    logger.info("=" * 80)
    logger.info(f"1. Strict Point-F1 (EVT):       {evt_metrics.point_f1:.4f} (Precision: {evt_metrics.point_precision:.4f}, Recall: {evt_metrics.point_recall:.4f})")
    logger.info(f"2. Legacy PA-F1 (EVT):          {evt_metrics.pa_f1:.4f} (Precision: {evt_metrics.pa_precision:.4f}, Recall: {evt_metrics.pa_recall:.4f})")
    logger.info(f"3. Strict Point-F1 (Conformal): {conformal_metrics.point_f1:.4f} (Precision: {conformal_metrics.point_precision:.4f}, Recall: {conformal_metrics.point_recall:.4f})")
    logger.info(f"4. Legacy PA-F1 (Conformal):    {conformal_metrics.pa_f1:.4f} (Precision: {conformal_metrics.pa_precision:.4f}, Recall: {conformal_metrics.pa_recall:.4f})")
    logger.info(f"5. Conformal Guarantees:        Target alpha={alpha:.4f} | Empirical False Alarm Rate={conf_fa_rate:.4f} | Anomaly Detection Rate={conf_recall:.4f}")
    if n_channels > 1 and len(top_causes_list) > 0:
        top_causes_arr = np.array(top_causes_list)
        # Find root cause channels during actual anomaly segments
        anom_window_indices = np.where(raw_window_scores > conf_res.threshold)[0]
        if len(anom_window_indices) > 0:
            active_top_causes = top_causes_arr[anom_window_indices, 0]
            top_ch, counts = np.unique(active_top_causes, return_counts=True)
            top_order = np.argsort(counts)[::-1]
            root_cause_str = ", ".join([f"Channel {top_ch[k]} ({100 * counts[top_order[k]] / len(active_top_causes):.1f}%)" for k in range(min(3, len(top_ch)))])
            logger.info(f"6. Causal Root-Cause Attribution: Top Fault Contributors: {root_cause_str}")

    logger.info(f"7. Learned Causal Graph Adjacency Shape: {mean_adj.shape} | Matrix Sparsity: {100 * np.mean(mean_adj < 0.05):.1f}%")
    logger.info("=" * 80)

    return {
        "dataset": dataset_name,
        "channel": channel_name,
        "elapsed_sec": elapsed,
        "point_f1_evt": evt_metrics.point_f1,
        "pa_f1_evt": evt_metrics.pa_f1,
        "point_f1_conformal": conformal_metrics.point_f1,
        "pa_f1_conformal": conformal_metrics.pa_f1,
        "conformal_fa_rate": conf_fa_rate,
        "conformal_recall": conf_recall,
        "mean_adjacency": mean_adj,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Causal SSM Flow-JEPA Experiment")
    parser.add_argument("--dataset", type=str, default="SMAP", help="Dataset name (e.g. SMAP, Daphnet, GHL, CalIt2)")
    parser.add_argument("--channel", type=str, default="A-7", help="Channel name (e.g. A-7, S08R01E3, 0)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--alpha", type=float, default=0.01, help="Conformal significance level")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, cpu, auto)")
    args = parser.parse_args()

    run_causal_ssm_experiment(
        dataset_name=args.dataset,
        channel_name=args.channel,
        epochs=args.epochs,
        batch_size=args.batch_size,
        alpha=args.alpha,
        device_str=args.device,
    )
