"""Rigorous Controlled Backbone Experiment: Observation Space vs. Latent Space.

Directly addresses Reviewer Major Concern 3.1 & 3.2:
Isolates the effect of prediction space by using the EXACT same HybridTCNEncoder backbone:
1. TCN-Obs-Recon: TCN Autoencoder reconstructing in raw observation space
2. TCN-Obs-Pred:  TCN Forecaster predicting upcoming window in raw observation space
3. TS-JEPA:       TCN Joint-Embedding Predictive Architecture predicting in latent space

Includes:
- Held-out nominal validation calibration (20% split)
- Residual tail diagnostics (excess kurtosis, GPD tail index, empirical FPR)
- Controlled noise-injection experiment (SNR = Inf, 20dB, 10dB, 0dB)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.data_loader import DataLoader
from src.models.encoders.tcn_encoder import HybridTCNEncoder
from src.scoring.event_fusion import aggregate_window_scores, calibrate_evt_threshold, compute_metrics


class TCNObsForecaster(nn.Module):
    """Forecaster predicting upcoming window in raw observation space."""
    def __init__(self, input_dim: int, context_len: int = 256, suspect_len: int = 64, latent_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.suspect_len = suspect_len
        self.encoder = HybridTCNEncoder(input_dim=input_dim, latent_dim=latent_dim, filters=48, tcn_layers=3)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, suspect_len * input_dim)
        )

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        z = self.encoder(ctx)
        pred = self.decoder(z)
        return pred.view(-1, self.suspect_len, self.input_dim)


class TCNObsReconstructor(nn.Module):
    """Autoencoder reconstructing the full window in raw observation space."""
    def __init__(self, input_dim: int, total_len: int = 320, latent_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.total_len = total_len
        self.encoder = HybridTCNEncoder(input_dim=input_dim, latent_dim=latent_dim, filters=48, tcn_layers=3)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, total_len * input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        rec = self.decoder(z)
        return rec.view(-1, self.total_len, self.input_dim)


def add_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Inject Gaussian noise at specified SNR (dB)."""
    if math.isinf(snr_db):
        return signal
    sig_power = np.mean(signal ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power + 1e-8), size=signal.shape)
    return signal + noise


def compute_tail_stats(residuals: np.ndarray) -> Dict[str, float]:
    """Compute empirical tail diagnostics: excess kurtosis and tail index."""
    kurt = float(stats.kurtosis(residuals, fisher=True))
    # Hill-type tail index proxy or 99th/95th ratio
    q95 = np.percentile(residuals, 95)
    q99 = np.percentile(residuals, 99)
    q999 = np.percentile(residuals, 99.9)
    tail_ratio = float((q999 - q95) / (q99 - q95 + 1e-8))
    return {"kurtosis": kurt, "tail_ratio": tail_ratio}


def load_data(dataset_name: str, channel: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds_dir = ROOT / "mTSBench_data" / dataset_name
    if channel == "default":
        train_p = ds_dir / f"{dataset_name}_train.csv"
        test_p = ds_dir / f"{dataset_name}_test.csv"
    else:
        train_p = ds_dir / f"{dataset_name}_{channel}_train.csv"
        if not train_p.exists():
            train_p = ds_dir / f"{channel}_train.csv"
        test_p = ds_dir / f"{dataset_name}_{channel}_test.csv"
        if not test_p.exists():
            test_p = ds_dir / f"{channel}_test.csv"

    train_df = pd.read_csv(train_p)
    test_df = pd.read_csv(test_p)

    numeric_cols = [c for c in train_df.columns if c not in ["timestamp", "is_anomaly", "label"]]
    train_raw = train_df[numeric_cols].values.astype(np.float32)
    test_raw = test_df[numeric_cols].values.astype(np.float32)

    lbl_col = [c for c in test_df.columns if c in ["is_anomaly", "label", "labels"]]
    labels = test_df[lbl_col[0]].values.astype(int) if lbl_col else np.zeros(len(test_raw), dtype=int)

    mean = np.mean(train_raw, axis=0, keepdims=True)
    std = np.std(train_raw, axis=0, keepdims=True)
    std = np.maximum(std, 1e-2)

    train_scaled = (train_raw - mean) / std
    test_scaled = (test_raw - mean) / std
    return train_scaled, test_scaled, labels


def run_experiment(
    dataset_name: str,
    channel: str,
    epochs: int = 20,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    snr_levels: List[float] = [float("inf"), 20.0, 10.0, 0.0]
) -> pd.DataFrame:
    dev = torch.device(device)
    c_len, s_len = 256, 64
    w_len = c_len + s_len

    train_norm, test_norm, labels = load_data(dataset_name, channel)
    input_dim = train_norm.shape[1] if train_norm.ndim > 1 else 1

    # Held-out nominal validation split (80% train, 20% validation)
    all_tr_windows = DataLoader.create_windows(train_norm, w_len, step=10, copy=False)
    if len(all_tr_windows) >= 10:
        n_tr = int(len(all_tr_windows) * 0.8)
        tr_windows = all_tr_windows[:n_tr]
        val_windows = all_tr_windows[n_tr:]
    else:
        all_tr_windows = DataLoader.create_windows(train_norm, w_len, step=1, copy=False)
        if len(all_tr_windows) >= 5:
            n_tr = max(1, int(len(all_tr_windows) * 0.8))
            tr_windows = all_tr_windows[:n_tr]
            val_windows = all_tr_windows[n_tr:]
        else:
            tr_windows = all_tr_windows
            val_windows = all_tr_windows

    test_windows = DataLoader.create_windows(test_norm, w_len, step=1, copy=False)

    results = []

    models_to_test = ["tcn_obs_recon", "tcn_obs_pred", "ts_jepa"]

    for m_name in models_to_test:
        print(f"  Training model [{m_name}]...", flush=True)
        # Set seeds
        torch.manual_seed(42)
        np.random.seed(42)

        if m_name == "tcn_obs_recon":
            model = TCNObsReconstructor(input_dim=input_dim, total_len=w_len, latent_dim=32).to(dev)
            opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for ep in range(epochs):
                model.train()
                perm = np.random.permutation(len(tr_windows))
                for b in range(0, len(perm), batch_size):
                    batch = torch.from_numpy(tr_windows[perm[b:b+batch_size]]).float().to(dev)
                    rec = model(batch)
                    loss = F.mse_loss(rec, batch)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            def get_scores(windows_arr):
                model.eval()
                scores = []
                with torch.no_grad():
                    for i in range(0, len(windows_arr), 128):
                        b = torch.from_numpy(windows_arr[i:i+128]).float().to(dev)
                        rec = model(b)
                        # Error across suspect region (context_len:)
                        err = torch.mean((rec[:, c_len:] - b[:, c_len:]) ** 2, dim=(-2, -1))
                        scores.append(err.cpu().numpy())
                return np.concatenate(scores)

        elif m_name == "tcn_obs_pred":
            model = TCNObsForecaster(input_dim=input_dim, context_len=c_len, suspect_len=s_len, latent_dim=32).to(dev)
            opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for ep in range(epochs):
                model.train()
                perm = np.random.permutation(len(tr_windows))
                for b in range(0, len(perm), batch_size):
                    batch = torch.from_numpy(tr_windows[perm[b:b+batch_size]]).float().to(dev)
                    ctx = batch[:, :c_len]
                    tgt = batch[:, c_len:]
                    pred = model(ctx)
                    loss = F.mse_loss(pred, tgt)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            def get_scores(windows_arr):
                model.eval()
                scores = []
                with torch.no_grad():
                    for i in range(0, len(windows_arr), 128):
                        b = torch.from_numpy(windows_arr[i:i+128]).float().to(dev)
                        ctx = b[:, :c_len]
                        tgt = b[:, c_len:]
                        pred = model(ctx)
                        err = torch.mean((pred - tgt) ** 2, dim=(-2, -1))
                        scores.append(err.cpu().numpy())
                return np.concatenate(scores)

        elif m_name == "ts_jepa":
            from src.config import CSMConfig
            from src.engine.trainer import build_ts_jepa_model
            cfg = CSMConfig(
                model_type="ts_jepa",
                context_size=c_len,
                suspect_size=s_len,
                latent_dim=32,
                filters=48,
                tcn_layers=3,
                epochs=epochs,
                batch_size=batch_size,
                use_mahalanobis=False,
            )
            model = build_ts_jepa_model(cfg, input_dim=input_dim, device=dev)
            opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for ep in range(epochs):
                model.train()
                perm = np.random.permutation(len(tr_windows))
                for b in range(0, len(perm), batch_size):
                    batch = torch.from_numpy(tr_windows[perm[b:b+batch_size]]).float().to(dev)
                    ctx = batch[:, :c_len]
                    tgt = batch[:, c_len:]
                    loss, _ = model.compute_objective(ctx, tgt, cfg)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            def get_scores(windows_arr):
                model.eval()
                scores = []
                with torch.no_grad():
                    for i in range(0, len(windows_arr), 128):
                        b = torch.from_numpy(windows_arr[i:i+128]).float().to(dev)
                        ctx = b[:, :c_len]
                        tgt = b[:, c_len:]
                        disc = model.compute_predictive_discrepancy(ctx, tgt, use_mahalanobis=False)
                        scores.append(disc.cpu().numpy())
                return np.concatenate(scores)

        # 1. Validation Calibration
        val_res = get_scores(val_windows)
        evt_res = calibrate_evt_threshold(val_res, risk_level=1e-3, init_percentile=98.0, adaptive_kurtosis=True)
        thresh = evt_res.threshold
        val_tail = compute_tail_stats(val_res)

        # 2. Test Evaluation across SNR levels
        for snr in snr_levels:
            if math.isinf(snr):
                test_in = test_windows
            else:
                test_in = add_noise(test_windows, snr)

            test_res = get_scores(test_in)
            test_tail = compute_tail_stats(test_res)

            # Align point scores
            pt_scores, valid_mask = aggregate_window_scores(
                test_res,
                n_points=len(test_norm),
                context_size=c_len,
                suspect_size=s_len,
                step=1,
                reducer="mean",
                mapping_method="middle",
            )
            y_pred = (pt_scores > thresh).astype(int)

            # Compute Point-F1, PA-F1, Precision, Recall, FPR
            m = compute_metrics(labels, y_pred, pt_scores)
            fp = m.get("fp", 0)
            tn = m.get("tn", 0)
            fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0

            results.append({
                "dataset": dataset_name,
                "channel": channel,
                "model": m_name,
                "snr_db": snr,
                "val_kurtosis": val_tail["kurtosis"],
                "test_kurtosis": test_tail["kurtosis"],
                "point_f1": m.get("point_f1", 0.0),
                "point_prec": m.get("point_precision", 0.0),
                "point_rec": m.get("point_recall", 0.0),
                "pa_f1": m.get("pa_f1", 0.0),
                "fp": fp,
                "tp": m.get("tp", 0),
                "empirical_fpr": fpr,
                "threshold": thresh,
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    test_runs = [
        ("SMAP", "P-3"),
        ("Daphnet", "S01R01E1"),
        ("GECCO", "water_quality"),
        ("CalIt2", "traffic"),
        ("MSL", "M-1"),
        ("SMD", "machine-1-2"),
        ("SMD", "machine-1-3"),
        ("SMAP", "A-1"),
        ("MSL", "C-1"),
        ("PSM", "default"),
        ("swan", "sf"),
        ("room-occupancy", "default"),
        ("Genesis", "default"),
        ("Daphnet", "S02R01E0"),
        ("OPPORTUNITY", "S1-ADL2"),
    ]
    out_path = ROOT / "reports" / "controlled_backbone_experiment.csv"
    existing_df = pd.DataFrame()
    completed = set()
    if out_path.exists():
        try:
            existing_df = pd.read_csv(out_path)
            for _, row in existing_df.iterrows():
                completed.add((str(row["dataset"]), str(row["channel"])))
        except Exception:
            existing_df = pd.DataFrame()

    all_dfs = [existing_df] if not existing_df.empty else []
    for d, c in test_runs:
        if (d, c) in completed:
            print(f"Skipping already completed {d}:{c}...", flush=True)
            continue
        print(f"Running controlled backbone experiment on {d}:{c}...", flush=True)
        df = run_experiment(d, c, epochs=10)
        all_dfs.append(df)
        current_df = pd.concat(all_dfs, ignore_index=True)
        current_df.to_csv(out_path, index=False)
        print(f"Finished and saved {d}:{c} to {out_path}!", flush=True)

    res_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    if not res_df.empty:
        res_df.to_csv(out_path, index=False)
        print(f"\nFinal results saved to {out_path}", flush=True)
        print("\nSummary at SNR = Inf (Clean):", flush=True)
        clean = res_df[res_df["snr_db"] == float("inf")]
        print(clean.groupby("model")[["point_f1", "point_prec", "point_rec", "pa_f1", "val_kurtosis", "test_kurtosis", "empirical_fpr"]].mean(), flush=True)
