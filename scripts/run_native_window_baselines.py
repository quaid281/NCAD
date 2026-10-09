"""Native-Window Baseline Experiment: TimesNet (W=96) & TranAD (W=10) vs. TS-JEPA (W=320).

Directly addresses Reviewer Major Concern 3.1 & 3.6:
Investigates whether TimesNet and TranAD performance was depressed by evaluating
them at W=320 rather than their canonical native window sizes:
- TimesNet at native W=96 vs. W=320
- TranAD at native W=10 vs. W=320
- TS-JEPA at C=256, S=64 (W=320)

Evaluates:
- Point-F1, Precision, Recall, PA-F1
- Nominal validation tail kurtosis vs test residual kurtosis
- Empirical False Positive Rate under EVT calibration (u=0.98, alpha=1e-3)
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

from src.config import CSMConfig
from src.data.data_loader import DataLoader
from src.engine.trainer import build_ts_jepa_model
from src.models.baselines import TimesNet, TranAD
from src.scoring.event_fusion import aggregate_window_scores, calibrate_evt_threshold, compute_metrics


def compute_tail_stats(residuals: np.ndarray) -> Dict[str, float]:
    """Compute excess kurtosis and tail ratio."""
    residuals = np.asarray(residuals, dtype=np.float64).flatten()
    kurt = float(stats.kurtosis(residuals, fisher=True)) if len(residuals) > 4 else 0.0
    q95 = np.percentile(residuals, 95)
    q99 = np.percentile(residuals, 99)
    q999 = np.percentile(residuals, 99.9)
    tail_ratio = float((q999 - q95) / (q99 - q95 + 1e-8))
    return {"kurtosis": kurt, "tail_ratio": tail_ratio}


def load_data(dataset_name: str, channel: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds_dir = ROOT / "mTSBench_data" / dataset_name
    if channel == "default":
        train_p = ds_dir / f"{dataset_name}_train.csv"
        if not train_p.exists():
            train_p = ds_dir / f"{dataset_name}.csv"
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


def run_window_experiment(
    dataset_name: str,
    channel: str,
    epochs: int = 10,
    batch_size: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> pd.DataFrame:
    dev = torch.device(device)
    train_norm, test_norm, labels = load_data(dataset_name, channel)
    input_dim = train_norm.shape[1] if train_norm.ndim > 1 else 1

    configs = [
        ("TimesNet_Native_W96", "timesnet", 96),
        ("TimesNet_Long_W320", "timesnet", 320),
        ("TranAD_Native_W10", "tranad", 10),
        ("TranAD_Long_W320", "tranad", 320),
        ("TS-JEPA_Ours_W320", "ts_jepa", 320),
    ]

    results = []

    for label, m_type, w_len in configs:
        print(f"  Evaluating [{label}] on {dataset_name}:{channel}...", flush=True)
        torch.manual_seed(42)
        np.random.seed(42)

        # Window creation
        all_tr_windows = DataLoader.create_windows(train_norm, w_len, step=max(1, w_len // 10), copy=False)
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

        if m_type == "timesnet":
            model = TimesNet(c_in=input_dim, d_model=48, d_ff=48, e_layers=2, top_k=3, dropout=0.1).to(dev)
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
                        # Mean squared reconstruction error over the window
                        err = torch.mean((rec - b) ** 2, dim=-1).mean(dim=-1)
                        scores.append(err.cpu().numpy())
                return np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)

        elif m_type == "tranad":
            model = TranAD(c_in=input_dim, d_model=48, n_heads=4, e_layers=2, d_layers=2, d_ff=96).to(dev)
            opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for ep in range(epochs):
                model.train()
                perm = np.random.permutation(len(tr_windows))
                for b in range(0, len(perm), batch_size):
                    rec1, rec2 = model(batch)
                    loss1, loss2 = TranAD.adversarial_loss(rec1, rec2, batch, epoch=ep + 1)
                    loss = loss1 + loss2
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            def get_scores(windows_arr):
                model.eval()
                scores = []
                with torch.no_grad():
                    for i in range(0, len(windows_arr), 128):
                        b = torch.from_numpy(windows_arr[i:i+128]).float().to(dev)
                        s = model.compute_anomaly_scores(b)
                        # Mean anomaly score across timesteps
                        scores.append(s.mean(dim=-1).cpu().numpy())
                return np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)

        elif m_type == "ts_jepa":
            c_len = int(w_len * 0.8)
            s_len = w_len - c_len
            cfg = CSMConfig(
                model_type="ts_jepa",
                context_size=c_len,
                suspect_size=s_len,
                latent_dim=32,
                filters=48,
                tcn_layers=3,
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
                return np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)

        # 1. Validation Calibration
        val_res = get_scores(val_windows)
        evt_res = calibrate_evt_threshold(val_res, risk_level=1e-3, init_percentile=98.0, adaptive_kurtosis=True)
        thresh = evt_res.threshold
        val_tail = compute_tail_stats(val_res)

        # 2. Test Scoring
        test_res = get_scores(test_windows)
        test_tail = compute_tail_stats(test_res)

        # Map to points
        pt_scores, _ = aggregate_window_scores(
            test_res,
            n_points=len(test_norm),
            context_size=int(w_len * 0.8) if m_type == "ts_jepa" else 0,
            suspect_size=w_len - int(w_len * 0.8) if m_type == "ts_jepa" else w_len,
            step=1,
            reducer="mean",
            mapping_method="middle",
        )
        y_pred = (pt_scores > thresh).astype(int)

        m = compute_metrics(labels, y_pred, pt_scores)
        fp = m.get("fp", 0)
        tn = m.get("tn", 0)
        fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        results.append({
            "dataset": dataset_name,
            "channel": channel,
            "model_config": label,
            "window_size": w_len,
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
        ("SMD", "machine-1-2"),
        ("MSL", "M-1"),
        ("Daphnet", "S01R01E1"),
        ("GECCO", "water_quality"),
    ]
    out_path = ROOT / "reports" / "native_window_baselines.csv"
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
        print(f"Running native-window baseline experiment on {d}:{c}...", flush=True)
        df = run_window_experiment(d, c, epochs=10)
        all_dfs.append(df)
        current_df = pd.concat(all_dfs, ignore_index=True)
        current_df.to_csv(out_path, index=False)
        print(f"Finished and saved {d}:{c} to {out_path}!", flush=True)

    res_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    if not res_df.empty:
        res_df.to_csv(out_path, index=False)
        print(f"\nFinal results saved to {out_path}", flush=True)
        print("\nSummary by Model Configuration across Streams:", flush=True)
        print(res_df.groupby("model_config")[["point_f1", "point_prec", "point_rec", "pa_f1", "test_kurtosis", "empirical_fpr"]].mean(), flush=True)
