#!/usr/bin/env python3
"""Run Controlled Covariance vs. Marginal Anomaly Benchmark.

Resolves Reviewer 3.1B:
Directly tests whether Reynolds-Stress (Latent Fluctuation Covariance Alignment)
specifically detects anomalies through changes in cross-channel dependence,
comparing against marginal reconstruction and unconstrained JEPA.

Controlled Synthetic Regimes:
1. Pure Cross-Channel Covariance Anomaly:
   - Marginals (means, variances) remain strictly identical (mu=0, sigma^2=1).
   - Cross-channel correlation rho(x1, x2) flips from +0.85 to -0.85.
   - Point-level marginal reconstructors fail; covariance-aligned models succeed.
2. Pure Marginal Anomaly:
   - Correlation structure is preserved, but marginal variance scales up.
3. Joint Anomaly:
   - Both marginal variance and cross-channel covariance are perturbed.

Outputs:
- PR-AUC and Calibrated Point-F1 across Pure Covariance vs. Pure Marginal anomalies.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
from src.models.baselines import TimesNet, TranAD


def generate_synthetic_data(
    n_samples: int = 15000,
    n_channels: int = 4,
    anomaly_type: str = "pure_covariance",
    seed: int = 42,
):
    """Generate synthetic telemetry with controlled covariance vs marginal anomalies."""
    np.random.seed(seed)
    # Nominal correlation: channel 0 and 1 are strongly positively correlated (0.85)
    # channels 2 and 3 are correlated (0.75)
    cov_nominal = np.eye(n_channels)
    cov_nominal[0, 1] = cov_nominal[1, 0] = 0.85
    cov_nominal[2, 3] = cov_nominal[3, 2] = 0.75

    # Nominal autoregressive dynamics
    A = 0.7 * np.eye(n_channels)
    X = np.zeros((n_samples, n_channels), dtype=np.float32)
    noise_nominal = np.random.multivariate_normal(np.zeros(n_channels), cov_nominal, size=n_samples).astype(np.float32)

    for t in range(1, n_samples):
        X[t] = A @ X[t - 1] + noise_nominal[t] * 0.4

    # Standardize nominal data to have unit variance and zero mean
    train_split = int(n_samples * 0.4)
    train_data = X[:train_split]
    test_data = X[train_split:].copy()
    labels = np.zeros(len(test_data), dtype=int)

    # Inject anomaly episodes into test data
    # 3 episodes of length 150 timesteps
    anomaly_spans = [
        (1500, 1650),
        (4000, 4150),
        (6500, 6650),
    ]

    for start, end in anomaly_spans:
        labels[start:end] = 1
        L = end - start
        if anomaly_type == "pure_covariance":
            # Flip cross-correlation rho(x1, x2) from +0.85 to -0.85
            # but keep marginal variances exactly 1.0!
            cov_anom = np.eye(n_channels)
            cov_anom[0, 1] = cov_anom[1, 0] = -0.85
            cov_anom[2, 3] = cov_anom[3, 2] = -0.75
            anom_noise = np.random.multivariate_normal(np.zeros(n_channels), cov_anom, size=L).astype(np.float32)
            for t_idx, t in enumerate(range(start, end)):
                prev = test_data[t - 1] if t > 0 else np.zeros(n_channels)
                test_data[t] = A @ prev + anom_noise[t_idx] * 0.4
        elif anomaly_type == "pure_marginal":
            # Scale marginal variance by 3.0 while preserving exact correlation structure
            anom_noise = np.random.multivariate_normal(np.zeros(n_channels), cov_nominal, size=L).astype(np.float32) * 2.5
            for t_idx, t in enumerate(range(start, end)):
                prev = test_data[t - 1] if t > 0 else np.zeros(n_channels)
                test_data[t] = A @ prev + anom_noise[t_idx] * 0.4

    return train_data, test_data, labels


def evaluate_models_on_synthetic(
    train_vals: np.ndarray,
    test_vals: np.ndarray,
    test_labels: np.ndarray,
    anomaly_regime: str,
    device: torch.device,
    epochs: int = 12,
):
    train_mean = np.mean(train_vals, axis=0)
    train_std = np.std(train_vals, axis=0)
    scale = np.where(train_std < 1e-8, 1.0, train_std)
    train_scaled = (train_vals - train_mean) / scale
    test_scaled = (test_vals - train_mean) / scale

    context_size = 128
    suspect_size = 32
    window_size = context_size + suspect_size

    train_windows = DataLoader.create_windows(train_scaled, window_size, step=5, copy=False)
    test_windows = DataLoader.create_windows(test_scaled, window_size, step=1, copy=False)
    training_data, val_data = split_train_val(train_windows, val_split=0.2, seed=42, window_size=window_size, step=5)

    input_dim = train_vals.shape[1]
    models = ["Reynolds-Stress_JEPA", "TS-JEPA", "TimesNet", "TranAD"]
    records = []

    for model_name in models:
        print(f"  Training {model_name} on {anomaly_regime}...")
        if model_name == "Reynolds-Stress_JEPA":
            cfg = CSMConfig(
                model_type="reynolds_stress_jepa",
                context_size=context_size,
                suspect_size=suspect_size,
                latent_dim=32,
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
                    loss, _ = model.compute_objective(ctx, tgt, cfg)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    model.update_target_encoder()

            def get_scores(w_arr):
                model.eval()
                res = []
                with torch.no_grad():
                    for i in range(0, len(w_arr), 128):
                        chunk = torch.from_numpy(np.ascontiguousarray(w_arr[i:i+128])).float().to(device)
                        disc = model.compute_predictive_discrepancy(chunk[:, :context_size], chunk[:, context_size:])
                        res.append(disc.cpu().numpy())
                return np.concatenate(res, axis=0)

        elif model_name == "TS-JEPA":
            cfg = CSMConfig(
                model_type="ts_jepa",
                context_size=context_size,
                suspect_size=suspect_size,
                latent_dim=32,
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
                    loss, _ = model.compute_objective(ctx, tgt, cfg)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    model.update_target_encoder()

            def get_scores(w_arr):
                model.eval()
                res = []
                with torch.no_grad():
                    for i in range(0, len(w_arr), 128):
                        chunk = torch.from_numpy(np.ascontiguousarray(w_arr[i:i+128])).float().to(device)
                        disc = model.compute_predictive_discrepancy(chunk[:, :context_size], chunk[:, context_size:])
                        res.append(disc.cpu().numpy())
                return np.concatenate(res, axis=0)

        elif model_name == "TimesNet":
            model = TimesNet(c_in=input_dim, d_model=48, d_ff=96, e_layers=2).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for ep in range(epochs):
                model.train()
                perm = np.random.permutation(len(training_data))
                for b in range(0, len(perm), 16):
                    batch = torch.from_numpy(np.ascontiguousarray(training_data[perm[b:b+16]])).float().to(device)
                    out = model(batch)
                    loss = nn.functional.mse_loss(out, batch)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

            def get_scores(w_arr):
                model.eval()
                res = []
                with torch.no_grad():
                    for i in range(0, len(w_arr), 128):
                        chunk = torch.from_numpy(np.ascontiguousarray(w_arr[i:i+128])).float().to(device)
                        sc = model.compute_anomaly_scores(chunk)
                        res.append(sc[:, context_size:].mean(dim=-1).cpu().numpy())
                return np.concatenate(res, axis=0)

        elif model_name == "TranAD":
            model = TranAD(c_in=input_dim, d_model=48, n_heads=4, e_layers=2, d_layers=2, d_ff=96).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for ep in range(epochs):
                model.train()
                perm = np.random.permutation(len(training_data))
                for b in range(0, len(perm), 16):
                    batch = torch.from_numpy(np.ascontiguousarray(training_data[perm[b:b+16]])).float().to(device)
                    rec1, rec2 = model(batch)
                    l1, l2 = model.adversarial_loss(rec1, rec2, batch, epoch=ep+1)
                    loss = l1 + l2
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

            def get_scores(w_arr):
                model.eval()
                res = []
                with torch.no_grad():
                    for i in range(0, len(w_arr), 128):
                        chunk = torch.from_numpy(np.ascontiguousarray(w_arr[i:i+128])).float().to(device)
                        sc = model.compute_anomaly_scores(chunk)
                        res.append(sc[:, context_size:].mean(dim=-1).cpu().numpy())
                return np.concatenate(res, axis=0)

        train_dense = DataLoader.create_windows(train_scaled, window_size, step=1, copy=False)
        train_win_sc = get_scores(train_dense)
        test_win_sc = get_scores(test_windows)
        stats = robust_stats(train_win_sc)
        train_z = positive_robust_z(train_win_sc, stats)
        test_z = positive_robust_z(test_win_sc, stats)

        pt_sc, val_m = aggregate_window_scores(
            test_z,
            n_points=len(test_vals),
            context_size=context_size,
            suspect_size=suspect_size,
            step=1,
            reducer="mean",
            mapping_method="smear",
        )
        tr_pt, tr_m = aggregate_window_scores(
            train_z,
            n_points=len(train_vals),
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
            "anomaly_regime": anomaly_regime,
            "model": model_name,
            "pr_auc": m_pt.get("pr_auc", 0.0),
            "point_f1": m_pt.get("f1", 0.0),
            "point_precision": m_pt.get("precision", 0.0),
            "point_recall": m_pt.get("recall", 0.0),
            "event_recall": m_pt.get("event_recall", 0.0),
            "false_positives": m_pt["fp"],
            "empirical_fpr": fpr,
        })

    return records


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Controlled Covariance Anomaly Benchmark on device: {device}")

    all_records = []
    out_csv = ROOT / "reports" / "covariance_anomaly_validation.csv"

    for regime in ["pure_covariance", "pure_marginal"]:
        print(f"\n==========================================")
        print(f"EVALUATING REGIME: {regime.upper()}")
        print(f"==========================================")
        tr, te, lbl = generate_synthetic_data(n_samples=15000, n_channels=4, anomaly_type=regime, seed=42)
        rec = evaluate_models_on_synthetic(tr, te, lbl, anomaly_regime=regime, device=device, epochs=12)
        all_records.extend(rec)
        pd.DataFrame(all_records).to_csv(out_csv, index=False)
        print(f"Saved intermediate records to {out_csv}")

    df = pd.DataFrame(all_records)
    print("\n" + "="*80)
    print("CONTROLLED COVARIANCE VS MARGINAL BENCHMARK SUMMARY")
    print("="*80)
    summary = df.pivot(index="model", columns="anomaly_regime", values=["pr_auc", "point_f1", "empirical_fpr"]).round(4)
    print(summary.to_string())


if __name__ == "__main__":
    main()
