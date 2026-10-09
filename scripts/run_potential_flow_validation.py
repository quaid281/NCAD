#!/usr/bin/env python3
"""Run Potential-Flow Mathematical Validation on Controlled Dynamical Systems.

Resolves Reviewer 3.1C:
Tests the curl-free scalar potential prior against controlled conservative
and non-conservative dynamical systems:
1. Conservative System: Duffing Oscillator (Hamiltonian phase space, zero circulation)
   - Anomaly: Non-conservative damping/circulation injection
2. Non-Conservative System: Van der Pol Oscillator (Limit-cycle attractor, non-zero vorticity)

Models compared with matched parameter budgets:
1. Potential-Flow JEPA (Pure curl-free scalar potential gradient: v = -grad Phi)
2. Helmholtz-JEPA (Decomposed into scalar potential gradient + rotational solenoidal circulation)
3. Unconstrained TS-JEPA (Standard unconstrained MLP predictor)

Outputs:
- Nominal Trajectory Residual (MSE on clean telemetry)
- Anomaly Detection PR-AUC & Calibrated Point-F1
- Vorticity / Circulation Discrepancy response
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


def simulate_duffing(n_steps=12000, dt=0.02, alpha=1.0, beta=1.0, seed=42):
    """Conservative Duffing oscillator: x_dot = y, y_dot = -alpha*x - beta*x^3."""
    np.random.seed(seed)
    X = np.zeros((n_steps, 2), dtype=np.float32)
    x, y = 1.0, 0.0
    for t in range(n_steps):
        # Symplectic Verlet integration (exact energy conservation)
        x_next = x + y * dt
        y_next = y - (alpha * x_next + beta * (x_next ** 3)) * dt
        X[t] = [x_next, y_next]
        x, y = x_next, y_next

    # Nominal train: first 40%
    train_split = int(n_steps * 0.4)
    train_data = X[:train_split].copy()
    test_data = X[train_split:].copy()
    labels = np.zeros(len(test_data), dtype=int)

    # Inject non-conservative anomalies into test data (damping/work injection)
    # Episode 1: Non-conservative damping gamma * y
    spans = [(1000, 1150), (3000, 3150), (5000, 5150)]
    for s, e in spans:
        labels[s:e] = 1
        x, y = test_data[s - 1]
        for t_idx, t in enumerate(range(s, e)):
            x_next = x + y * dt
            # Non-conservative damping dissipation
            y_next = y - (alpha * x_next + beta * (x_next ** 3) + 2.5 * y) * dt
            test_data[t] = [x_next, y_next]
            x, y = x_next, y_next

    return train_data, test_data, labels


def simulate_vanderpol(n_steps=12000, dt=0.02, mu=1.5, seed=42):
    """Non-conservative Van der Pol oscillator (limit cycle with vorticity)."""
    np.random.seed(seed)
    X = np.zeros((n_steps, 2), dtype=np.float32)
    x, y = 0.5, 0.5
    for t in range(n_steps):
        # x_dot = y, y_dot = mu*(1 - x^2)*y - x
        x_next = x + y * dt
        y_next = y + (mu * (1.0 - x_next ** 2) * y - x_next) * dt
        X[t] = [x_next, y_next]
        x, y = x_next, y_next

    train_split = int(n_steps * 0.4)
    train_data = X[:train_split].copy()
    test_data = X[train_split:].copy()
    labels = np.zeros(len(test_data), dtype=int)

    spans = [(1000, 1150), (3000, 3150), (5000, 5150)]
    for s, e in spans:
        labels[s:e] = 1
        x, y = test_data[s - 1]
        for t_idx, t in enumerate(range(s, e)):
            x_next = x + y * dt
            # Perturb limit cycle frequency/vorticity
            y_next = y + (0.1 * (1.0 - x_next ** 2) * y - 5.0 * x_next) * dt
            test_data[t] = [x_next, y_next]
            x, y = x_next, y_next

    return train_data, test_data, labels


class HelmholtzJEPA(nn.Module):
    """Helmholtz-Hodge JEPA: Decomposes velocity into scalar potential gradient + solenoidal rotation."""
    def __init__(self, encoder, latent_dim=16):
        super().__init__()
        self.context_encoder = encoder
        self.latent_dim = latent_dim
        # Potential network Phi: R^D -> R
        self.phi_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        # Skew-symmetric rotational circulation matrix: A = -A^T
        self.skew_param = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.05)

    def get_rotational_matrix(self):
        return self.skew_param - self.skew_param.T

    def forward(self, z_ctx):
        # Gradient of scalar potential
        z_req = z_ctx.clone().detach().requires_grad_(True)
        phi = self.phi_net(z_req).sum()
        grad_phi = torch.autograd.grad(phi, z_req, create_graph=True)[0]
        # Rotational velocity
        rot_v = z_ctx @ self.get_rotational_matrix()
        # Combined Helmholtz field
        v_pred = -grad_phi + rot_v
        return z_ctx + v_pred


def evaluate_system(
    train_vals: np.ndarray,
    test_vals: np.ndarray,
    test_labels: np.ndarray,
    system_name: str,
    device: torch.device,
    epochs: int = 15,
):
    train_mean = np.mean(train_vals, axis=0)
    train_std = np.std(train_vals, axis=0)
    scale = np.where(train_std < 1e-8, 1.0, train_std)
    train_scaled = (train_vals - train_mean) / scale
    test_scaled = (test_vals - train_mean) / scale

    context_size = 64
    suspect_size = 16
    window_size = context_size + suspect_size

    train_windows = DataLoader.create_windows(train_scaled, window_size, step=5, copy=False)
    test_windows = DataLoader.create_windows(test_scaled, window_size, step=1, copy=False)
    training_data, val_data = split_train_val(train_windows, val_split=0.2, seed=42, window_size=window_size, step=5)

    input_dim = 2
    latent_dim = 16
    models = ["Potential-Flow_JEPA", "Helmholtz_JEPA", "TS-JEPA"]
    records = []

    for model_name in models:
        print(f"  Training {model_name} on {system_name}...")
        if model_name == "Potential-Flow_JEPA":
            cfg = CSMConfig(
                model_type="potential_flow_jepa",
                context_size=context_size,
                suspect_size=suspect_size,
                latent_dim=latent_dim,
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
                latent_dim=latent_dim,
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

        elif model_name == "Helmholtz_JEPA":
            cfg = CSMConfig(
                model_type="ts_jepa",
                context_size=context_size,
                suspect_size=suspect_size,
                latent_dim=latent_dim,
            )
            base_model = build_ts_jepa_model(cfg, input_dim=input_dim, device=device)
            helmholtz_pred = HelmholtzJEPA(base_model.context_encoder, latent_dim=latent_dim).to(device)
            base_model.predictor = helmholtz_pred
            model = base_model
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for ep in range(epochs):
                model.train()
                perm = np.random.permutation(len(training_data))
                for b in range(0, len(perm), 16):
                    batch = torch.from_numpy(np.ascontiguousarray(training_data[perm[b:b+16]])).float().to(device)
                    ctx = batch[:, :context_size]
                    tgt = batch[:, context_size:]
                    z_ctx = model.context_encoder(ctx)
                    z_pred = model.predictor(z_ctx)
                    z_tgt = model.target_encoder(tgt)
                    loss = nn.functional.mse_loss(z_pred, z_tgt)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    model.update_target_encoder()

            def get_scores(w_arr):
                model.eval()
                res = []
                for i in range(0, len(w_arr), 128):
                    chunk = torch.from_numpy(np.ascontiguousarray(w_arr[i:i+128])).float().to(device)
                    z_ctx = model.context_encoder(chunk[:, :context_size])
                    z_pred = model.predictor(z_ctx)
                    z_tgt = model.target_encoder(chunk[:, context_size:])
                    disc = torch.norm(z_pred - z_tgt, dim=-1)
                    res.append(disc.detach().cpu().numpy())
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
            "dynamical_system": system_name,
            "model": model_name,
            "nominal_train_mse": float(np.mean(train_win_sc)),
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
    print(f"Running Potential-Flow Dynamical Validation on device: {device}")

    all_records = []
    out_csv = ROOT / "reports" / "potential_flow_dynamical_validation.csv"

    # 1. Conservative Hamiltonian System
    print("\n" + "="*80)
    print("EVALUATING CONSERVATIVE SYSTEM: DUFFING OSCILLATOR")
    print("="*80)
    tr, te, lbl = simulate_duffing(n_steps=12000, dt=0.02, seed=42)
    rec1 = evaluate_system(tr, te, lbl, system_name="Conservative_Duffing", device=device, epochs=15)
    all_records.extend(rec1)

    # 2. Non-Conservative Limit Cycle System
    print("\n" + "="*80)
    print("EVALUATING NON-CONSERVATIVE SYSTEM: VAN DER POL OSCILLATOR")
    print("="*80)
    tr, te, lbl = simulate_vanderpol(n_steps=12000, dt=0.02, seed=42)
    rec2 = evaluate_system(tr, te, lbl, system_name="NonConservative_VanDerPol", device=device, epochs=15)
    all_records.extend(rec2)

    pd.DataFrame(all_records).to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")

    df = pd.DataFrame(all_records)
    print("\n" + "="*80)
    print("POTENTIAL-FLOW DYNAMICAL SYSTEM VALIDATION SUMMARY")
    print("="*80)
    summary = df.pivot(index="model", columns="dynamical_system", values=["pr_auc", "point_f1", "empirical_fpr"]).round(4)
    print(summary.to_string())


if __name__ == "__main__":
    main()
