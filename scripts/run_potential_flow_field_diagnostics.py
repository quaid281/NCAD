#!/usr/bin/env python3
"""Potential-Flow Velocity Field Diagnostics (Reviewer Major Concern 3).

Numerically verifies the mathematical properties claimed for
PotentialFlowJEPAModel:

1. Gradient-field constraint: v_raw = -grad_z Phi must have a symmetric
   Jacobian (it is the negative Hessian of Phi). We verify
   ||J - J^T||_F / ||J||_F ~ 0 for the pre-projection field and quantify the
   antisymmetric component introduced by the tangent projection that the
   deployed field applies afterwards.
2. Finite-difference agreement between autodiff gradients and central
   differences of Phi (implementation sanity check).
3. Midpoint scoring rule vs. multi-step quadrature: correlation and
   rank-agreement between the deployed t=0.5 single-midpoint score and
   K-step integrated scores (K=4,8), plus calibrated held-out FPR and
   Point-F1 under each integration rule.

Output: reports/potential_flow_field_diagnostics.csv
        reports/potential_flow_integration_comparison.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CSMConfig
from src.data.data_loader import DataLoader
from src.engine.trainer import build_ts_jepa_model
from src.scoring.event_fusion import aggregate_window_scores, calibrate_evt_threshold, compute_metrics

C_LEN, S_LEN = 256, 64
W_LEN = C_LEN + S_LEN
EPOCHS = 12


def jacobian_of_field(v_fn, z: torch.Tensor) -> torch.Tensor:
    """Per-sample Jacobian J[b, i, j] = d v_i / d z_j via reverse-mode looping."""
    B, D = z.shape
    z = z.detach().clone().requires_grad_(True)
    v = v_fn(z)
    J = torch.zeros(B, D, D, device=z.device, dtype=z.dtype)
    for i in range(D):
        g = torch.autograd.grad(v[:, i].sum(), z, retain_graph=True)[0]
        J[:, i, :] = g
    return J


def symmetry_stats(J: torch.Tensor) -> dict:
    sym = 0.5 * (J + J.transpose(-1, -2))
    asym = 0.5 * (J - J.transpose(-1, -2))
    nJ = J.norm(dim=(-1, -2))
    return {
        "jacobian_frobenius_mean": float(nJ.mean()),
        "antisym_frac": float((asym.norm(dim=(-1, -2)) / nJ.clamp_min(1e-12)).mean()),
        "sym_frac": float((sym.norm(dim=(-1, -2)) / nJ.clamp_min(1e-12)).mean()),
    }


def finite_difference_check(model, z_t, t, z_ctx, p_regime, eps: float = 1e-4) -> float:
    """Relative error between autodiff -grad Phi and central differences."""
    z_in = z_t.detach().clone().requires_grad_(True)
    phi = model.flow_predictor.potential_field(z_in, t, z_ctx, p_regime)
    g_auto = torch.autograd.grad(phi.sum(), z_in)[0]
    g_fd = torch.zeros_like(g_auto)
    for d in range(z_t.shape[1]):
        e = torch.zeros_like(z_t)
        e[:, d] = eps
        plus = model.flow_predictor.potential_field(z_t + e, t, z_ctx, p_regime)
        minus = model.flow_predictor.potential_field(z_t - e, t, z_ctx, p_regime)
        g_fd[:, d] = (plus - minus).squeeze(-1) / (2 * eps)
    num = (g_auto - g_fd).norm(dim=-1)
    den = g_auto.norm(dim=-1).clamp_min(1e-12)
    return float((num / den).mean())


def run(dataset: str, channel: str, device: torch.device) -> tuple[list[dict], list[dict]]:
    ds_dir = ROOT / "mTSBench_data" / dataset
    train_p = ds_dir / f"{dataset}_{channel}_train.csv"
    if not train_p.exists():
        train_p = ds_dir / f"{channel}_train.csv"
    test_p = ds_dir / f"{dataset}_{channel}_test.csv"
    train_df = pd.read_csv(train_p)
    test_df = pd.read_csv(test_p)
    numeric_cols = [c for c in train_df.columns if c not in ["timestamp", "is_anomaly", "label"]]
    lbl_col = [c for c in test_df.columns if c in ["is_anomaly", "label", "labels"]]
    labels = test_df[lbl_col[0]].to_numpy().astype(int) if lbl_col else np.zeros(len(test_df), dtype=int)
    train_vals = train_df[numeric_cols].to_numpy(np.float32)
    test_vals = test_df[numeric_cols].to_numpy(np.float32)
    mean, std = train_vals.mean(0), np.maximum(train_vals.std(0), 1e-2)
    train_scaled, test_scaled = (train_vals - mean) / std, (test_vals - mean) / std
    K = train_scaled.shape[1]

    tr_win = DataLoader.create_windows(train_scaled, W_LEN, step=10, copy=False)
    n_tr = int(len(tr_win) * 0.8)
    fit_win, val_win = tr_win[:n_tr], tr_win[n_tr:]
    test_win = DataLoader.create_windows(test_scaled, W_LEN, step=5, copy=False)  # stride 5 for speed

    cfg = CSMConfig(model_type="potential_flow_jepa", context_size=C_LEN, suspect_size=S_LEN,
                    latent_dim=32, filters=48, tcn_layers=3)
    model = build_ts_jepa_model(cfg, input_dim=K, device=device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    torch.manual_seed(42)
    np.random.seed(42)
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(len(fit_win))
        for b in range(0, len(perm), 32):
            batch = torch.from_numpy(np.ascontiguousarray(fit_win[perm[b:b + 32]])).float().to(device)
            loss, _ = model.compute_objective(batch[:, :C_LEN], batch[:, C_LEN:], cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.update_target_encoder()
        print(f"    epoch {ep+1}/{EPOCHS} loss={float(loss):.4f}", flush=True)

    model.eval()

    # ---------------- Field diagnostics on a probe batch ----------------
    probe = torch.from_numpy(test_win[:64]).float().to(device)
    ctx_b, tgt_b = probe[:, :C_LEN], probe[:, C_LEN:]
    with torch.no_grad():
        z_ctx = model.context_encoder(ctx_b)
        z_tgt = model.target_encoder(tgt_b)
        if model.grassmannian_codebook is not None:
            p_regime, _, _ = model.grassmannian_codebook(z_ctx, hard=model.eval_hard_routing)
        else:
            p_regime = torch.zeros_like(z_ctx)
    z_eval = 0.5 * z_tgt
    t_eval = torch.full((z_eval.shape[0],), 0.5, device=device, dtype=z_eval.dtype)

    def v_raw_fn(z):
        z_in = z.clone().requires_grad_(True)
        phi = model.flow_predictor.potential_field(z_in, t_eval, z_ctx, p_regime)
        return -torch.autograd.grad(phi.sum(), z_in, create_graph=True)[0]

    def v_proj_fn(z):
        return model.flow_predictor(z, t_eval, z_ctx, p_regime=p_regime, create_graph=True)

    J_raw = jacobian_of_field(v_raw_fn, z_eval)
    J_proj = jacobian_of_field(v_proj_fn, z_eval)
    fd_err = finite_difference_check(model, z_eval, t_eval, z_ctx, p_regime)

    diag_rows = [{
        "dataset": dataset, "channel": channel,
        "field": "raw_gradient",
        **symmetry_stats(J_raw),
        "fd_rel_err": fd_err,
    }, {
        "dataset": dataset, "channel": channel,
        "field": "tangent_projected",
        **symmetry_stats(J_proj),
        "fd_rel_err": float("nan"),
    }]

    # ---------------- Midpoint vs integrated scoring ----------------
    def get_scores(w_arr, steps):
        out = []
        with torch.no_grad():
            for i in range(0, len(w_arr), 128):
                chunk = torch.from_numpy(np.ascontiguousarray(w_arr[i:i + 128])).float().to(device)
                out.append(model.compute_predictive_discrepancy(
                    chunk[:, :C_LEN], chunk[:, C_LEN:], integration_steps=steps).cpu().numpy())
        return np.concatenate(out)

    val_scores = get_scores(val_win, 1)
    evt = calibrate_evt_threshold(val_scores, risk_level=1e-3, init_percentile=98.0, adaptive_kurtosis=True)
    score_by_k = {}
    for k in [1, 4, 8]:
        score_by_k[k] = get_scores(test_win, k)

    integ_rows = []
    s1 = score_by_k[1]
    for k in [1, 4, 8]:
        s_k = score_by_k[k]
        rho = float(sstats.spearmanr(s1, s_k).statistic) if k != 1 else 1.0
        pt, mask = aggregate_window_scores(s_k, n_points=len(test_scaled), context_size=C_LEN,
                                         suspect_size=S_LEN, step=5, reducer="mean",
                                         mapping_method="middle")
        preds = (pt > evt.threshold).astype(int)
        m = compute_metrics(labels, preds, pt)
        fp, tn = m.get("fp", 0), m.get("tn", 0)
        integ_rows.append({
            "dataset": dataset, "channel": channel, "integration_steps": k,
            "spearman_vs_midpoint": rho,
            "point_f1": m.get("f1", 0.0),
            "point_precision": m.get("precision", 0.0),
            "point_recall": m.get("recall", 0.0),
            "empirical_fpr": fp / max(fp + tn, 1),
        })
    return diag_rows, integ_rows


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Potential-flow field diagnostics on {device}")
    streams = [("SMAP", "P-3"), ("MSL", "M-1")]
    diag_rows, integ_rows = [], []
    for ds, ch in streams:
        print(f"\n=== {ds}:{ch} ===")
        try:
            d, i = run(ds, ch, device)
            diag_rows += d
            integ_rows += i
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"FAILED {ds}:{ch}: {e}")
        pd.DataFrame(diag_rows).to_csv(ROOT / "reports" / "potential_flow_field_diagnostics.csv", index=False)
        pd.DataFrame(integ_rows).to_csv(ROOT / "reports" / "potential_flow_integration_comparison.csv", index=False)

    print("\n=== Jacobian symmetry diagnostics ===")
    print(pd.DataFrame(diag_rows).to_string())
    print("\n=== Integration-rule comparison ===")
    print(pd.DataFrame(integ_rows).to_string())


if __name__ == "__main__":
    main()
