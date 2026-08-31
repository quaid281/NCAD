import os, sys, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))

from src.models import PatchFlowJEPA, FlowTSJEPA
from src.data.data_loader import DataLoader
from src.models.train_model import split_train_validation as split_train_val
from src.utils.event_fusion import (
    aggregate_window_scores,
    calibrate_evt_threshold,
    compute_metrics,
    event_level_filter,
    fuse_evidence_scores,
    local_deviation_scores,
    moving_average,
    positive_robust_z,
    robust_stats,
)

def von_neumann_operator_entropy_loss(z: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    if z.ndim == 3:
        z = z.reshape(-1, z.size(-1))
    D = z.size(-1)
    N = z.size(0)
    z_c = z - z.mean(dim=0, keepdim=True)
    cov = (z_c.T @ z_c) / max(N - 1, 1)
    tr = torch.trace(cov) + eps
    rho = cov / tr + (eps / D) * torch.eye(D, device=z.device)
    rho = rho / torch.trace(rho)
    evals = torch.linalg.eigvalsh(rho)
    evals = torch.clamp(evals, min=eps)
    p = evals / evals.sum()
    vn_entropy = -torch.sum(p * torch.log(p))
    max_entropy = math.log(float(D))
    return torch.clamp(max_entropy - vn_entropy, min=0.0)

def math_flow_loss(v_pred, v_target, z_ctx, z_tgt_true, use_op_entropy=True, cov_weight=0.5):
    loss_flow = F.mse_loss(v_pred, v_target)
    def var_l(x):
        if x.ndim == 3: x = x.reshape(-1, x.size(-1))
        std = torch.sqrt(torch.var(x, dim=0, unbiased=False) + 1e-4)
        return torch.mean(F.relu(1.0 - std))
    
    var_loss = 0.5 * (var_l(z_ctx) + var_l(z_tgt_true))
    
    if use_op_entropy:
        cov_loss = 0.5 * (von_neumann_operator_entropy_loss(z_ctx) + von_neumann_operator_entropy_loss(z_tgt_true))
    else:
        def std_cov(x):
            if x.ndim == 3: x = x.reshape(-1, x.size(-1))
            x_c = x - x.mean(dim=0, keepdim=True)
            C = (x_c.T @ x_c) / max(len(x) - 1, 1)
            off_diag = C - torch.diag(torch.diag(C))
            return (off_diag ** 2).sum() / x.size(-1)
        cov_loss = 0.5 * (std_cov(z_ctx) + std_cov(z_tgt_true))
        
    return loss_flow + 1.0 * var_loss + cov_weight * cov_loss

def get_chebyshev_lobatto_nodes(K: int = 4, device = torch.device('cpu')):
    if K == 2:
        return [0.5], [1.0]
    elif K == 4:
        t1 = 0.5 * (1.0 - math.sqrt(2) / 2.0)
        t2 = 0.5
        t3 = 0.5 * (1.0 + math.sqrt(2) / 2.0)
        w = [0.2761, 0.4478, 0.2761]
        sum_w = sum(w)
        return [t1, t2, t3], [x / sum_w for x in w]
    raise ValueError(f"Unsupported K: {K}")

def compute_chebyshev_discrepancy(model, ctx_windows, tgt_windows, K=4, use_mahalanobis=False):
    model.eval()
    device = ctx_windows.device
    nodes, weights = get_chebyshev_lobatto_nodes(K=K, device=device)
    n_samples = len(ctx_windows)
    res_list = []
    with torch.no_grad():
        for i in range(0, n_samples, 1024):
            ctx_b = ctx_windows[i : i + 1024]
            tgt_b = tgt_windows[i : i + 1024]
            h_ctx = model.context_encoder(ctx_b)
            z_tgt_obs = model.target_encoder(tgt_b)
            B_b, N_tgt, D = z_tgt_obs.shape
            total_disc = torch.zeros(B_b, device=device)
            for t_val, w_val in zip(nodes, weights):
                t_tensor = torch.full((B_b,), float(t_val), device=device, dtype=torch.float32)
                z_node = float(t_val) * z_tgt_obs
                v_pred = model.flow_predictor(z_node, t_tensor, h_ctx)
                diff = v_pred - z_tgt_obs
                if use_mahalanobis and bool(model.precision_fitted.item()):
                    diff_c = diff - model.residual_mean
                    mahal = torch.sum((diff_c @ model.precision_matrix) * diff_c, dim=-1)
                    patch_scores = torch.sqrt(torch.clamp(mahal, min=1e-8))
                else:
                    patch_scores = torch.linalg.norm(diff, dim=-1)
                total_disc = total_disc + float(w_val) * patch_scores.mean(dim=-1)
            res_list.append(total_disc.cpu().numpy())
    return np.concatenate(res_list, axis=0)

def run_suite(epochs=25, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_dir = ROOT / "mTSBench_data" / "Daphnet"
    channels = ["S01R01E1", "S02R01E0", "S02R02E0", "S03R01E0", "S03R01E1", "S03R02E0"]

    context_size = 256
    suspect_size = 64
    window_size = context_size + suspect_size
    patch_size = 16
    n_tgt_patches = suspect_size // patch_size

    all_rows = []

    for chan in channels:
        print(f"\n=======================================================")
        print(f"BENCHMARKING CHANNEL: {chan}")
        print(f"=======================================================")
        train_path = ds_dir / f"Daphnet_{chan}_train.csv"
        test_path = ds_dir / f"Daphnet_{chan}_test.csv"

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
        train_windows_dense = DataLoader.create_windows(train_scaled, window_size, step=1)
        test_windows = DataLoader.create_windows(test_scaled, window_size, step=1)
        training_data, _ = split_train_val(train_windows, val_split=0.1, seed=seed)

        input_dim = len(numeric_cols)

        for use_op_entropy in [False, True]:
            reg_name = "Op-Entropy" if use_op_entropy else "Std-VICReg"
            torch.manual_seed(seed)
            np.random.seed(seed)

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

            for epoch in range(1, epochs + 1):
                model.train()
                perm = np.random.permutation(len(training_data))
                for b in range(0, len(perm), batch_size):
                    global_step += 1
                    batch_arr = training_data[perm[b : b + batch_size]]
                    ctx = torch.from_numpy(batch_arr[:, :context_size]).float().to(device)
                    tgt = torch.from_numpy(batch_arr[:, context_size:]).float().to(device)
                    h_ctx, z_tgt_true, v_pred, v_target = model(ctx, tgt)
                    loss = math_flow_loss(v_pred, v_target, h_ctx, z_tgt_true, use_op_entropy=use_op_entropy, cov_weight=0.5)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    ema_val = 0.996 + (0.9995 - 0.996) * 0.5 * (1.0 - np.cos(np.pi * global_step / total_steps))
                    model.update_target_encoder(decay=ema_val)
                scheduler.step()

            ctx_all = torch.from_numpy(train_windows[:, :context_size]).float().to(device)
            tgt_all = torch.from_numpy(train_windows[:, context_size:]).float().to(device)
            ctx_test = torch.from_numpy(test_windows[:, :context_size]).float().to(device)
            tgt_test = torch.from_numpy(test_windows[:, context_size:]).float().to(device)

            test_local = local_deviation_scores(test_windows, context_size)
            train_local = local_deviation_scores(train_windows, context_size)
            local_stats = robust_stats(train_local)
            loc_z = positive_robust_z(test_local, local_stats, clip=20.0)
            train_loc_z = positive_robust_z(train_local, local_stats, clip=20.0)

            # Compare Midpoint vs Chebyshev
            for score_mode, K_nodes in [("Midpoint (t=0.5)", 2), ("Chebyshev-3pt", 4)]:
                train_disc = compute_chebyshev_discrepancy(model, ctx_all, tgt_all, K=K_nodes, use_mahalanobis=False)
                test_disc = compute_chebyshev_discrepancy(model, ctx_test, tgt_test, K=K_nodes, use_mahalanobis=False)

                disc_stats = robust_stats(train_disc)
                disc_z = positive_robust_z(test_disc, disc_stats, clip=20.0)
                train_disc_z = positive_robust_z(train_disc, disc_stats, clip=20.0)

                win_scores = 0.7 * disc_z + 0.3 * loc_z
                train_win_scores = 0.7 * train_disc_z + 0.3 * train_loc_z

                pt_scores, valid_mask = aggregate_window_scores(
                    win_scores,
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
                    train_win_scores,
                    n_points=(len(train_windows) - 1) * 10 + window_size,
                    context_size=context_size,
                    suspect_size=suspect_size,
                    step=10,
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

                row = {
                    "channel": chan,
                    "model": f"{reg_name} + {score_mode}",
                    "reg": reg_name,
                    "scoring": score_mode,
                    "pa_f1": m_pa.get("f1", 0.0),
                    "point_f1": m_pt.get("f1", 0.0),
                    "oracle_pa_f1": float(best_pa_f1),
                }
                all_rows.append(row)
                print(f"  [{row['model']:30s}] -> PA-F1: {row['pa_f1']:.4f} | Pt-F1: {row['point_f1']:.4f} | Oracle: {row['oracle_pa_f1']:.4f}")

    df_out = pd.DataFrame(all_rows)
    print("\n" + "=" * 90)
    print("CONSOLIDATED RESULTS ACROSS DAPHNET:")
    print("=" * 90)
    piv = df_out.pivot(index="channel", columns="model", values="pa_f1")
    print(piv.to_string())
    print("\nMean PA-F1 by Variant:")
    print(df_out.groupby("model")["pa_f1"].mean().to_string())
    print("\nMean Oracle PA-F1 by Variant:")
    print(df_out.groupby("model")["oracle_pa_f1"].mean().to_string())

if __name__ == "__main__":
    run_suite(epochs=25)
