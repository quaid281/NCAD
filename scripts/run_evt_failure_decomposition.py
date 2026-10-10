#!/usr/bin/env python3
"""Run Empirical EVT Calibration Failure Decomposition.

Directly addresses Reviewer Major Concern 5:
Isolates and quantifies the four potential causes of EVT threshold failure:
1. Autocorrelation & Extremal Clustering: Lag-1 rho_1, Extremal index theta, Effective tail sample size N_eff
2. Distribution Shift / Nonstationarity: 2-sample KS test (D_KS, p_val), Wasserstein-1 distance W_1 between Calibration and Evaluation splits
3. Tail Model Misspecification: Cramér-von Mises goodness-of-fit test (W^2, p_val) on GPD tail excesses
4. Finite-Sample Estimation Variance: Bootstrap SE of threshold tau, and actual held-out FPR overshoot (FPR_eval / q)

Evaluates TS-JEPA, TimesNet, and TranAD across diverse physical telemetry streams on GPU.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CSMConfig
from src.data.data_loader import DataLoader
from src.engine.trainer import build_ts_jepa_model
from src.models.baselines import TimesNet, TranAD
from src.scoring.event_fusion import aggregate_window_scores
from src.scoring.evt_calibrator import fit_gpd, compute_excess_kurtosis


def compute_extremal_index(x: np.ndarray, u: float) -> Tuple[float, int, float]:
    """Compute Ferro-Segers (2003) extremal index on threshold excesses."""
    exc_idx = np.where(x > u)[0]
    n_u = len(exc_idx)
    if n_u <= 2:
        return 1.0, n_u, float(n_u)
    T = np.diff(exc_idx)
    if np.max(T) <= 2:
        theta = min(1.0, 2.0 * (np.sum(T) ** 2) / ((n_u - 1) * np.sum(T ** 2)))
    else:
        denom = (n_u - 1) * np.sum((T - 1) * (T - 2))
        theta = min(1.0, 2.0 * (np.sum(T - 1) ** 2) / denom) if denom > 0 else 1.0
    theta = float(np.clip(theta, 0.01, 1.0))
    n_eff = float(theta * n_u)
    return theta, n_u, n_eff


def compute_lag1_autocorr(x: np.ndarray) -> float:
    """Compute sample lag-1 autocorrelation."""
    if len(x) < 3:
        return 0.0
    x_c = x - np.mean(x)
    var = np.sum(x_c ** 2)
    if var < 1e-12:
        return 0.0
    return float(np.sum(x_c[:-1] * x_c[1:]) / var)


def compute_gpd_gof(excesses: np.ndarray, sigma: float, xi: float) -> Tuple[float, float]:
    """Compute Cramér-von Mises goodness-of-fit test for fitted Generalized Pareto Distribution."""
    if len(excesses) < 5 or sigma <= 1e-6:
        return 0.0, 1.0
    try:
        # scipy genpareto parameterization: c = -xi, scale = sigma
        # In our EVT formulation: P(Y <= y) = 1 - (1 + xi * y / sigma)^(-1/xi)
        # scipy: F(y) = 1 - (1 + c * y / scale)^(-1/c) with c = xi
        res = stats.cramervonmises(excesses, 'genpareto', args=(xi, 0, sigma))
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return 0.0, 1.0


def compute_threshold_bootstrap_se(x: np.ndarray, u: float, q: float, n_boot: int = 200) -> float:
    """Compute bootstrap standard error of the SPOT threshold estimate."""
    excesses = x[x > u] - u
    if len(excesses) < 10:
        return 0.0
    n = len(x)
    n_u = len(excesses)
    boot_thresh = []
    block_len = 16
    n_blocks = max(1, n_u // block_len)
    blocks = [excesses[i*block_len : (i+1)*block_len] for i in range(n_blocks)]
    
    for _ in range(n_boot):
        sample_blocks = [blocks[idx] for idx in np.random.choice(len(blocks), size=len(blocks), replace=True)]
        b_exc = np.concatenate(sample_blocks)
        fit_res = fit_gpd(b_exc)
        sigma, xi = fit_res.sigma, fit_res.gamma
        if abs(xi) < 1e-6:
            tau_b = u - sigma * math.log(q * n / n_u)
        else:
            term = (q * n / n_u) ** (-xi) - 1.0
            tau_b = u + (sigma / xi) * term
        if np.isfinite(tau_b):
            boot_thresh.append(tau_b)
    if len(boot_thresh) < 10:
        return 0.0
    return float(np.std(boot_thresh))


def load_partitioned_telemetry(dataset: str, channel: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load stream and partition nominal telemetry chronologically into Train (65%), Cal (17.5%), and Eval (17.5%)."""
    d = ROOT / "mTSBench_data" / dataset
    if channel == "default":
        tr_p = d / f"{dataset}_train.csv"
        tr_p = tr_p if tr_p.exists() else d / f"{dataset}.csv"
    else:
        tr_p = d / f"{dataset}_{channel}_train.csv"
        tr_p = tr_p if tr_p.exists() else d / f"{channel}_train.csv"
    
    df = pd.read_csv(tr_p)
    cols = [c for c in df.columns if c not in ["timestamp", "is_anomaly", "label", "labels"]]
    vals = df[cols].to_numpy(dtype=np.float32)
    
    mean = np.mean(vals, axis=0, keepdims=True)
    std = np.std(vals, axis=0, keepdims=True)
    std = np.maximum(std, 1e-2)
    scaled = (vals - mean) / std
    
    N = len(scaled)
    n_tr = int(N * 0.65)
    n_cal = int(N * 0.175)
    
    train_part = scaled[:n_tr]
    cal_part = scaled[n_tr : n_tr + n_cal]
    # 64-step isolation gap between cal and eval
    eval_start = n_tr + n_cal + 64
    eval_part = scaled[eval_start:] if eval_start < N else scaled[n_tr + n_cal:]
    return train_part, cal_part, eval_part


def run_failure_decomposition() -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing EVT Failure Decomposition on {device}...")
    
    streams = [
        ("SMAP", "P-3"),
        ("SMAP", "A-1"),
        ("MSL", "M-1"),
        ("MSL", "C-1"),
        ("SMD", "machine-1-2"),
        ("Daphnet", "S01R01E1"),
        ("GECCO", "water_quality"),
        ("CalIt2", "traffic"),
        ("Genesis", "default"),
        ("room-occupancy", "default"),
    ]
    
    models = ["TS-JEPA", "TimesNet", "TranAD"]
    c_len, s_len = 256, 64
    w_len = c_len + s_len
    q_design = 1e-3
    
    records = []
    
    for ds, ch in streams:
        print(f"\n================ Processing Stream {ds}:{ch} ================")
        try:
            tr_raw, cal_raw, eval_raw = load_partitioned_telemetry(ds, ch)
        except Exception as e:
            print(f"Skipping {ds}:{ch} due to load error: {e}")
            continue
            
        k = tr_raw.shape[1]
        tr_win = DataLoader.create_windows(tr_raw, w_len, step=5, copy=False)
        cal_win = DataLoader.create_windows(cal_raw, w_len, step=1, copy=False)
        eval_win = DataLoader.create_windows(eval_raw, w_len, step=1, copy=False)
        
        if len(cal_win) < 20 or len(eval_win) < 20:
            print(f"Skipping {ds}:{ch}: insufficient windows (cal={len(cal_win)}, eval={len(eval_win)})")
            continue
            
        for m_name in models:
            print(f"  Training {m_name}...")
            torch.manual_seed(42)
            np.random.seed(42)
            
            if m_name == "TS-JEPA":
                cfg = CSMConfig(model_type="ts_jepa", context_size=c_len, suspect_size=s_len, latent_dim=32, filters=48, tcn_layers=3)
                m = build_ts_jepa_model(cfg, input_dim=k, device=device)
                opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
                for ep in range(12):
                    m.train()
                    perm = np.random.permutation(len(tr_win))
                    for b in range(0, len(perm), 32):
                        batch = torch.from_numpy(tr_win[perm[b:b+32]]).float().to(device)
                        loss, _ = m.compute_objective(batch[:, :c_len], batch[:, c_len:], cfg)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                        m.update_target_encoder()
                        
                def score_fn(w_arr):
                    m.eval()
                    sc = []
                    with torch.no_grad():
                        for i in range(0, len(w_arr), 128):
                            chunk = torch.from_numpy(w_arr[i:i+128]).float().to(device)
                            d = m.compute_predictive_discrepancy(chunk[:, :c_len], chunk[:, c_len:])
                            sc.append(d.cpu().numpy())
                    return np.concatenate(sc)

            elif m_name == "TimesNet":
                m = TimesNet(c_in=k, d_model=48, d_ff=96, e_layers=2).to(device)
                opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
                for ep in range(12):
                    m.train()
                    perm = np.random.permutation(len(tr_win))
                    for b in range(0, len(perm), 32):
                        batch = torch.from_numpy(tr_win[perm[b:b+32]]).float().to(device)
                        rec = m(batch)
                        loss = nn.functional.mse_loss(rec, batch)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                        
                def score_fn(w_arr):
                    m.eval()
                    sc = []
                    with torch.no_grad():
                        for i in range(0, len(w_arr), 128):
                            chunk = torch.from_numpy(w_arr[i:i+128]).float().to(device)
                            rec = m(chunk)
                            err = torch.mean((rec[:, c_len:] - chunk[:, c_len:]) ** 2, dim=(-2, -1))
                            sc.append(err.cpu().numpy())
                    return np.concatenate(sc)

            elif m_name == "TranAD":
                m = TranAD(c_in=k, d_model=48, n_heads=4, e_layers=2, d_layers=2, d_ff=96).to(device)
                opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
                for ep in range(12):
                    m.train()
                    perm = np.random.permutation(len(tr_win))
                    for b in range(0, len(perm), 32):
                        batch = torch.from_numpy(tr_win[perm[b:b+32]]).float().to(device)
                        rec1, rec2 = m(batch)
                        l1, l2 = m.adversarial_loss(rec1, rec2, batch, epoch=ep+1)
                        loss = l1 + l2
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                        
                def score_fn(w_arr):
                    m.eval()
                    sc = []
                    with torch.no_grad():
                        for i in range(0, len(w_arr), 128):
                            chunk = torch.from_numpy(w_arr[i:i+128]).float().to(device)
                            rec1, rec2 = m(chunk)
                            err = torch.mean((rec2[:, c_len:] - chunk[:, c_len:]) ** 2, dim=(-2, -1))
                            sc.append(err.cpu().numpy())
                    return np.concatenate(sc)
            
            # 1. Get raw scores
            cal_scores = score_fn(cal_win)
            eval_scores = score_fn(eval_win)
            
            # Map window scores to point timeline (trailing mode to isolate purely historical)
            pt_cal, _ = aggregate_window_scores(cal_scores, n_points=len(cal_raw), context_size=c_len, suspect_size=s_len, step=1, mapping_method="trailing")
            pt_eval, _ = aggregate_window_scores(eval_scores, n_points=len(eval_raw), context_size=c_len, suspect_size=s_len, step=1, mapping_method="trailing")
            
            # Filter valid scores (points that received window coverage)
            valid_cal = pt_cal[pt_cal > 0]
            valid_eval = pt_eval[pt_eval > 0]
            
            if len(valid_cal) < 30 or len(valid_eval) < 30:
                continue
                
            # Mechanism 1: Autocorrelation & Clustering
            rho1 = compute_lag1_autocorr(valid_cal)
            u98 = float(np.percentile(valid_cal, 98.0))
            theta, n_u, n_eff = compute_extremal_index(valid_cal, u98)
            mean_cluster_size = 1.0 / max(theta, 1e-4)
            
            # Mechanism 2: Nonstationarity / Distribution Shift
            ks_stat, ks_pval = stats.ks_2samp(valid_cal, valid_eval)
            w1_dist = float(stats.wasserstein_distance(valid_cal, valid_eval))
            cal_q98 = float(np.percentile(valid_cal, 98.0))
            eval_q98 = float(np.percentile(valid_eval, 98.0))
            delta_q98_pct = float((eval_q98 - cal_q98) / max(abs(cal_q98), 1e-6) * 100.0)
            
            # Mechanism 3: GPD Goodness of Fit
            excesses = valid_cal[valid_cal > u98] - u98
            fit_res = fit_gpd(excesses)
            sigma, xi = fit_res.sigma, fit_res.gamma
            cvm_stat, cvm_pval = compute_gpd_gof(excesses, sigma, xi)
            
            # Mechanism 4: Threshold Estimation & Held-out FPR Overshoot
            n = len(valid_cal)
            if abs(xi) < 1e-6:
                tau_spot = u98 - sigma * math.log(q_design * n / n_u)
            else:
                term = (q_design * n / n_u) ** (-xi) - 1.0
                tau_spot = u98 + (sigma / xi) * term
                
            tau_se = compute_threshold_bootstrap_se(valid_cal, u98, q_design)
            
            # Held-out nominal FPR
            hold_fp = int(np.sum(valid_eval > tau_spot))
            hold_fpr = float(hold_fp / len(valid_eval))
            overshoot = float(hold_fpr / q_design)
            
            records.append({
                "dataset": ds,
                "channel": ch,
                "model": m_name,
                "rho1_autocorr": rho1,
                "extremal_index_theta": theta,
                "mean_cluster_size": mean_cluster_size,
                "raw_tail_Nu": n_u,
                "effective_tail_Neff": n_eff,
                "ks_stat_shift": ks_stat,
                "ks_pval_shift": ks_pval,
                "wasserstein1_dist": w1_dist,
                "delta_q98_pct": delta_q98_pct,
                "gpd_xi": xi,
                "gpd_sigma": sigma,
                "cvm_stat_gof": cvm_stat,
                "cvm_pval_gof": cvm_pval,
                "threshold_spot": tau_spot,
                "threshold_se": tau_se,
                "held_out_nominal_fpr": hold_fpr,
                "overshoot_ratio": overshoot,
            })
            
    df_out = pd.DataFrame(records)
    out_csv = ROOT / "reports" / "evt_failure_decomposition.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\nSuccessfully wrote {len(df_out)} rows to {out_csv}")
    return df_out


if __name__ == "__main__":
    df = run_failure_decomposition()
    print("\n=== EVT Calibration Failure Decomposition Summary ===")
    summary = df.groupby("model")[[
        "rho1_autocorr", "extremal_index_theta", "mean_cluster_size",
        "ks_stat_shift", "wasserstein1_dist", "delta_q98_pct",
        "cvm_pval_gof", "held_out_nominal_fpr", "overshoot_ratio"
    ]].mean()
    print(summary.to_string())
