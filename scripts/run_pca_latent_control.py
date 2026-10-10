#!/usr/bin/env python3
"""Non-Neural Latent Control for the Controlled Backbone Experiment.

Addresses Reviewer Major Concern 2: "Would a simpler learned representation,
dimensionality reduction method, or non-neural latent predictor produce similar
residual conditioning?"

Adds two controls to the matched-backbone protocol of
`scripts/run_controlled_backbone_experiment.py`:

1. pca_latent_pred: PCA (D=32) projection of context/target windows followed by
   ridge regression predicting the target latent from the context latent.
   Score: ||z_tgt_pred - z_tgt||_2.
2. rand_latent_pred: fixed random Gaussian projection (D=32) + ridge regression.
   Controls whether the benefit comes from any compressed latent space or
   specifically from the learned JEPA representation.

Identical protocol: same 15 streams, same 80/20 chronological validation split,
same SPOT calibration (q=1e-3, u=p98), same SNR levels, same metrics schema.
Output rows merge cleanly with reports/controlled_backbone_experiment.csv.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.data_loader import DataLoader
from src.scoring.event_fusion import aggregate_window_scores, calibrate_evt_threshold, compute_metrics

from scripts.run_controlled_backbone_experiment import add_noise, compute_tail_stats, load_data


def ridge_fit(Z_ctx: np.ndarray, Z_tgt: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """Closed-form ridge map W: z_tgt ~= z_ctx @ W."""
    d = Z_ctx.shape[1]
    A = Z_ctx.T @ Z_ctx + lam * np.eye(d)
    B = Z_ctx.T @ Z_tgt
    return np.linalg.solve(A, B)


def run_control(
    dataset_name: str,
    channel: str,
    latent_dim: int = 32,
    snr_levels: List[float] = [float("inf"), 20.0, 10.0, 0.0],
) -> pd.DataFrame:
    c_len, s_len = 256, 64
    w_len = c_len + s_len
    rng = np.random.default_rng(42)

    train_norm, test_norm, labels = load_data(dataset_name, channel)
    input_dim = train_norm.shape[1]

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

    # Subsample training windows for linear solves if extremely large
    n_fit = min(len(tr_windows), 20000)
    fit_idx = rng.choice(len(tr_windows), size=n_fit, replace=False)
    fit_windows = tr_windows[fit_idx]

    X_ctx = fit_windows[:, :c_len, :].reshape(n_fit, -1).astype(np.float64)
    X_tgt = fit_windows[:, c_len:, :].reshape(n_fit, -1).astype(np.float64)

    # Center once on the training fit set
    mu_ctx = X_ctx.mean(axis=0, keepdims=True)
    mu_tgt = X_tgt.mean(axis=0, keepdims=True)

    d_ctx = min(latent_dim, X_ctx.shape[1] - 1)
    d_tgt = min(latent_dim, X_tgt.shape[1] - 1)

    # PCA via SVD (deterministic)
    Uc, Sc, Vct = np.linalg.svd(X_ctx - mu_ctx, full_matrices=False)
    Vc = Vct[:d_ctx].T  # (ctx_flat, d_ctx)
    Ut, Stt, Vtt = np.linalg.svd(X_tgt - mu_tgt, full_matrices=False)
    Vt = Vtt[:d_tgt].T  # (tgt_flat, d_tgt)

    # Fixed random Gaussian projections (non-learned latent space control)
    R_ctx = rng.standard_normal((X_ctx.shape[1], d_ctx)) / math.sqrt(X_ctx.shape[1])
    R_tgt = rng.standard_normal((X_tgt.shape[1], d_tgt)) / math.sqrt(X_tgt.shape[1])

    def encode(win: np.ndarray, proj_ctx: np.ndarray, proj_tgt: np.ndarray):
        ctx_flat = win[:, :c_len, :].reshape(len(win), -1).astype(np.float64) - mu_ctx
        tgt_flat = win[:, c_len:, :].reshape(len(win), -1).astype(np.float64) - mu_tgt
        return ctx_flat @ proj_ctx, tgt_flat @ proj_tgt

    results = []
    controls = [
        ("pca_latent_pred", Vc, Vt),
        ("rand_latent_pred", R_ctx, R_tgt),
    ]

    for m_name, P_ctx, P_tgt in controls:
        print(f"  Fitting control [{m_name}]...", flush=True)
        Zc_fit, Zt_fit = encode(fit_windows, P_ctx, P_tgt)
        W = ridge_fit(Zc_fit, Zt_fit)

        def get_scores(windows_arr):
            Zc, Zt = encode(windows_arr, P_ctx, P_tgt)
            resid = Zt - Zc @ W
            return np.linalg.norm(resid, axis=1).astype(np.float32)

        val_res = get_scores(val_windows)
        evt_res = calibrate_evt_threshold(val_res, risk_level=1e-3, init_percentile=98.0, adaptive_kurtosis=True)
        thresh = evt_res.threshold
        val_tail = compute_tail_stats(val_res)

        for snr in snr_levels:
            test_in = test_windows if math.isinf(snr) else add_noise(test_windows, snr)
            test_res = get_scores(test_in)
            test_tail = compute_tail_stats(test_res)

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
                "point_f1": m.get("f1", m.get("point_f1", 0.0)),
                "point_prec": m.get("precision", m.get("point_prec", 0.0)),
                "point_rec": m.get("recall", m.get("point_rec", 0.0)),
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
    out_path = ROOT / "reports" / "controlled_backbone_latent_controls.csv"
    all_dfs = []
    for d, c in test_runs:
        try:
            print(f"Running latent controls on {d}:{c}...", flush=True)
            df = run_control(d, c)
            all_dfs.append(df)
            pd.concat(all_dfs, ignore_index=True).to_csv(out_path, index=False)
            print(f"Finished {d}:{c}", flush=True)
        except Exception as e:
            print(f"FAILED {d}:{c}: {e}", flush=True)

    res_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    if not res_df.empty:
        res_df.to_csv(out_path, index=False)
        print(f"\nFinal results saved to {out_path}", flush=True)
        clean = res_df[res_df["snr_db"] == float("inf")]
        print(
            clean.groupby("model")[
                ["point_f1", "point_prec", "point_rec", "pa_f1", "val_kurtosis", "test_kurtosis", "empirical_fpr"]
            ].mean(),
            flush=True,
        )
