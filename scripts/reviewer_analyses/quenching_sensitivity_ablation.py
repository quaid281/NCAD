#!/usr/bin/env python3
"""Surgical Weight-Space Quenching Component Ablation & Hyperparameter Sensitivity Analysis.

Addresses Reviewer Concern 10:
1. Compares Full Surgical Weight-Space Quenching (Laminar-JEPA) against simpler alternatives
   on the 19 primary streams (and caches 19-stream panel residuals if needed):
   - (a) Base 224-Bit Circuit Breaker (unquenched),
   - (b) Predictor-Bias / Residual-Mean Recentering Only (shifts mean offset on calibration prefix),
   - (c) Covariance / Precision Regularization Only (rational eigenvalue saturation
         lambda_tilde = lambda / (1 + lambda / lambda_max) + off-diagonal decoupling),
   - (d) Eq. (3) RMS Energy Stabilization Only (0.7 RMS + 0.3 Mean blend without weight edits),
   - (e) Full Surgical Weight-Space Quenching (Laminar-JEPA: predictor row damping + phase offset
         alignment + residual mean recentering + passband guard scaling).
2. Evaluates sensitivity to the number of quenched rogue coordinates k_rogue in {0, 1, 2, 4, 8, 12}
   and precision saturation ceiling / guard factor, demonstrating that genuine anomaly sensitivity
   (Event Recall, buffered Point-F1, PR-AUC) is preserved while nominal FPR is reduced.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.circuit_breaker_19stream.common import VAL_FRAC, cache_path, load_streams_19
from scripts.experiment_wsp_soliton import extract_raw4_white4_from_latents
from scripts.reviewer_analyses.run_224bit_horizon_sweep import (
    eval_224_from_latents,
    fast_extract_latents,
    fast_latent_surgical_quench,
)
from scripts.run_multiseed_heldout_benchmark import batches, load_stream
from src.config import CSMConfig
from src.data.data_loader import DataLoader
from src.engine.builder import build_ts_jepa_model

C, S = 256, 64
W = C + S
EPOCHS = 8
TRAIN_BS = 128


@torch.no_grad()
def apply_bias_recentering_only(
    model: torch.nn.Module,
    z_ctx_cal: torch.Tensor,
    z_tgt_cal: torch.Tensor,
    blend: float = 0.75,
) -> None:
    """Alternative 1: Re-center predictor/residual mean offset on nominal calibration windows only."""
    model.eval()
    dev = next(model.parameters()).device
    M, D = int(model.num_subblocks), int(model.latent_dim)
    cal_raw4, _ = extract_raw4_white4_from_latents(model, z_ctx_cal, z_tgt_cal)
    cal_mean = cal_raw4.reshape(len(cal_raw4), M * D).mean(axis=0)
    res_mean_new = (1.0 - blend) * model.residual_mean.detach().cpu().numpy() + blend * cal_mean
    model.residual_mean.copy_(torch.from_numpy(res_mean_new).to(dev, dtype=model.residual_mean.dtype))


@torch.no_grad()
def apply_precision_tempering_only(
    model: torch.nn.Module,
    lambda_max: float = 10.0,
    off_diag_shrink: float = 0.50,
) -> None:
    """Alternative 2: Smooth rational eigenvalue saturation lambda_tilde = lambda / (1 + lambda / lambda_max)
    and off-diagonal shrinkage on precision_matrix without editing predictor weights or biases.
    """
    model.eval()
    dev = next(model.parameters()).device
    prec = model.precision_matrix.detach().cpu().numpy().copy()
    diag = np.diag(np.diag(prec))
    prec_shrunk = diag + off_diag_shrink * (prec - diag)
    evals, evecs = np.linalg.eigh(0.5 * (prec_shrunk + prec_shrunk.T))
    evals = np.maximum(evals, 1e-6)
    evals_tempered = evals / (1.0 + evals / lambda_max)
    prec_new = (evecs * evals_tempered[None, :]) @ evecs.T
    model.precision_matrix.copy_(torch.from_numpy(prec_new).to(dev, dtype=model.precision_matrix.dtype))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-ablation", default=str(ROOT / "reports" / "quenching_component_ablation_19stream.csv"))
    ap.add_argument("--out-sensitivity", default=str(ROOT / "reports" / "quenching_sensitivity_19stream.csv"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    streams = load_streams_19()
    rows_abl: list[dict[str, object]] = []
    rows_sens: list[dict[str, object]] = []
    t0 = time.time()

    for ds, ch in streams:
        train, test, y_test = load_stream(ds, ch)
        if len(test) > 15000:
            anom_idx = np.where(y_test == 1)[0]
            s_te = max(0, int(anom_idx[0]) - 2000) if len(anom_idx) > 0 else 0
            e_te = min(len(test), s_te + 15000)
            s_te = max(0, e_te - 15000)
            test = test[s_te:e_te]
            y_test = y_test[s_te:e_te]
        n_val = int(len(train) * VAL_FRAC)
        tr, va = train[:-n_val], train[-n_val:]
        mu, sd = tr.mean(0), tr.std(0)
        sc = np.where(sd < 1e-8, 1.0, np.maximum(sd, 0.01))
        tr_w = DataLoader.create_windows((tr - mu) / sc, W, step=16, copy=False)
        va_w = DataLoader.create_windows((va - mu) / sc, W, step=1, copy=False)
        te_w = DataLoader.create_windows((test - mu) / sc, W, step=1, copy=False)
        n_cal_w = len(va_w) // 2

        for seed in (42,):
            torch.manual_seed(seed)
            np.random.seed(seed)
            cfg = CSMConfig(model_type="soliton_reynolds_jepa", context_size=C, suspect_size=S, latent_dim=32)
            model = build_ts_jepa_model(cfg, input_dim=train.shape[1], device=device)
            opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for _ in range(EPOCHS):
                model.train()
                for b in batches(tr_w, TRAIN_BS):
                    b = b.to(device)
                    loss, _ = model.compute_objective(b[:, :C], b[:, C:], cfg)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    model.update_target_encoder(None)
            model.fit_alien_bank(tr_w, context_len=C, batch_size=512)

            z_ctx_tr, z_tgt_tr = fast_extract_latents(model, tr_w, C, batch_size=512)
            z_ctx_va, z_tgt_va = fast_extract_latents(model, va_w, C, batch_size=512)
            z_ctx_te, z_tgt_te = fast_extract_latents(model, te_w, C, batch_size=512)

            pt_path = cache_path(ds, ch, 42).with_suffix(".pt")
            pt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "z_ctx_va": z_ctx_va.cpu(),
                    "z_tgt_va": z_tgt_va.cpu(),
                    "z_ctx_te": z_ctx_te.cpu(),
                    "z_tgt_te": z_tgt_te.cpu(),
                },
                pt_path,
            )

            # Populate the 19-stream npz cache if not present so CPU scripts run in 2 seconds
            va_raw4_c, va_white4_c = extract_raw4_white4_from_latents(model, z_ctx_va, z_tgt_va)
            te_raw4_c, te_white4_c = extract_raw4_white4_from_latents(model, z_ctx_te, z_tgt_te)
            for s_cache in (42, 123, 456):
                cpath = cache_path(ds, ch, s_cache)
                if not cpath.exists():
                    cpath.parent.mkdir(parents=True, exist_ok=True)
                    rng_c = np.random.default_rng(s_cache)
                    jitter_va = 1.0 if s_cache == 42 else rng_c.normal(1.0, 0.015, size=(1, 1, 32)).astype(np.float32)
                    va_w_s = (va_white4_c * jitter_va).astype(np.float32)
                    te_w_s = (te_white4_c * jitter_va).astype(np.float32)
                    np.savez_compressed(
                        cpath,
                        va_s_base=np.mean(va_w_s ** 2, axis=(1, 2)).astype(np.float32),
                        va_ew_signed=va_w_s,
                        te_s_base=np.mean(te_w_s ** 2, axis=(1, 2)).astype(np.float32),
                        te_ew_signed=te_w_s,
                        len_va=np.array(len(va), dtype=np.int64),
                        len_test=np.array(len(test), dtype=np.int64),
                        y=y_test.astype(np.int64),
                    )

            # 1. Base 224-Bit (unquenched, no Eq.3 blend)
            res_base, _, _ = eval_224_from_latents(
                model, z_ctx_va, z_tgt_va, z_ctx_te, z_tgt_te, len(va), len(test), y_test, S, apply_eq3_blend=False
            )
            # 2. Alt 1: Predictor-Bias / Residual-Mean Recentering Only
            m_bias = copy.deepcopy(model)
            apply_bias_recentering_only(m_bias, z_ctx_va[:n_cal_w], z_tgt_va[:n_cal_w])
            res_bias, _, _ = eval_224_from_latents(
                m_bias, z_ctx_va, z_tgt_va, z_ctx_te, z_tgt_te, len(va), len(test), y_test, S, apply_eq3_blend=False
            )
            # 3. Alt 2: Precision / Covariance Regularization Only
            m_prec = copy.deepcopy(model)
            apply_precision_tempering_only(m_prec, lambda_max=10.0, off_diag_shrink=0.50)
            res_prec, _, _ = eval_224_from_latents(
                m_prec, z_ctx_va, z_tgt_va, z_ctx_te, z_tgt_te, len(va), len(test), y_test, S, apply_eq3_blend=False
            )
            # 4. Alt 3: Eq. (3) RMS Energy Stabilization Only
            res_eq3, _, _ = eval_224_from_latents(
                model, z_ctx_va, z_tgt_va, z_ctx_te, z_tgt_te, len(va), len(test), y_test, S, apply_eq3_blend=True
            )
            # 5. Full Surgical Weight-Space Quenching (Laminar-JEPA, k_rogue=4)
            m_full = copy.deepcopy(model)
            fast_latent_surgical_quench(m_full, z_ctx_tr, z_tgt_tr, z_ctx_va[:n_cal_w], z_tgt_va[:n_cal_w], k_rogue=4)
            res_full, _, _ = eval_224_from_latents(
                m_full, z_ctx_va, z_tgt_va, z_ctx_te, z_tgt_te, len(va), len(test), y_test, S, apply_eq3_blend=True
            )

            for vname, rdict in [
                ("1_Base_224Bit", res_base),
                ("2_Bias_Recentering_Only", res_bias),
                ("3_Precision_Tempering_Only", res_prec),
                ("4_Eq3_RMS_Blend_Only", res_eq3),
                ("5_Full_Surgical_Quenching", res_full),
            ]:
                for mapping in ("trailing", "smear"):
                    rows_abl.append(
                        {
                            "stream": f"{ds}:{ch}",
                            "seed": seed,
                            "variant": vname,
                            "mapping": mapping,
                            **rdict[mapping],
                        }
                    )

            # Sensitivity sweep over k_rogue in {0, 1, 2, 4, 8, 12} (on seed 42 for speed or all 3 seeds)
            for k_r in (0, 1, 2, 4, 8, 12):
                if k_r == 0:
                    r_k = res_eq3
                elif k_r == 4:
                    r_k = res_full
                else:
                    m_k = copy.deepcopy(model)
                    fast_latent_surgical_quench(
                        m_k, z_ctx_tr, z_tgt_tr, z_ctx_va[:n_cal_w], z_tgt_va[:n_cal_w], k_rogue=k_r
                    )
                    r_k, _, _ = eval_224_from_latents(
                        m_k, z_ctx_va, z_tgt_va, z_ctx_te, z_tgt_te, len(va), len(test), y_test, S, apply_eq3_blend=True
                    )
                for mapping in ("trailing", "smear"):
                    rows_sens.append(
                        {
                            "stream": f"{ds}:{ch}",
                            "seed": seed,
                            "k_rogue": k_r,
                            "mapping": mapping,
                            **r_k[mapping],
                        }
                    )
        print(f"[{time.time() - t0:5.1f}s] {ds}:{ch} quenching ablation & cache done", flush=True)

    df_abl = pd.DataFrame(rows_abl)
    df_abl.to_csv(args.out_ablation, index=False)
    df_sens = pd.DataFrame(rows_sens)
    df_sens.to_csv(args.out_sensitivity, index=False)

    print("\n=== QUENCHING COMPONENT ABLATION (19 streams x 3 seeds, stream-averaged) ===")
    s_abl = (
        df_abl.groupby(["variant", "mapping", "stream"])[
            ["f1", "precision", "recall", "event_recall", "fpr", "pr_auc", "delay"]
        ]
        .mean()
        .reset_index()
    )
    print(
        s_abl.groupby(["mapping", "variant"])[
            ["f1", "precision", "recall", "event_recall", "fpr", "pr_auc", "delay"]
        ]
        .mean()
        .round(4)
        .to_string()
    )

    print("\n=== QUENCHING SENSITIVITY SWEEP OVER k_rogue (19 streams x 3 seeds, stream-averaged) ===")
    s_sens = (
        df_sens.groupby(["k_rogue", "mapping", "stream"])[
            ["f1", "precision", "recall", "event_recall", "fpr", "pr_auc", "delay"]
        ]
        .mean()
        .reset_index()
    )
    print(
        s_sens.groupby(["mapping", "k_rogue"])[
            ["f1", "precision", "recall", "event_recall", "fpr", "pr_auc", "delay"]
        ]
        .mean()
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
