#!/usr/bin/env python3
"""Comprehensive Layer-to-Layer Activation Trace & Latent Space Diagnosis for OpEntropy.

Performs:
1. Fine-grained layer-to-layer activation tracing of all submodules in OperatorEntropyJEPAModel
   using LayerActivationTracer and LayerAutopsy:
   - Numerical health (mean, std, L2 norm, min/max, sparsity, stable rank, batch variance).
   - Health flags (HEALTHY, DEAD_NEURON, COLLAPSED, EXPLODING, NAN_INF).
   - Sequential drift (cosine similarity between adjacent layers).
   - Nominal vs. anomalous layer perturbation divergence on real telemetry.
2. Forensic Latent Space Diagnosis:
   - Covariance spectrum (eigenvalues, condition number, RankMe, Participation Ratio).
   - Koopman operator spectral analysis (singular values, spectral radius, operator RankMe).
   - Representation rank comparison: Upgraded OpEntropy vs. Unconstrained TS-JEPA vs. Historical OpEntropy (alpha_Z=0).
   - Anomaly separability: Mahalanobis distance, centroid displacement, residual AUROC, bilateral energy discrepancy.
   - Latent dimension fault localization (coordinate saliency).

Outputs:
  reports/latent_diagnosis/opentropy_activation_trace.json
  reports/latent_diagnosis/opentropy_latent_diagnosis.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CSMConfig
from src.engine.activation_tracer import LayerActivationTracer, LayerAutopsy
from src.engine.trainer import build_ts_jepa_model
from src.models.jepa.operator_entropy_jepa import (
    OperatorEntropyJEPAModel,
    compute_empirical_covariance,
    covariance_spectral_entropy,
    spectral_operator_entropy_loss,
)


def load_telemetry_stream(
    dataset: str = "SMAP",
    channel: str = "P-3",
    context_len: int = 256,
    target_len: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load real telemetry traces, create sliding windows and separate nominal vs anomalous windows."""
    d = ROOT / "mTSBench_data" / dataset
    tr_path = d / f"{dataset}_{channel}_train.csv"
    if not tr_path.exists():
        tr_path = d / f"{channel}_train.csv"
    te_path = d / f"{dataset}_{channel}_test.csv"
    if not te_path.exists():
        te_path = d / f"{channel}_test.csv"

    df_tr = pd.read_csv(tr_path)
    df_te = pd.read_csv(te_path)

    feat_cols = [c for c in df_tr.columns if c not in ["timestamp", "is_anomaly", "label"]]
    lbl_cols = [c for c in df_te.columns if c in ["is_anomaly", "label", "labels"]]

    tr_data = np.nan_to_num(df_tr[feat_cols].to_numpy(), nan=0.0).astype(np.float32)
    te_data = np.nan_to_num(df_te[feat_cols].to_numpy(), nan=0.0).astype(np.float32)
    labels = df_te[lbl_cols[0]].to_numpy().astype(int) if lbl_cols else np.zeros(len(df_te), dtype=int)

    # Z-score standardize using strictly the training split statistics
    mu = tr_data.mean(axis=0, keepdims=True)
    sigma = tr_data.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    tr_data = (tr_data - mu) / sigma
    te_data = (te_data - mu) / sigma

    W = context_len + target_len

    # Build windows from test set
    te_wins = []
    win_labels = []
    for i in range(len(te_data) - W + 1):
        te_wins.append(te_data[i : i + W])
        # A window is anomalous if target horizon has anomaly
        win_labels.append(int(labels[i + context_len : i + W].any()))

    te_wins = np.array(te_wins, dtype=np.float32)
    win_labels = np.array(win_labels, dtype=int)

    nom_mask = win_labels == 0
    anom_mask = win_labels == 1

    nom_wins = te_wins[nom_mask]
    anom_wins = te_wins[anom_mask]

    return tr_data, te_wins, win_labels, nom_wins, anom_wins


def compute_spectral_metrics(cov: torch.Tensor) -> Dict[str, float]:
    """Compute RankMe, Participation Ratio, Condition Number, and Top-PC share from covariance."""
    cov_sym = 0.5 * (cov + cov.T).float()
    evals = torch.linalg.eigvalsh(cov_sym)
    evals = torch.clamp(evals, min=1e-12)
    evals_desc = torch.sort(evals, descending=True).values

    total_var = torch.sum(evals_desc)
    p = evals_desc / total_var

    # Von Neumann entropy & RankMe
    vn_entropy = -torch.sum(p * torch.log(p))
    rank_me = float(torch.exp(vn_entropy).item())

    # Participation Ratio: (Tr Sigma)^2 / Tr(Sigma^2)
    pr = float(((total_var ** 2) / torch.sum(evals_desc ** 2)).item())

    # Condition number
    cond_num = float((evals_desc[0] / evals_desc[-1]).item())

    # Top PC shares
    top1_share = float(p[0].item())
    top3_share = float(torch.sum(p[:3]).item())

    return {
        "rank_me": rank_me,
        "participation_ratio": pr,
        "condition_number": cond_num,
        "top1_pc_share": top1_share,
        "top3_pc_share": top3_share,
        "total_variance": float(total_var.item()),
        "min_eigenvalue": float(evals_desc[-1].item()),
        "max_eigenvalue": float(evals_desc[0].item()),
    }


def train_model(
    model_type: str,
    tr_data: np.ndarray,
    channels: int,
    device: torch.device,
    epochs: int = 5,
    alpha_rep_entropy: float = 0.02,
    seed: int = 42,
) -> nn.Module:
    """Train a JEPA model on nominal telemetry windows with locked RNG seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cfg = CSMConfig(
        model_type=model_type,
        feature_dim=channels,
        latent_dim=32,
        use_koopman_matrix=(model_type == "operator_entropy_jepa"),
        alpha_rep_entropy=alpha_rep_entropy,
    )
    model = build_ts_jepa_model(cfg, channels, device).to(device)

    C, S = 256, 64
    W = C + S
    n_win = len(tr_data) - W + 1
    tr_wins = np.array([tr_data[i : i + W] for i in range(n_win)], dtype=np.float32)

    bs = min(64, n_win)
    idx = np.arange(n_win)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    for ep in range(epochs):
        model.train()
        np.random.shuffle(idx)
        for s in range(0, n_win, bs):
            b = torch.from_numpy(tr_wins[idx[s : s + bs]]).to(device)
            loss, _ = model.compute_objective(b[:, :C], b[:, C:], config=cfg)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            if hasattr(model, "update_target_encoder"):
                model.update_target_encoder()

    model.eval()
    if hasattr(model, "fit_mahalanobis_covariance"):
        tr_t = torch.from_numpy(tr_wins)
        model.fit_mahalanobis_covariance(tr_t[:, :C], tr_t[:, C:], batch_size=256)
    return model


def main():
    print("=" * 70)
    print(" FORENSIC LAYER ACTIVATION TRACE & LATENT DIAGNOSIS: OpEntropy")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load real telemetry stream (SMAP:P-3, K=25 channels)
    dataset, channel = "SMAP", "P-3"
    print(f"Loading stream {dataset}:{channel}...")
    tr_data, te_wins, win_labels, nom_wins, anom_wins = load_telemetry_stream(
        dataset=dataset, channel=channel
    )
    channels = tr_data.shape[-1]
    print(f"   Train observations: {len(tr_data):,}, Channels: {channels}")
    print(f"   Total test windows: {len(te_wins):,} (Nominal: {len(nom_wins):,}, Anomalous: {len(anom_wins):,})")

    # 2. Train models:
    #   (a) Upgraded OpEntropy (alpha_rep_entropy = 0.02)
    #   (b) Historical OpEntropy (alpha_rep_entropy = 0.0, single-operator entropy)
    #   (c) TS-JEPA (unconstrained MLP predictor baseline)
    print("\n--- Training Models for Diagnostic Comparison ---")
    print("1. Training Upgraded OpEntropy (with dual spectral entropy coupling)...")
    model_upgraded = train_model("operator_entropy_jepa", tr_data, channels, device, epochs=5, alpha_rep_entropy=0.02)

    print("2. Training Historical OpEntropy (alpha_rep=0, single-operator entropy)...")
    model_historical = train_model("operator_entropy_jepa", tr_data, channels, device, epochs=5, alpha_rep_entropy=0.0)

    print("3. Training Vanilla TS-JEPA (unconstrained control)...")
    model_jepa = train_model("ts_jepa", tr_data, channels, device, epochs=5)

    # ----------------------------------------------------------------------- #
    # PART 1: LAYER-TO-LAYER ACTIVATION TRACE & AUTOPSY
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print(" PART 1: LAYER-TO-LAYER ACTIVATION TRACE (Upgraded OpEntropy)")
    print("=" * 70)

    C, S = 256, 64
    eval_bs = 64
    ctx_nom = torch.from_numpy(nom_wins[:eval_bs, :C]).to(device)
    tgt_nom = torch.from_numpy(nom_wins[:eval_bs, C:]).to(device)

    ctx_anom = torch.from_numpy(anom_wins[:eval_bs, :C]).to(device)
    tgt_anom = torch.from_numpy(anom_wins[:eval_bs, C:]).to(device)

    with LayerActivationTracer(model_upgraded, leaf_only=True, capture_tensors=True, cpu_offload=True) as nom_tracer:
        with torch.no_grad():
            model_upgraded(ctx_nom, tgt_nom)

    with LayerActivationTracer(model_upgraded, leaf_only=True, capture_tensors=True, cpu_offload=True) as anom_tracer:
        with torch.no_grad():
            model_upgraded(ctx_anom, tgt_anom)

    autopsy = LayerAutopsy.inspect(
        nominal_tracer=nom_tracer,
        anomaly_tracer=anom_tracer,
        model_name="Upgraded OpEntropy JEPA",
    )

    print(autopsy.summary())
    df_layers = autopsy.to_dataframe()

    # Save activation trace JSON
    out_dir = ROOT / "reports" / "latent_diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_json_path = out_dir / "opentropy_activation_trace.json"

    trace_records_data = []
    for r in autopsy.records:
        rec_dict = {
            "name": r.name,
            "module_type": r.module_type,
            "output_shape": r.output_shape,
            "mean": r.mean,
            "std": r.std,
            "norm": r.norm,
            "min_val": r.min_val,
            "max_val": r.max_val,
            "sparsity": r.sparsity,
            "stable_rank": r.stable_rank,
            "batch_variance": r.batch_variance,
            "health_flag": r.health_flag,
            "sequential_cosine_sim": autopsy.get_sequential_cosine_sim(r.name),
            "anomaly_divergence": autopsy.anomaly_divergence.get(r.name, None),
        }
        trace_records_data.append(rec_dict)

    with open(trace_json_path, "w") as f:
        json.dump(
            {
                "model": "Upgraded OperatorEntropyJEPAModel",
                "dataset": f"{dataset}:{channel}",
                "global_health": autopsy.global_health,
                "collapsed_layers": autopsy.collapsed_layers,
                "dead_layers": autopsy.dead_layers,
                "unstable_layers": autopsy.unstable_layers,
                "records": trace_records_data,
            },
            f,
            indent=2,
        )
    print(f"\n[Saved layer trace to {trace_json_path}]")

    print("\nSelected Layer Profiling Table:")
    cols_to_show = ["layer_name", "module_type", "output_shape", "l2_norm", "stable_rank", "sparsity", "health_flag"]
    if "anomaly_divergence" in df_layers.columns:
        cols_to_show.append("anomaly_divergence")
    print(df_layers[cols_to_show].head(20).to_string(index=False))

    # ----------------------------------------------------------------------- #
    # PART 2: LATENT SPACE SPECTRUM & METRIC GEOMETRY DIAGNOSIS
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print(" PART 2: LATENT SPACE SPECTRUM & DYNAMICAL DIAGNOSIS")
    print("=" * 70)

    # Extract full-batch representations on test windows
    def extract_all_latents(model: nn.Module, wins: np.ndarray) -> Dict[str, torch.Tensor]:
        all_z_ctx, all_z_tgt, all_z_pred, all_res, all_disc = [], [], [], [], []
        with torch.no_grad():
            for i in range(0, len(wins), 128):
                b = torch.from_numpy(wins[i : i + 128]).to(device)
                ctx_b, tgt_b = b[:, :C], b[:, C:]
                if isinstance(model, OperatorEntropyJEPAModel):
                    z_pred, z_tgt, _ = model(ctx_b, tgt_b)
                    z_ctx = model._apply_filters(model.context_encoder(ctx_b))
                else:
                    z_ctx, z_tgt, z_pred = model(ctx_b, tgt_b)
                res = z_tgt - z_pred
                disc = model.compute_predictive_discrepancy(ctx_b, tgt_b)

                all_z_ctx.append(z_ctx.cpu())
                all_z_tgt.append(z_tgt.cpu())
                all_z_pred.append(z_pred.cpu())
                all_res.append(res.cpu())
                all_disc.append(disc.cpu())

        return {
            "z_ctx": torch.cat(all_z_ctx, dim=0),
            "z_tgt": torch.cat(all_z_tgt, dim=0),
            "z_pred": torch.cat(all_z_pred, dim=0),
            "residual": torch.cat(all_res, dim=0),
            "discrepancy": torch.cat(all_disc, dim=0),
        }

    print("Extracting latent manifolds on full test sequence...")
    lat_up = extract_all_latents(model_upgraded, te_wins)
    lat_hist = extract_all_latents(model_historical, te_wins)
    lat_jepa = extract_all_latents(model_jepa, te_wins)

    # Covariance spectra on nominal test samples
    nom_idx = torch.from_numpy(win_labels == 0)
    anom_idx = torch.from_numpy(win_labels == 1)

    cov_z_up = compute_empirical_covariance(lat_up["z_ctx"][nom_idx])
    cov_z_hist = compute_empirical_covariance(lat_hist["z_ctx"][nom_idx])
    cov_z_jepa = compute_empirical_covariance(lat_jepa["z_ctx"][nom_idx])

    spec_up = compute_spectral_metrics(cov_z_up)
    spec_hist = compute_spectral_metrics(cov_z_hist)
    spec_jepa = compute_spectral_metrics(cov_z_jepa)

    print("\n--- Latent Representation Rank & Conditioning (Z_ctx) ---")
    spec_comp_df = pd.DataFrame([
        {
            "Model": "Upgraded OpEntropy (Coupled)",
            "RankMe (exp H)": spec_up["rank_me"],
            "Part. Ratio": spec_up["participation_ratio"],
            "Condition #": spec_up["condition_number"],
            "Top-1 PC %": spec_up["top1_pc_share"] * 100,
            "Top-3 PC %": spec_up["top3_pc_share"] * 100,
            "Total Var": spec_up["total_variance"],
        },
        {
            "Model": "Historical OpEntropy (alpha_Z=0)",
            "RankMe (exp H)": spec_hist["rank_me"],
            "Part. Ratio": spec_hist["participation_ratio"],
            "Condition #": spec_hist["condition_number"],
            "Top-1 PC %": spec_hist["top1_pc_share"] * 100,
            "Top-3 PC %": spec_hist["top3_pc_share"] * 100,
            "Total Var": spec_hist["total_variance"],
        },
        {
            "Model": "Vanilla TS-JEPA (Unconstrained)",
            "RankMe (exp H)": spec_jepa["rank_me"],
            "Part. Ratio": spec_jepa["participation_ratio"],
            "Condition #": spec_jepa["condition_number"],
            "Top-1 PC %": spec_jepa["top1_pc_share"] * 100,
            "Top-3 PC %": spec_jepa["top3_pc_share"] * 100,
            "Total Var": spec_jepa["total_variance"],
        },
    ])
    print(spec_comp_df.round(3).to_string(index=False))

    # Koopman Operator Analysis
    print("\n--- Koopman Operator Properties ---")
    K_up = model_upgraded.K.detach().cpu().float()
    K_hist = model_historical.K.detach().cpu().float()

    s_up = torch.linalg.svdvals(K_up)
    s_hist = torch.linalg.svdvals(K_hist)

    eig_up = torch.linalg.eigvals(K_up)
    eig_hist = torch.linalg.eigvals(K_hist)

    def koopman_stats(K: torch.Tensor, s: torch.Tensor, eigs: torch.Tensor) -> Dict[str, float]:
        p = (s ** 2) / torch.sum(s ** 2)
        vn_K = float(torch.exp(-torch.sum(p * torch.log(torch.clamp(p, min=1e-12)))).item())
        rho = float(torch.max(torch.abs(eigs)).item())
        frob = float(torch.linalg.norm(K).item())
        s_max = float(s[0].item())
        s_min = float(s[-1].item())
        return {
            "operator_rank_me": vn_K,
            "spectral_radius": rho,
            "frobenius_norm": frob,
            "max_singular_val": s_max,
            "min_singular_val": s_min,
            "condition_number": s_max / (s_min + 1e-12),
        }

    kstats_up = koopman_stats(K_up, s_up, eig_up)
    kstats_hist = koopman_stats(K_hist, s_hist, eig_hist)

    k_df = pd.DataFrame([
        {"Model": "Upgraded OpEntropy", **kstats_up},
        {"Model": "Historical OpEntropy", **kstats_hist},
    ])
    print(k_df.round(4).to_string(index=False))

    # Anomaly Separation & Geometry
    print("\n--- Anomaly Separability & Geometry ---")
    def separation_stats(lat: Dict[str, torch.Tensor], y: np.ndarray) -> Dict[str, float]:
        res = lat["residual"]
        res_norm = torch.linalg.norm(res, dim=-1).numpy()
        disc = lat["discrepancy"].numpy()

        nom_res = res[y == 0]
        anom_res = res[y == 1]

        mu_nom = nom_res.mean(dim=0)
        mu_anom = anom_res.mean(dim=0)
        centroid_dist = float(torch.linalg.norm(mu_anom - mu_nom).item())

        # Residual covariance on nominal
        cov_res = compute_empirical_covariance(nom_res)
        cov_inv = torch.linalg.pinv(cov_res + 1e-4 * torch.eye(cov_res.size(0)))
        diff = (mu_anom - mu_nom).unsqueeze(0)
        mahal_dist = float(torch.sqrt(torch.clamp(diff @ cov_inv @ diff.T, min=0.0)).item())

        # AUROCs
        auroc_res = float(roc_auc_score(y, res_norm))
        auroc_disc = float(roc_auc_score(y, disc))

        # Separation Ratio
        sep_ratio = float((res_norm[y == 1].mean() / (res_norm[y == 0].mean() + 1e-6)))

        return {
            "auroc_discrepancy": auroc_disc,
            "auroc_residual_l2": auroc_res,
            "separation_ratio": sep_ratio,
            "centroid_displacement": centroid_dist,
            "mahalanobis_displacement": mahal_dist,
        }

    y_arr = win_labels
    sep_up = separation_stats(lat_up, y_arr)
    sep_hist = separation_stats(lat_hist, y_arr)
    sep_jepa = separation_stats(lat_jepa, y_arr)

    sep_df = pd.DataFrame([
        {"Model": "Upgraded OpEntropy", **sep_up},
        {"Model": "Historical OpEntropy", **sep_hist},
        {"Model": "Vanilla TS-JEPA", **sep_jepa},
    ])
    print(sep_df.round(4).to_string(index=False))

    # Dimension-wise fault localization
    print("\n--- Top Latent Coordinate Saliency on Anomalies (Upgraded OpEntropy) ---")
    nom_res = lat_up["residual"][nom_idx]
    anom_res = lat_up["residual"][anom_idx]
    dim_signal = (anom_res.pow(2).mean(dim=0) / (nom_res.pow(2).mean(dim=0) + 1e-6)).numpy()
    top_dims = np.argsort(dim_signal)[::-1]

    top_dims_summary = []
    for rank, dim_i in enumerate(top_dims[:8], 1):
        top_dims_summary.append({
            "Rank": rank,
            "Latent Dim": int(dim_i),
            "Signal-to-Noise Ratio": float(dim_signal[dim_i]),
            "Nominal Variance": float(nom_res[:, dim_i].var().item()),
            "Anomalous Energy": float(anom_res[:, dim_i].pow(2).mean().item()),
        })
    print(pd.DataFrame(top_dims_summary).round(3).to_string(index=False))

    # Save full latent diagnosis JSON
    lat_json_path = out_dir / "opentropy_latent_diagnosis.json"
    with open(lat_json_path, "w") as f:
        json.dump(
            {
                "dataset": f"{dataset}:{channel}",
                "representation_spectrum": {
                    "upgraded": spec_up,
                    "historical": spec_hist,
                    "ts_jepa": spec_jepa,
                },
                "koopman_properties": {
                    "upgraded": kstats_up,
                    "historical": kstats_hist,
                },
                "anomaly_separation": {
                    "upgraded": sep_up,
                    "historical": sep_hist,
                    "ts_jepa": sep_jepa,
                },
                "top_saliency_dimensions": top_dims_summary,
            },
            f,
            indent=2,
        )
    print(f"\n[Saved latent diagnosis to {lat_json_path}]")
    print("=" * 70)
    print(" DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
