"""Activation Tracing Harness for the Baseline Architectures:
1. TS-JEPA (Unconstrained Latent Predictive Control)
2. TranAD (Adversarial Transformer Reconstructive SOTA 2022)
3. TimesNet (2D Inception Reconstructive SOTA 2023)

Traces:
- Layer-by-layer tensor dimensions, means, standard deviations, and Frobenius norms.
- Reconstructive observation-space residuals vs. Latent predictive discrepancies.
- Nominal vs. Anomalous response and empirical demonstration of Noise Drowning.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import (
    HybridTCNEncoder,
    TimesNet,
    TranAD,
    TSJEPAModel,
)
try:
    from scripts.trace_model_activations import ActivationTracer, generate_synthetic_telemetry
except ModuleNotFoundError:
    from trace_model_activations import ActivationTracer, generate_synthetic_telemetry



def trace_ts_jepa(
    ctx_nom: torch.Tensor, tgt_nom: torch.Tensor,
    ctx_anom: torch.Tensor, tgt_anom: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    channels = ctx_nom.size(-1)
    latent_dim = 32
    encoder = HybridTCNEncoder(input_dim=channels, latent_dim=latent_dim, filters=48, tcn_layers=3).to(device)
    model = TSJEPAModel(
        context_encoder=encoder,
        latent_dim=latent_dim,
        predictor_hidden_dim=64,
        predictor_layers=2,
    ).to(device)
    model.eval()

    model.fit_mahalanobis_covariance(ctx_nom, tgt_nom, batch_size=4)

    tracer = ActivationTracer()
    tracer.register("context_encoder", model.context_encoder)
    tracer.register("predictor", model.predictor)
    tracer.register("target_encoder", model.target_encoder)
    tracer.register("saliency_gate", model.saliency_gate)

    # 1. Forward Nominal
    with torch.no_grad():
        disc_nom = model.compute_predictive_discrepancy(ctx_nom.to(device), tgt_nom.to(device), use_mahalanobis=True)
        z_ctx_nom = model.context_encoder(ctx_nom.to(device))
        z_pred_nom = model.predictor(z_ctx_nom)
        z_tgt_nom = model.target_encoder(tgt_nom.to(device))
    records_nom = list(tracer.records)
    tracer.clear()

    # 2. Forward Anomalous
    with torch.no_grad():
        disc_anom = model.compute_predictive_discrepancy(ctx_anom.to(device), tgt_anom.to(device), use_mahalanobis=True)
        z_ctx_anom = model.context_encoder(ctx_anom.to(device))
        z_pred_anom = model.predictor(z_ctx_anom)
        z_tgt_anom = model.target_encoder(tgt_anom.to(device))
    records_anom = list(tracer.records)
    tracer.remove_hooks()

    return {
        "model_name": "TS-JEPA (Vanilla Control)",
        "type": "Predictive Latent",
        "records_nom": records_nom,
        "records_anom": records_anom,
        "metrics": {
            "discrepancy_nominal": disc_nom.mean().item(),
            "discrepancy_anomalous": disc_anom.mean().item(),
            "discrepancy_ratio": (disc_anom.mean() / (disc_nom.mean() + 1e-6)).item(),
            "z_ctx_norm_nom": z_ctx_nom.norm(dim=-1).mean().item(),
            "z_pred_norm_nom": z_pred_nom.norm(dim=-1).mean().item(),
            "z_tgt_norm_nom": z_tgt_nom.norm(dim=-1).mean().item(),
            "latent_diff_norm_nom": (z_tgt_nom - z_pred_nom).norm(dim=-1).mean().item(),
            "latent_diff_norm_anom": (z_tgt_anom - z_pred_anom).norm(dim=-1).mean().item(),
        },
    }


def trace_tranad(
    ctx_nom: torch.Tensor, tgt_nom: torch.Tensor,
    ctx_anom: torch.Tensor, tgt_anom: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    channels = ctx_nom.size(-1)
    context_len = ctx_nom.size(1)
    
    # Full window input [B, 320, 9]
    full_nom = torch.cat([ctx_nom, tgt_nom], dim=1).to(device)
    full_anom = torch.cat([ctx_anom, tgt_anom], dim=1).to(device)

    model = TranAD(
        c_in=channels,
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_layers=2,
        d_ff=128,
        dropout=0.10,
    ).to(device)
    model.eval()

    tracer = ActivationTracer()
    tracer.register("embedding", model.embedding)
    tracer.register("pos_enc", model.pos_enc)
    for i, blk in enumerate(model.encoder_blocks):
        tracer.register(f"encoder_block_{i}", blk)
    for i, blk in enumerate(model.decoder1_blocks):
        tracer.register(f"decoder1_block_{i}", blk)
    tracer.register("proj1 (coarse rec)", model.proj1)
    for i, blk in enumerate(model.decoder2_blocks):
        tracer.register(f"decoder2_block_{i}", blk)
    tracer.register("proj2 (fine rec)", model.proj2)

    with torch.no_grad():
        rec1_nom, rec2_nom = model(full_nom)
        scores_nom = model.compute_anomaly_scores(full_nom)  # [B, 320]
        # Suspect horizon score
        suspect_score_nom = scores_nom[:, context_len:].mean(dim=-1)
        # Context horizon score (ambient noise)
        context_score_nom = scores_nom[:, :context_len].mean(dim=-1)
    records_nom = list(tracer.records)
    tracer.clear()

    with torch.no_grad():
        rec1_anom, rec2_anom = model(full_anom)
        scores_anom = model.compute_anomaly_scores(full_anom)
        suspect_score_anom = scores_anom[:, context_len:].mean(dim=-1)
        context_score_anom = scores_anom[:, :context_len].mean(dim=-1)
    records_anom = list(tracer.records)
    tracer.remove_hooks()

    # Noise drowning ratio: ratio of anomaly error in target horizon vs normal noise in context horizon
    return {
        "model_name": "TranAD (VLDB 2022)",
        "type": "Reconstructive Adversarial Transformer",
        "records_nom": records_nom,
        "records_anom": records_anom,
        "metrics": {
            "score_nominal_context": context_score_nom.mean().item(),
            "score_nominal_suspect": suspect_score_nom.mean().item(),
            "score_anomalous_suspect": suspect_score_anom.mean().item(),
            "anomaly_amplification_ratio": (suspect_score_anom.mean() / (suspect_score_nom.mean() + 1e-6)).item(),
            "coarse_rec1_error_nom": torch.mean((rec1_nom - full_nom) ** 2).item(),
            "coarse_rec1_error_anom": torch.mean((rec1_anom - full_anom) ** 2).item(),
            "fine_rec2_error_nom": torch.mean((rec2_nom - full_nom) ** 2).item(),
            "fine_rec2_error_anom": torch.mean((rec2_anom - full_anom) ** 2).item(),
        },
    }


def trace_timesnet(
    ctx_nom: torch.Tensor, tgt_nom: torch.Tensor,
    ctx_anom: torch.Tensor, tgt_anom: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    channels = ctx_nom.size(-1)
    context_len = ctx_nom.size(1)

    full_nom = torch.cat([ctx_nom, tgt_nom], dim=1).to(device)
    full_anom = torch.cat([ctx_anom, tgt_anom], dim=1).to(device)

    model = TimesNet(
        c_in=channels,
        d_model=64,
        d_ff=64,
        e_layers=2,
        top_k=3,
        num_kernels=3,
        dropout=0.10,
    ).to(device)
    model.eval()

    tracer = ActivationTracer()
    tracer.register("embedding", model.embedding)
    tracer.register("layer_norm", model.layer_norm)
    for i, blk in enumerate(model.blocks):
        tracer.register(f"times_block_{i}", blk)
        for j, conv in enumerate(blk.conv):
            tracer.register(f"times_block_{i}_inception_conv_{j}", conv)
    tracer.register("projection (reconstruction)", model.projection)

    with torch.no_grad():
        rec_nom = model(full_nom)
        scores_nom = model.compute_anomaly_scores(full_nom)  # [B, 320]
        suspect_score_nom = scores_nom[:, context_len:].mean(dim=-1)
        context_score_nom = scores_nom[:, :context_len].mean(dim=-1)
    records_nom = list(tracer.records)
    tracer.clear()

    with torch.no_grad():
        rec_anom = model(full_anom)
        scores_anom = model.compute_anomaly_scores(full_anom)
        suspect_score_anom = scores_anom[:, context_len:].mean(dim=-1)
        context_score_anom = scores_anom[:, :context_len].mean(dim=-1)
    records_anom = list(tracer.records)
    tracer.remove_hooks()

    # Extract dominant FFT periods
    from src.models.baselines.timesnet import FFT_for_Period
    with torch.no_grad():
        periods_nom, weights_nom = FFT_for_Period(model.embedding(full_nom), k=3)
        periods_anom, weights_anom = FFT_for_Period(model.embedding(full_anom), k=3)

    return {
        "model_name": "TimesNet (ICLR 2023)",
        "type": "Reconstructive 2D Temporal Variation",
        "records_nom": records_nom,
        "records_anom": records_anom,
        "metrics": {
            "score_nominal_context": context_score_nom.mean().item(),
            "score_nominal_suspect": suspect_score_nom.mean().item(),
            "score_anomalous_suspect": suspect_score_anom.mean().item(),
            "anomaly_amplification_ratio": (suspect_score_anom.mean() / (suspect_score_nom.mean() + 1e-6)).item(),
            "reconstruction_error_nom": torch.mean((rec_nom - full_nom) ** 2).item(),
            "reconstruction_error_anom": torch.mean((rec_anom - full_anom) ** 2).item(),
            "dominant_periods_nom": periods_nom,
            "period_weights_nom": weights_nom.mean(dim=0).cpu().tolist() if weights_nom.ndim > 1 else weights_nom.cpu().tolist(),
            "dominant_periods_anom": periods_anom,
            "period_weights_anom": weights_anom.mean(dim=0).cpu().tolist() if weights_anom.ndim > 1 else weights_anom.cpu().tolist(),
        },
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Baseline Activation Trace on device: {device}")

    # Generate synthetic telemetry testbatch (B=4, C=256, S=64, K=9)
    ctx_nom, tgt_nom, ctx_anom, tgt_anom = generate_synthetic_telemetry(
        batch_size=4, context_len=256, target_len=64, channels=9, seed=42
    )

    # 1. TS-JEPA Trace
    print("\n" + "=" * 90)
    print("ACTIVATION TRACE: TS-JEPA (Unconstrained Latent Predictive Control)")
    print("=" * 90)
    res_ts = trace_ts_jepa(ctx_nom, tgt_nom, ctx_anom, tgt_anom, device)
    print(f"{'Module / Sub-layer':<32} | {'Nominal Output Shape':<22} | {'Nom Norm':<10} | {'Anom Norm':<10} | {'Ratio':<8}")
    print("-" * 90)
    for r_nom, r_anom in zip(res_ts["records_nom"], res_ts["records_anom"]):
        nom_norm = r_nom.get("norm", 0.0)
        anom_norm = r_anom.get("norm", 0.0)
        ratio = anom_norm / (nom_norm + 1e-6)
        print(f"{r_nom['name']:<32} | {str(r_nom['output_shape']):<22} | {nom_norm:<10.3f} | {anom_norm:<10.3f} | {ratio:<8.2f}x")

    print("\n--- Diagnostic Metrics ---")
    for k, v in res_ts["metrics"].items():
        if isinstance(v, float):
            print(f"  • {k:<30}: {v:.4f}")
        else:
            print(f"  • {k:<30}: {v}")

    # 2. TranAD Trace
    print("\n" + "=" * 90)
    print("ACTIVATION TRACE: TranAD (Adversarial Transformer SOTA 2022)")
    print("=" * 90)
    res_tranad = trace_tranad(ctx_nom, tgt_nom, ctx_anom, tgt_anom, device)
    print(f"{'Module / Sub-layer':<32} | {'Nominal Output Shape':<22} | {'Nom Norm':<10} | {'Anom Norm':<10} | {'Ratio':<8}")
    print("-" * 90)
    for r_nom, r_anom in zip(res_tranad["records_nom"], res_tranad["records_anom"]):
        nom_norm = r_nom.get("norm", 0.0)
        anom_norm = r_anom.get("norm", 0.0)
        ratio = anom_norm / (nom_norm + 1e-6)
        print(f"{r_nom['name']:<32} | {str(r_nom['output_shape']):<22} | {nom_norm:<10.3f} | {anom_norm:<10.3f} | {ratio:<8.2f}x")

    print("\n--- Diagnostic Metrics & Reconstructive Error Breakdown ---")
    for k, v in res_tranad["metrics"].items():
        if isinstance(v, float):
            print(f"  • {k:<30}: {v:.4f}")
        else:
            print(f"  • {k:<30}: {v}")

    # 3. TimesNet Trace
    print("\n" + "=" * 90)
    print("ACTIVATION TRACE: TimesNet (2D Inception Reconstructive SOTA 2023)")
    print("=" * 90)
    res_timesnet = trace_timesnet(ctx_nom, tgt_nom, ctx_anom, tgt_anom, device)
    print(f"{'Module / Sub-layer':<32} | {'Nominal Output Shape':<22} | {'Nom Norm':<10} | {'Anom Norm':<10} | {'Ratio':<8}")
    print("-" * 90)
    for r_nom, r_anom in zip(res_timesnet["records_nom"], res_timesnet["records_anom"]):
        nom_norm = r_nom.get("norm", 0.0)
        anom_norm = r_anom.get("norm", 0.0)
        ratio = anom_norm / (nom_norm + 1e-6)
        print(f"{r_nom['name']:<32} | {str(r_nom['output_shape']):<22} | {nom_norm:<10.3f} | {anom_norm:<10.3f} | {ratio:<8.2f}x")

    print("\n--- Diagnostic Metrics & FFT Frequency Decomposition ---")
    for k, v in res_timesnet["metrics"].items():
        if isinstance(v, float):
            print(f"  • {k:<30}: {v:.4f}")
        else:
            print(f"  • {k:<30}: {v}")

    # Save summary
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "activation_trace_baselines.json"
    with open(out_file, "w") as f:
        json.dump(
            {
                "ts_jepa": res_ts,
                "tranad": res_tranad,
                "timesnet": res_timesnet,
            },
            f,
            indent=2,
        )
    print(f"\nBaseline trace artifacts saved to: {out_file.as_posix()}")


if __name__ == "__main__":
    main()
