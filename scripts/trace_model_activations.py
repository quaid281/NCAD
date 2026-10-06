"""Activation Tracing Harness for the Physics-Informed JEPA Triad.

Performs fine-grained forward-pass activation tracing across:
1. OperatorEntropyJEPAModel (Koopman Spectral Mechanics)
2. ReynoldsStressJEPAModel (Hydrodynamic Transport & Turbulent Stress)
3. PotentialFlowJEPAModel (Conservative Potential Field Theory)

Traces:
- Layer-by-layer tensor dimensions, means, standard deviations, and Frobenius norms.
- Physical invariant checks (von Neumann entropy, stress cone positive semi-definiteness,
  conservative velocity curl, Grassmannian regime routing, and coordinate saliency weights).
- Nominal vs. Anomalous activation responses.
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
    OperatorEntropyJEPAModel,
    PotentialFlowJEPAModel,
    ReynoldsStressJEPAModel,
)
from src.models.jepa.operator_entropy_jepa import von_neumann_entropy


class ActivationTracer:
    """Hooks into PyTorch modules to capture activation statistics during forward passes."""

    def __init__(self):
        self.records: List[Dict[str, Any]] = []
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def register(self, name: str, module: nn.Module):
        def hook(m, inp, out):
            rec = {
                "name": name,
                "class": m.__class__.__name__,
            }
            if isinstance(inp, tuple) and len(inp) > 0 and isinstance(inp[0], torch.Tensor):
                rec["input_shape"] = list(inp[0].shape)
            elif isinstance(inp, torch.Tensor):
                rec["input_shape"] = list(inp.shape)
            else:
                rec["input_shape"] = "N/A"

            if isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                t = out[0].detach()
                rec["output_shape"] = list(t.shape)
                rec["mean"] = float(t.mean().item())
                rec["std"] = float(t.std().item()) if t.numel() > 1 else 0.0
                rec["norm"] = float(torch.linalg.norm(t).item())
                rec["min"] = float(t.min().item())
                rec["max"] = float(t.max().item())
            elif isinstance(out, torch.Tensor):
                t = out.detach()
                rec["output_shape"] = list(t.shape)
                rec["mean"] = float(t.mean().item())
                rec["std"] = float(t.std().item()) if t.numel() > 1 else 0.0
                rec["norm"] = float(torch.linalg.norm(t).item())
                rec["min"] = float(t.min().item())
                rec["max"] = float(t.max().item())
            else:
                rec["output_shape"] = str(type(out))

            self.records.append(rec)

        handle = module.register_forward_hook(hook)
        self.hooks.append(handle)

    def clear(self):
        self.records.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def generate_synthetic_telemetry(
    batch_size: int = 4,
    context_len: int = 256,
    target_len: int = 64,
    channels: int = 9,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate realistic physical telemetry:

    - Nominal: coupled harmonic limit cycles + mild stochastic jitter.
    - Anomalous: contains abrupt phase slip, shear turbulence, and amplitude blowout.
    """
    rng = np.random.RandomState(seed)
    total_len = context_len + target_len
    t = np.linspace(0, 10 * np.pi, total_len)

    # Base coupled dynamics
    signals_nom = []
    signals_anom = []

    for b in range(batch_size):
        chan_nom = []
        chan_anom = []
        for c in range(channels):
            freq = 0.5 + 0.1 * c
            phase = (c * np.pi) / channels
            # Nominal limit cycle
            base = np.sin(freq * t + phase) + 0.3 * np.cos(2 * freq * t)
            noise = rng.normal(0, 0.05, total_len)
            s_nom = base + noise

            # Anomalous copy: inject fault in the suspect target horizon
            s_anom = s_nom.copy()
            # Anomaly starts at context_len
            fault_idx = np.arange(context_len, total_len)
            # High-frequency turbulent burst + phase reversal
            s_anom[fault_idx] = -1.8 * np.sin(3.5 * freq * t[fault_idx] + phase) + rng.normal(0, 0.3, len(fault_idx))

            chan_nom.append(s_nom)
            chan_anom.append(s_anom)

        signals_nom.append(np.stack(chan_nom, axis=-1))
        signals_anom.append(np.stack(chan_anom, axis=-1))

    arr_nom = np.array(signals_nom, dtype=np.float32)
    arr_anom = np.array(signals_anom, dtype=np.float32)

    ctx_nom = torch.from_numpy(arr_nom[:, :context_len, :])
    tgt_nom = torch.from_numpy(arr_nom[:, context_len:, :])

    ctx_anom = torch.from_numpy(arr_anom[:, :context_len, :])
    tgt_anom = torch.from_numpy(arr_anom[:, context_len:, :])

    return ctx_nom, tgt_nom, ctx_anom, tgt_anom


def trace_operator_entropy(
    ctx_nom: torch.Tensor, tgt_nom: torch.Tensor,
    ctx_anom: torch.Tensor, tgt_anom: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    channels = ctx_nom.size(-1)
    latent_dim = 32
    encoder = HybridTCNEncoder(input_dim=channels, latent_dim=latent_dim, filters=48, tcn_layers=3).to(device)
    model = OperatorEntropyJEPAModel(
        context_encoder=encoder,
        latent_dim=latent_dim,
        hidden_dim=64,
        predictor_layers=2,
        use_radial_heat=True,
    ).to(device)
    model.eval()

    # Fit covariance for realistic Mahalanobis scoring
    model.fit_mahalanobis_covariance(ctx_nom, tgt_nom, batch_size=4)

    tracer = ActivationTracer()
    tracer.register("context_encoder", model.context_encoder)
    if model.radial_heat is not None:
        tracer.register("radial_heat_exterior", model.radial_heat)
    tracer.register("cohn_elkies_filter", model.cohn_elkies)
    tracer.register("predictor_mlp", model.predictor)
    tracer.register("target_encoder", model.target_encoder)
    tracer.register("saliency_gate", model.saliency_gate)

    # 1. Forward Nominal
    with torch.no_grad():
        z_pred_nom, z_tgt_nom, _ = model(ctx_nom.to(device), tgt_nom.to(device))
        disc_nom = model.compute_predictive_discrepancy(ctx_nom.to(device), tgt_nom.to(device), use_mahalanobis=True)
    records_nom = list(tracer.records)
    tracer.clear()

    # 2. Forward Anomalous
    with torch.no_grad():
        z_pred_anom, z_tgt_anom, _ = model(ctx_anom.to(device), tgt_anom.to(device))
        disc_anom = model.compute_predictive_discrepancy(ctx_anom.to(device), tgt_anom.to(device), use_mahalanobis=True)
    records_anom = list(tracer.records)
    tracer.remove_hooks()

    # Invariant diagnostics
    vn_nom = von_neumann_entropy(z_pred_nom).item()
    vn_anom = von_neumann_entropy(z_pred_anom).item()

    energy_leak_nom = torch.clamp(torch.norm(z_tgt_nom, dim=-1) - torch.norm(z_pred_nom, dim=-1), min=0.0).mean().item()
    energy_leak_anom = torch.clamp(torch.norm(z_tgt_anom, dim=-1) - torch.norm(z_pred_anom, dim=-1), min=0.0).mean().item()

    return {
        "model_name": "OperatorEntropyJEPA",
        "records_nom": records_nom,
        "records_anom": records_anom,
        "metrics": {
            "discrepancy_nominal": disc_nom.mean().item(),
            "discrepancy_anomalous": disc_anom.mean().item(),
            "discrepancy_ratio": (disc_anom.mean() / (disc_nom.mean() + 1e-6)).item(),
            "von_neumann_deficiency_nom": vn_nom,
            "von_neumann_deficiency_anom": vn_anom,
            "energy_leak_nom": energy_leak_nom,
            "energy_leak_anom": energy_leak_anom,
            "z_pred_norm_nom": z_pred_nom.norm(dim=-1).mean().item(),
            "z_pred_norm_anom": z_pred_anom.norm(dim=-1).mean().item(),
            "z_tgt_norm_nom": z_tgt_nom.norm(dim=-1).mean().item(),
            "z_tgt_norm_anom": z_tgt_anom.norm(dim=-1).mean().item(),
        },
    }


def trace_reynolds_stress(
    ctx_nom: torch.Tensor, tgt_nom: torch.Tensor,
    ctx_anom: torch.Tensor, tgt_anom: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    channels = ctx_nom.size(-1)
    latent_dim = 32
    stress_dim = 16
    encoder = HybridTCNEncoder(input_dim=channels, latent_dim=latent_dim, filters=48, tcn_layers=3).to(device)
    model = ReynoldsStressJEPAModel(
        context_encoder=encoder,
        latent_dim=latent_dim,
        stress_dim=stress_dim,
        use_shearing_pulses=True,
        use_admissible_cone=True,
    ).to(device)
    model.eval()

    model.fit_mahalanobis_covariance(ctx_nom, tgt_nom, batch_size=4)

    tracer = ActivationTracer()
    tracer.register("context_encoder", model.context_encoder)
    if model.shearing_pulse is not None:
        tracer.register("shearing_wavelet_pulses", model.shearing_pulse)
    tracer.register("hankel_moment_filter", model.hankel_filter)
    tracer.register("predictor_mlp", model.predictor)
    tracer.register("cohn_elkies_filter", model.cohn_elkies)
    tracer.register("stress_head", model.stress_head)
    tracer.register("target_encoder", model.target_encoder)
    tracer.register("saliency_gate", model.saliency_gate)

    with torch.no_grad():
        z_pred_nom, z_tgt_nom, stress_diff_nom = model(ctx_nom.to(device), tgt_nom.to(device))
        disc_nom = model.compute_predictive_discrepancy(ctx_nom.to(device), tgt_nom.to(device), use_mahalanobis=True)
    records_nom = list(tracer.records)
    tracer.clear()

    with torch.no_grad():
        z_pred_anom, z_tgt_anom, stress_diff_anom = model(ctx_anom.to(device), tgt_anom.to(device))
        disc_anom = model.compute_predictive_discrepancy(ctx_anom.to(device), tgt_anom.to(device), use_mahalanobis=True)
    records_anom = list(tracer.records)
    tracer.remove_hooks()

    # Stress tensor physical invariant checks
    with torch.no_grad():
        sigma_obs_nom = model.compute_observed_stress(z_tgt_nom)
        sigma_pred_nom = model.stress_head(z_pred_nom)
        evals_nom = torch.linalg.eigvalsh(sigma_obs_nom)
        min_eig_nom = evals_nom.min().item()

        sigma_obs_anom = model.compute_observed_stress(z_tgt_anom)
        sigma_pred_anom = model.stress_head(z_pred_anom)
        evals_anom = torch.linalg.eigvalsh(sigma_obs_anom)
        min_eig_anom = evals_anom.min().item()

    return {
        "model_name": "ReynoldsStressJEPA",
        "records_nom": records_nom,
        "records_anom": records_anom,
        "metrics": {
            "discrepancy_nominal": disc_nom.mean().item(),
            "discrepancy_anomalous": disc_anom.mean().item(),
            "discrepancy_ratio": (disc_anom.mean() / (disc_nom.mean() + 1e-6)).item(),
            "stress_diff_nominal": stress_diff_nom.mean().item(),
            "stress_diff_anomalous": stress_diff_anom.mean().item(),
            "stress_diff_ratio": (stress_diff_anom.mean() / (stress_diff_nom.mean() + 1e-6)).item(),
            "cone_min_eig_nom": min_eig_nom,
            "cone_min_eig_anom": min_eig_anom,
            "stress_norm_nom": sigma_obs_nom.norm().item(),
            "stress_norm_anom": sigma_obs_anom.norm().item(),
        },
    }


def trace_potential_flow(
    ctx_nom: torch.Tensor, tgt_nom: torch.Tensor,
    ctx_anom: torch.Tensor, tgt_anom: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    channels = ctx_nom.size(-1)
    latent_dim = 32
    encoder = HybridTCNEncoder(input_dim=channels, latent_dim=latent_dim, filters=48, tcn_layers=3).to(device)
    model = PotentialFlowJEPAModel(
        context_encoder=encoder,
        latent_dim=latent_dim,
        predictor_hidden_dim=64,
        predictor_layers=2,
        n_regimes=4,
        subspace_dim=8,
        use_regimes=True,
    ).to(device)
    model.eval()

    model.fit_mahalanobis_covariance(ctx_nom, tgt_nom, batch_size=4)

    tracer = ActivationTracer()
    tracer.register("context_encoder", model.context_encoder)
    if model.grassmannian_codebook is not None:
        tracer.register("harmonic_grassmannian_codebook", model.grassmannian_codebook)
    tracer.register("scalar_potential_field", model.flow_predictor.potential_field)
    tracer.register("flow_predictor", model.flow_predictor)
    tracer.register("target_encoder", model.target_encoder)
    tracer.register("saliency_gate", model.saliency_gate)

    # 1. Nominal
    with torch.no_grad():
        disc_nom = model.compute_predictive_discrepancy(
            ctx_nom.to(device), tgt_nom.to(device), use_mahalanobis=True, include_curvature=True
        )
    records_nom = list(tracer.records)
    tracer.clear()

    # 2. Anomalous
    with torch.no_grad():
        disc_anom = model.compute_predictive_discrepancy(
            ctx_anom.to(device), tgt_anom.to(device), use_mahalanobis=True, include_curvature=True
        )
    records_anom = list(tracer.records)
    tracer.remove_hooks()

    # Extract regime routing and energy curvature
    with torch.no_grad():
        z_ctx_nom = model.context_encoder(ctx_nom.to(device))
        p_ctx_nom, probs_nom, loss_ortho = model.grassmannian_codebook(z_ctx_nom, hard=False)
        z_tgt_nom = model.target_encoder(tgt_nom.to(device))

        z_ctx_anom = model.context_encoder(ctx_anom.to(device))
        p_ctx_anom, probs_anom, _ = model.grassmannian_codebook(z_ctx_anom, hard=False)
        z_tgt_anom = model.target_encoder(tgt_anom.to(device))

        # Check conservative gradient field at midpoint t=0.5
        t_mid = torch.full((ctx_nom.size(0),), 0.5, device=device)
        v_nom = model.flow_predictor(z_tgt_nom * 0.5, t_mid, z_ctx_nom, p_ctx_nom, create_graph=False)
        v_anom = model.flow_predictor(z_tgt_anom * 0.5, t_mid, z_ctx_anom, p_ctx_anom, create_graph=False)

        # Hutchinson curvature
        curv_nom = model.flow_predictor.compute_energy_laplacian(z_tgt_nom * 0.5, t_mid, z_ctx_nom, p_ctx_nom, n_probes=4)
        curv_anom = model.flow_predictor.compute_energy_laplacian(z_tgt_anom * 0.5, t_mid, z_ctx_anom, p_ctx_anom, n_probes=4)

    return {
        "model_name": "PotentialFlowJEPA",
        "records_nom": records_nom,
        "records_anom": records_anom,
        "metrics": {
            "discrepancy_nominal": disc_nom.mean().item(),
            "discrepancy_anomalous": disc_anom.mean().item(),
            "discrepancy_ratio": (disc_anom.mean() / (disc_nom.mean() + 1e-6)).item(),
            "delsarte_frame_ortho_loss": loss_ortho.item(),
            "regime_entropy_nom": float((-torch.sum(probs_nom * torch.log(probs_nom + 1e-8), dim=-1)).mean().item()),
            "regime_entropy_anom": float((-torch.sum(probs_anom * torch.log(probs_anom + 1e-8), dim=-1)).mean().item()),
            "v_midpoint_norm_nom": v_nom.norm(dim=-1).mean().item(),
            "v_midpoint_norm_anom": v_anom.norm(dim=-1).mean().item(),
            "energy_curvature_nom": curv_nom.mean().item(),
            "energy_curvature_anom": curv_anom.mean().item(),
            "curvature_ratio": (curv_anom.abs().mean() / (curv_nom.abs().mean() + 1e-6)).item(),
        },
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Activation Trace on device: {device}")

    # Generate synthetic multi-sensor telemetry (B=4, C=256, S=64, K=9)
    ctx_nom, tgt_nom, ctx_anom, tgt_anom = generate_synthetic_telemetry(
        batch_size=4, context_len=256, target_len=64, channels=9, seed=42
    )

    print("\n" + "=" * 90)
    print("ACTIVATION TRACE 1: Operator-Entropy JEPA (Koopman Spectral Mechanics)")
    print("=" * 90)
    res_oe = trace_operator_entropy(ctx_nom, tgt_nom, ctx_anom, tgt_anom, device)

    print(f"{'Module / Sub-layer':<32} | {'Nominal Output Shape':<22} | {'Nom Norm':<10} | {'Anom Norm':<10} | {'Ratio':<8}")
    print("-" * 90)
    for r_nom, r_anom in zip(res_oe["records_nom"], res_oe["records_anom"]):
        nom_norm = r_nom.get("norm", 0.0)
        anom_norm = r_anom.get("norm", 0.0)
        ratio = anom_norm / (nom_norm + 1e-6)
        print(f"{r_nom['name']:<32} | {str(r_nom['output_shape']):<22} | {nom_norm:<10.3f} | {anom_norm:<10.3f} | {ratio:<8.2f}x")

    print("\n--- Physical Diagnostic Metrics ---")
    for k, v in res_oe["metrics"].items():
        print(f"  • {k:<30}: {v:.4f}")

    print("\n" + "=" * 90)
    print("ACTIVATION TRACE 2: Reynolds-Stress JEPA (Continuum Mechanics & Turbulent Transport)")
    print("=" * 90)
    res_rs = trace_reynolds_stress(ctx_nom, tgt_nom, ctx_anom, tgt_anom, device)

    print(f"{'Module / Sub-layer':<32} | {'Nominal Output Shape':<22} | {'Nom Norm':<10} | {'Anom Norm':<10} | {'Ratio':<8}")
    print("-" * 90)
    for r_nom, r_anom in zip(res_rs["records_nom"], res_rs["records_anom"]):
        nom_norm = r_nom.get("norm", 0.0)
        anom_norm = r_anom.get("norm", 0.0)
        ratio = anom_norm / (nom_norm + 1e-6)
        print(f"{r_nom['name']:<32} | {str(r_nom['output_shape']):<22} | {nom_norm:<10.3f} | {anom_norm:<10.3f} | {ratio:<8.2f}x")

    print("\n--- Physical Diagnostic Metrics ---")
    for k, v in res_rs["metrics"].items():
        print(f"  • {k:<30}: {v:.4f}")

    print("\n" + "=" * 90)
    print("ACTIVATION TRACE 3: Potential-Flow JEPA (Conservative Field Theory)")
    print("=" * 90)
    res_pf = trace_potential_flow(ctx_nom, tgt_nom, ctx_anom, tgt_anom, device)

    print(f"{'Module / Sub-layer':<32} | {'Nominal Output Shape':<22} | {'Nom Norm':<10} | {'Anom Norm':<10} | {'Ratio':<8}")
    print("-" * 90)
    for r_nom, r_anom in zip(res_pf["records_nom"], res_pf["records_anom"]):
        nom_norm = r_nom.get("norm", 0.0)
        anom_norm = r_anom.get("norm", 0.0)
        ratio = anom_norm / (nom_norm + 1e-6)
        print(f"{r_nom['name']:<32} | {str(r_nom['output_shape']):<22} | {nom_norm:<10.3f} | {anom_norm:<10.3f} | {ratio:<8.2f}x")

    print("\n--- Physical Diagnostic Metrics ---")
    for k, v in res_pf["metrics"].items():
        print(f"  • {k:<30}: {v:.4f}")

    # Save complete JSON summary
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "activation_trace_summary.json"
    with open(out_file, "w") as f:
        json.dump(
            {
                "operator_entropy": res_oe,
                "reynolds_stress": res_rs,
                "potential_flow": res_pf,
            },
            f,
            indent=2,
        )
    print(f"\nTrace artifacts saved to: {out_file.as_posix()}")


if __name__ == "__main__":
    main()
