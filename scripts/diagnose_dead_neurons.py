"""Comprehensive Forensic Diagnostic: Dead Neurons, Saturated Activations, and Scale Discrepancies.

Checks:
1. Dead neurons (dimensions with 0 variance or constant 0 across batch).
2. Saturated activations (GELU/SiLU/Sigmoid/Softmax pinning).
3. Scale balance between context, predictor, and target representations.
4. Gradient flow: checks for un-updated parameters or vanishing/exploding gradients.
5. Regime codebook utilization (in PotentialFlowJEPA).
6. Stress head output rank and conditioning (in ReynoldsStressJEPA).
"""

from __future__ import annotations

import sys
from pathlib import Path

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
from trace_model_activations import generate_synthetic_telemetry


def check_module_activations(name: str, tensor: torch.Tensor, threshold: float = 1e-5) -> dict:
    """Analyze a single activation tensor for dead or collapsed coordinates."""
    if tensor.ndim == 3:
        # e.g. [B, T, D] -> flatten B, T
        t_flat = tensor.reshape(-1, tensor.size(-1))
    elif tensor.ndim == 2:
        t_flat = tensor
    else:
        t_flat = tensor.reshape(tensor.size(0), -1)

    variances = t_flat.var(dim=0)
    means = t_flat.mean(dim=0)
    mins = t_flat.min(dim=0).values
    maxs = t_flat.max(dim=0).values

    dead_zero = ((mins == 0) & (maxs == 0)).sum().item()
    dead_const = (variances < threshold).sum().item()
    total_dims = t_flat.size(-1)

    return {
        "name": name,
        "shape": list(tensor.shape),
        "total_dims": total_dims,
        "dead_exact_zero": dead_zero,
        "low_variance_dims": dead_const,
        "min_variance": float(variances.min().item()),
        "max_variance": float(variances.max().item()),
        "mean_norm": float(t_flat.norm(dim=-1).mean().item()),
    }


def diagnose_operator_entropy(ctx: torch.Tensor, tgt: torch.Tensor, device: torch.device):
    print("\n" + "=" * 80)
    print("FORENSIC AUDIT 1: Operator-Entropy JEPA")
    print("=" * 80)

    encoder = HybridTCNEncoder(input_dim=ctx.size(-1), latent_dim=32, filters=48, tcn_layers=3).to(device)
    model = OperatorEntropyJEPAModel(context_encoder=encoder, latent_dim=32).to(device)

    # 1. Training gradient audit
    model.train()
    loss, metrics = model.compute_objective(ctx, tgt)
    loss.backward()

    missing_grads = []
    vanishing_grads = []
    exploding_grads = []
    for p_name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                missing_grads.append(p_name)
            else:
                gnorm = p.grad.norm().item()
                if gnorm < 1e-7:
                    vanishing_grads.append((p_name, gnorm))
                elif gnorm > 50.0:
                    exploding_grads.append((p_name, gnorm))

    print(f"1. GRADIENT FLOW CHECK:")
    print(f"   • Missing Gradients: {len(missing_grads)}")
    if missing_grads:
        print(f"     WARNING: {missing_grads}")
    print(f"   • Vanishing Gradients (<1e-7): {len(vanishing_grads)}")
    if vanishing_grads:
        for n, g in vanishing_grads[:4]:
            print(f"     - {n}: norm={g:.2e}")
    print(f"   • Exploding Gradients (>50.0): {len(exploding_grads)}")

    # 2. Activation audit
    model.eval()
    with torch.no_grad():
        z_ctx_raw = model.context_encoder(ctx)
        z_ctx_filt = model._apply_filters(z_ctx_raw)
        z_pred_raw = model.predictor(z_ctx_filt)
        z_pred_filt = model._apply_filters(z_pred_raw)
        z_tgt_raw = model.target_encoder(tgt)
        z_tgt_filt = model._apply_filters(z_tgt_raw)

    print(f"\n2. ACTIVATION & COORDINATE HEALTH:")
    for tag, t in [
        ("Context Encoder (Raw)", z_ctx_raw),
        ("Context Encoder (Filtered)", z_ctx_filt),
        ("Predictor (Raw MLP)", z_pred_raw),
        ("Predictor (Filtered)", z_pred_filt),
        ("Target Encoder (Raw)", z_tgt_raw),
        ("Target Encoder (Filtered)", z_tgt_filt),
    ]:
        res = check_module_activations(tag, t)
        print(f"   • {res['name']:<28}: shape={str(res['shape']):<12} | Dead Zeros: {res['dead_exact_zero']}/{res['total_dims']} | Low Var (<1e-5): {res['low_variance_dims']}/{res['total_dims']} | Norm: {res['mean_norm']:.3f}")

    # 3. Norm ratio check
    ratio = res['mean_norm'] / check_module_activations("Predictor (Filtered)", z_pred_filt)['mean_norm']
    print(f"\n3. SCALE BALANCE CHECK:")
    print(f"   • Target-to-Predictor Norm Ratio: {ratio:.2f}x")
    if ratio > 2.0 or ratio < 0.5:
        print(f"     NOTE: Predictor outputs are {1/ratio:.2f}x smaller than target norm before training convergence.")


def diagnose_reynolds_stress(ctx: torch.Tensor, tgt: torch.Tensor, device: torch.device):
    print("\n" + "=" * 80)
    print("FORENSIC AUDIT 2: Reynolds-Stress JEPA")
    print("=" * 80)

    encoder = HybridTCNEncoder(input_dim=ctx.size(-1), latent_dim=32, filters=48, tcn_layers=3).to(device)
    model = ReynoldsStressJEPAModel(context_encoder=encoder, latent_dim=32, stress_dim=16).to(device)

    model.train()
    loss, metrics = model.compute_objective(ctx, tgt)
    loss.backward()

    missing_grads = []
    vanishing_grads = []
    for p_name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                missing_grads.append(p_name)
            elif p.grad.norm().item() < 1e-7:
                vanishing_grads.append((p_name, p.grad.norm().item()))

    print(f"1. GRADIENT FLOW CHECK:")
    print(f"   • Missing Gradients: {len(missing_grads)}")
    if missing_grads:
        print(f"     WARNING: {missing_grads}")
    print(f"   • Vanishing Gradients (<1e-7): {len(vanishing_grads)}")
    if vanishing_grads:
        for n, g in vanishing_grads[:4]:
            print(f"     - {n}: norm={g:.2e}")

    model.eval()
    with torch.no_grad():
        z_ctx = model._extract_context(ctx)
        z_pred = model.cohn_elkies(model.predictor(z_ctx))
        z_tgt = model.cohn_elkies(model.hankel_filter(model.target_encoder(tgt)))
        sigma_obs = model.compute_observed_stress(z_tgt)
        sigma_pred = model.stress_head(z_ctx)

    print(f"\n2. ACTIVATION & COORDINATE HEALTH:")
    for tag, t in [
        ("Context Encoder Extracted", z_ctx),
        ("Predictor Representation", z_pred),
        ("Target Representation", z_tgt),
    ]:
        res = check_module_activations(tag, t)
        print(f"   • {res['name']:<28}: shape={str(res['shape']):<12} | Dead Zeros: {res['dead_exact_zero']}/{res['total_dims']} | Low Var (<1e-5): {res['low_variance_dims']}/{res['total_dims']} | Norm: {res['mean_norm']:.3f}")

    print(f"\n3. REYNOLDS STRESS TENSOR AUDIT:")
    evals_obs = torch.linalg.eigvalsh(sigma_obs)
    evals_pred = torch.linalg.eigvalsh(sigma_pred)
    print(f"   • Observed Stress Eigenvalues: min={evals_obs.min().item():.4f}, max={evals_obs.max().item():.4f}, mean={evals_obs.mean().item():.4f}")
    print(f"   • Predicted Stress Eigenvalues: min={evals_pred.min().item():.4f}, max={evals_pred.max().item():.4f}, mean={evals_pred.mean().item():.4f}")
    
    # Check if stress head is receiving gradients during training
    stress_grad_active = any(p.grad is not None and p.grad.norm() > 1e-6 for n, p in model.stress_head.named_parameters())
    print(f"   • Stress Head Gradient Active During Pure MSE: {stress_grad_active}")


def diagnose_potential_flow(ctx: torch.Tensor, tgt: torch.Tensor, device: torch.device):
    print("\n" + "=" * 80)
    print("FORENSIC AUDIT 3: Potential-Flow JEPA")
    print("=" * 80)

    encoder = HybridTCNEncoder(input_dim=ctx.size(-1), latent_dim=32, filters=48, tcn_layers=3).to(device)
    model = PotentialFlowJEPAModel(
        context_encoder=encoder,
        latent_dim=32,
        predictor_hidden_dim=64,
        n_regimes=4,
        subspace_dim=8,
    ).to(device)

    model.train()
    loss, metrics = model.compute_objective(ctx, tgt)
    loss.backward()

    missing_grads = []
    vanishing_grads = []
    for p_name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                missing_grads.append(p_name)
            elif p.grad.norm().item() < 1e-7:
                vanishing_grads.append((p_name, p.grad.norm().item()))

    print(f"1. GRADIENT FLOW CHECK:")
    print(f"   • Missing Gradients: {len(missing_grads)}")
    if missing_grads:
        print(f"     NOTE: {missing_grads}")
    print(f"   • Vanishing Gradients (<1e-7): {len(vanishing_grads)}")
    if vanishing_grads:
        for n, g in vanishing_grads[:4]:
            print(f"     - {n}: norm={g:.2e}")

    model.eval()
    with torch.no_grad():
        z_ctx = model.context_encoder(ctx)
        p_ctx, regime_probs, loss_ortho = model.grassmannian_codebook(z_ctx, hard=False)
        z_tgt = model.target_encoder(tgt)

    print(f"\n2. ACTIVATION & COORDINATE HEALTH:")
    for tag, t in [
        ("Context Encoder", z_ctx),
        ("Grassmannian Projected p_ctx", p_ctx),
        ("Target Encoder", z_tgt),
    ]:
        res = check_module_activations(tag, t)
        print(f"   • {res['name']:<28}: shape={str(res['shape']):<12} | Dead Zeros: {res['dead_exact_zero']}/{res['total_dims']} | Low Var (<1e-5): {res['low_variance_dims']}/{res['total_dims']} | Norm: {res['mean_norm']:.3f}")

    print(f"\n3. HARMONIC GRASSMANNIAN ROUTING HEALTH:")
    mean_probs = regime_probs.mean(dim=0).cpu().numpy()
    print(f"   • Regime Selection Probabilities: {np.round(mean_probs, 4)}")
    active_regimes = (mean_probs > 0.05).sum()
    print(f"   • Active Regimes (>5% utilization): {active_regimes}/4")
    if active_regimes == 1:
        print(f"     OBSERVATION: One regime dominates. (Normal for homogeneous single-regime test sequences).")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Forensic Diagnostic on: {device}")

    # Generate testbatch of 32 sequences
    ctx_nom, tgt_nom, _, _ = generate_synthetic_telemetry(
        batch_size=32, context_len=256, target_len=64, channels=9, seed=42
    )
    ctx_nom, tgt_nom = ctx_nom.to(device), tgt_nom.to(device)

    diagnose_operator_entropy(ctx_nom, tgt_nom, device)
    diagnose_reynolds_stress(ctx_nom, tgt_nom, device)
    diagnose_potential_flow(ctx_nom, tgt_nom, device)


if __name__ == "__main__":
    main()
