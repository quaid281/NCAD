"""Tests for CoordinateSaliencyGate and anti-drowning discrepancy integration."""

import math
import pytest
import torch
import torch.nn as nn

from src.models.geometric_layers import CoordinateSaliencyGate, CausalChannelSaliencyGate
from src.models.encoders.tcn_encoder import HybridTCNEncoder
from src.models.jepa.reynolds_stress_jepa import ReynoldsStressJEPAModel
from src.models.jepa.operator_entropy_jepa import OperatorEntropyJEPAModel
from src.models.jepa.potential_flow_jepa import PotentialFlowJEPAModel
from src.models.jepa.ts_jepa import TSJEPAModel
from src.models.jepa.harmonic_spring_jepa import HarmonicSpringJEPAModel


class DummyEncoder(nn.Module):
    def __init__(self, in_features=4, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.proj = nn.Linear(in_features, latent_dim)

    def forward(self, x):
        return self.proj(x.mean(dim=1))


# =============================================================================
# Unit Tests for CoordinateSaliencyGate
# =============================================================================

def test_saliency_gate_isotropic_scale_preservation():
    """Verify that when errors are identical across all coordinates,
    the saliency score exactly matches the standard L2 norm."""
    B, D = 4, 32
    gate = CoordinateSaliencyGate(dim=D, tau=0.5, alpha=1.0)
    
    # Isotropic residual: every coordinate has value 2.5
    e = torch.full((B, D), 2.5)
    
    l2_expected = torch.linalg.norm(e, dim=-1)
    saliency_score = gate(e)
    
    assert torch.allclose(saliency_score, l2_expected, atol=1e-5), (
        f"Expected {l2_expected[0].item()}, got {saliency_score[0].item()}"
    )


def test_saliency_gate_sparse_anomaly_amplification():
    """Verify that a single-coordinate spike is amplified relative to isotropic background noise."""
    B, D = 1, 32
    gate = CoordinateSaliencyGate(dim=D, tau=0.2, alpha=1.0)
    
    # Case 1: Sparse spike of magnitude 10 on coordinate 0, 0 elsewhere
    e_spike = torch.zeros(B, D)
    e_spike[0, 0] = 10.0
    
    l2_spike = torch.linalg.norm(e_spike, dim=-1)  # 10.0
    saliency_spike = gate(e_spike)                # ~ 10.0 * sqrt(32) = 56.57
    
    expected_amplified = 10.0 * math.sqrt(D)
    assert saliency_spike.item() > l2_spike.item() * 4.0, (
        f"Expected significant amplification, got {saliency_spike.item()} vs {l2_spike.item()}"
    )
    assert abs(saliency_spike.item() - expected_amplified) < 1.0


def test_saliency_gate_temperature_limits():
    """Test asymptotic behavior as tau -> infinity (L2 norm) and tau -> 0 (L_inf * sqrt(D))."""
    B, D = 2, 16
    e = torch.randn(B, D)
    
    # Large tau -> L2
    gate_inf = CoordinateSaliencyGate(dim=D, tau=1e5, alpha=1.0)
    score_inf = gate_inf(e)
    l2_norm = torch.linalg.norm(e, dim=-1)
    assert torch.allclose(score_inf, l2_norm, atol=1e-3)
    
    # Small tau -> Linf * sqrt(D)
    gate_zero = CoordinateSaliencyGate(dim=D, tau=1e-3, alpha=1.0)
    score_zero = gate_zero(e)
    linf_expected = torch.max(torch.abs(e), dim=-1).values * math.sqrt(D)
    assert torch.allclose(score_zero, linf_expected, atol=1e-3)


def test_saliency_gate_blending_interpolation():
    """Test smooth interpolation between L2 norm (alpha=0) and pure saliency (alpha=1)."""
    B, D = 3, 16
    e = torch.randn(B, D)
    
    gate_l2 = CoordinateSaliencyGate(dim=D, alpha=0.0)
    gate_half = CoordinateSaliencyGate(dim=D, alpha=0.5)
    gate_full = CoordinateSaliencyGate(dim=D, alpha=1.0)
    
    s0 = gate_l2(e)
    s05 = gate_half(e)
    s1 = gate_full(e)
    
    expected_half = 0.5 * s0 + 0.5 * s1
    assert torch.allclose(s05, expected_half, atol=1e-5)


def test_saliency_gate_gradient_flow():
    """Verify that gradients pass cleanly through the saliency gate."""
    B, D = 4, 16
    gate = CoordinateSaliencyGate(dim=D, tau=0.5, alpha=0.5)
    
    e = torch.randn(B, D, requires_grad=True)
    score = gate(e).sum()
    score.backward()
    
    assert e.grad is not None
    assert torch.all(torch.isfinite(e.grad))
    assert not torch.all(e.grad == 0)


# =============================================================================
# Integration Tests Across Top 5 Models
# =============================================================================

def test_reynolds_stress_jepa_saliency_discrepancy():
    B, T_ctx, T_tgt, C, D = 2, 32, 16, 4, 16
    encoder = DummyEncoder(in_features=C, latent_dim=D)
    model = ReynoldsStressJEPAModel(context_encoder=encoder, latent_dim=D, stress_dim=8)
    
    ctx = torch.randn(B, T_ctx, C)
    tgt = torch.randn(B, T_tgt, C)
    
    disc = model.compute_predictive_discrepancy(ctx, tgt)
    assert disc.shape == (B,)
    assert torch.all(torch.isfinite(disc))
    assert torch.all(disc >= 0.0)


def test_operator_entropy_jepa_saliency_discrepancy():
    B, T_ctx, T_tgt, C, D = 2, 32, 16, 4, 16
    encoder = DummyEncoder(in_features=C, latent_dim=D)
    model = OperatorEntropyJEPAModel(context_encoder=encoder, latent_dim=D)
    
    ctx = torch.randn(B, T_ctx, C)
    tgt = torch.randn(B, T_tgt, C)
    
    disc = model.compute_predictive_discrepancy(ctx, tgt)
    assert disc.shape == (B,)
    assert torch.all(torch.isfinite(disc))
    assert torch.all(disc >= 0.0)


def test_potential_flow_jepa_saliency_discrepancy():
    B, T_ctx, T_tgt, C, D = 2, 32, 16, 4, 16
    encoder = DummyEncoder(in_features=C, latent_dim=D)
    model = PotentialFlowJEPAModel(context_encoder=encoder, latent_dim=D, predictor_layers=2)
    
    ctx = torch.randn(B, T_ctx, C)
    tgt = torch.randn(B, T_tgt, C)
    
    disc = model.compute_predictive_discrepancy(ctx, tgt)
    assert disc.shape == (B,)
    assert torch.all(torch.isfinite(disc))
    assert torch.all(disc >= 0.0)


def test_ts_jepa_saliency_discrepancy():
    B, T_ctx, T_tgt, C, D = 2, 32, 16, 4, 16
    encoder = DummyEncoder(in_features=C, latent_dim=D)
    model = TSJEPAModel(context_encoder=encoder, latent_dim=D)
    
    ctx = torch.randn(B, T_ctx, C)
    tgt = torch.randn(B, T_tgt, C)
    
    disc = model.compute_predictive_discrepancy(ctx, tgt)
    assert disc.shape == (B,)
    assert torch.all(torch.isfinite(disc))
    assert torch.all(disc >= 0.0)


def test_harmonic_spring_jepa_saliency_discrepancy():
    B, T_ctx, T_tgt, C, D = 2, 32, 16, 4, 16
    encoder = DummyEncoder(in_features=C, latent_dim=D)
    model = HarmonicSpringJEPAModel(context_encoder=encoder, latent_dim=D)
    
    ctx = torch.randn(B, T_ctx, C)
    tgt = torch.randn(B, T_tgt, C)
    
    disc = model.compute_predictive_discrepancy(ctx, tgt)
    assert disc.shape == (B,)
    assert torch.all(torch.isfinite(disc))
    assert torch.all(disc >= 0.0)


# =============================================================================
# Unit Tests for CausalChannelSaliencyGate (Major Reviewer Issue #12)
# =============================================================================

def test_causal_channel_saliency_zero_target_leakage():
    """Verify that gating weights and gated context are strictly invariant
    to variations in the suspect future target window (zero leakage)."""
    B, T_ctx, T_tgt, K = 4, 64, 16, 8
    gate = CausalChannelSaliencyGate()

    x_ctx = torch.randn(B, T_ctx, K, requires_grad=True)
    x_tgt_1 = torch.randn(B, T_tgt, K, requires_grad=True)
    x_tgt_2 = x_tgt_1.clone().detach() * 100.0  # Massive target anomaly burst

    # Forward with target 1
    ctx_gated_1, tgt_gated_1, g_1 = gate(x_ctx, x_tgt_1, mode="causal_context")

    # Forward with massive target anomaly
    ctx_gated_2, tgt_gated_2, g_2 = gate(x_ctx, x_tgt_2, mode="causal_context")

    # 1. Gating weights g MUST be identical regardless of target content
    assert torch.allclose(g_1, g_2, atol=1e-7), "Gate weights leaked target window information!"
    assert torch.allclose(ctx_gated_1, ctx_gated_2, atol=1e-7), "Context gating was contaminated by target!"

    # 2. Gradient of gating weights w.r.t target MUST be None or identically zero
    loss = g_1.sum()
    loss.backward()
    assert x_tgt_1.grad is None or torch.all(x_tgt_1.grad == 0), (
        "Non-zero gradient flow from target window into causal gating weights!"
    )


def test_causal_channel_saliency_univariate_identity():
    """Verify that for single-channel streams (K=1), gating smoothly defaults to identity."""
    B, T_ctx, T_tgt, K = 2, 32, 16, 1
    gate = CausalChannelSaliencyGate()

    x_ctx = torch.randn(B, T_ctx, K)
    x_tgt = torch.randn(B, T_tgt, K)

    ctx_gated, tgt_gated, g = gate(x_ctx, x_tgt, mode="causal_context")
    assert torch.allclose(g, torch.ones_like(g))
    assert torch.allclose(ctx_gated, x_ctx)
    assert torch.allclose(tgt_gated, x_tgt)


def test_causal_channel_saliency_amplification():
    """Verify that high-variance channels in context are prioritized over quiescent noise."""
    B, T_ctx, K = 2, 128, 4
    gate = CausalChannelSaliencyGate()

    x_ctx = torch.zeros(B, T_ctx, K)
    # Channel 0: High variance active signal (sinusoid + noise)
    t = torch.linspace(0, 10, T_ctx).unsqueeze(0).repeat(B, 1)
    x_ctx[:, :, 0] = torch.sin(t) * 5.0
    # Channel 1..3: Low amplitude quiescent sensor jitter
    x_ctx[:, :, 1:] = torch.randn(B, T_ctx, K - 1) * 0.05

    g = gate.compute_gate(x_ctx)  # (B, 1, K)
    assert g.shape == (B, 1, K)

    # Active channel 0 must receive higher saliency weight than quiescent channels
    for b in range(B):
        assert g[b, 0, 0] > g[b, 0, 1]
        assert g[b, 0, 0] > g[b, 0, 2]
        assert g[b, 0, 0] > g[b, 0, 3]


def test_causal_channel_saliency_ablation_modes():
    """Test all four reviewer-requested ablation modes:
    causal_context, target_only, independent, uniform."""
    B, T_ctx, T_tgt, K = 2, 32, 16, 6
    gate = CausalChannelSaliencyGate()

    x_ctx = torch.randn(B, T_ctx, K)
    x_tgt = torch.randn(B, T_tgt, K)

    # Mode 1: causal_context
    c1, t1, g1 = gate(x_ctx, x_tgt, mode="causal_context")
    assert c1.shape == x_ctx.shape and t1.shape == x_tgt.shape

    # Mode 2: target_only
    c2, t2, g2 = gate(x_ctx, x_tgt, mode="target_only")
    assert c2.shape == x_ctx.shape and t2.shape == x_tgt.shape

    # Mode 3: independent
    c3, t3, g3 = gate(x_ctx, x_tgt, mode="independent")
    assert c3.shape == x_ctx.shape and t3.shape == x_tgt.shape

    # Mode 4: uniform
    c4, t4, g4 = gate(x_ctx, x_tgt, mode="uniform")
    assert torch.allclose(c4, x_ctx) and torch.allclose(t4, x_tgt)
    assert torch.allclose(g4, torch.ones_like(g4))


def test_coordinate_saliency_gate_static_delegation():
    """Verify that CoordinateSaliencyGate.gate_channels static method delegates correctly."""
    B, T_ctx, T_tgt, K = 2, 32, 16, 4
    x_ctx = torch.randn(B, T_ctx, K)
    x_tgt = torch.randn(B, T_tgt, K)

    c, t, g = CoordinateSaliencyGate.gate_channels(x_ctx, x_tgt, mode="causal_context")
    assert c.shape == x_ctx.shape
    assert t.shape == x_tgt.shape
    assert g.shape == (B, 1, K)
