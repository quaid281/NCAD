"""Tests for Mellin Two-Shell Velocity Damping in Flow Matching (Chapter 1 of Ten Advances)."""

import math
import pytest
import torch

from src.models.encoders.tcn_encoder import HybridTCNEncoder
from src.models.jepa.flow_ts_jepa import FlowLatentPredictor, FlowTSJEPAModel, MellinShellDamping


def test_mellin_shell_damping_bounds():
    latent_dim = 16
    damper = MellinShellDamping(latent_dim=latent_dim, damping_factor=0.5)

    r_crit = (1.0 / math.pi) * math.sqrt(float(latent_dim))
    r_outer = 2.0 * r_crit

    # 1. State inside nominal radius: no damping
    v = torch.ones(2, latent_dim)
    z_inside = torch.zeros(2, latent_dim)  # norm = 0 < r_outer
    v_out = damper(v, z_inside)
    assert torch.allclose(v_out, v, atol=1e-5)

    # 2. State at critical boundary
    z_crit = torch.ones(2, latent_dim) * (r_crit / math.sqrt(latent_dim))
    v_crit = damper(v, z_crit)
    assert torch.allclose(v_crit, v, atol=1e-5)

    # 3. State escaping outer barrier: norm > 2 * r_crit
    z_outside = torch.ones(2, latent_dim) * (3.0 * r_outer / math.sqrt(latent_dim))
    v_damped = damper(v, z_outside)
    # Norm of damped velocity must be strictly smaller than original
    assert torch.linalg.norm(v_damped) < torch.linalg.norm(v)


def test_flow_predictor_with_mellin_damping():
    latent_dim = 16
    predictor = FlowLatentPredictor(
        latent_dim=latent_dim,
        hidden_dim=32,
        num_layers=2,
        use_mellin_damping=True,
    )

    B = 4
    z_t = torch.randn(B, latent_dim)
    t = torch.rand(B)
    z_ctx = torch.randn(B, latent_dim)

    v = predictor(z_t, t, z_ctx)
    assert v.shape == (B, latent_dim)
    assert torch.all(torch.isfinite(v))


def test_flow_ts_jepa_with_mellin_damping():
    input_dim = 3
    latent_dim = 12
    encoder = HybridTCNEncoder(input_dim=input_dim, latent_dim=latent_dim, filters=16, tcn_layers=2)
    model = FlowTSJEPAModel(
        context_encoder=encoder,
        latent_dim=latent_dim,
        predictor_hidden_dim=24,
        use_mellin_damping=True,
    )

    ctx = torch.randn(4, 20, input_dim)
    tgt = torch.randn(4, 10, input_dim)

    # Forward
    z_ctx, z_tgt, v_pred, v_tgt = model(ctx, tgt)
    assert v_pred.shape == (4, latent_dim)
    assert v_tgt.shape == (4, latent_dim)

    # ODE sample
    z_sampled = model.sample_target(ctx, n_steps=2, solver="midpoint")
    assert z_sampled.shape == (4, latent_dim)
    assert torch.all(torch.isfinite(z_sampled))
