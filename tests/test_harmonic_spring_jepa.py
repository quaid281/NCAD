"""Unit tests for HarmonicSpringJEPA architecture."""

import pytest
import torch
import torch.nn as nn

from src.models import (
    HarmonicSpringHead,
    HarmonicSpringJEPA,
    HarmonicSpringJEPAModel,
    HybridTCNEncoder,
)


def test_harmonic_spring_head():
    B, D, rank = 8, 32, 4
    head = HarmonicSpringHead(latent_dim=D, hidden_dim=48, rank=rank, eps=1e-3)
    z_ctx = torch.randn(B, D)

    mu, L, d = head(z_ctx)
    assert mu.shape == (B, D)
    assert L.shape == (B, D, rank)
    assert d.shape == (B, D)
    assert (d > 0).all(), "Diagonal stiffness must be strictly positive"


def test_harmonic_spring_energy_and_curvature():
    B, D, rank = 8, 32, 4
    encoder = HybridTCNEncoder(input_dim=4, latent_dim=D, filters=32, tcn_layers=3)
    model = HarmonicSpringJEPAModel(context_encoder=encoder, latent_dim=D, hidden_dim=48, rank=rank)

    z_ctx = torch.randn(B, D)
    z_tgt = torch.randn(B, D)

    energy, curvature, log_det = model.compute_energy_and_curvature(z_ctx, z_tgt)
    assert energy.shape == (B,)
    assert (energy >= 0).all(), "Potential energy must be non-negative"

    assert curvature.shape == (B,)
    assert (curvature > 0).all(), "Curvature trace must be positive"

    assert log_det.shape == (B,)
    assert not torch.isnan(log_det).any()


def test_harmonic_spring_jepa_end_to_end():
    B, L_ctx, L_tgt, C, D = 8, 128, 32, 4, 32
    encoder = HybridTCNEncoder(input_dim=C, latent_dim=D, filters=32, tcn_layers=3)
    model = HarmonicSpringJEPAModel(
        context_encoder=encoder,
        latent_dim=D,
        hidden_dim=48,
        rank=4,
    )

    ctx = torch.randn(B, L_ctx, C)
    tgt = torch.randn(B, L_tgt, C)

    # 1. Training step
    model.train()
    loss, metrics = model.compute_objective(ctx, tgt)
    assert loss.item() > 0.0
    assert "loss_energy" in metrics
    assert "loss_det" in metrics
    assert "mean_curvature" in metrics

    loss.backward()

    # 2. Target EMA update
    model.update_target_encoder(decay=0.99)

    # 3. Fit nominal stats
    model.eval()
    model.fit_mahalanobis_covariance(ctx.numpy(), tgt.numpy(), batch_size=4)
    assert model.curv_count.item() > 0

    # 4. Fast closed-form discrepancy
    disc = model.compute_predictive_discrepancy(ctx, tgt, include_curvature=True)
    assert disc.shape == (B,)
    assert (disc >= 0).all()
