"""Tests for Tangent-Harmonic Subspace Projection JEPA (Chapter 2 of Ten Advances)."""

import pytest
import torch
import torch.nn as nn

from src.models.encoders.tcn_encoder import HybridTCNEncoder
from src.models.jepa.tangent_harmonic_jepa import TangentHarmonicJEPAModel, TangentHarmonicProjector


def test_tangent_harmonic_projector_identities():
    latent_dim = 16
    projector = TangentHarmonicProjector(latent_dim=latent_dim, alpha=1.0, beta=0.5)

    B = 10
    z = torch.randn(B, latent_dim)
    z = torch.nn.functional.normalize(z, p=2, dim=-1)

    # Identical vectors: discrepancy must be exactly zero
    disc_zero, metrics_zero = projector(z, z)
    assert torch.allclose(disc_zero, torch.zeros(B), atol=1e-5)
    assert abs(metrics_zero["d0_angular"].item()) < 1e-5
    assert abs(metrics_zero["d1_tangent_overlap"].item()) < 1e-5
    assert abs(metrics_zero["d2_gegenbauer"].item()) < 1e-5

    # Orthogonal vectors: <z1, z2> = 0
    # Construct orthogonal pairs
    z_orth = torch.randn(B, latent_dim)
    z_orth = z_orth - torch.sum(z_orth * z, dim=-1, keepdim=True) * z
    z_orth = torch.nn.functional.normalize(z_orth, p=2, dim=-1)

    disc_orth, metrics_orth = projector(z, z_orth)
    assert torch.all(disc_orth > 0.0)
    # When s = 0: d0 = 1.0, d1 = 1.0 / (D - 1), c2 = -1 / (D - 1) => d2 = 1.0 + 1 / (D - 1)
    expected_d1 = 1.0 / (latent_dim - 1.0)
    assert torch.allclose(metrics_orth["d1_tangent_overlap"], torch.tensor(expected_d1), atol=1e-5)


def test_tangent_harmonic_jepa_forward_and_loss():
    input_dim = 4
    latent_dim = 16
    context_size = 32
    target_size = 16
    batch_size = 8

    encoder = HybridTCNEncoder(input_dim=input_dim, latent_dim=latent_dim, filters=16, tcn_layers=2)
    model = TangentHarmonicJEPAModel(context_encoder=encoder, latent_dim=latent_dim, hidden_dim=32)

    ctx = torch.randn(batch_size, context_size, input_dim)
    tgt = torch.randn(batch_size, target_size, input_dim)

    # Forward
    z_pred, z_tgt, disc = model(ctx, tgt)
    assert z_pred.shape == (batch_size, latent_dim)
    assert z_tgt.shape == (batch_size, latent_dim)
    assert disc.shape == (batch_size,)

    # Compute objective
    loss, metrics = model.compute_objective(ctx, tgt)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert "harmonic_loss" in metrics
    assert "std_loss" in metrics
    assert "cov_loss" in metrics

    # Backprop
    loss.backward()
    for p in model.context_encoder.parameters():
        assert p.grad is not None
    for p in model.target_encoder.parameters():
        assert p.grad is None


def test_tangent_harmonic_jepa_covariance_and_scoring():
    input_dim = 3
    latent_dim = 8
    encoder = HybridTCNEncoder(input_dim=input_dim, latent_dim=latent_dim, filters=16, tcn_layers=2)
    model = TangentHarmonicJEPAModel(context_encoder=encoder, latent_dim=latent_dim, hidden_dim=16)

    ctx = torch.randn(20, 20, input_dim)
    tgt = torch.randn(20, 10, input_dim)

    # Fit covariance using resolvent operator
    model.fit_covariance(ctx, tgt, batch_size=10, method="resolvent")
    assert model.precision_fitted.item() is True

    # Test discrepancy with and without Mahalanobis/resolvent weighting
    s_raw = model.compute_predictive_discrepancy(ctx[:5], tgt[:5], use_mahalanobis=False)
    s_whitened = model.compute_predictive_discrepancy(ctx[:5], tgt[:5], use_mahalanobis=True)

    assert s_raw.shape == (5,)
    assert s_whitened.shape == (5,)
    assert torch.all(s_raw >= 0.0)
    assert torch.all(s_whitened >= 0.0)
