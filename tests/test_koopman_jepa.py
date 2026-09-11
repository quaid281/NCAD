"""Unit tests for Koopman-Spectral & Lyapunov JEPA (KoopmanJEPA)."""

import pytest
import torch
import torch.nn as nn

from src.models.encoders.tcn_encoder import HybridTCNEncoder
from src.models.jepa.koopman_jepa import (
    KoopmanJEPA,
    KoopmanJEPAModel,
    KoopmanOperatorPredictor,
)


def test_koopman_operator_predictor():
    latent_dim = 16
    hidden_dim = 32
    B = 4
    predictor = KoopmanOperatorPredictor(latent_dim=latent_dim, hidden_dim=hidden_dim)

    z_ctx = torch.randn(B, latent_dim)
    K = predictor(z_ctx)

    assert K.shape == (B, latent_dim, latent_dim), f"Expected shape ({B}, {latent_dim}, {latent_dim}), got {K.shape}"

    # Linear transformation
    z_pred = torch.bmm(K, z_ctx.unsqueeze(-1)).squeeze(-1)
    assert z_pred.shape == (B, latent_dim)


def test_koopman_jepa_forward_and_loss():
    input_dim = 5
    latent_dim = 16
    B = 8
    seq_len = 30

    encoder = HybridTCNEncoder(input_dim=input_dim, latent_dim=latent_dim, filters=16, tcn_layers=2)
    model = KoopmanJEPA(
        context_encoder=encoder,
        latent_dim=latent_dim,
        hidden_dim=32,
        ema_decay=0.99,
    )

    ctx = torch.randn(B, seq_len, input_dim)
    tgt = torch.randn(B, seq_len, input_dim)

    # Forward pass
    z_ctx, z_tgt, z_pred, spectral_instability, lyapunov_exp = model(ctx, tgt)
    assert z_ctx.shape == (B, latent_dim)
    assert z_tgt.shape == (B, latent_dim)
    assert z_pred.shape == (B, latent_dim)
    assert spectral_instability.shape == (B,)
    assert lyapunov_exp.shape == (B,)

    # Compute objective
    loss, metrics = model.compute_objective(ctx, tgt, stability_weight=0.10, cov_weight=0.5)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0
    assert "loss_pred" in metrics
    assert "loss_stability" in metrics

    # Backward pass
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad and "target_encoder" not in name:
            assert param.grad is not None, f"Parameter {name} has no gradient!"


def test_koopman_jepa_discrepancy_and_mahalanobis():
    input_dim = 4
    latent_dim = 8
    B = 6
    seq_len = 20

    encoder = HybridTCNEncoder(input_dim=input_dim, latent_dim=latent_dim, filters=8, tcn_layers=2)
    model = KoopmanJEPA(
        context_encoder=encoder,
        latent_dim=latent_dim,
        hidden_dim=16,
    )

    ctx = torch.randn(B, seq_len, input_dim)
    tgt = torch.randn(B, seq_len, input_dim)

    # Discrepancy
    disc = model.compute_predictive_discrepancy(ctx, tgt, include_spectral=True)
    assert disc.shape == (B,)
    assert torch.all(disc >= 0.0)

    # Mahalanobis fit
    ctx_large = torch.randn(32, seq_len, input_dim)
    tgt_large = torch.randn(32, seq_len, input_dim)
    model.fit_mahalanobis_covariance(ctx_large, tgt_large, batch_size=16)
    assert bool(model.precision_fitted.item()) is True
