"""Unit tests for SelectiveLatentWorldJEPA (Selective Gated Memory Latent World Model)."""

import pytest
import torch
import torch.nn as nn

from src.models.jepa.selective_latent_world_jepa import (
    SelectiveLatentCore,
    SelectiveLatentWorldJEPA,
    SelectiveLatentWorldJEPAModel,
)


def test_selective_latent_core_step():
    B, D, N = 4, 16, 8
    core = SelectiveLatentCore(latent_dim=D, d_state=N, hidden_dim=32)

    z_0 = torch.randn(B, D)
    h_0 = torch.zeros(B, D, N)

    z_hat_next, h_next = core.step(z_0, h_0)
    assert z_hat_next.shape == (B, D)
    assert h_next.shape == (B, D, N)
    assert torch.all(torch.isfinite(z_hat_next))
    assert torch.all(torch.isfinite(h_next))


def test_selective_latent_world_jepa_forward_loss_and_discrepancy():
    B, C_len, H_len, C_dim = 4, 25, 12, 3
    latent_dim = 16

    model = SelectiveLatentWorldJEPA(input_dim=C_dim, latent_dim=latent_dim, d_state=8, hidden_dim=32)

    ctx = torch.randn(B, C_len, C_dim)
    tgt = torch.randn(B, H_len, C_dim)

    # Forward
    Z_ctx, Z_tgt, Z_ctx_pred, Z_sus_pred = model(ctx, tgt)
    assert Z_ctx.shape == (B, C_len, latent_dim)
    assert Z_tgt.shape == (B, H_len, latent_dim)
    assert Z_sus_pred.shape == (B, H_len, latent_dim)

    # Loss
    loss, metrics = model.compute_objective(ctx, tgt)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert "loss_track" in metrics
    assert "loss_rollout" in metrics

    # Backprop
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad and "target_encoder" not in name:
            assert param.grad is not None, f"Parameter {name} has no gradient!"

    # Discrepancy
    disc = model.compute_predictive_discrepancy(ctx, tgt)
    assert disc.shape == (B,)
    assert torch.all(disc >= 0.0)

    # Mahalanobis
    ctx_large = torch.randn(16, C_len, C_dim)
    tgt_large = torch.randn(16, H_len, C_dim)
    model.fit_mahalanobis_covariance(ctx_large, tgt_large, batch_size=8)
    assert bool(model.precision_fitted.item()) is True
