"""Unit tests for LatentWorldJEPA (Pure Latent-Space Recurrence & World Model)."""

import pytest
import torch
import torch.nn as nn

from src.models.jepa.latent_world_jepa import (
    LatentRecurrentCore,
    LatentWorldJEPA,
    LatentWorldJEPAModel,
    SpatialFrameEncoder,
)


def test_spatial_frame_encoder():
    B, T, C = 4, 30, 5
    latent_dim = 16
    encoder = SpatialFrameEncoder(input_dim=C, latent_dim=latent_dim, hidden_dim=32)

    x = torch.randn(B, T, C)
    z = encoder(x)
    assert z.shape == (B, T, latent_dim)


def test_latent_recurrent_core():
    B, D, H = 4, 16, 32
    core = LatentRecurrentCore(latent_dim=D, hidden_dim=H)

    z_0 = torch.randn(B, D)
    h_0 = torch.zeros(B, H)

    z_hat_1, h_1 = core.step(z_0, h_0)
    assert z_hat_1.shape == (B, D)
    assert h_1.shape == (B, H)


def test_latent_world_jepa_forward_loss_and_discrepancy():
    B, C_len, H_len, C_dim = 4, 30, 10, 3
    latent_dim = 16

    model = LatentWorldJEPA(input_dim=C_dim, latent_dim=latent_dim, hidden_dim=32)

    ctx = torch.randn(B, C_len, C_dim)
    tgt = torch.randn(B, H_len, C_dim)

    # 1. Forward Trajectory and Rollout
    Z_ctx, Z_tgt, Z_ctx_pred, Z_sus_pred = model(ctx, tgt)
    assert Z_ctx.shape == (B, C_len, latent_dim)
    assert Z_tgt.shape == (B, H_len, latent_dim)
    assert Z_sus_pred.shape == (B, H_len, latent_dim)

    # 2. Objective & Loss
    loss, metrics = model.compute_objective(ctx, tgt)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert "loss_track" in metrics
    assert "loss_rollout" in metrics

    # 3. Backprop
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad and "target_encoder" not in name:
            assert param.grad is not None, f"Parameter {name} has no gradient!"

    # 4. Discrepancy Scoring
    disc = model.compute_predictive_discrepancy(ctx, tgt)
    assert disc.shape == (B,)
    assert torch.all(disc >= 0.0)

    # 5. Mahalanobis Covariance Fit
    ctx_large = torch.randn(16, C_len, C_dim)
    tgt_large = torch.randn(16, H_len, C_dim)
    model.fit_mahalanobis_covariance(ctx_large, tgt_large, batch_size=8)
    assert bool(model.precision_fitted.item()) is True
