"""Unit tests for DualTimescaleLatentWorldJEPA (Dual-Timescale Cerebellar-Cortical Latent World Model)."""

import pytest
import torch
import torch.nn as nn

from src.models.jepa.dual_timescale_latent_world_jepa import (
    DualTimescaleLatentCore,
    DualTimescaleLatentJEPA,
    DualTimescaleLatentWorldJEPA,
    DualTimescaleLatentWorldJEPAModel,
)


def test_dual_timescale_latent_core_step():
    B, D, H = 4, 16, 32
    core = DualTimescaleLatentCore(latent_dim=D, k_interval=4, hidden_dim=H)

    z_0 = torch.randn(B, D)
    h_fast = torch.zeros(B, H)
    h_slow = torch.zeros(B, H)

    z_hat_next, h_f, h_s = core.step(z_0, h_fast, h_slow, step_idx=0)
    assert z_hat_next.shape == (B, D)
    assert h_f.shape == (B, H)
    assert h_s.shape == (B, H)


def test_dual_timescale_latent_world_jepa_forward_loss_and_discrepancy():
    B, C_len, H_len, C_dim = 4, 25, 12, 3
    latent_dim = 16

    model = DualTimescaleLatentWorldJEPA(input_dim=C_dim, latent_dim=latent_dim, k_interval=4, hidden_dim=32)

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
