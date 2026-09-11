"""Unit tests for MemoryBankLatentWorldJEPA (Episodic Memory Bank Latent World Model)."""

import pytest
import torch
import torch.nn as nn

from src.models.jepa.memory_bank_latent_world_jepa import (
    LatentMemoryBankCore,
    MemoryBankLatentWorldJEPA,
    MemoryBankLatentWorldJEPAModel,
)


def test_latent_memory_bank_core_step():
    B, C_len, D, H = 4, 20, 16, 32
    core = LatentMemoryBankCore(latent_dim=D, num_heads=2, hidden_dim=H)

    Z_ctx = torch.randn(B, C_len, D)
    K_mem, V_mem = core.encode_memory_bank(Z_ctx)
    assert K_mem.shape == (B, C_len, D)
    assert V_mem.shape == (B, C_len, D)

    z_0 = torch.randn(B, D)
    h_0 = torch.zeros(B, H)

    z_hat_next, h_next = core.step_with_memory(z_0, h_0, K_mem, V_mem)
    assert z_hat_next.shape == (B, D)
    assert h_next.shape == (B, H)


def test_memory_bank_latent_world_jepa_forward_loss_and_discrepancy():
    B, C_len, H_len, C_dim = 4, 25, 12, 3
    latent_dim = 16

    model = MemoryBankLatentWorldJEPA(input_dim=C_dim, latent_dim=latent_dim, num_heads=2, hidden_dim=32)

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
