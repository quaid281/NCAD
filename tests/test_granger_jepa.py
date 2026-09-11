"""Unit tests for GrangerCausalJEPA (Cross-Channel Causal JEPA)."""

import pytest
import torch
import torch.nn as nn

from src.models.jepa.granger_jepa import (
    CausalCrossAttentionHead,
    ChannelTCNEncoder,
    GrangerCausalJEPA,
    GrangerCausalJEPAModel,
)


def test_channel_tcn_encoder():
    B, L, C = 4, 30, 5
    latent_dim = 16
    encoder = ChannelTCNEncoder(in_channels=1, latent_dim=latent_dim)

    x = torch.randn(B, L, C)
    h = encoder(x)
    assert h.shape == (B, C, latent_dim)


def test_causal_cross_attention_head():
    B, C, D = 4, 5, 16
    head = CausalCrossAttentionHead(latent_dim=D, num_heads=2)

    H_ctx = torch.randn(B, C, D)
    H_pred, G = head(H_ctx)

    assert H_pred.shape == (B, C, D)
    assert G.shape == (B, C, C)
    # Rows of G should sum to 1.0 (softmax attention)
    row_sums = torch.sum(G, dim=-1)
    assert torch.allclose(row_sums, torch.ones(B, C), atol=1e-5)


def test_granger_causal_jepa_forward_loss_and_discrepancy():
    B, L, C = 4, 25, 3
    latent_dim = 16

    model = GrangerCausalJEPA(input_dim=C, latent_dim=latent_dim, num_heads=2)

    ctx = torch.randn(B, L, C)
    tgt = torch.randn(B, L, C)

    # Loss
    loss, metrics = model.compute_objective(ctx, tgt)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert "loss_pred" in metrics
    assert "loss_sparse" in metrics

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
    ctx_large = torch.randn(16, L, C)
    tgt_large = torch.randn(16, L, C)
    model.fit_mahalanobis_covariance(ctx_large, tgt_large, batch_size=8)
    assert bool(model.precision_fitted.item()) is True
    assert bool(model.G_nominal_fitted.item()) is True
