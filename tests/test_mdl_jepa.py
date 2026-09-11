"""Unit tests for MDLCompressorJEPA (Neural Entropy Compressor JEPA)."""

import pytest
import torch
import torch.nn as nn

from src.models.encoders.tcn_encoder import HybridTCNEncoder
from src.models.jepa.mdl_jepa import (
    EntropyPriorHead,
    MDLCompressorJEPA,
    MDLCompressorJEPAModel,
)


def test_entropy_prior_head():
    latent_dim = 16
    B = 8
    head = EntropyPriorHead(latent_dim=latent_dim, hidden_dim=32)

    z_ctx = torch.randn(B, latent_dim)
    mu, log_var = head(z_ctx)

    assert mu.shape == (B, latent_dim)
    assert log_var.shape == (B, latent_dim)
    assert torch.all(torch.isfinite(mu))
    assert torch.all(torch.isfinite(log_var))


def test_mdl_compressor_jepa_forward_loss_and_discrepancy():
    latent_dim = 16
    B = 6
    seq_len = 25
    input_dim = 3

    encoder = HybridTCNEncoder(input_dim=input_dim, latent_dim=latent_dim, filters=16, tcn_layers=2)
    model = MDLCompressorJEPA(
        context_encoder=encoder,
        latent_dim=latent_dim,
        hidden_dim=32,
    )

    ctx = torch.randn(B, seq_len, input_dim)
    tgt = torch.randn(B, seq_len, input_dim)

    # Loss
    loss, metrics = model.compute_objective(ctx, tgt)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert "loss_nll" in metrics
    assert "bitrate_bits" in metrics

    # Backprop
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad and "target_encoder" not in name:
            assert param.grad is not None, f"Parameter {name} has no gradient!"

    # Discrepancy
    disc = model.compute_predictive_discrepancy(ctx, tgt)
    assert disc.shape == (B,)
    assert torch.all(torch.isfinite(disc))

    # Mahalanobis
    ctx_large = torch.randn(32, seq_len, input_dim)
    tgt_large = torch.randn(32, seq_len, input_dim)
    model.fit_mahalanobis_covariance(ctx_large, tgt_large, batch_size=16)
    assert bool(model.precision_fitted.item()) is True
