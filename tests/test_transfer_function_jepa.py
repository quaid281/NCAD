"""Unit tests for TransferFunctionJEPA (Spectral Coherence JEPA)."""

import pytest
import torch
import torch.nn as nn

from src.models.encoders.tcn_encoder import HybridTCNEncoder
from src.models.jepa.transfer_function_jepa import (
    SpectralTransferHead,
    TransferFunctionJEPA,
    TransferFunctionJEPAModel,
)


def test_spectral_transfer_head_shapes():
    latent_dim = 16
    B = 8
    head = SpectralTransferHead(latent_dim=latent_dim, hidden_dim=32)

    z_ctx = torch.randn(B, latent_dim)
    gain, phase_delay = head(z_ctx)

    assert gain.shape == (B, latent_dim // 2)
    assert phase_delay.shape == (B, latent_dim // 2)
    assert torch.all(gain > 0.0), "Gain must be strictly positive"
    assert torch.all(phase_delay >= -torch.pi) and torch.all(phase_delay <= torch.pi), "Phase delay must be in [-pi, pi]"


def test_polar_cartesian_invertibility():
    B, D = 6, 16
    z = torch.randn(B, D)

    r, theta = TransferFunctionJEPA.to_polar(z)
    z_rec = TransferFunctionJEPA.from_polar(r, theta)

    assert torch.allclose(z, z_rec, atol=1e-5), f"Polar reconstruction failed: max diff {torch.max(torch.abs(z - z_rec))}"


def test_transfer_function_jepa_forward_loss_and_discrepancy():
    latent_dim = 16
    B = 6
    seq_len = 25
    input_dim = 3

    encoder = HybridTCNEncoder(input_dim=input_dim, latent_dim=latent_dim, filters=16, tcn_layers=2)
    model = TransferFunctionJEPA(
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
    assert loss.item() > 0.0
    assert "loss_gain" in metrics
    assert "loss_phase" in metrics

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
    ctx_large = torch.randn(32, seq_len, input_dim)
    tgt_large = torch.randn(32, seq_len, input_dim)
    model.fit_mahalanobis_covariance(ctx_large, tgt_large, batch_size=16)
    assert bool(model.precision_fitted.item()) is True
