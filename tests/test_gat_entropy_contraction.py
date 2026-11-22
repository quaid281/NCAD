"""Tests for GAT Layer-Wise Entropy Contraction Regularizer (Chapter 10 of Ten Advances)."""

import math
import pytest
import torch

from src.models.encoders.relational_gat_encoder import (
    RelationalGATEncoder,
    compute_layer_entropy_potential,
    layer_entropy_contraction_loss,
)


def test_compute_layer_entropy_potential_bounds():
    B, H, N = 2, 4, 8

    # 1. Uniform attention: maximum entropy = log2(N)
    uniform_attn = torch.full((B, H, N, N), 1.0 / N)
    max_entropy = compute_layer_entropy_potential(uniform_attn)
    expected_max = math.log2(float(N))
    assert abs(max_entropy.item() - expected_max) < 1e-4

    # 2. One-hot / deterministic attention: minimum entropy = 0
    one_hot_attn = torch.zeros((B, H, N, N))
    for i in range(N):
        one_hot_attn[:, :, i, i] = 1.0
    min_entropy = compute_layer_entropy_potential(one_hot_attn)
    assert min_entropy.item() < 1e-4


def test_layer_entropy_contraction_loss():
    B, H, N = 2, 4, 8

    # Case A: Entropy decreases from Layer 1 to Layer 2 (contraction / refining)
    att_layer1 = torch.full((B, H, N, N), 1.0 / N)  # High entropy
    att_layer2 = torch.zeros((B, H, N, N))
    for i in range(N):
        att_layer2[:, :, i, i] = 1.0  # Zero entropy
    loss_contracting = layer_entropy_contraction_loss([att_layer1, att_layer2], margin=0.0)
    assert loss_contracting.item() == 0.0

    # Case B: Entropy increases from Layer 1 to Layer 2 (dispersion / over-smoothing)
    loss_expanding = layer_entropy_contraction_loss([att_layer2, att_layer1], margin=0.0)
    assert loss_expanding.item() > 0.0


def test_relational_gat_encoder_with_entropy_contraction():
    input_dim = 4
    latent_dim = 16
    encoder = RelationalGATEncoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        filters=16,
        tcn_layers=2,
        gat_layers=3,
        dropout=0.0,
    )

    inputs = torch.randn(4, 32, input_dim)
    latent, attentions = encoder(inputs, return_attention=True)

    assert latent.shape == (4, latent_dim)
    assert len(attentions) == 3
    for att in attentions:
        assert att.ndim == 4  # (B, H, N, N)

    # Compute contraction loss
    loss = layer_entropy_contraction_loss(attentions, margin=0.01)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0

    # Test backprop
    loss.backward()
    for gat in encoder.gat_blocks:
        for p in gat.parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.all(torch.isfinite(p.grad))
