"""Unit tests for Relational Graph Attention Network (GAT) Encoder."""

import pytest
import torch
import torch.optim as optim

from src.models.encoders.relational_gat_encoder import (
    RelationalGATEncoder,
    RelationalGraphAttentionLayer,
)
from src.models.encoders.tcn_encoder import contrastive_loss


def test_relational_gat_layer():
    """Verify forward pass and attention extraction of RelationalGraphAttentionLayer."""
    batch_size = 4
    num_nodes = 9
    node_dim = 32

    layer = RelationalGraphAttentionLayer(node_dim=node_dim, num_heads=4, dropout=0.0)
    x = torch.randn(batch_size, num_nodes, node_dim)

    out, attn = layer(x, return_attention=True)
    assert out.shape == (batch_size, num_nodes, node_dim)
    assert attn.shape == (batch_size, 4, num_nodes, num_nodes)

    # Softmax row sums must be 1.0
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_relational_gat_encoder_univariate_and_multivariate():
    """Verify forward pass on univariate and multivariate inputs."""
    # Univariate test (1 channel, 100 timesteps)
    encoder_uni = RelationalGATEncoder(input_dim=1, latent_dim=16, filters=32, tcn_layers=3, gat_layers=2)
    x_uni = torch.randn(8, 100, 1)
    z_uni = encoder_uni(x_uni)
    assert z_uni.shape == (8, 16)
    assert not torch.isnan(z_uni).any()

    # Multivariate test (Daphnet 9 channels, 256 timesteps)
    encoder_multi = RelationalGATEncoder(input_dim=9, latent_dim=32, filters=48, tcn_layers=3, gat_layers=2)
    x_multi = torch.randn(4, 256, 9)
    z_multi, attns = encoder_multi(x_multi, return_attention=True)
    assert z_multi.shape == (4, 32)
    assert len(attns) == 2
    assert not torch.isnan(z_multi).any()

    # High-dimensional multivariate test (OPPORTUNITY 77 channels, 128 timesteps)
    encoder_high = RelationalGATEncoder(input_dim=77, latent_dim=16, filters=64, tcn_layers=2, gat_layers=1)
    x_high = torch.randn(2, 128, 77)
    z_high = encoder_high(x_high)
    assert z_high.shape == (2, 16)


def test_relational_gat_optimization_step():
    """Verify backpropagation and parameter updates via contrastive loss."""
    encoder = RelationalGATEncoder(input_dim=6, latent_dim=16, filters=32, tcn_layers=2, gat_layers=1)
    optimizer = optim.AdamW(encoder.parameters(), lr=1e-3)

    full_windows = torch.randn(8, 80, 6)
    context_windows = full_windows[:, :64]
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])

    z_full = encoder(full_windows)
    z_ctx = encoder(context_windows)
    loss = contrastive_loss(z_full, z_ctx, labels, margin=1.0)

    assert loss.item() > 0.0
    optimizer.zero_grad()
    loss.backward()

    # Verify gradients exist across both TCN and GAT blocks
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    optimizer.step()
