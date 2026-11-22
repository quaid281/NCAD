"""Unit tests for PrototypeGraphJEPA (Anchor Transition Geometry)."""

import pytest
import torch
import torch.nn as nn

from src.models.encoders.tcn_encoder import HybridTCNEncoder
from src.models.jepa.prototype_graph_jepa import (
    PrototypeGraphJEPA,
    PrototypeGraphJEPAModel,
    PrototypeGraphModule,
)


def test_prototype_graph_module_properties():
    num_prototypes = 8
    latent_dim = 16
    B = 6
    module = PrototypeGraphModule(num_prototypes=num_prototypes, latent_dim=latent_dim)

    # 1. Transition matrix is row-stochastic (rows sum to 1.0)
    T = module.transition_matrix
    assert T.shape == (num_prototypes, num_prototypes)
    row_sums = torch.sum(T, dim=-1)
    assert torch.allclose(row_sums, torch.ones(num_prototypes), atol=1e-5)

    # 2. Soft assignments are valid probability distributions
    z = torch.randn(B, latent_dim)
    w, sq_dists = module.compute_soft_assignments(z)
    assert w.shape == (B, num_prototypes)
    assert sq_dists.shape == (B, num_prototypes)
    w_sums = torch.sum(w, dim=-1)
    assert torch.allclose(w_sums, torch.ones(B), atol=1e-5)


def test_prototype_graph_jepa_forward_loss_and_discrepancy():
    num_prototypes = 8
    latent_dim = 16
    B = 6
    seq_len = 25
    input_dim = 4

    encoder = HybridTCNEncoder(input_dim=input_dim, latent_dim=latent_dim, filters=16, tcn_layers=2)
    model = PrototypeGraphJEPA(
        context_encoder=encoder,
        latent_dim=latent_dim,
        num_prototypes=num_prototypes,
    )

    ctx = torch.randn(B, seq_len, input_dim)
    tgt = torch.randn(B, seq_len, input_dim)

    # Forward pass
    z_ctx, z_tgt, w_ctx, sq_dists_ctx = model(ctx, tgt)
    assert z_ctx.shape == (B, latent_dim)
    assert z_tgt.shape == (B, latent_dim)

    # Loss computation
    loss, metrics = model.compute_objective(ctx, tgt)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0
    assert "loss_anchor" in metrics
    assert "loss_trans" in metrics
    assert "loss_diversity" in metrics

    # Backward pass
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad and "target_encoder" not in name:
            assert param.grad is not None, f"Parameter {name} has no gradient!"

    # Discrepancy computation
    disc = model.compute_predictive_discrepancy(ctx, tgt)
    assert disc.shape == (B,)
    assert torch.all(disc >= 0.0)

    # Mahalanobis covariance fit
    ctx_large = torch.randn(32, seq_len, input_dim)
    tgt_large = torch.randn(32, seq_len, input_dim)
    model.fit_mahalanobis_covariance(ctx_large, tgt_large, batch_size=16)
    assert bool(model.precision_fitted.item()) is True
