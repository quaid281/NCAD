"""Unit tests for TangentNormalJEPA (Orthogonal Subspace Decomposition)."""

import pytest
import torch
import torch.nn as nn

from src.models.encoders.tcn_encoder import HybridTCNEncoder
from src.models.jepa.tangent_normal_jepa import (
    TangentNormalJEPA,
    TangentNormalJEPAModel,
    TangentSubspaceHead,
)


def test_tangent_subspace_head_orthonormality():
    latent_dim = 16
    subspace_dim = 4
    B = 8
    head = TangentSubspaceHead(latent_dim=latent_dim, subspace_dim=subspace_dim, hidden_dim=32)

    z_ctx = torch.randn(B, latent_dim)
    U, v_tangent = head(z_ctx)

    assert U.shape == (B, latent_dim, subspace_dim)
    assert v_tangent.shape == (B, subspace_dim)

    # Verify orthonormality: U.T @ U = I_d
    UtU = torch.bmm(U.transpose(1, 2), U)
    I_d = torch.eye(subspace_dim).unsqueeze(0).expand(B, -1, -1)
    assert torch.allclose(UtU, I_d, atol=1e-5), f"U is not orthonormal: max diff {torch.max(torch.abs(UtU - I_d))}"


def test_tangent_normal_orthogonal_decomposition():
    latent_dim = 16
    subspace_dim = 4
    B = 6
    seq_len = 30
    input_dim = 4

    encoder = HybridTCNEncoder(input_dim=input_dim, latent_dim=latent_dim, filters=16, tcn_layers=2)
    model = TangentNormalJEPA(
        context_encoder=encoder,
        latent_dim=latent_dim,
        subspace_dim=subspace_dim,
        hidden_dim=32,
    )

    ctx = torch.randn(B, seq_len, input_dim)
    tgt = torch.randn(B, seq_len, input_dim)

    z_ctx, z_tgt, U, v_tangent = model(ctx, tgt)
    e = z_tgt - z_ctx

    c_parallel = torch.bmm(U.transpose(1, 2), e.unsqueeze(-1)).squeeze(-1)
    e_parallel = torch.bmm(U, c_parallel.unsqueeze(-1)).squeeze(-1)
    e_perp = e - e_parallel

    # 1. Reconstruction sum: e_parallel + e_perp == e
    assert torch.allclose(e_parallel + e_perp, e, atol=1e-5)

    # 2. Orthogonality: e_perp . e_parallel == 0
    dot_product = torch.sum(e_perp * e_parallel, dim=-1)
    assert torch.allclose(dot_product, torch.zeros(B), atol=1e-5), f"e_perp and e_parallel not orthogonal: max dot {torch.max(torch.abs(dot_product))}"


def test_tangent_normal_jepa_forward_loss_and_discrepancy():
    latent_dim = 16
    subspace_dim = 4
    B = 6
    seq_len = 25
    input_dim = 3

    encoder = HybridTCNEncoder(input_dim=input_dim, latent_dim=latent_dim, filters=16, tcn_layers=2)
    model = TangentNormalJEPA(
        context_encoder=encoder,
        latent_dim=latent_dim,
        subspace_dim=subspace_dim,
        hidden_dim=32,
    )

    ctx = torch.randn(B, seq_len, input_dim)
    tgt = torch.randn(B, seq_len, input_dim)

    # Loss computation
    loss, metrics = model.compute_objective(ctx, tgt)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0
    assert "loss_normal" in metrics
    assert "loss_tangent" in metrics

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
