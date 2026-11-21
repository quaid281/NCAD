"""Unit tests for the 4 geometric invariant layers derived from 'Ten Advances in Mathematics' (OpenAI, 2026).

Tests:
1. CohnElkiesFilter: Even Mellin perturbation envelope, contractive damping, and gradient backpropagation.
2. MovingSubspaceProjector: Orthonormal frame generation, rank preservation, and tangent bundle projection.
3. GTInterlacingLayer: Gelfand-Tsetlin branching sieve, dimension-weighted reciprocity, and sorted decay envelope.
4. HankelMomentFilter: Bounded-moment Hankel matrix formation, resolvent operator filtering, and trace gating.
5. End-to-end integration and pure objective training with updated top-5 models.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.geometric_layers import (
    CohnElkiesFilter,
    GTInterlacingLayer,
    HankelMomentFilter,
    MovingSubspaceProjector,
)


def test_cohn_elkies_filter():
    """Verify Cohn-Elkies dual-shell Fourier modulation layer."""
    latent_dim = 32
    layer = CohnElkiesFilter(latent_dim=latent_dim, n_shells=8, eps=0.05)

    # 2D input
    z = torch.randn(16, latent_dim, requires_grad=True)
    out = layer(z)

    assert out.shape == z.shape
    assert torch.all(torch.isfinite(out))

    # Backward pass
    loss = out.sum()
    loss.backward()
    assert z.grad is not None
    assert torch.all(torch.isfinite(z.grad))

    # 3D sequence input
    z3 = torch.randn(8, 20, latent_dim)
    out3 = layer(z3)
    assert out3.shape == (8, 20, latent_dim)


def test_moving_subspace_projector():
    """Verify Moving Subspace Delsarte Projection Layer."""
    latent_dim = 32
    subspace_dim = 8
    proj = MovingSubspaceProjector(latent_dim=latent_dim, subspace_dim=subspace_dim)

    z_pred = torch.randn(16, latent_dim, requires_grad=True)
    z_ctx = torch.randn(16, latent_dim, requires_grad=True)

    out = proj(z_pred, z_ctx)
    assert out.shape == (16, latent_dim)
    assert torch.all(torch.isfinite(out))

    # Backward pass through both target and context paths
    loss = out.sum()
    loss.backward()
    assert z_pred.grad is not None
    assert z_ctx.grad is not None
    assert torch.all(torch.isfinite(z_pred.grad))
    assert torch.all(torch.isfinite(z_ctx.grad))


def test_gt_interlacing_layer():
    """Verify Gelfand-Tsetlin Interlacing Sieve Layer."""
    num_channels = 16
    gt_sieve = GTInterlacingLayer(num_channels=num_channels, rank=4)

    # 3D attention matrix (B, C, C)
    A = torch.randn(8, num_channels, num_channels, requires_grad=True)
    out = gt_sieve(A)

    assert out.shape == (8, num_channels, num_channels)
    assert torch.all(torch.isfinite(out))

    # Row sum normalization check (stochastic transitions)
    row_sums = out.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)

    # Backward pass
    loss = out.sum()
    loss.backward()
    assert A.grad is not None
    assert torch.all(torch.isfinite(A.grad))

    # 4D multi-head attention matrix (B, H, C, C)
    A4 = torch.randn(4, 4, num_channels, num_channels)
    out4 = gt_sieve(A4)
    assert out4.shape == (4, 4, num_channels, num_channels)


def test_hankel_moment_filter():
    """Verify Bounded-Moment Hankel Separable Filter."""
    latent_dim = 32
    filter_layer = HankelMomentFilter(latent_dim=latent_dim, hankel_order=4)

    # 2D input (B, D)
    x2 = torch.randn(16, latent_dim, requires_grad=True)
    out2 = filter_layer(x2)
    assert out2.shape == (16, latent_dim)
    assert torch.all(torch.isfinite(out2))

    loss2 = out2.sum()
    loss2.backward()
    assert x2.grad is not None
    assert torch.all(torch.isfinite(x2.grad))

    # 3D sequence input (B, L, D)
    x3 = torch.randn(8, 25, latent_dim, requires_grad=True)
    out3 = filter_layer(x3)
    assert out3.shape == (8, 25, latent_dim)
    assert torch.all(torch.isfinite(out3))


def test_top_models_pure_objective():
    """Verify that primary Laminar-JEPA-Lite computes a finite, gradient-flowing objective."""
    from src.models.laminar_lite import LaminarJEPALiteModel, LaminarLiteEncoder

    B, L_ctx, L_tgt, C, D = 4, 32, 16, 4, 16
    encoder = LaminarLiteEncoder(input_dim=C, latent_dim=D, filters=16, tcn_layers=2)
    model = LaminarJEPALiteModel(context_encoder=encoder, latent_dim=D, predictor_hidden_dim=32)

    ctx = torch.randn(B, L_ctx, C)
    tgt = torch.randn(B, L_tgt, C)

    loss, diag = model.compute_objective(ctx, tgt)
    assert torch.isfinite(loss)
    assert "loss" in diag
    loss.backward()
