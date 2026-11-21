r"""Tests for Geometric Invariant Layers and Simplified Tier 1 Models.

Validates:
1. MovingTangentProjection (Chapter 2, §4-5):
   - Orthogonality: u^T (P_u v) = 0
   - Idempotency: P_u(P_u v) = P_u v
2. ResolventPurification (Chapter 6, §4.2):
   - Symmetry: \Gamma(F) = \Gamma(F)^T
   - Positive semi-definiteness: min eigenvalue >= 0
   - Bounded spectrum: max eigenvalue < 1
   - Backward pass gradient flow without eigvalsh
3. Integration in ReynoldsStressJEPA, PotentialFlowJEPA, HarmonicSpringJEPA.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.geometric_layers import (
    HankelPolynomialFilter,
    MovingTangentProjection,
    ResolventPurification,
)


def test_moving_tangent_projection_orthogonality():
    B, D = 8, 32
    layer = MovingTangentProjection(latent_dim=D)
    
    # Random pole and velocity vectors
    z_pole = torch.randn(B, D)
    v = torch.randn(B, D)
    
    v_tangent = layer(v, z_pole=z_pole)
    
    # Normalize pole to unit sphere
    u = z_pole / torch.norm(z_pole, p=2, dim=-1, keepdim=True)
    
    # Check orthogonality <u, v_tangent> == 0
    inner = torch.sum(u * v_tangent, dim=-1)
    assert torch.allclose(inner, torch.zeros_like(inner), atol=1e-5), f"Max inner product: {inner.abs().max()}"


def test_moving_tangent_projection_idempotency():
    B, D = 4, 16
    layer = MovingTangentProjection(latent_dim=D)
    z_pole = torch.randn(B, D)
    v = torch.randn(B, D)
    
    v_proj1 = layer(v, z_pole=z_pole)
    v_proj2 = layer(v_proj1, z_pole=z_pole)
    assert torch.allclose(v_proj1, v_proj2, atol=1e-5)


def test_resolvent_purification_spectral_bounds():
    B, D = 6, 16
    gamma = 0.05
    layer = ResolventPurification(dim=D, gamma=gamma)
    
    # Unconstrained random matrix factor
    L = torch.randn(B, D, D, requires_grad=True)
    purified = layer(L)
    
    # 1. Symmetry check: Gamma = Gamma^T
    assert torch.allclose(purified, purified.transpose(-1, -2), atol=1e-5)
    
    # 2. Spectral bounds check via eigenvalues
    evals = torch.linalg.eigvalsh(purified)
    
    # All eigenvalues must be >= 0 (SPSD)
    assert (evals >= -1e-6).all(), f"Negative eigenvalue detected: {evals.min().item()}"
    
    # All eigenvalues must be < 1.0 (strict contractivity)
    assert (evals < 1.0).all(), f"Eigenvalue exceeds 1: {evals.max().item()}"
    
    # 3. Gradient backward pass check
    loss = purified.sum()
    loss.backward()
    assert L.grad is not None
    assert not torch.isnan(L.grad).any()


def test_hankel_polynomial_filter_properties():
    B, L, C = 4, 32, 8
    degree = 3
    filter_layer = HankelPolynomialFilter(channels=C, window_len=L, degree=degree, learnable_fusion=False)
    
    # 1. Check idempotency of projection matrix: P^2 == P
    P = filter_layer.P_poly
    P_sq = P @ P
    assert torch.allclose(P, P_sq, atol=1e-5), f"Max projection error: {(P - P_sq).abs().max()}"
    
    # 2. Check exact polynomial preservation:
    # Construct a pure quadratic polynomial signal x(t) = 3 t^2 - t + 2
    t = torch.linspace(-1.0, 1.0, steps=L).unsqueeze(0).unsqueeze(-1)  # (1, L, 1)
    poly_signal = (3.0 * t**2 - t + 2.0).expand(B, L, C)
    
    out, x_poly, x_res = filter_layer(poly_signal, return_decomposition=True)
    # The polynomial signal must be preserved exactly by the projection
    assert torch.allclose(x_poly, poly_signal, atol=1e-4), f"Poly error: {(x_poly - poly_signal).abs().max()}"
    # The residual component must be approximately 0
    assert torch.allclose(x_res, torch.zeros_like(x_res), atol=1e-4), f"Residual max: {x_res.abs().max()}"
    
    # 3. Check orthogonality: <x_poly, x_res> == 0 on arbitrary signal
    rand_signal = torch.randn(B, L, C)
    _, rand_poly, rand_res = filter_layer(rand_signal, return_decomposition=True)
    inner = torch.sum(rand_poly * rand_res, dim=1)  # (B, C)
    assert torch.allclose(inner, torch.zeros_like(inner), atol=1e-4), f"Orthogonality error: {inner.abs().max()}"
    
    # 4. Check moments computation
    moments = filter_layer.compute_hankel_moments(rand_signal)
    assert moments.shape == (B, degree + 1, C)


def test_resolvent_purification_bottleneck():
    from src.models.geometric_layers import ResolventPurificationBottleneck
    B, D = 16, 32
    layer = ResolventPurificationBottleneck(latent_dim=D, gamma=1e-2)
    z = torch.randn(B, D, requires_grad=True)
    out = layer(z)
    assert out.shape == (B, D)
    assert not torch.isnan(out).any()
    loss = out.sum()
    loss.backward()
    assert z.grad is not None


def test_cayley_orthogonal_gate_spectral_radius():
    from src.models.geometric_layers import CayleyOrthogonalGate
    D = 16
    gate = CayleyOrthogonalGate(dim=D)
    W = gate.transition_matrix
    assert W.shape == (D, D)
    # Check spectral radius: max |lambda_i| <= 1.0001
    evals = torch.linalg.eigvals(W)
    max_radius = torch.max(torch.abs(evals)).item()
    assert max_radius <= 1.0001, f"Cayley spectral radius {max_radius} exceeds 1.0"
    # Check matrix 2-norm <= 1.0001
    norm_2 = torch.linalg.norm(W, ord=2).item()
    assert norm_2 <= 1.0001, f"Cayley matrix 2-norm {norm_2} exceeds 1.0"


def test_symplectic_leapfrog_block():
    from src.models.geometric_layers import SymplecticLeapfrogBlock
    B, coord_dim = 8, 16
    latent_dim = coord_dim * 2
    block = SymplecticLeapfrogBlock(coord_dim=coord_dim, hidden_dim=32, num_steps=2)
    z = torch.randn(B, latent_dim, requires_grad=True)
    z_next = block(z)
    assert z_next.shape == (B, latent_dim)
    assert not torch.isnan(z_next).any()
    loss = z_next.sum()
    loss.backward()
    assert z.grad is not None


def test_hybrid_tcn_encoder_with_geometric_layers():
    from src.models.encoders.tcn_encoder import HybridTCNEncoder
    B, L, C = 4, 128, 5
    latent_dim = 24
    enc = HybridTCNEncoder(
        input_dim=C,
        latent_dim=latent_dim,
        filters=32,
        tcn_layers=3,
        use_resolvent=True,
        use_hankel=True,
        window_len=L,
    )
    x = torch.randn(B, L, C)
    z = enc(x)
    assert z.shape == (B, latent_dim)
    assert not torch.isnan(z).any()


