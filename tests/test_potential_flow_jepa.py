"""Unit tests for Potential-Flow JEPA (PF-JEPA) architecture."""

import pytest
import torch
import torch.nn as nn

from src.models import (
    HarmonicGrassmannianCodebook,
    HybridTCNEncoder,
    PotentialFlowJEPA,
    PotentialFlowJEPAModel,
    PotentialFlowPredictor,
    ScalarPotentialField,
)


def test_harmonic_grassmannian_codebook():
    B, D = 16, 32
    n_regimes, subspace_dim = 4, 8
    codebook = HarmonicGrassmannianCodebook(latent_dim=D, n_regimes=n_regimes, subspace_dim=subspace_dim)

    # Check orthonormal bases: U_k^T U_k == I_d
    frames = codebook.get_orthonormal_frames()  # (K, D, d)
    assert frames.shape == (n_regimes, D, subspace_dim)
    for k in range(n_regimes):
        gram = frames[k].T @ frames[k]
        eye = torch.eye(subspace_dim)
        assert torch.allclose(gram, eye, atol=1e-5), f"Frame {k} not orthonormal"

    # Forward pass (soft routing)
    z_ctx = torch.randn(B, D)
    p_ctx, regime_probs, loss_ortho = codebook(z_ctx, hard=False)
    assert p_ctx.shape == (B, D)
    assert regime_probs.shape == (B, n_regimes)
    assert torch.allclose(regime_probs.sum(dim=-1), torch.ones(B), atol=1e-5)
    assert loss_ortho.item() >= 0.0

    # Forward pass (hard routing)
    p_ctx_hard, regime_probs_hard, _ = codebook(z_ctx, hard=True)
    assert p_ctx_hard.shape == (B, D)
    assert regime_probs_hard.shape == (B, n_regimes)


def test_scalar_potential_field():
    B, D = 8, 32
    potential_field = ScalarPotentialField(latent_dim=D, hidden_dim=64, num_layers=2)
    z_t = torch.randn(B, D)
    t = torch.rand(B)
    z_ctx = torch.randn(B, D)
    p_regime = torch.randn(B, D)

    phi = potential_field(z_t, t, z_ctx, p_regime)
    assert phi.shape == (B, 1)


def test_potential_flow_predictor_and_laplacian():
    B, D = 8, 32
    predictor = PotentialFlowPredictor(latent_dim=D, hidden_dim=64, num_layers=2)
    z_t = torch.randn(B, D, requires_grad=True)
    t = torch.rand(B)
    z_ctx = torch.randn(B, D)
    p_regime = torch.randn(B, D)

    # Compute conservative velocity: v = -nabla Phi
    v_pred = predictor(z_t, t, z_ctx, p_regime, create_graph=True)
    assert v_pred.shape == (B, D)

    # Check backpropagation through velocity field
    loss = torch.sum(v_pred ** 2)
    loss.backward()

    # Compute Energy Laplacian Curvature via Hutchinson estimator
    laplacian = predictor.compute_energy_laplacian(z_t.detach(), t, z_ctx, p_regime, n_probes=2)
    assert laplacian.shape == (B,)
    assert not torch.isnan(laplacian).any()
    assert not torch.isinf(laplacian).any()


def test_potential_flow_jepa_end_to_end():
    B, L_ctx, L_tgt, C, D = 8, 128, 32, 4, 32
    encoder = HybridTCNEncoder(input_dim=C, latent_dim=D, filters=32, tcn_layers=3)
    model = PotentialFlowJEPAModel(
        context_encoder=encoder,
        latent_dim=D,
        predictor_hidden_dim=48,
        predictor_layers=2,
        n_regimes=3,
        subspace_dim=8,
        use_regimes=True,
    )

    ctx = torch.randn(B, L_ctx, C)
    tgt = torch.randn(B, L_tgt, C)

    # Training forward pass
    model.train()
    loss, metrics = model.compute_objective(ctx, tgt)
    assert loss.item() > 0.0
    assert "loss_flow" in metrics
    assert "loss_grassmann" in metrics

    loss.backward()

    # Target encoder EMA update
    model.update_target_encoder(decay=0.99)

    # Eval mode & Covariance fitting
    model.eval()
    model.fit_mahalanobis_covariance(ctx, tgt, batch_size=4)
    assert bool(model.precision_fitted.item())

    # Discrepancy with and without curvature
    disc_base = model.compute_predictive_discrepancy(ctx, tgt, use_mahalanobis=True, include_curvature=False)
    assert disc_base.shape == (B,)
    assert (disc_base >= 0).all()

    disc_curv = model.compute_predictive_discrepancy(
        ctx, tgt, use_mahalanobis=True, include_curvature=True, curvature_weight=0.1
    )
    assert disc_curv.shape == (B,)
    assert (disc_curv >= 0).all()
