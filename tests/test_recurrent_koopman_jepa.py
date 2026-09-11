"""Unit tests for Recurrent State-Space Koopman JEPA (RecurrentKoopmanJEPA)."""

import pytest
import torch
import torch.nn as nn

from src.models.jepa.recurrent_koopman_jepa import (
    HarmonicStateSpaceCell,
    RecurrentKoopmanJEPA,
    RecurrentKoopmanJEPAModel,
    RecurrentSSMEncoder,
)


def test_harmonic_state_space_cell():
    latent_dim = 16
    B = 4
    cell = HarmonicStateSpaceCell(latent_dim=latent_dim)

    assert cell.num_modes == 8
    mags, phases = cell.eigenvalues
    assert mags.shape == (8,)
    assert phases.shape == (8,)
    assert torch.all(mags <= 1.0)  # Stable/dissipative dynamics

    h = torch.randn(B, latent_dim)
    h_next = cell.step(h)
    assert h_next.shape == (B, latent_dim)

    # Rollout
    H = 32
    traj = cell.rollout(h, steps=H)
    assert traj.shape == (B, H, latent_dim)


def test_recurrent_ssm_encoder():
    input_dim = 6
    latent_dim = 16
    B = 4
    T = 50

    encoder = RecurrentSSMEncoder(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=32)
    x = torch.randn(B, T, input_dim)

    traj, h_final = encoder(x)
    assert traj.shape == (B, T, latent_dim)
    assert h_final.shape == (B, latent_dim)
    # The final step of the trajectory should match h_final
    assert torch.allclose(traj[:, -1, :], h_final)


def test_recurrent_koopman_jepa_end_to_end():
    input_dim = 5
    latent_dim = 16
    B = 4
    C = 40
    H = 20

    model = RecurrentKoopmanJEPA(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=32,
        ema_decay=0.99,
    )

    ctx = torch.randn(B, C, input_dim)
    tgt = torch.randn(B, H, input_dim)

    # Forward pass
    h_ctx_traj, h_tgt_traj, h_pred_traj = model(ctx, tgt)
    assert h_ctx_traj.shape == (B, C, latent_dim)
    assert h_tgt_traj.shape == (B, H, latent_dim)
    assert h_pred_traj.shape == (B, H, latent_dim)

    # Loss computation
    loss, metrics = model.compute_objective(ctx, tgt, cov_weight=0.5, freq_div_weight=0.05)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0
    assert "loss_rollout" in metrics
    assert "mean_eigenvalue_mag" in metrics

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
    ctx_large = torch.randn(32, C, input_dim)
    tgt_large = torch.randn(32, H, input_dim)
    model.fit_mahalanobis_covariance(ctx_large, tgt_large, batch_size=16)
    assert bool(model.precision_fitted.item()) is True
