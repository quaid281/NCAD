"""Unit tests for HamiltonianSymplecticJEPA (Symplectic Dynamics JEPA)."""

import pytest
import torch
import torch.nn as nn

from src.models.encoders.tcn_encoder import HybridTCNEncoder
from src.models.jepa.hamiltonian_jepa import (
    HamiltonianSymplecticJEPA,
    HamiltonianSymplecticJEPAModel,
    PotentialEnergyNet,
)


def test_potential_energy_net_and_grad():
    B, D_coord = 8, 8
    net = PotentialEnergyNet(coord_dim=D_coord, hidden_dim=32)

    q = torch.randn(B, D_coord)
    V = net(q)
    assert V.shape == (B,)

    grad = net.grad_V(q)
    assert grad.shape == (B, D_coord)
    assert torch.all(torch.isfinite(grad))


def test_symplectic_leapfrog_energy_conservation():
    B, D_coord = 6, 8
    net = PotentialEnergyNet(coord_dim=D_coord, hidden_dim=32)

    # Simple harmonic potential V(q) = 1/2 * ||q||^2
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()

    q_0 = torch.randn(B, D_coord)
    p_0 = torch.randn(B, D_coord)

    encoder = HybridTCNEncoder(input_dim=3, latent_dim=16, filters=16, tcn_layers=2)
    model = HamiltonianSymplecticJEPA(
        context_encoder=encoder,
        latent_dim=16,
        step_size=0.01,
        num_leapfrog_steps=10,
    )

    H_0 = model.compute_hamiltonian(q_0, p_0)
    q_K, p_K = model.symplectic_rollout(q_0, p_0)
    H_K = model.compute_hamiltonian(q_K, p_K)

    # Leapfrog is symplectic: energy should be closely conserved over small dt
    rel_energy_err = torch.abs(H_K - H_0) / (torch.abs(H_0) + 1e-4)
    assert torch.mean(rel_energy_err).item() < 0.1, f"Symplectic energy drift too high: {torch.mean(rel_energy_err).item()}"


def test_hamiltonian_jepa_forward_loss_and_discrepancy():
    latent_dim = 16
    B = 6
    seq_len = 25
    input_dim = 3

    encoder = HybridTCNEncoder(input_dim=input_dim, latent_dim=latent_dim, filters=16, tcn_layers=2)
    model = HamiltonianSymplecticJEPA(
        context_encoder=encoder,
        latent_dim=latent_dim,
        hidden_dim=32,
        step_size=0.05,
        num_leapfrog_steps=2,
    )

    ctx = torch.randn(B, seq_len, input_dim)
    tgt = torch.randn(B, seq_len, input_dim)

    # Loss
    loss, metrics = model.compute_objective(ctx, tgt)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert "loss_state" in metrics
    assert "loss_energy" in metrics

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
