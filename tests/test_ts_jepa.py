"""Unit tests for Time-Series Joint Embedding Predictive Architecture (TS-JEPA)."""

import pytest
import torch
import torch.optim as optim

from src.models.relational_gat_encoder import RelationalGATEncoder
from src.models.ts_jepa import (
    LatentPredictor,
    TSJEPAModel,
    jepa_vicreg_loss,
)


def test_latent_predictor_shape():
    """Verify LatentPredictor preserves batch and latent dimensions."""
    latent_dim = 16
    predictor = LatentPredictor(latent_dim=latent_dim, hidden_dim=32, num_layers=2)
    z_ctx = torch.randn(8, latent_dim)
    z_pred = predictor(z_ctx)
    
    assert z_pred.shape == (8, latent_dim)
    assert not torch.isnan(z_pred).any()


def test_ts_jepa_model_forward_and_ema():
    """Verify forward pass and EMA target encoder updates."""
    encoder = RelationalGATEncoder(input_dim=4, latent_dim=16, filters=32, tcn_layers=2, gat_layers=1)
    jepa = TSJEPAModel(context_encoder=encoder, latent_dim=16, ema_decay=0.99)
    
    ctx = torch.randn(4, 64, 4)
    target = torch.randn(4, 16, 4)
    
    z_ctx, z_target_true, z_target_pred = jepa(ctx, target)
    
    assert z_ctx.shape == (4, 16)
    assert z_target_true.shape == (4, 16)
    assert z_target_pred.shape == (4, 16)
    
    # Store initial target encoder param
    init_param = next(jepa.target_encoder.parameters()).clone()
    
    # Modify context encoder and perform EMA step
    with torch.no_grad():
        for p in jepa.context_encoder.parameters():
            p.add_(torch.randn_like(p) * 0.1)
            
    jepa.update_target_encoder()
    updated_param = next(jepa.target_encoder.parameters())
    
    # Target parameter should have moved towards context encoder
    assert not torch.equal(init_param, updated_param)


def test_jepa_vicreg_loss():
    """Verify VICReg loss components, variance preservation, and gradient flow."""
    z_pred = torch.randn(16, 8, requires_grad=True)
    z_true = torch.randn(16, 8)
    z_ctx = torch.randn(16, 8, requires_grad=True)
    
    loss = jepa_vicreg_loss(z_pred, z_true, z_ctx, sim_weight=1.0, var_weight=1.0, cov_weight=0.1)
    
    assert loss.ndim == 0
    assert loss.item() > 0.0
    assert not torch.isnan(loss).item()
    
    loss.backward()
    assert z_pred.grad is not None
    assert z_ctx.grad is not None


def test_ts_jepa_training_loop():
    """Verify end-to-end training optimization and discrepancy computation."""
    torch.manual_seed(42)
    encoder = RelationalGATEncoder(input_dim=3, latent_dim=16, filters=32, tcn_layers=2, gat_layers=1)
    jepa = TSJEPAModel(context_encoder=encoder, latent_dim=16, ema_decay=0.95)
    optimizer = optim.AdamW(jepa.parameters(), lr=5e-3)
    
    # Train on 30 steps of nominal harmonic trajectories
    t = torch.linspace(0, 4 * 3.14159, 80).unsqueeze(0).unsqueeze(-1)  # (1, 80, 1)
    phases = torch.tensor([0.0, 1.0, 2.0]).view(1, 1, 3)
    clean_trajectory = torch.sin(t + phases).repeat(8, 1, 1)
    
    initial_loss = None
    for step in range(30):
        batch = clean_trajectory + 0.05 * torch.randn_like(clean_trajectory)
        ctx = batch[:, :64]
        target = batch[:, 64:]
        
        optimizer.zero_grad()
        z_ctx, z_target_true, z_target_pred = jepa(ctx, target)
        loss = jepa_vicreg_loss(z_target_pred, z_target_true, z_ctx)
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
        jepa.update_target_encoder()
        
    final_loss = loss.item()
    assert final_loss < initial_loss
    assert final_loss < 2.0
    
    # Verify predictive discrepancy computation
    test_ctx = clean_trajectory[:, :64]
    test_target = clean_trajectory[:, 64:]
    discrepancy = jepa.compute_predictive_discrepancy(test_ctx, test_target)
    
    assert discrepancy.shape == (8,)
    assert torch.isfinite(discrepancy).all()
    assert (discrepancy >= 0.0).all()

