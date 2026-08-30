"""Unit tests for Conditional Flow Matching TS-JEPA (FlowTSJEPA & PatchFlowJEPA)."""

import pytest
import torch
import torch.optim as optim

from src.models import (
    FlowLatentPredictor,
    FlowTSJEPA,
    FlowTSJEPAModel,
    HybridTCNEncoder,
    PatchFlowJEPA,
    PatchFlowPredictor,
    RelationalGATEncoder,
    flow_matching_vicreg_loss,
)
from src.models.flow_ts_jepa import TimestepEmbedding


def test_timestep_embedding():
    """Verify continuous sinusoidal timestep embeddings for flow times in [0, 1]."""
    embed = TimestepEmbedding(embed_dim=32)
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    emb = embed(t)
    assert emb.shape == (5, 32)
    assert not torch.isnan(emb).any()
    # Embeddings for distinct times should be distinct
    assert not torch.allclose(emb[0], emb[1])


def test_flow_latent_predictor():
    """Verify continuous velocity field network v_psi(z_t, t, z_ctx)."""
    latent_dim = 16
    predictor = FlowLatentPredictor(latent_dim=latent_dim, hidden_dim=32, num_layers=2)
    
    z_t = torch.randn(8, latent_dim, requires_grad=True)
    t = torch.rand(8)
    z_ctx = torch.randn(8, latent_dim, requires_grad=True)
    
    v = predictor(z_t, t, z_ctx)
    assert v.shape == (8, latent_dim)
    assert not torch.isnan(v).any()
    
    # Check backward gradient flow
    loss = v.sum()
    loss.backward()
    assert z_t.grad is not None
    assert z_ctx.grad is not None


def test_flow_ts_jepa_forward_and_loss():
    """Verify forward pass and Optimal Transport Flow Matching loss."""
    encoder = HybridTCNEncoder(input_dim=4, latent_dim=16, filters=32, tcn_layers=2)
    model = FlowTSJEPA(context_encoder=encoder, latent_dim=16, ema_decay=0.99)
    
    ctx = torch.randn(4, 64, 4)
    target = torch.randn(4, 16, 4)
    
    z_ctx, z_tgt_true, v_pred, v_target = model(ctx, target)
    
    assert z_ctx.shape == (4, 16)
    assert z_tgt_true.shape == (4, 16)
    assert v_pred.shape == (4, 16)
    assert v_target.shape == (4, 16)
    
    loss, metrics = flow_matching_vicreg_loss(
        v_pred=v_pred,
        v_target=v_target,
        z_ctx=z_ctx,
        z_tgt_true=z_tgt_true,
    )
    
    assert loss.ndim == 0
    assert loss.item() > 0.0
    assert "loss_flow" in metrics
    assert "loss_var" in metrics
    assert "loss_cov" in metrics


def test_flow_ts_jepa_sample_ode_solvers():
    """Verify ODE integration generation with Euler, Midpoint, and RK4 solvers."""
    encoder = HybridTCNEncoder(input_dim=3, latent_dim=16, filters=32, tcn_layers=2)
    model = FlowTSJEPA(context_encoder=encoder, latent_dim=16)
    
    ctx = torch.randn(4, 64, 3)
    
    for solver in ["euler", "midpoint", "rk4"]:
        z_gen = model.sample_target(ctx, n_steps=4, solver=solver)
        assert z_gen.shape == (4, 16)
        assert not torch.isnan(z_gen).any()


def test_flow_ts_jepa_discrepancy_and_mahalanobis():
    """Verify instantaneous and trajectory anomaly scoring with covariance whitening."""
    encoder = RelationalGATEncoder(input_dim=3, latent_dim=16, filters=32, tcn_layers=2, gat_layers=1)
    model = FlowTSJEPA(context_encoder=encoder, latent_dim=16)
    
    ctx = torch.randn(8, 64, 3)
    target = torch.randn(8, 16, 3)
    
    # 1. Instantaneous energy discrepancy (fast forward pass)
    score_fast = model.compute_instantaneous_energy_discrepancy(ctx, target, n_eval_times=2)
    assert score_fast.shape == (8,)
    assert (score_fast >= 0).all()
    
    # 2. Trajectory ODE discrepancy
    score_traj = model.compute_trajectory_discrepancy(ctx, target, n_steps=2)
    assert score_traj.shape == (8,)
    assert (score_traj >= 0).all()
    
    # 3. Fit Mahalanobis covariance and score with whitening
    model.fit_mahalanobis_covariance(ctx, target, batch_size=4)
    assert bool(model.precision_fitted.item()) is True
    
    score_maha = model.compute_instantaneous_energy_discrepancy(ctx, target, use_mahalanobis=True)
    assert score_maha.shape == (8,)
    assert (score_maha >= 0).all()


def test_flow_ts_jepa_training_loop():
    """Verify optimization and loss reduction on harmonic time series."""
    torch.manual_seed(42)
    encoder = HybridTCNEncoder(input_dim=2, latent_dim=16, filters=32, tcn_layers=2)
    model = FlowTSJEPA(context_encoder=encoder, latent_dim=16, ema_decay=0.95)
    optimizer = optim.AdamW(model.parameters(), lr=5e-3)
    
    # Generate harmonic sequence
    t_seq = torch.linspace(0, 4 * 3.14159, 80).unsqueeze(0).unsqueeze(-1)
    phases = torch.tensor([0.0, 1.5]).view(1, 1, 2)
    traj = torch.sin(t_seq + phases).repeat(8, 1, 1)
    
    initial_loss = None
    final_loss = None
    for step in range(25):
        batch = traj + 0.05 * torch.randn_like(traj)
        ctx = batch[:, :64]
        target = batch[:, 64:]
        
        optimizer.zero_grad()
        z_ctx, z_tgt_true, v_pred, v_target = model(ctx, target)
        loss, _ = flow_matching_vicreg_loss(v_pred, v_target, z_ctx, z_tgt_true)
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
        model.update_target_encoder()
        final_loss = loss.item()
        
    assert final_loss < initial_loss


def test_patch_flow_jepa():
    """Verify PatchFlowJEPA patch tokenization, cross-attention flow, and localized scoring."""
    model = PatchFlowJEPA(
        input_dim=4,
        patch_size=16,
        d_model=32,
        n_heads=4,
        n_layers=2,
        n_target_patches=3,
        predictor_layers=2,
    )
    
    ctx = torch.randn(4, 64, 4)   # 64 / 16 = 4 context patches
    target = torch.randn(4, 48, 4) # 48 / 16 = 3 target patches
    
    # 1. Forward training pass
    h_ctx, z_tgt_true, v_pred, v_target = model(ctx, target)
    assert h_ctx.shape == (4, 4, 32)
    assert z_tgt_true.shape == (4, 3, 32)
    assert v_pred.shape == (4, 3, 32)
    assert v_target.shape == (4, 3, 32)
    
    # 2. ODE target patch generation
    z_gen = model.sample_target_patches(ctx, n_steps=3, solver="midpoint")
    assert z_gen.shape == (4, 3, 32)
    assert not torch.isnan(z_gen).any()
    
    # 3. Patch-level and window-level discrepancy scoring
    patch_scores, win_scores = model.compute_patch_instantaneous_discrepancy(ctx, target, n_eval_times=2)
    assert patch_scores.shape == (4, 3) # Localized anomaly score for each of 3 target patches
    assert win_scores.shape == (4,)
    
    patch_scores_t, win_scores_t = model.compute_patch_trajectory_discrepancy(ctx, target, n_steps=2)
    assert patch_scores_t.shape == (4, 3)
    assert win_scores_t.shape == (4,)
    
    # 4. Target EMA update
    init_param = next(model.target_encoder.parameters()).clone()
    with torch.no_grad():
        for p in model.context_encoder.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    model.update_target_encoder()
    updated_param = next(model.target_encoder.parameters())
    assert not torch.equal(init_param, updated_param)
