"""Unit tests for Time-Series Joint Embedding Predictive Architecture (TS-JEPA)."""

import pytest
import torch
import torch.optim as optim

from src.models import (
    HybridTCNEncoder,
    LatentPredictor,
    RelationalGATEncoder,
    RelationalGAT_JEPAModel,
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


def test_ts_jepa_mahalanobis_covariance_scoring():
    """Verify fitting empirical residual covariance matrix and Mahalanobis scoring."""
    torch.manual_seed(42)
    encoder = HybridTCNEncoder(input_dim=3, latent_dim=16, filters=32, tcn_layers=2)
    jepa = TSJEPAModel(context_encoder=encoder, latent_dim=16)
    
    ctx = torch.randn(20, 64, 3)
    target = torch.randn(20, 16, 3)
    
    assert not jepa.precision_fitted
    jepa.fit_mahalanobis_covariance(ctx, target)
    assert jepa.precision_fitted
    assert jepa.precision_matrix.shape == (16, 16)
    
    mahal_disc = jepa.compute_predictive_discrepancy(ctx, target, use_mahalanobis=True)
    assert mahal_disc.shape == (20,)
    assert torch.isfinite(mahal_disc).all()
    assert (mahal_disc >= 0.0).all()


def test_relational_gat_jepa_model():
    """Verify RelationalGAT_JEPAModel forward, compute_loss, and discrepancy."""
    torch.manual_seed(42)
    model = RelationalGAT_JEPAModel(input_dim=4, latent_dim=16, filters=32, tcn_layers=2, gat_layers=1)
    
    ctx = torch.randn(8, 64, 4)
    target = torch.randn(8, 16, 4)
    
    loss = model.compute_loss(ctx, target)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    
    disc = model.compute_predictive_discrepancy(ctx, target)
    assert disc.shape == (8,)
    assert torch.isfinite(disc).all()


def test_teacher_determinism_in_train_mode():
    """Verify target encoder outputs are 100% deterministic even when model.train() is active."""
    encoder = HybridTCNEncoder(input_dim=4, latent_dim=16, filters=16, tcn_layers=2, dropout=0.20)
    model = TSJEPAModel(context_encoder=encoder, latent_dim=16, dropout=0.20)

    model.train()  # Explicitly put model into training mode
    assert model.training
    assert not model.target_encoder.training

    x_tgt = torch.randn(8, 16, 4)
    _, z_true_1, _ = model(torch.randn(8, 32, 4), x_tgt)
    _, z_true_2, _ = model(torch.randn(8, 32, 4), x_tgt)

    assert torch.allclose(z_true_1, z_true_2, atol=1e-7)


def test_mahalanobis_checkpoint_persistence():
    """Verify precision_fitted buffer and precision_matrix persist across state_dict save/load."""
    import io
    encoder = HybridTCNEncoder(input_dim=4, latent_dim=16, filters=16, tcn_layers=2)
    model = TSJEPAModel(context_encoder=encoder, latent_dim=16)

    ctx = torch.randn(64, 32, 4)
    tgt = torch.randn(64, 16, 4)
    model.fit_mahalanobis_covariance(ctx, tgt)
    assert bool(model.precision_fitted.item())

    # Save to buffer
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)

    # Load into fresh model instance
    fresh_encoder = HybridTCNEncoder(input_dim=4, latent_dim=16, filters=16, tcn_layers=2)
    fresh_model = TSJEPAModel(context_encoder=fresh_encoder, latent_dim=16)
    assert not bool(fresh_model.precision_fitted.item())

    fresh_model.load_state_dict(torch.load(buffer))
    assert bool(fresh_model.precision_fitted.item())

    # Scores should match identically
    score_orig = model.compute_predictive_discrepancy(ctx[:4], tgt[:4], use_mahalanobis=True)
    score_fresh = fresh_model.compute_predictive_discrepancy(ctx[:4], tgt[:4], use_mahalanobis=True)
    assert torch.allclose(score_orig, score_fresh, atol=1e-6)


def test_unfitted_mahalanobis_raises_error():
    """Verify calling Mahalanobis scoring without fitting covariance raises RuntimeError."""
    encoder = HybridTCNEncoder(input_dim=4, latent_dim=16, filters=16, tcn_layers=2)
    model = TSJEPAModel(context_encoder=encoder, latent_dim=16)
    ctx = torch.randn(4, 32, 4)
    tgt = torch.randn(4, 16, 4)

    with pytest.raises(RuntimeError):
        model.compute_predictive_discrepancy(ctx, tgt, use_mahalanobis=True)


def test_vicreg_branch_independence():
    """Verify branch-wise VICReg penalizes constant representations regardless of other branches."""
    latent_dim = 16
    z_collapsed = torch.ones(32, latent_dim)
    z_good_ctx = torch.randn(32, latent_dim)
    z_true = torch.randn(32, latent_dim)

    loss_collapsed = jepa_vicreg_loss(z_collapsed, z_true, z_context=z_good_ctx)
    z_varied_pred = torch.randn(32, latent_dim)
    loss_varied = jepa_vicreg_loss(z_varied_pred, z_true, z_context=z_good_ctx)

    assert loss_collapsed.item() > loss_varied.item()


def test_orchestrator_ts_jepa_pipeline(tmp_path):
    """Verify run_channel executes cleanly with TS-JEPA model_type."""
    import numpy as np
    from src.config import CSMConfig
    from src.data.data_loader import ChannelData
    from src.engine.orchestrator import run_channel

    n_train = 300
    n_test = 150
    train_raw = np.random.randn(n_train).astype(np.float32)
    test_raw = np.random.randn(n_test).astype(np.float32)
    test_raw[60:80] += 5.0
    test_labels = np.zeros(n_test, dtype=np.int32)
    test_labels[60:80] = 1

    from src.data.data_loader import NormalizationStats

    channel_data = ChannelData(
        channel_id="test_channel",
        train_raw=train_raw,
        test_raw=test_raw,
        train_normalized=train_raw,
        test_normalized=test_raw,
        labels=test_labels,
        anomaly_sequences=[(60, 80)],
        norm_stats=NormalizationStats(mean=0.0, std=1.0),
    )

    config = CSMConfig(
        model_type="ts_jepa",
        context_size=32,
        suspect_size=8,
        step=2,
        epochs=2,
        batch_size=16,
        latent_dim=16,
        filters=16,
        tcn_layers=2,
        save_plots=False,
    )

    device = torch.device("cpu")
    results = run_channel(channel_data, tmp_path, config, device)

    assert "point_metrics" in results
    assert "pa_metrics" in results
    assert "f1" in results["point_metrics"]
    assert 0.0 <= results["point_metrics"]["f1"] <= 1.0



