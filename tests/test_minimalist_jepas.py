"""Unit tests for CosineJEPA, CycleJEPA, and CosineCycleJEPA."""

import pytest
import torch

from src.models import (
    CosineCycleJEPA,
    CosineCycleJEPAModel,
    CosineJEPA,
    CosineJEPAModel,
    CycleJEPA,
    CycleJEPAModel,
    HybridTCNEncoder,
)


def test_cosine_jepa_end_to_end():
    B, L_ctx, L_tgt, C, D = 8, 128, 32, 4, 32
    encoder = HybridTCNEncoder(input_dim=C, latent_dim=D, filters=32, tcn_layers=3)
    model = CosineJEPAModel(context_encoder=encoder, latent_dim=D, hidden_dim=48)

    ctx = torch.randn(B, L_ctx, C)
    tgt = torch.randn(B, L_tgt, C)

    # 1. Training step
    model.train()
    loss, metrics = model.compute_objective(ctx, tgt)
    assert loss.item() > 0.0
    assert "loss_cosine" in metrics
    loss.backward()

    # 2. Update target encoder
    model.update_target_encoder(decay=0.99)

    # 3. Discrepancy scoring bounded in [0, 2]
    model.eval()
    disc = model.compute_predictive_discrepancy(ctx, tgt)
    assert disc.shape == (B,)
    assert (disc >= 0.0).all() and (disc <= 2.0001).all()


def test_cycle_jepa_end_to_end():
    B, L_ctx, L_tgt, C, D = 8, 128, 32, 4, 32
    encoder = HybridTCNEncoder(input_dim=C, latent_dim=D, filters=32, tcn_layers=3)
    model = CycleJEPAModel(context_encoder=encoder, latent_dim=D, hidden_dim=48)

    ctx = torch.randn(B, L_ctx, C)
    tgt = torch.randn(B, L_tgt, C)

    # 1. Training step
    model.train()
    loss, metrics = model.compute_objective(ctx, tgt)
    assert loss.item() > 0.0
    assert "loss_fwd" in metrics
    assert "loss_bwd" in metrics
    assert "loss_cycle" in metrics
    loss.backward()

    # 2. Update target encoder
    model.update_target_encoder(decay=0.99)

    # 3. Discrepancy scoring
    model.eval()
    disc = model.compute_predictive_discrepancy(ctx, tgt)
    assert disc.shape == (B,)
    assert (disc >= 0.0).all()


def test_cosine_cycle_jepa_end_to_end():
    B, L_ctx, L_tgt, C, D = 8, 128, 32, 4, 32
    encoder = HybridTCNEncoder(input_dim=C, latent_dim=D, filters=32, tcn_layers=3)
    model = CosineCycleJEPAModel(context_encoder=encoder, latent_dim=D, hidden_dim=48)

    ctx = torch.randn(B, L_ctx, C)
    tgt = torch.randn(B, L_tgt, C)

    # 1. Training step
    model.train()
    loss, metrics = model.compute_objective(ctx, tgt)
    assert loss.item() > 0.0
    assert "loss_fwd" in metrics
    assert "loss_bwd" in metrics
    assert "loss_cycle" in metrics
    loss.backward()

    # 2. Update target encoder
    model.update_target_encoder(decay=0.99)

    # 3. Discrepancy scoring bounded in [0, 4]
    model.eval()
    disc = model.compute_predictive_discrepancy(ctx, tgt)
    assert disc.shape == (B,)
    assert (disc >= 0.0).all() and (disc <= 4.0001).all()
