"""Unit tests for TS-JEPA architectural ablations."""

import pytest
import torch

from src.models import (
    HybridTCNEncoder,
    MultiScaleTSJEPA,
    PatchTSJEPA,
    RelationalGAT_JEPAModel,
    TSJEPAModel,
)


def test_patch_ts_jepa_forward_and_loss():
    batch_size = 4
    context_len = 256
    target_len = 64
    input_dim = 9
    patch_size = 16
    d_model = 32

    model = PatchTSJEPA(
        input_dim=input_dim,
        patch_size=patch_size,
        d_model=d_model,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        n_target_patches=target_len // patch_size,
    )

    ctx = torch.randn(batch_size, context_len, input_dim)
    tgt = torch.randn(batch_size, target_len, input_dim)

    h_ctx, h_tgt_true, h_tgt_pred = model(ctx, tgt)

    assert h_ctx.shape == (batch_size, context_len // patch_size, d_model)
    assert h_tgt_true.shape == (batch_size, target_len // patch_size, d_model)
    assert h_tgt_pred.shape == (batch_size, target_len // patch_size, d_model)

    loss = model.compute_patch_loss(h_tgt_pred, h_tgt_true, h_ctx)
    assert torch.isfinite(loss)
    loss.backward()

    # Test covariance fitting & discrepancy scoring
    model.fit_mahalanobis_covariance(ctx, tgt)
    assert model.precision_fitted

    scores_euc = model.compute_predictive_discrepancy(ctx, tgt, use_mahalanobis=False)
    scores_mah = model.compute_predictive_discrepancy(ctx, tgt, use_mahalanobis=True)

    assert scores_euc.shape == (batch_size,)
    assert scores_mah.shape == (batch_size,)
    assert torch.all(torch.isfinite(scores_euc))
    assert torch.all(torch.isfinite(scores_mah))


def test_multiscale_ts_jepa_forward_and_loss():
    batch_size = 4
    context_len = 256
    target_len = 64
    input_dim = 9
    latent_dim = 16

    model = MultiScaleTSJEPA(
        input_dim=input_dim,
        latent_dim=latent_dim,
        horizons=(16, 64),
        filters=32,
        tcn_layers=2,
    )

    ctx = torch.randn(batch_size, context_len, input_dim)
    tgt = torch.randn(batch_size, target_len, input_dim)

    z_ctx, z_tgt_true_dict, z_tgt_pred_dict = model(ctx, tgt)

    assert z_ctx.shape == (batch_size, latent_dim)
    assert 16 in z_tgt_pred_dict and 64 in z_tgt_pred_dict
    assert z_tgt_pred_dict[16].shape == (batch_size, latent_dim)
    assert z_tgt_true_dict[16].shape == (batch_size, latent_dim)

    loss = model.compute_multiscale_loss(z_tgt_pred_dict, z_tgt_true_dict, z_ctx)
    assert torch.isfinite(loss)
    loss.backward()

    model.fit_mahalanobis_covariance(ctx, tgt)
    assert model.precision_fitted

    scores_euc = model.compute_predictive_discrepancy(ctx, tgt, use_mahalanobis=False)
    scores_mah = model.compute_predictive_discrepancy(ctx, tgt, use_mahalanobis=True)

    assert scores_euc.shape == (batch_size,)
    assert scores_mah.shape == (batch_size,)
    assert torch.all(torch.isfinite(scores_euc))
    assert torch.all(torch.isfinite(scores_mah))


def test_gat_jepa_centered_mahalanobis():
    batch_size = 4
    context_len = 256
    target_len = 64
    input_dim = 5
    latent_dim = 16

    model = RelationalGAT_JEPAModel(
        input_dim=input_dim,
        latent_dim=latent_dim,
        filters=32,
        tcn_layers=2,
        gat_layers=1,
        gat_heads=2,
    )

    ctx = torch.randn(batch_size, context_len, input_dim)
    tgt = torch.randn(batch_size, target_len, input_dim)

    loss = model.compute_loss(ctx, tgt)
    assert torch.isfinite(loss)
    loss.backward()

    model.fit_mahalanobis_covariance(ctx, tgt)
    assert model.precision_fitted
    assert hasattr(model, "residual_mean")
    assert model.residual_mean.shape == (latent_dim,)

    scores_mah = model.compute_predictive_discrepancy(ctx, tgt, use_mahalanobis=True)
    assert scores_mah.shape == (batch_size,)
    assert torch.all(torch.isfinite(scores_mah))
