"""Unit tests for multi-scale TS-JEPA architectural ablation."""

import torch

from src.models import MultiScaleTSJEPA


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
