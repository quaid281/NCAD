"""Tests for batched covariance fitting accepting numpy arrays."""

import numpy as np
import torch

from src.config import CSMConfig
from src.engine.trainer import build_ts_jepa_model


def test_fit_mahalanobis_covariance_accepts_numpy():
    """Covariance fitting should accept numpy arrays without eager GPU transfer."""
    config = CSMConfig(
        model_type="ts_jepa",
        context_size=32,
        suspect_size=8,
        latent_dim=16,
        filters=16,
        tcn_layers=2,
        epochs=1,
    )
    model = build_ts_jepa_model(config, input_dim=3, device=torch.device("cpu"))

    n = 40
    ctx_np = np.random.randn(n, config.context_size, 3).astype(np.float32)
    tgt_np = np.random.randn(n, config.suspect_size, 3).astype(np.float32)

    model.fit_mahalanobis_covariance(ctx_np, tgt_np, batch_size=8)
    assert bool(model.precision_fitted.item()) is True
    assert model.precision_matrix.shape == (config.latent_dim, config.latent_dim)
    assert model.residual_mean.shape == (config.latent_dim,)
    # Precision matrix should be finite
    assert torch.isfinite(model.precision_matrix).all()


def test_fit_mahalanobis_covariance_numpy_matches_tensor():
    """Numpy and tensor inputs should produce equivalent covariance fits."""
    config = CSMConfig(
        model_type="ts_jepa",
        context_size=32,
        suspect_size=8,
        latent_dim=8,
        filters=16,
        tcn_layers=2,
        epochs=1,
    )
    torch.manual_seed(0)
    model_a = build_ts_jepa_model(config, input_dim=2, device=torch.device("cpu"))
    torch.manual_seed(0)
    model_b = build_ts_jepa_model(config, input_dim=2, device=torch.device("cpu"))

    n = 30
    ctx = torch.randn(n, config.context_size, 2)
    tgt = torch.randn(n, config.suspect_size, 2)

    model_a.fit_mahalanobis_covariance(ctx.numpy(), tgt.numpy(), batch_size=16)
    model_b.fit_mahalanobis_covariance(ctx, tgt, batch_size=16)

    assert torch.allclose(model_a.precision_matrix, model_b.precision_matrix, atol=1e-5)
    assert torch.allclose(model_a.residual_mean, model_b.residual_mean, atol=1e-5)
