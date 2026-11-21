import numpy as np
import pytest
import torch

from src.models.encoders.multi_scale_tcn_encoder import MultiScaleTCNEncoder
from src.models.encoders.tcn_encoder import HybridTCNEncoder, contrastive_loss


def test_hybrid_tcn_encoder():
    batch_size = 4
    window_size = 100
    feature_dim = 16
    latent_dim = 8

    model = HybridTCNEncoder(
        input_dim=feature_dim,
        latent_dim=latent_dim,
        filters=16,
        tcn_layers=2,
        kernel_size=3,
        dropout=0.1
    )

    # Input shape: (Batch, Window, Feature)
    x = torch.randn(batch_size, window_size, feature_dim)
    output = model(x)

    # Output shape: (Batch, LatentDim)
    assert output.shape == (batch_size, latent_dim)


def test_multi_scale_tcn_encoder():
    batch_size = 4
    window_size = 100
    feature_dim = 16
    latent_dim = 8

    model = MultiScaleTCNEncoder(
        input_dim=feature_dim,
        latent_dim=latent_dim,
        filters=16,
        tcn_layers=2,
        kernel_size=3,
        dropout=0.1
    )

    x = torch.randn(batch_size, window_size, feature_dim)
    output = model(x)

    assert output.shape == (batch_size, latent_dim)


def test_contrastive_loss():
    batch_size = 2
    latent_dim = 8

    emb_full = torch.randn(batch_size, latent_dim)
    emb_context = torch.randn(batch_size, latent_dim)

    # Labels: 0 = clean context/suspect match, 1 = injected anomaly
    labels = torch.tensor([0.0, 1.0])

    loss = contrastive_loss(emb_full, emb_context, labels, margin=1.0)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_selective_ssm_encoder():
    from src.models.encoders.selective_ssm_encoder import SelectiveSSMContextEncoder

    batch_size = 4
    window_size = 64
    feature_dim = 8
    latent_dim = 16

    model = SelectiveSSMContextEncoder(input_dim=feature_dim, latent_dim=latent_dim, hidden_dim=32, layers=2)
    x = torch.randn(batch_size, window_size, feature_dim)
    out = model(x)

    assert out.shape == (batch_size, latent_dim)
    assert torch.isfinite(out).all()


def test_laminar_jepa_lite_forward():
    from src.models.laminar_lite import LaminarJEPALiteModel, LaminarLiteEncoder

    enc = LaminarLiteEncoder(input_dim=4, latent_dim=16, filters=16, tcn_layers=2)
    model = LaminarJEPALiteModel(context_encoder=enc, latent_dim=16, predictor_hidden_dim=32)
    ctx = torch.randn(4, 64, 4)
    target = torch.randn(4, 16, 4)

    loss, diag = model.compute_objective(ctx, target)
    assert torch.isfinite(loss)
    assert "loss" in diag

    discrepancy = model.compute_predictive_discrepancy(ctx, target)
    assert discrepancy.shape == (4,)
    assert torch.isfinite(discrepancy).all()

