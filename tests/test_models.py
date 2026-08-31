import numpy as np
import pytest
import torch

from src.models.multi_scale_tcn_encoder import MultiScaleTCNEncoder
from src.models.tcn_encoder import HybridTCNEncoder, contrastive_loss


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
    from src.models.selective_ssm_encoder import SelectiveSSMContextEncoder

    batch_size = 4
    window_size = 64
    feature_dim = 8
    latent_dim = 16

    model = SelectiveSSMContextEncoder(input_dim=feature_dim, latent_dim=latent_dim, hidden_dim=32, layers=2)
    x = torch.randn(batch_size, window_size, feature_dim)
    out = model(x)

    assert out.shape == (batch_size, latent_dim)
    assert torch.isfinite(out).all()


def test_ncad_jepa_model_forward():
    from src.models.ncad_jepa import NCADJEPAModel

    model = NCADJEPAModel(input_dim=4, latent_dim=16, filters=16, tcn_layers=2)
    ctx = torch.randn(4, 64, 4)
    target = torch.randn(4, 16, 4)

    z_ctx, z_target_true, z_target_pred = model(ctx, target)
    assert z_ctx.shape == (4, 16)
    assert z_target_true.shape == (4, 16)
    assert z_target_pred.shape == (4, 16)

    discrepancy = model.compute_predictive_discrepancy(ctx, target)
    assert discrepancy.shape == (4,)
    assert torch.isfinite(discrepancy).all()


def test_gat_jepa_model_forward():
    from src.models.gat_jepa import RelationalGAT_JEPAModel

    model = RelationalGAT_JEPAModel(input_dim=4, latent_dim=16, filters=16, tcn_layers=2, gat_layers=1)
    ctx = torch.randn(4, 64, 4)
    target = torch.randn(4, 16, 4)

    z_ctx, z_target_true, z_target_pred = model(ctx, target)
    assert z_ctx.shape == (4, 16)
    assert z_target_true.shape == (4, 16)
    assert z_target_pred.shape == (4, 16)

    discrepancy = model.compute_predictive_discrepancy(ctx, target)
    assert discrepancy.shape == (4,)
    assert torch.isfinite(discrepancy).all()

