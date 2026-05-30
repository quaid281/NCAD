import pytest
import torch
import numpy as np

from src.models.tcn_encoder import HybridTCNEncoder, contrastive_loss
from src.models.multi_scale_tcn_encoder import MultiScaleTCNEncoder


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
