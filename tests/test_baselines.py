"""Unit tests for modern SOTA baselines:
- Anomaly Transformer (ICLR 2022)
- TimesNet (ICLR 2023)
- DCdetector (KDD 2023)
- TranAD (VLDB 2022)
"""

import pytest
import torch
from src.models.baselines import (
    AnomalyTransformer,
    DCdetector,
    TimesNet,
    TranAD,
)


@pytest.fixture
def dummy_data():
    batch_size = 4
    seq_len = 64
    num_channels = 5
    return torch.randn(batch_size, seq_len, num_channels)


def test_anomaly_transformer(dummy_data):
    B, L, C = dummy_data.shape
    model = AnomalyTransformer(c_in=C, d_model=32, n_heads=2, e_layers=2, d_ff=64)
    rec, series_list, prior_list = model(dummy_data)

    assert rec.shape == (B, L, C)
    assert len(series_list) == 2
    assert len(prior_list) == 2
    assert series_list[0].shape == (B, 2, L, L)
    assert prior_list[0].shape == (B, 2, L, L)

    # Loss computation & backward
    loss_rec = torch.mean((rec - dummy_data) ** 2)
    ass_dis = model.association_discrepancy(prior_list, series_list)
    loss = loss_rec + 0.01 * ass_dis.mean()
    loss.backward()

    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None

    # Anomaly scoring
    scores = model.compute_anomaly_scores(dummy_data)
    assert scores.shape == (B, L)
    assert (scores >= 0).all()


def test_timesnet(dummy_data):
    B, L, C = dummy_data.shape
    model = TimesNet(c_in=C, d_model=32, d_ff=32, e_layers=2, top_k=2)
    rec = model(dummy_data)

    assert rec.shape == (B, L, C)

    # Loss computation & backward
    loss = torch.mean((rec - dummy_data) ** 2)
    loss.backward()

    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None

    # Anomaly scoring
    scores = model.compute_anomaly_scores(dummy_data)
    assert scores.shape == (B, L)
    assert (scores >= 0).all()


def test_dcdetector(dummy_data):
    B, L, C = dummy_data.shape
    model = DCdetector(c_in=C, patch_size1=8, patch_size2=16, d_model=32, n_heads=2, e_layers=2)
    z1, z2 = model(dummy_data)

    assert z1.shape[0] == B
    assert z2.shape[0] == B
    assert z1.shape[-1] == 32
    assert z2.shape[-1] == 32

    # Contrastive loss & backward
    loss = model.contrastive_loss(z1, z2)
    loss.backward()

    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None

    # Anomaly scoring
    scores = model.compute_anomaly_scores(dummy_data)
    assert scores.shape == (B, L)


def test_tranad(dummy_data):
    B, L, C = dummy_data.shape
    model = TranAD(c_in=C, d_model=32, n_heads=2, e_layers=2, d_layers=2, d_ff=64)
    rec1, rec2 = model(dummy_data)

    assert rec1.shape == (B, L, C)
    assert rec2.shape == (B, L, C)

    # Adversarial loss & backward
    l1, l2 = model.adversarial_loss(rec1, rec2, dummy_data, epoch=1)
    loss = l1 + l2
    loss.backward()

    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None

    # Anomaly scoring
    scores = model.compute_anomaly_scores(dummy_data)
    assert scores.shape == (B, L)
    assert (scores >= 0).all()
