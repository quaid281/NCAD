"""Tests for the shared JEPABase lifecycle behaviour."""

import torch

from src.models._jepa_utils import JEPABase
from src.models.jepa.multiscale_ts_jepa import MultiScaleTSJEPA
from src.models.jepa.ts_jepa import TSJEPAModel
from src.models.laminar_lite import LaminarJEPALiteModel


def _make_ts_jepa():
    from src.models.encoders.tcn_encoder import HybridTCNEncoder

    enc = HybridTCNEncoder(input_dim=3, latent_dim=16, filters=16, tcn_layers=2)
    return TSJEPAModel(context_encoder=enc, latent_dim=16, predictor_hidden_dim=32, predictor_layers=1)


def test_all_jepa_models_inherit_jepa_base():
    assert issubclass(TSJEPAModel, JEPABase)
    assert issubclass(MultiScaleTSJEPA, JEPABase)
    assert issubclass(LaminarJEPALiteModel, JEPABase)


def test_target_encoder_is_frozen_and_eval():
    model = _make_ts_jepa()
    for p in model.target_encoder.parameters():
        assert p.requires_grad is False
    assert model.target_encoder.training is False


def test_train_mode_keeps_target_encoder_in_eval():
    model = _make_ts_jepa()
    model.train()
    assert model.training is True
    assert model.target_encoder.training is False


def test_update_target_encoder_ema():
    model = _make_ts_jepa()
    with torch.no_grad():
        for p in model.context_encoder.parameters():
            p.add_(0.1)
    tgt_before = [p.clone() for p in model.target_encoder.parameters()]

    model.update_target_encoder(decay=0.5)

    for p_q, p_k, before_k in zip(
        model.context_encoder.parameters(),
        model.target_encoder.parameters(),
        tgt_before,
    ):
        expected = 0.5 * before_k + 0.5 * p_q
        assert torch.allclose(p_k, expected, atol=1e-6)
