"""Tests for CSMConfig validation and canonical model instantiation."""

import pytest
import torch

from src.config import CSMConfig


class TestConfigValidation:
    def test_default_config_is_valid(self):
        c = CSMConfig()
        assert c.model_type == "ts_jepa"
        assert c.patch_size == 16

    def test_laminar_jepa_lite_config_is_valid(self):
        c = CSMConfig(model_type="laminar_jepa_lite", context_size=128, suspect_size=32)
        assert c.model_type == "laminar_jepa_lite"
        assert c.full_window_size == 160

    def test_unknown_model_type_rejected(self):
        with pytest.raises(ValueError, match="Unknown model_type"):
            CSMConfig(model_type="bogus")

    def test_unknown_encoder_rejected(self):
        with pytest.raises(ValueError, match="Unknown encoder_architecture"):
            CSMConfig(encoder_architecture="bogus")

    def test_invalid_percentile_rejected(self):
        with pytest.raises(ValueError, match="event_threshold_percentile"):
            CSMConfig(event_threshold_percentile=0.0)

    def test_non_positive_batch_size_rejected(self):
        with pytest.raises(ValueError, match="batch_size must be positive"):
            CSMConfig(batch_size=0)


class TestLaminarLiteBuild:
    def test_laminar_lite_config_builds_and_forwards(self):
        from src.engine.trainer import build_ts_jepa_model

        config = CSMConfig(
            model_type="laminar_jepa_lite",
            context_size=64,
            suspect_size=16,
            latent_dim=16,
            filters=32,
            tcn_layers=2,
        )
        model = build_ts_jepa_model(config, input_dim=4, device=torch.device("cpu"))
        ctx = torch.randn(2, config.context_size, 4)
        tgt = torch.randn(2, config.suspect_size, 4)
        loss, diag = model.compute_objective(ctx, tgt)
        assert torch.isfinite(loss)
        assert "loss" in diag
