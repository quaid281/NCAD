"""Tests for CSMConfig validation and patch_ts_jepa compatibility."""

import pytest
import torch

from src.config import CSMConfig


class TestConfigValidation:
    def test_default_config_is_valid(self):
        c = CSMConfig()
        assert c.model_type == "ts_jepa"
        assert c.patch_size == 16

    def test_patch_ts_jepa_rejects_indivisible_context(self):
        with pytest.raises(ValueError, match="context_size .* must be divisible by patch_size"):
            CSMConfig(model_type="patch_ts_jepa", context_size=284, suspect_size=16, patch_size=16)

    def test_patch_ts_jepa_rejects_indivisible_suspect(self):
        with pytest.raises(ValueError, match="suspect_size .* must be divisible by patch_size"):
            CSMConfig(model_type="patch_ts_jepa", context_size=288, suspect_size=18, patch_size=16)

    def test_patch_ts_jepa_accepts_compatible_sizes(self):
        c = CSMConfig(model_type="patch_ts_jepa", context_size=288, suspect_size=32, patch_size=16)
        assert c.context_size % c.patch_size == 0
        assert c.suspect_size % c.patch_size == 0

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


class TestPatchTSJEPABuild:
    def test_compatible_patch_config_builds_and_forwards(self):
        from src.engine.trainer import build_ts_jepa_model

        config = CSMConfig(
            model_type="patch_ts_jepa",
            context_size=288,
            suspect_size=16,
            patch_size=16,
            filters=32,
            tcn_layers=2,
        )
        model = build_ts_jepa_model(config, input_dim=4, device=torch.device("cpu"))
        ctx = torch.randn(2, config.context_size, 4)
        tgt = torch.randn(2, config.suspect_size, 4)
        h_ctx, h_tgt_true, h_tgt_pred = model(ctx, tgt)
        assert h_ctx.shape[0] == 2
        assert h_tgt_pred.shape[0] == 2
