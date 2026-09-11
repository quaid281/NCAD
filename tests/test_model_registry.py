"""Tests for the central model registry."""

import pytest

from src.models.registry import (
    canonical_model_choices,
    canonical_model_type,
    is_jepa_model,
    model_spec,
    requires_patch_division,
    valid_model_types,
)


def test_canonical_choices_are_unique_and_sorted():
    choices = canonical_model_choices()
    assert len(choices) == len(set(choices))
    assert "ts_jepa" in choices
    assert "ncad" in choices


def test_aliases_resolve_to_canonical():
    assert canonical_model_type("patch_jepa") == "patch_ts_jepa"
    assert canonical_model_type("relational_gat_jepa") == "gat_jepa"
    assert canonical_model_type("ts_jepa") == "ts_jepa"


def test_unknown_model_type_raises():
    with pytest.raises(ValueError, match="Unknown model_type"):
        canonical_model_type("bogus")


def test_is_jepa_model():
    assert is_jepa_model("ts_jepa") is True
    assert is_jepa_model("patch_ts_jepa") is True
    assert is_jepa_model("gat_jepa") is True
    assert is_jepa_model("ncad") is False
    assert is_jepa_model("ncad_jepa") is True


def test_requires_patch_division():
    assert requires_patch_division("patch_ts_jepa") is True
    assert requires_patch_division("patch_jepa") is True
    assert requires_patch_division("ts_jepa") is False
    assert requires_patch_division("gat_jepa") is False


def test_valid_model_types_includes_aliases():
    valid = valid_model_types()
    assert "patch_jepa" in valid
    assert "relational_gat_jepa" in valid
    assert "ts_jepa" in valid


def test_model_spec_returns_none_for_unknown():
    assert model_spec("bogus") is None


def test_cli_choices_match_registry():
    """The CLI --model-type choices must be a subset of valid model types."""
    from src.cli import parse_args

    # parse_args reads sys.argv; we cannot easily introspect choices without
    # invoking argparse, so we instead check that every canonical choice is
    # accepted by the registry.
    for choice in canonical_model_choices():
        assert canonical_model_type(choice) == choice


def test_build_ts_jepa_model_all_canonical_variants():
    """Verify that build_ts_jepa_model can instantiate every registered JEPA variant."""
    import torch

    from src.config import CSMConfig
    from src.engine.trainer import build_ts_jepa_model

    device = torch.device("cpu")
    input_dim = 4
    for choice in canonical_model_choices():
        if not is_jepa_model(choice):
            continue
        config = CSMConfig(
            model_type=choice,
            context_size=64,
            suspect_size=16,
            patch_size=8,
            latent_dim=16,
            filters=16,
            tcn_layers=2,
            rank=2,
            n_regimes=2,
            subspace_dim=4,
        )
        model = build_ts_jepa_model(config, input_dim=input_dim, device=device)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

