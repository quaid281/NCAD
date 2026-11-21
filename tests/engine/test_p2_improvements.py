"""Unit tests for hierarchical multi-layer predictive discrepancy scoring."""

import numpy as np
import torch

from src.config import CSMConfig
from src.engine.evaluator import compute_hierarchical_jepa_discrepancy, compute_jepa_discrepancy
from src.engine.trainer import build_ts_jepa_model


def test_hierarchical_jepa_discrepancy():
    """Verify multi-layer hierarchical discrepancy computation on TS-JEPA."""
    config = CSMConfig(
        model_type="ts_jepa",
        context_size=64,
        suspect_size=16,
        latent_dim=16,
        filters=16,
        tcn_layers=2,
    )
    device = torch.device("cpu")
    model = build_ts_jepa_model(config, input_dim=4, device=device)

    windows = np.random.randn(8, 80, 4).astype(np.float32)

    fused_scores, layer_dict = compute_hierarchical_jepa_discrepancy(
        model=model,
        windows=windows,
        config=config,
        device=device,
        batch_size=4,
        intermediate_weight=0.40,
    )
    assert fused_scores.shape == (8,)
    assert np.all(fused_scores >= 0.0)
    assert len(layer_dict) > 0
    for scores in layer_dict.values():
        assert len(scores) == 8

    wrapper_scores = compute_jepa_discrepancy(
        model=model,
        windows=windows,
        config=config,
        device=device,
        batch_size=4,
        hierarchical=True,
    )
    assert wrapper_scores.shape == (8,)
    assert np.all(np.isfinite(wrapper_scores))
