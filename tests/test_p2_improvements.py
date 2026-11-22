"""Unit tests for P2 architectural improvements:
- Observation-gated Kalman belief correction in LatentWorldJEPA.
- Multi-layer hierarchical predictive discrepancy scoring in Engine Evaluator.
"""

import numpy as np
import pytest
import torch

from src.config import CSMConfig
from src.engine.evaluator import compute_hierarchical_jepa_discrepancy, compute_jepa_discrepancy
from src.engine.trainer import build_ts_jepa_model
from src.models.jepa.latent_world_jepa import LatentRecurrentCore, LatentWorldJEPA


def test_latent_world_observation_gating():
    """Verify observation-gated Kalman updates and backward compatibility."""
    B, D, H = 4, 16, 32
    core = LatentRecurrentCore(latent_dim=D, hidden_dim=H)

    z_t = torch.randn(B, D)
    h_0 = torch.zeros(B, H)

    # 1. Backward compatibility check (pure autonomous step)
    z_hat_next, h_next = core.step(z_t, h_0)
    assert z_hat_next.shape == (B, D)
    assert h_next.shape == (B, H)

    # 2. Closed-loop observation step with innovation return
    z_obs = z_hat_next + 1.5 * torch.randn(B, D)
    z_hat_2, h_2, innovation = core.step(z_t, h_0, z_obs=z_obs, return_innovation=True)
    assert z_hat_2.shape == (B, D)
    assert h_2.shape == (B, H)
    assert innovation.shape == (B, D)
    assert torch.all(torch.isfinite(innovation))
    assert float(torch.norm(innovation).item()) > 0.0

    # 3. Backprop check through observation gate
    loss = (innovation ** 2).sum() + h_2.sum()
    loss.backward()
    assert core.obs_gate[0].weight.grad is not None
    assert core.obs_proj.weight.grad is not None


def test_latent_world_jepa_closed_loop_discrepancy():
    """Verify closed-loop innovation scoring on LatentWorldJEPA."""
    B, C_len, H_len, C_dim = 4, 32, 16, 4
    model = LatentWorldJEPA(input_dim=C_dim, latent_dim=16, hidden_dim=32)

    ctx = torch.randn(B, C_len, C_dim)
    tgt = torch.randn(B, H_len, C_dim)

    # Compute discrepancy
    disc = model.compute_predictive_discrepancy(ctx, tgt)
    assert disc.shape == (B,)
    assert torch.all(disc >= 0.0)
    assert torch.all(torch.isfinite(disc))


def test_hierarchical_jepa_discrepancy():
    """Verify multi-layer hierarchical discrepancy computation."""
    config = CSMConfig(
        model_type="flow_jepa",
        context_size=64,
        suspect_size=16,
        latent_dim=16,
        filters=16,
        tcn_layers=2,
    )
    device = torch.device("cpu")
    model = build_ts_jepa_model(config, input_dim=4, device=device)

    windows = np.random.randn(8, 80, 4).astype(np.float32)

    # 1. Direct hierarchical evaluator call
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
    for l_name, scores in layer_dict.items():
        assert len(scores) == 8

    # 2. Standard evaluator wrapper call with hierarchical=True
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
