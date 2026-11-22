"""Unit tests for P1 architectural improvements:
- Context-conditioned query modulation and frequency scaling in PatchTSJEPA.
- Residual skip projections in CausalSSMContextEncoder.
- Highway dimension preservation in FlowLatentPredictor.
"""

import pytest
import torch

from src.config import CSMConfig
from src.engine.activation_tracer import autopsy_model
from src.engine.trainer import build_ts_jepa_model
from src.models.jepa.causal_ssm_flow_jepa import CausalSSMContextEncoder, CausalSSMFlowJEPA
from src.models.jepa.flow_ts_jepa import FlowLatentPredictor
from src.models.jepa.patch_ts_jepa import PatchTSJEPA


def test_patch_jepa_no_variance_collapse():
    """Verify that PatchTSJEPA exhibits strictly positive batch variance and no collapsed layers."""
    device = torch.device("cpu")
    config = CSMConfig(
        model_type="patch_ts_jepa",
        context_size=64,
        suspect_size=16,
        patch_size=8,
        latent_dim=16,
        filters=16,
        tcn_layers=2,
    )
    model = build_ts_jepa_model(config, input_dim=4, device=device)

    # Distinct batch samples
    B = 4
    x_c = torch.randn(B, 64, 4)
    x_t = torch.randn(B, 16, 4)

    report = autopsy_model(model, sample_input=(x_c, x_t), model_name="PatchTSJEPA")
    # All layers should have positive batch variance (zero collapsed layers)
    assert len(report.collapsed_layers) == 0
    assert report.global_health == "HEALTHY"


def test_causal_ssm_residual_skip_proj():
    """Verify that CausalSSMContextEncoder uses skip_proj and backpropagates through it."""
    B, L, C = 4, 64, 3
    latent_dim = 16
    node_dim = 12
    hidden_dim = 24
    encoder = CausalSSMContextEncoder(
        in_channels=C,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        node_dim=node_dim,
        ssm_layers=1,
        gat_layers=1,
        num_heads=2,
    )

    x = torch.randn(B, L, C)
    z_ctx = encoder(x, return_graph=False)
    assert z_ctx.shape == (B, latent_dim)

    # Verify both global_proj and skip_proj receive gradients
    loss = z_ctx.sum()
    loss.backward()
    assert encoder.global_proj[0].weight.grad is not None
    assert encoder.skip_proj[0].weight.grad is not None


def test_flow_latent_predictor_highway():
    """Verify FlowLatentPredictor dimension highway and gradient flow to highway_scale."""
    B, D = 4, 16
    predictor = FlowLatentPredictor(latent_dim=D, hidden_dim=32, num_layers=2)

    z_t = torch.randn(B, D)
    t = torch.rand(B)
    z_ctx = torch.randn(B, D)

    v = predictor(z_t, t, z_ctx)
    assert v.shape == (B, D)

    loss = v.sum()
    loss.backward()
    assert predictor.highway_scale.grad is not None
    assert predictor.input_proj.weight.grad is not None
    assert predictor.out_proj.weight.grad is not None
