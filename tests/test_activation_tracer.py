"""Unit tests for the Layer Activation Tracer and Model Autopsy engine."""

import pytest
import torch
import torch.nn as nn

from src.config import CSMConfig
from src.engine.activation_tracer import (
    LayerActivationTracer,
    LayerAutopsy,
    _compute_batch_variance,
    _compute_stable_rank,
    autopsy_model,
)
from src.engine.trainer import build_encoder, build_ts_jepa_model
from src.models.baselines import AnomalyTransformer


class SimpleTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_tracer_hook_lifecycle():
    """Verify hooks are created on enter and fully removed on exit."""
    model = SimpleTestNet()
    x = torch.randn(4, 8)

    tracer = LayerActivationTracer(model)
    assert len(tracer._handles) == 0

    with tracer:
        assert len(tracer._handles) > 0
        _ = model(x)
        assert len(tracer.records) > 0

    # Handles should be removed and empty
    assert len(tracer._handles) == 0

    # Ensure no leftover hooks remain on submodules
    for module in model.modules():
        assert len(module._forward_hooks) == 0


def test_tracer_statistics_accuracy():
    """Verify numerical health stats match known tensor math."""
    # Test stable rank on rank-1 outer product matrix
    u = torch.randn(10, 1)
    v = torch.randn(1, 10)
    rank1_mat = u @ v
    s_rank = _compute_stable_rank(rank1_mat)
    assert 0.9 <= s_rank <= 1.1

    # Test batch variance on constant tensor vs varying tensor
    constant_t = torch.ones(5, 8) * 3.14
    assert _compute_batch_variance(constant_t) == 0.0

    varying_t = torch.randn(20, 8)
    assert _compute_batch_variance(varying_t) > 0.1


def test_tracer_on_single_input_encoder():
    """Verify tracing on a standard temporal sequence encoder."""
    config = CSMConfig(
        encoder_architecture="hybrid_tcn",
        latent_dim=16,
        filters=16,
        tcn_layers=2,
    )
    device = torch.device("cpu")
    encoder = build_encoder(config, input_dim=4, device=device)
    x = torch.randn(2, 64, 4)

    with LayerActivationTracer(encoder) as tracer:
        _ = encoder(x)

    assert len(tracer.records) > 0
    for name, rec in tracer.records.items():
        assert rec.health_flag in ("HEALTHY", "HIGH_SPARSITY", "COLLAPSED")
        assert rec.norm > 0.0
        assert rec.output_shape is not None


def test_tracer_on_dual_input_jepa():
    """Verify tracing on a JEPA model accepting (context, target)."""
    config = CSMConfig(
        model_type="ts_jepa",
        latent_dim=16,
        filters=16,
        tcn_layers=2,
    )
    device = torch.device("cpu")
    jepa = build_ts_jepa_model(config, input_dim=4, device=device)

    x_c = torch.randn(2, 64, 4)
    x_t = torch.randn(2, 16, 4)

    report = autopsy_model(jepa, sample_input=(x_c, x_t), model_name="TS-JEPA")
    assert len(report.records) > 0
    assert report.global_health in ("HEALTHY", "WARNING_COLLAPSE", "WARNING_HIGH_DEAD_NEURONS")

    summary_str = report.summary()
    assert "TS-JEPA" in summary_str
    assert "Total Submodules Traced" in summary_str

    df = report.to_dataframe()
    assert len(df) == len(report.records)
    assert "layer_name" in df.columns
    assert "l2_norm" in df.columns


def test_tracer_on_tuple_output_baseline():
    """Verify tracing on baselines returning tuples of outputs."""
    baseline = AnomalyTransformer(c_in=4, d_model=16, n_heads=2, e_layers=2, d_ff=32)
    x = torch.randn(2, 64, 4)

    report = autopsy_model(baseline, sample_input=x, model_name="AnomalyTransformer")
    assert len(report.records) > 0
    assert report.global_health != "CRITICAL_UNSTABLE"

    table_md = report.markdown_table(max_rows=10)
    assert "| layer_name |" in table_md


def test_autopsy_anomaly_divergence():
    """Verify anomaly divergence tracks perturbation propagation through layers."""
    model = SimpleTestNet()
    x_nom = torch.randn(4, 8)
    x_anom = x_nom.clone()
    # Inject large perturbation in first feature
    x_anom[:, 0] += 10.0

    report = autopsy_model(
        model,
        sample_input=x_nom,
        anomalous_input=x_anom,
        model_name="PerturbedNet",
    )
    assert len(report.anomaly_divergence) > 0
    # Every layer receiving perturbed input should show positive divergence
    for layer, div in report.anomaly_divergence.items():
        assert div > 0.0
