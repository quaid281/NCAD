"""Tests for CausalSSMFlowJEPA and SplitConformalCalibrator."""

import numpy as np
import pytest
import torch

from src.models.jepa.causal_ssm_flow_jepa import CausalSSMContextEncoder, CausalSSMFlowJEPA
from src.scoring.conformal_calibrator import SplitConformalCalibrator


def test_causal_ssm_context_encoder_shapes():
    B, L, C = 4, 64, 5
    latent_dim = 16
    node_dim = 12
    hidden_dim = 24
    encoder = CausalSSMContextEncoder(
        in_channels=C,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        node_dim=node_dim,
        ssm_layers=2,
        gat_layers=1,
        num_heads=2,
    )

    x = torch.randn(B, L, C)
    z_ctx = encoder(x, return_graph=False)
    assert z_ctx.shape == (B, latent_dim)

    z_ctx, z_nodes, attn = encoder(x, return_graph=True)
    assert z_ctx.shape == (B, latent_dim)
    assert z_nodes.shape == (B, C, node_dim)
    assert attn.shape == (B, C, C)
    assert torch.all(torch.isfinite(z_ctx))
    assert torch.all(torch.isfinite(z_nodes))


def test_causal_ssm_flow_jepa_forward_and_backward():
    B, L_ctx, L_tgt, C = 4, 64, 16, 3
    latent_dim = 16
    model = CausalSSMFlowJEPA(
        in_channels=C,
        latent_dim=latent_dim,
        hidden_dim=32,
        node_dim=16,
        ssm_layers=1,
        gat_layers=1,
        flow_layers=2,
        num_heads=2,
    )

    # Verify target encoder parameters are frozen
    for param in model.target_encoder.parameters():
        assert not param.requires_grad

    x_ctx = torch.randn(B, L_ctx, C)
    x_tgt = torch.randn(B, L_tgt, C)

    loss, diag = model(x_ctx, x_tgt, return_diagnostics=True)
    assert torch.isfinite(loss)
    assert "cfm_loss" in diag
    assert "vic_loss" in diag
    assert "graph_loss" in diag

    loss.backward()

    # Check gradients exist for trainable modules
    assert model.context_encoder.channel_proj.weight.grad is not None
    assert model.flow_predictor.out_proj.weight.grad is not None
    # Verify target encoder has no gradients
    for param in model.target_encoder.parameters():
        assert param.grad is None


def test_causal_ssm_flow_jepa_scoring_and_counterfactual():
    B, L_ctx, L_tgt, C = 2, 32, 16, 4
    model = CausalSSMFlowJEPA(
        in_channels=C,
        latent_dim=16,
        hidden_dim=32,
        node_dim=16,
        ssm_layers=1,
        gat_layers=1,
        flow_layers=2,
        num_heads=2,
    )

    x_ctx = torch.randn(B, L_ctx, C)
    x_tgt = torch.randn(B, L_tgt, C)

    # Anomaly score
    scores = model.compute_anomaly_score(x_ctx, x_tgt)
    assert scores.shape == (B,)
    assert torch.all(scores >= 0)

    # Counterfactual root-cause attribution
    ch_scores, top_causes = model.counterfactual_root_cause_attribution(x_ctx, x_tgt)
    assert ch_scores.shape == (B, C)
    assert top_causes.shape == (B, C)

    # Causal graph extraction
    graph = model.get_causal_graph(x_ctx)
    assert graph.shape == (B, C, C)


def test_split_conformal_calibrator_guarantees():
    np.random.seed(42)
    # Generate 1,000 nominal calibration residuals from standard distribution
    cal_scores = np.random.exponential(scale=1.0, size=1000)
    alpha = 0.05
    calibrator = SplitConformalCalibrator(alpha=alpha)
    result = calibrator.calibrate(cal_scores)

    assert result.is_calibrated
    assert result.threshold > 0.0
    assert result.significance_level == alpha

    # Test on 2,000 holdout nominal samples
    test_nominal_scores = np.random.exponential(scale=1.0, size=2000)
    emp_fa_rate = calibrator.empirical_false_alarm_rate(test_nominal_scores)

    # Empirical false alarm rate should be close to alpha (e.g. ~ 5% +/- sampling error)
    assert abs(emp_fa_rate - alpha) < 0.02

    # Test on clear anomaly spikes
    anomaly_scores = test_nominal_scores + 10.0
    anom_preds = calibrator.predict(anomaly_scores)
    assert np.mean(anom_preds) > 0.95  # >95% detected

    # P-values
    p_vals = calibrator.compute_p_values(anomaly_scores)
    assert np.all((p_vals >= 0.0) & (p_vals <= 1.0))
    assert np.mean(p_vals < alpha) > 0.95
