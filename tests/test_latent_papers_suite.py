"""Unit test suite for the 2025/2026 Latent Space Papers Suite.

Validates:
1. KinematicAdaptiveLatentJEPA (CMU AHEAD 2026)
2. LatentDeliberationJEPA (DMLR & Monet CVPR 2026)
3. SpectralPhysicsLatentJEPA (Polymathic AI NeurIPS 2025)
4. MultiAgentConsensusLatentJEPA (Interlat ACL 2026)
5. Model registry and builder integration across configurations.
"""

from __future__ import annotations

import pytest
import torch

from src.config import CSMConfig
from src.engine.trainer import build_ts_jepa_model
from src.models import (
    KinematicAdaptiveLatentJEPA,
    KinematicAdaptiveLatentJEPAModel,
    LatentDeliberationJEPA,
    LatentDeliberationJEPAModel,
    MultiAgentConsensusLatentJEPA,
    MultiAgentConsensusLatentJEPAModel,
    SpectralPhysicsLatentJEPA,
    SpectralPhysicsLatentJEPAModel,
)
from src.models.registry import canonical_model_type, is_jepa_model, model_spec


@pytest.fixture
def dummy_batch():
    """Generates synthetic context and target windows."""
    torch.manual_seed(42)
    B, C_len, H_len, C = 8, 32, 16, 4
    ctx = torch.randn(B, C_len, C)
    tgt = torch.randn(B, H_len, C)
    return ctx, tgt


class TestKinematicAdaptiveLatentJEPA:
    """Test suite for KinematicAdaptiveLatentJEPA."""

    def test_forward_shapes_and_backward(self, dummy_batch):
        ctx, tgt = dummy_batch
        B, C_len, C = ctx.shape
        H_len = tgt.shape[1]
        latent_dim = 24

        model = KinematicAdaptiveLatentJEPAModel(
            input_dim=C,
            latent_dim=latent_dim,
            hidden_dim=48,
            dropout=0.0,
        )

        # Forward
        z_ctx, z_tgt, z_ctx_pred, z_sus_pred, sig_sus = model(ctx, tgt)
        assert z_ctx.shape == (B, C_len, latent_dim)
        assert z_tgt.shape == (B, H_len, latent_dim)
        assert z_ctx_pred.shape == (B, C_len, latent_dim)
        assert z_sus_pred.shape == (B, H_len, latent_dim)
        assert sig_sus.shape == (B, H_len, latent_dim)

        # Objective & Backward
        loss, metrics = model.compute_objective(ctx, tgt)
        assert torch.isfinite(loss)
        assert "mean_dispersion" in metrics
        assert "loss_rollout" in metrics

        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and not name.startswith("target_encoder"):
                assert param.grad is not None, f"Missing grad for {name}"

        # Anomaly scoring
        scores = model.compute_predictive_discrepancy(ctx, tgt)
        assert scores.shape == (B,)
        assert torch.all(scores > 0)

        # Covariance fitting
        model.fit_mahalanobis_covariance(ctx, tgt)
        assert model.precision_fitted.item() is True


class TestLatentDeliberationJEPA:
    """Test suite for LatentDeliberationJEPA."""

    def test_thought_steps_and_turbulence(self, dummy_batch):
        ctx, tgt = dummy_batch
        B, C_len, C = ctx.shape
        H_len = tgt.shape[1]
        latent_dim = 32

        model = LatentDeliberationJEPAModel(
            input_dim=C,
            latent_dim=latent_dim,
            hidden_dim=48,
            num_thought_steps=3,
            num_heads=4,
            dropout=0.0,
        )

        z_ctx, z_tgt, z_ctx_pred, z_sus_pred, turb = model(ctx, tgt)
        assert z_ctx.shape == (B, C_len, latent_dim)
        assert z_tgt.shape == (B, H_len, latent_dim)
        assert z_sus_pred.shape == (B, H_len, latent_dim)
        assert turb.shape == (B,)

        loss, metrics = model.compute_objective(ctx, tgt)
        assert torch.isfinite(loss)
        assert "loss_turb" in metrics

        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and not name.startswith("target_encoder"):
                assert param.grad is not None, f"Missing grad for {name}"

        scores = model.compute_predictive_discrepancy(ctx, tgt)
        assert scores.shape == (B,)
        assert torch.all(scores > 0)

        model.fit_mahalanobis_covariance(ctx, tgt)
        assert model.precision_fitted.item() is True


class TestSpectralPhysicsLatentJEPA:
    """Test suite for SpectralPhysicsLatentJEPA."""

    def test_spectral_psd_and_oscillators(self, dummy_batch):
        ctx, tgt = dummy_batch
        B, C_len, C = ctx.shape
        H_len = tgt.shape[1]
        latent_dim = 32  # divisible by 2

        model = SpectralPhysicsLatentJEPAModel(
            input_dim=C,
            latent_dim=latent_dim,
            hidden_dim=48,
            dropout=0.0,
        )

        z_ctx, z_tgt, z_ctx_pred, z_sus_pred = model(ctx, tgt)
        assert z_ctx.shape == (B, C_len, latent_dim)
        assert z_tgt.shape == (B, H_len, latent_dim)
        assert z_sus_pred.shape == (B, H_len, latent_dim)

        loss, metrics = model.compute_objective(ctx, tgt)
        assert torch.isfinite(loss)
        assert "loss_spectral" in metrics
        assert metrics["loss_spectral"] >= 0.0

        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and not name.startswith("target_encoder"):
                assert param.grad is not None, f"Missing grad for {name}"

        scores = model.compute_predictive_discrepancy(ctx, tgt)
        assert scores.shape == (B,)
        assert torch.all(scores > 0)

        model.fit_mahalanobis_covariance(ctx, tgt)
        assert model.precision_fitted.item() is True


class TestMultiAgentConsensusLatentJEPA:
    """Test suite for MultiAgentConsensusLatentJEPA."""

    def test_multiagent_communication_and_consensus(self, dummy_batch):
        ctx, tgt = dummy_batch
        B, C_len, C = ctx.shape
        H_len = tgt.shape[1]
        d_agent = 16
        expected_dim = C * d_agent

        model = MultiAgentConsensusLatentJEPAModel(
            input_dim=C,
            d_agent=d_agent,
            hidden_dim=32,
            num_heads=2,
            dropout=0.0,
        )

        z_ctx, z_tgt, z_ctx_pred, z_sus_pred, dissonance = model(ctx, tgt)
        assert z_ctx.shape == (B, C_len, expected_dim)
        assert z_tgt.shape == (B, H_len, expected_dim)
        assert z_sus_pred.shape == (B, H_len, expected_dim)
        assert dissonance.shape == (B,)

        loss, metrics = model.compute_objective(ctx, tgt)
        assert torch.isfinite(loss)
        assert "loss_consensus" in metrics

        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and not name.startswith("target_encoder"):
                assert param.grad is not None, f"Missing grad for {name}"

        scores = model.compute_predictive_discrepancy(ctx, tgt)
        assert scores.shape == (B,)
        assert torch.all(scores > 0)

        model.fit_mahalanobis_covariance(ctx, tgt)
        assert model.precision_fitted.item() is True


class TestRegistryAndBuilderIntegration:
    """Test registry recognition and builder construction for all 4 models."""

    @pytest.mark.parametrize(
        "model_key,expected_canonical",
        [
            ("kinematic_adaptive_latent_jepa", "kinematic_adaptive_latent_jepa"),
            ("kinematic_latent_jepa", "kinematic_adaptive_latent_jepa"),
            ("ahead_jepa", "kinematic_adaptive_latent_jepa"),
            ("latent_deliberation_jepa", "latent_deliberation_jepa"),
            ("deliberation_jepa", "latent_deliberation_jepa"),
            ("monet_jepa", "latent_deliberation_jepa"),
            ("spectral_physics_latent_jepa", "spectral_physics_latent_jepa"),
            ("spectral_latent_jepa", "spectral_physics_latent_jepa"),
            ("multiagent_consensus_latent_jepa", "multiagent_consensus_latent_jepa"),
            ("interlat_jepa", "multiagent_consensus_latent_jepa"),
        ],
    )
    def test_registry_resolution(self, model_key, expected_canonical):
        canon = canonical_model_type(model_key)
        assert canon == expected_canonical
        assert is_jepa_model(model_key) is True
        spec = model_spec(model_key)
        assert spec is not None
        assert spec.canonical_name == expected_canonical

    @pytest.mark.parametrize(
        "model_name",
        [
            "kinematic_adaptive_latent_jepa",
            "latent_deliberation_jepa",
            "spectral_physics_latent_jepa",
            "multiagent_consensus_latent_jepa",
        ],
    )
    def test_build_ts_jepa_model(self, model_name):
        cfg = CSMConfig(model_type=model_name, latent_dim=32, dropout=0.05)
        model = build_ts_jepa_model(cfg, input_dim=5, device=torch.device("cpu"))
        assert model is not None
        assert hasattr(model, "compute_objective")
        assert hasattr(model, "compute_predictive_discrepancy")
        assert hasattr(model, "fit_mahalanobis_covariance")
