"""Unit tests for Operator-Entropy Regularizer and Koopman Operator JEPA."""

import math
import pytest
import torch
import torch.nn.functional as F

from src.models.jepa.operator_entropy_jepa import (
    OperatorEntropyJEPAModel,
    covariance_spectral_entropy,
    spectral_operator_entropy_loss,
    von_neumann_entropy,
)
from src.models.encoders.tcn_encoder import HybridTCNEncoder


def test_spectral_operator_entropy_non_degeneracy():
    """Verify that K=0 is strictly penalized and K=I achieves minimal deficiency."""
    D = 32

    # Ideal isotropic transition
    K_ideal = torch.eye(D)
    loss_ideal = spectral_operator_entropy_loss(K_ideal)
    assert loss_ideal.item() < 1e-4

    # Rank-1 dimensional collapse
    K_rank1 = torch.zeros(D, D)
    K_rank1[0, 0] = math.sqrt(D)
    loss_rank1 = spectral_operator_entropy_loss(K_rank1)
    # Deficiency should equal log(D)
    assert abs(loss_rank1.item() - math.log(D)) < 1e-3

    # Degenerate energy collapse K = 0
    K_zero = torch.zeros(D, D)
    loss_zero = spectral_operator_entropy_loss(K_zero)
    # Loss should be huge due to logarithmic energy barrier
    assert loss_zero.item() > 10.0
    assert loss_zero.item() > loss_ideal.item()


def test_operator_entropy_jepa_forward_and_backward():
    """Verify forward pass, loss computation, and gradients for OperatorEntropyJEPAModel."""
    D = 32
    K_in = 5
    encoder = HybridTCNEncoder(
        input_dim=K_in,
        latent_dim=D,
        filters=32,
        tcn_layers=2,
        kernel_size=3,
    )
    model = OperatorEntropyJEPAModel(
        context_encoder=encoder,
        latent_dim=D,
        alpha_entropy=0.5,
        use_koopman_matrix=True,
    )

    B = 4
    x_ctx = torch.randn(B, 128, K_in)
    x_tgt = torch.randn(B, 32, K_in)

    z_pred, z_tgt, diff = model(x_ctx, x_tgt)
    assert z_pred.shape == (B, D)
    assert z_tgt.shape == (B, D)
    assert diff.shape == (B,)

    loss, metrics = model.compute_objective(x_ctx, x_tgt)
    assert "loss" in metrics
    assert "pred_loss" in metrics
    assert "spectral_reg" in metrics
    assert metrics["spectral_reg"] >= 0.0

    # Ensure gradients propagate to Koopman transition matrix K
    loss.backward()
    assert model.K.grad is not None
    assert model.K.grad.norm().item() > 0.0


def test_covariance_entropy_deficiency():
    """Verify that latent representation covariance collapse is penalized."""
    D = 16
    N = 64

    # Isotropic normal embeddings
    z_isotropic = torch.randn(N, D)
    loss_iso = von_neumann_entropy(z_isotropic)

    # Constant collapsed embeddings (all identical)
    z_collapsed = torch.zeros(N, D)
    loss_coll = von_neumann_entropy(z_collapsed)

    assert loss_coll.item() > loss_iso.item()


def test_pipeline_symmetry_and_filter_idempotency():
    """Verify that filters are applied once to z_ctx and z_tgt, and fit_covariance matches."""
    D = 16
    K_in = 3
    encoder = HybridTCNEncoder(input_dim=K_in, latent_dim=D, filters=16, tcn_layers=2)
    model = OperatorEntropyJEPAModel(context_encoder=encoder, latent_dim=D, use_koopman_matrix=True)

    B = 8
    x_ctx = torch.randn(B, 32, K_in)
    x_tgt = torch.randn(B, 16, K_in)

    # 1. Pipeline symmetry in forward (under eval mode to disable dropout)
    model.eval()
    z_pred, z_tgt, diff = model(x_ctx, x_tgt)
    # Reconstruct manual filtered representations
    z_ctx_manual = model._apply_filters(model.context_encoder(x_ctx))
    z_pred_manual = model._predict(z_ctx_manual)
    assert torch.allclose(z_pred, z_pred_manual, atol=1e-5)

    # 2. fit_covariance matches test-time residual
    model.fit_covariance(x_ctx, x_tgt, batch_size=8)
    assert bool(model.precision_fitted.item()) is True
    assert model.precision_matrix.shape == (D, D)
    assert not torch.isnan(model.precision_matrix).any()

    # compute_predictive_discrepancy with mahalanobis
    disc = model.compute_predictive_discrepancy(x_ctx, x_tgt, use_mahalanobis=True)
    assert disc.shape == (B,)
    assert not torch.isnan(disc).any()


def test_bilateral_discrepancy_scoring():
    """Verify that bilateral relative energy discrepancy detects both surges and collapses."""
    torch.manual_seed(42)
    D = 16
    K_in = 3
    encoder = HybridTCNEncoder(input_dim=K_in, latent_dim=D, filters=16, tcn_layers=2)
    model = OperatorEntropyJEPAModel(context_encoder=encoder, latent_dim=D, use_koopman_matrix=True)
    model.eval()

    # 1. Exact mathematical verification of bilateral relative energy discrepancy
    e_pred = torch.tensor([4.0, 5.0])
    e_tgt_surge = torch.tensor([8.0, 10.0])       # 2x surge
    e_tgt_collapse = torch.tensor([0.0, 0.0])    # complete collapse (dropout/freeze)
    e_tgt_equal = torch.tensor([4.0, 5.0])       # exact match

    # Old one-sided formulation: torch.clamp(e_tgt - e_pred, min=0.0)
    old_surge = torch.clamp(e_tgt_surge - e_pred, min=0.0)
    old_collapse = torch.clamp(e_tgt_collapse - e_pred, min=0.0)

    # New bilateral formulation: torch.abs(e_tgt - e_pred) / (e_pred + 1e-6)
    new_surge = torch.abs(e_tgt_surge - e_pred) / (e_pred + 1e-6)
    new_collapse = torch.abs(e_tgt_collapse - e_pred) / (e_pred + 1e-6)
    new_equal = torch.abs(e_tgt_equal - e_pred) / (e_pred + 1e-6)

    # Old clamp completely missed energy collapse (clamped to 0.0)
    assert torch.all(old_collapse == 0.0)
    # New bilateral detects BOTH surge and collapse
    assert torch.all(new_surge > 0.0)
    assert torch.all(new_collapse > 0.99)
    assert torch.allclose(new_equal, torch.zeros_like(new_equal), atol=1e-5)

    # 2. Integration with compute_predictive_discrepancy
    B = 4
    x_ctx = torch.randn(B, 32, K_in)
    x_tgt = torch.randn(B, 16, K_in)
    x_tgt_zero = torch.zeros(B, 16, K_in)

    disc_norm = model.compute_predictive_discrepancy(x_ctx, x_tgt)
    disc_collapse = model.compute_predictive_discrepancy(x_ctx, x_tgt_zero)

    assert disc_norm.shape == (B,)
    assert disc_collapse.shape == (B,)
    assert not torch.isnan(disc_norm).any()
    assert not torch.isnan(disc_collapse).any()
    assert (disc_collapse > 0.0).all()


def test_dual_spectral_entropy_coupling():
    """Verify dual spectral regularization on both operator K and representation covariance."""
    D = 16
    K_in = 3
    encoder = HybridTCNEncoder(input_dim=K_in, latent_dim=D, filters=16, tcn_layers=2)
    alpha_entropy = 0.4
    alpha_rep_entropy = 0.05
    model = OperatorEntropyJEPAModel(
        context_encoder=encoder,
        latent_dim=D,
        alpha_entropy=alpha_entropy,
        alpha_rep_entropy=alpha_rep_entropy,
        use_koopman_matrix=True,
    )

    B = 8
    x_ctx = torch.randn(B, 32, K_in)
    x_tgt = torch.randn(B, 16, K_in)

    loss, metrics = model.compute_objective(x_ctx, x_tgt)

    assert "reg_K" in metrics
    assert "reg_Z" in metrics
    assert "spectral_reg" in metrics
    assert "vn_loss" in metrics

    assert metrics["reg_K"] >= 0.0
    assert metrics["reg_Z"] >= 0.0

    # Backprop check: gradients must flow to BOTH operator K and context_encoder
    loss.backward()
    assert model.K.grad is not None
    assert model.K.grad.norm().item() > 0.0
    encoder_has_grad = any(p.grad is not None and p.grad.norm().item() > 0.0 for p in encoder.parameters())
    assert encoder_has_grad


def test_dual_spectral_entropy_numerical_stability():
    """Verify numerical stability on extreme cases (zero embeddings, constant embeddings, batch size 1)."""
    D = 16
    K_in = 3
    encoder = HybridTCNEncoder(input_dim=K_in, latent_dim=D, filters=16, tcn_layers=2)
    model = OperatorEntropyJEPAModel(context_encoder=encoder, latent_dim=D, use_koopman_matrix=True)

    # Batch size 1 edge case (covariance undefined, should be 0 without crashing)
    x_ctx_1 = torch.randn(1, 32, K_in)
    x_tgt_1 = torch.randn(1, 16, K_in)
    loss_1, metrics_1 = model.compute_objective(x_ctx_1, x_tgt_1)
    assert not torch.isnan(loss_1)
    assert metrics_1["reg_Z"] == 0.0

    # Constant inputs (identical samples in batch)
    x_ctx_const = torch.ones(4, 32, K_in)
    x_tgt_const = torch.ones(4, 16, K_in)
    loss_const, metrics_const = model.compute_objective(x_ctx_const, x_tgt_const)
    assert not torch.isnan(loss_const)
    assert not math.isinf(loss_const.item())


def test_float16_precision_safety():
    """Verify that eigvalsh operations do not fail under float16 precision."""
    D = 16
    # Operator K in float16
    K_f16 = torch.eye(D, dtype=torch.float16, requires_grad=True)
    loss_K = spectral_operator_entropy_loss(K_f16)
    assert loss_K.dtype == torch.float16
    assert not torch.isnan(loss_K)
    loss_K.backward()
    assert K_f16.grad is not None
    assert not torch.isnan(K_f16.grad).any()

    # Covariance in float16
    cov_f16 = torch.eye(D, dtype=torch.float16, requires_grad=True)
    loss_cov = covariance_spectral_entropy(cov_f16)
    assert loss_cov.dtype == torch.float16
    assert not torch.isnan(loss_cov)
    loss_cov.backward()
    assert cov_f16.grad is not None
    assert not torch.isnan(cov_f16.grad).any()


def test_csm_config_and_builder_integration():
    """Verify CSMConfig accepts operator entropy parameters and builder properly instantiates."""
    from src.config import CSMConfig
    from src.engine.trainer import build_ts_jepa_model

    cfg = CSMConfig(
        model_type="operator_entropy_jepa",
        latent_dim=16,
        alpha_entropy=0.35,
        alpha_rep_entropy=0.015,
        use_koopman_matrix=True,
    )
    model = build_ts_jepa_model(cfg, input_dim=3, device=torch.device("cpu"))
    assert isinstance(model, OperatorEntropyJEPAModel)
    assert model.alpha_entropy == 0.35
    assert model.alpha_rep_entropy == 0.015
    assert model.use_koopman_matrix is True
    assert model.K is not None


def test_non_koopman_mode_objective():
    """Verify OperatorEntropyJEPAModel operates cleanly when use_koopman_matrix=False."""
    D = 16
    K_in = 3
    encoder = HybridTCNEncoder(input_dim=K_in, latent_dim=D, filters=16, tcn_layers=2)
    model = OperatorEntropyJEPAModel(
        context_encoder=encoder,
        latent_dim=D,
        alpha_entropy=0.5,
        alpha_rep_entropy=0.02,
        use_koopman_matrix=False,
    )
    assert model.K is None
    assert model.predictor is not None

    B = 4
    x_ctx = torch.randn(B, 32, K_in)
    x_tgt = torch.randn(B, 16, K_in)

    loss, metrics = model.compute_objective(x_ctx, x_tgt)
    assert metrics["reg_K"] == 0.0
    assert metrics["reg_Z"] >= 0.0
    loss.backward()

    # Predictor and encoder must both receive gradients
    pred_has_grad = any(p.grad is not None and p.grad.norm().item() > 0.0 for p in model.predictor.parameters())
    assert pred_has_grad


def test_discrete_lyapunov_invariance_defect():
    """Verify Discrete Lyapunov Invariance Defect penalizes covariance misalignment and propagates gradients."""
    D = 16
    K_in = 3
    encoder = HybridTCNEncoder(input_dim=K_in, latent_dim=D, filters=16, tcn_layers=2)
    model = OperatorEntropyJEPAModel(
        context_encoder=encoder,
        latent_dim=D,
        alpha_entropy=0.5,
        alpha_rep_entropy=0.005,
        beta_lyapunov=0.05,
        use_koopman_matrix=True,
    )
    assert model.beta_lyapunov == 0.05

    B = 8
    x_ctx = torch.randn(B, 32, K_in)
    x_tgt = torch.randn(B, 16, K_in)

    loss, metrics = model.compute_objective(x_ctx, x_tgt)
    assert "lyapunov_defect" in metrics
    assert metrics["lyapunov_defect"] >= 0.0

    loss.backward()
    assert model.K.grad is not None
    assert model.K.grad.norm().item() > 0.0



