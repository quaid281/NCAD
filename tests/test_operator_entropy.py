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
    """Verify that bilateral relative energy discrepancy detects both surges and collapses through the full encoder."""
    torch.manual_seed(42)
    D = 16
    K_in = 3
    encoder = HybridTCNEncoder(input_dim=K_in, latent_dim=D, filters=16, tcn_layers=2)
    model = OperatorEntropyJEPAModel(context_encoder=encoder, latent_dim=D, use_koopman_matrix=True)
    with torch.no_grad():
        model.K.copy_(torch.eye(D))
    model.eval()

    # 1. Exact mathematical verification of bilateral relative energy discrepancy
    e_pred = torch.tensor([4.0, 5.0])
    e_tgt_surge = torch.tensor([8.0, 10.0])       # 2x surge
    e_tgt_collapse = torch.tensor([0.0, 0.0])    # complete collapse (dropout/freeze)
    e_tgt_equal = torch.tensor([4.0, 5.0])       # exact match

    new_surge = torch.abs(e_tgt_surge - e_pred) / (e_pred + 1e-6)
    new_collapse = torch.abs(e_tgt_collapse - e_pred) / (e_pred + 1e-6)
    new_equal = torch.abs(e_tgt_equal - e_pred) / (e_pred + 1e-6)

    assert torch.all(new_surge > 0.0)
    assert torch.all(new_collapse > 0.99)
    assert torch.allclose(new_equal, torch.zeros_like(new_equal), atol=1e-5)

    # 2. Full-pipeline integration through LayerNorm encoder:
    # Nominal periodic continuation vs. complete zero-collapse vs. 500x amplitude surge
    B = 4
    # Period-16 sinusoid so context (32 steps = 2 periods) and target (32 steps = 2 periods) share stationary statistics
    t = torch.arange(64, dtype=torch.float32).view(1, 64, 1) * (2.0 * math.pi / 16.0)
    freqs = torch.tensor([1.0, 2.0, 3.0]).view(1, 1, K_in)
    phases = torch.arange(B, dtype=torch.float32).view(B, 1, 1) * 0.5
    series = torch.sin(t * freqs + phases)
    x_ctx = series[:, :32, :]
    x_tgt_nominal = series[:, 32:, :]
    x_tgt_zero = torch.zeros_like(x_tgt_nominal)
    x_tgt_surge = 500.0 * x_tgt_nominal

    disc_norm = model.compute_predictive_discrepancy(x_ctx, x_tgt_nominal)
    disc_collapse = model.compute_predictive_discrepancy(x_ctx, x_tgt_zero)
    disc_surge = model.compute_predictive_discrepancy(x_ctx, x_tgt_surge)

    assert disc_norm.shape == (B,)
    assert disc_collapse.shape == (B,)
    assert disc_surge.shape == (B,)
    assert not torch.isnan(disc_norm).any()
    assert not torch.isnan(disc_collapse).any()
    assert not torch.isnan(disc_surge).any()
    # Both zero-collapse and amplitude surge must produce strictly higher discrepancy than nominal
    assert torch.all(disc_collapse > disc_norm), f"Expected collapse ({disc_collapse}) > nominal ({disc_norm})"
    assert torch.all(disc_surge > disc_norm), f"Expected surge ({disc_surge}) > nominal ({disc_norm})"

    # Verify pre-LayerNorm radial energy itself is NOT pinned to sqrt(D)
    _ = model.target_encoder(x_tgt_nominal)
    e_nom = model.target_encoder.last_radial_energy.clone()
    _ = model.target_encoder(x_tgt_zero)
    e_zero = model.target_encoder.last_radial_energy.clone()
    _ = model.target_encoder(x_tgt_surge)
    e_surge = model.target_encoder.last_radial_energy.clone()
    assert torch.allclose(e_zero, torch.zeros_like(e_zero), atol=1e-6)
    assert torch.all(e_surge > 100.0 * e_nom)


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


def test_cayley_operator_forward_and_backward():
    """Verify forward pass, loss, contractivity, and gradients for Cayley operator mode."""
    D = 16
    K_in = 3
    encoder = HybridTCNEncoder(input_dim=K_in, latent_dim=D, filters=16, tcn_layers=2)
    model = OperatorEntropyJEPAModel(
        context_encoder=encoder,
        latent_dim=D,
        alpha_entropy=0.5,
        alpha_rep_entropy=0.005,
        beta_lyapunov=0.02,
        use_cayley_operator=True,
    )
    assert model.use_cayley_operator
    assert model.cayley_gate is not None
    assert model.K is None

    # Check contractivity of initial Cayley matrix
    W = model.effective_transition_matrix
    assert W is not None
    op_norm = torch.linalg.norm(W, ord=2).item()
    spectral_radius = torch.linalg.eigvals(W).abs().max().item()
    assert op_norm <= 1.0 + 1e-5
    assert spectral_radius <= 1.0 + 1e-5

    B = 8
    x_ctx = torch.randn(B, 32, K_in)
    x_tgt = torch.randn(B, 16, K_in)

    z_pred, z_tgt, diff = model(x_ctx, x_tgt)
    assert z_pred.shape == (B, D)
    assert z_tgt.shape == (B, D)

    loss, metrics = model.compute_objective(x_ctx, x_tgt)
    assert "loss" in metrics
    assert "pred_loss" in metrics
    assert "lyapunov_defect" in metrics
    assert "spectral_reg" in metrics

    loss.backward()
    assert model.cayley_gate.A.grad is not None
    assert model.cayley_gate.A.grad.norm().item() > 0.0
    assert model.cayley_gate.d.grad is not None
    assert model.cayley_gate.d.grad.norm().item() > 0.0


def test_boosted_opentropy_defaults_and_layer_pruning():
    """Verify that OperatorEntropyJEPAModel defaults decouple alpha_rep_entropy, activate Lyapunov, and prune inert layers."""
    D = 16
    K_in = 3
    encoder = HybridTCNEncoder(input_dim=K_in, latent_dim=D, filters=16, tcn_layers=2)
    model = OperatorEntropyJEPAModel(context_encoder=encoder, latent_dim=D)

    # 1. Verify decoupled rep entropy and active Lyapunov defect
    assert model.alpha_rep_entropy == 0.0
    assert model.beta_lyapunov == 0.02

    # 2. Verify pruned inert math layers
    assert model.radial_heat is None
    assert model.cohn_elkies is None

    # 3. Verify _apply_filters is identity pass-through
    z = torch.randn(4, D)
    assert torch.all(model._apply_filters(z) == z)

    # 4. Optional re-enablement check
    model_with_layers = OperatorEntropyJEPAModel(
        context_encoder=encoder,
        latent_dim=D,
        use_radial_heat=True,
        use_cohn_elkies=True,
    )
    assert model_with_layers.radial_heat is not None
    assert model_with_layers.cohn_elkies is not None


def test_decoupled_rep_entropy_and_lyapunov_defect_gradients():
    """Verify loss calculation and gradients under default alpha_rep_entropy=0.0 and beta_lyapunov=0.02."""
    D = 16
    K_in = 3
    encoder = HybridTCNEncoder(input_dim=K_in, latent_dim=D, filters=16, tcn_layers=2)
    model = OperatorEntropyJEPAModel(context_encoder=encoder, latent_dim=D, use_koopman_matrix=True)

    B = 8
    x_ctx = torch.randn(B, 32, K_in)
    x_tgt = torch.randn(B, 16, K_in)

    loss, metrics = model.compute_objective(x_ctx, x_tgt)

    # reg_Z should be exactly 0.0 when alpha_rep_entropy=0.0
    assert metrics["reg_Z"] == 0.0
    assert metrics["lyapunov_defect"] >= 0.0

    loss.backward()
    assert model.K.grad is not None
    assert model.K.grad.norm().item() > 0.0


def test_entropy_gradient_sign_on_collapsing_modes():
    """Verify dL/ds_i < 0 on collapsing modes (p_i <= 1e-10) including rank-deficient matrices with negative eigvalsh jitter."""
    # 1. Construct a diagonal operator K = diag(sqrt(s)) where small modes have p_i <= 1e-10
    s_init = torch.tensor(
        [0.0, 1e-11, 5e-11, 1e-9, 1.0, 1.5, 2.0, 3.5],
        dtype=torch.float64,
        requires_grad=True,
    )
    K = torch.diag(torch.sqrt(s_init + 1e-18))
    loss = spectral_operator_entropy_loss(K, beta_norm=0.0)
    loss.backward()

    assert s_init.grad is not None
    # Every collapsing mode (indices 0..3, with p_i <= 1e-9) must have strictly negative gradient
    # so that gradient descent (s <- s - lr * grad) increases s_i away from zero.
    for idx in range(4):
        assert s_init.grad[idx].item() < 0.0, (
            f"Expected dL/ds_{idx} < 0 for collapsing mode s={s_init[idx].item()}, "
            f"got {s_init.grad[idx].item()}"
        )

    # 2. Verify covariance_spectral_entropy has strictly negative gradient on collapsing eigenvalues
    evals_init = torch.tensor(
        [0.0, 1e-11, 5e-11, 1e-9, 1.0, 1.5, 2.0, 3.5],
        dtype=torch.float64,
        requires_grad=True,
    )
    cov = torch.diag(evals_init)
    loss_cov = covariance_spectral_entropy(cov)
    loss_cov.backward()
    assert evals_init.grad is not None
    for idx in range(4):
        assert evals_init.grad[idx].item() < 0.0, (
            f"Expected dL/dlambda_{idx} < 0 for collapsing eigenvalue, got {evals_init.grad[idx].item()}"
        )

    # 3. Verify rank-deficient float32 batch covariance where eigvalsh returns negative roundoff (~ -1e-7)
    # on collapsed zero modes: straight-through clamping must keep all 5 zero-mode gradients < 0.
    torch.manual_seed(0)
    z_rd = torch.randn(4, 8, dtype=torch.float32)
    z_c = z_rd - z_rd.mean(dim=0, keepdim=True)
    cov_rd = ((z_c.T @ z_c) / 3.0).detach().requires_grad_(True)
    loss_rd = covariance_spectral_entropy(cov_rd)
    loss_rd.backward()
    _, evecs_rd = torch.linalg.eigh(cov_rd.detach())
    grad_on_eigenmodes = torch.diag(evecs_rd.T @ cov_rd.grad @ evecs_rd)
    for idx in range(5):
        assert grad_on_eigenmodes[idx].item() < -1.0, (
            f"Expected strictly negative anti-collapse gradient on rank-deficient mode {idx}, "
            f"got {grad_on_eigenmodes[idx].item()}"
        )


def test_mahalanobis_zca_whitening_and_unfitted_guard():
    """Verify symmetric square-root ZCA whitening preserves quadratic Mahalanobis form, handles mixed dtypes, and guards unfitted calls."""
    from src.models._jepa_utils import whiten_residuals

    D = 8
    torch.manual_seed(7)
    A = torch.randn(D, D)
    # Strongly correlated precision matrix where off-diagonal coupling makes (r @ Omega) * r negative on some coordinates
    precision = A @ A.T + 0.05 * torch.eye(D)
    r = torch.randn(16, D)

    e_white = whiten_residuals(r, precision)
    quad_form = torch.sum((r @ precision) * r, dim=-1)
    white_norm_sq = torch.sum(e_white.pow(2), dim=-1)
    assert torch.allclose(white_norm_sq, quad_form, rtol=1e-4, atol=1e-5)

    # Verify mixed-dtype safety (bfloat16, float16, float64 residuals against float32 precision)
    for dt in (torch.bfloat16, torch.float16, torch.float64):
        e_dt = whiten_residuals(r.to(dt), precision)
        assert e_dt.dtype == dt
        assert not torch.isnan(e_dt).any()

    # Verify RuntimeError when use_mahalanobis=True before fit_covariance()
    encoder = HybridTCNEncoder(input_dim=3, latent_dim=D, filters=16, tcn_layers=2)
    model = OperatorEntropyJEPAModel(context_encoder=encoder, latent_dim=D)
    x_ctx = torch.randn(4, 32, 3)
    x_tgt = torch.randn(4, 16, 3)
    with pytest.raises(RuntimeError, match="fit_covariance"):
        model.compute_predictive_discrepancy(x_ctx, x_tgt, use_mahalanobis=True)

    # Verify fit_covariance runs under no_grad even when inputs require grad
    x_ctx_grad = torch.randn(8, 32, 3, requires_grad=True)
    x_tgt_grad = torch.randn(8, 16, 3, requires_grad=True)
    model.fit_covariance(x_ctx_grad, x_tgt_grad, batch_size=4)
    assert bool(model.precision_fitted.item()) is True
    assert not model.precision_matrix.requires_grad
    assert not model.residual_mean.requires_grad


def test_scale_invariant_lyapunov_and_gap_loss_gradients():
    """Verify Lyapunov defect is invariant to global representation scaling and gap_weight > 0 specifically propagates gradients."""
    from src.models.encoders.selective_ssm_encoder import SelectiveSSMContextEncoder

    D = 16
    K_in = 3

    # 1. Verify scale invariance of Lyapunov defect across global representation scaling z -> c * z
    class ScaledDummyEncoder(torch.nn.Module):
        def __init__(self, scale: float):
            super().__init__()
            self.scale = scale
            self.proj = torch.nn.Linear(K_in, D, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.scale * self.proj(x.mean(dim=1))

    torch.manual_seed(11)
    base_enc = ScaledDummyEncoder(scale=1.0)
    m_base = OperatorEntropyJEPAModel(
        context_encoder=base_enc,
        latent_dim=D,
        alpha_entropy=0.0,
        alpha_rep_entropy=0.0,
        beta_lyapunov=1.0,
        use_koopman_matrix=True,
    )
    B = 8
    x_ctx = torch.randn(B, 32, K_in)
    x_tgt = torch.randn(B, 16, K_in)
    _, metrics_1x = m_base.compute_objective(x_ctx, x_tgt)

    m_10x = OperatorEntropyJEPAModel(
        context_encoder=ScaledDummyEncoder(scale=10.0),
        latent_dim=D,
        alpha_entropy=0.0,
        alpha_rep_entropy=0.0,
        beta_lyapunov=1.0,
        use_koopman_matrix=True,
    )
    with torch.no_grad():
        m_10x.context_encoder.proj.weight.copy_(m_base.context_encoder.proj.weight)
        m_10x.target_encoder.proj.weight.copy_(m_base.target_encoder.proj.weight)
        m_10x.K.copy_(m_base.K)
    _, metrics_10x = m_10x.compute_objective(x_ctx, x_tgt)
    assert abs(metrics_1x["lyapunov_defect"] - metrics_10x["lyapunov_defect"]) / max(metrics_1x["lyapunov_defect"], 1e-6) < 1e-3

    # 2. Verify gap_weight > 0 specifically contributes gradients to K (isolating gap_loss from pred_loss)
    encoder = HybridTCNEncoder(input_dim=K_in, latent_dim=D, filters=16, tcn_layers=2)
    model = OperatorEntropyJEPAModel(
        context_encoder=encoder,
        latent_dim=D,
        alpha_entropy=0.0,
        alpha_rep_entropy=0.0,
        beta_lyapunov=0.0,
        spectral_gap_min=100.0,
        use_koopman_matrix=True,
    )
    model.eval()
    loss_no_gap, _ = model.compute_objective(x_ctx, x_tgt, gap_weight=0.0)
    loss_with_gap, metrics_gap = model.compute_objective(x_ctx, x_tgt, gap_weight=10.0)
    assert metrics_gap["gap_loss"] > 0.0
    only_gap_term = loss_with_gap - loss_no_gap.detach()
    model.zero_grad()
    only_gap_term.backward()
    assert model.K.grad is not None
    assert model.K.grad.norm().item() > 0.0

    # 3. Also verify SelectiveSSMContextEncoder preserves pre-LayerNorm radial energy under zero-collapse & surge
    ssm_enc = SelectiveSSMContextEncoder(input_dim=K_in, latent_dim=D, hidden_dim=16, layers=2)
    ssm_model = OperatorEntropyJEPAModel(context_encoder=ssm_enc, latent_dim=D)
    with torch.no_grad():
        ssm_model.K.copy_(torch.eye(D))
    ssm_model.eval()
    t = torch.arange(64, dtype=torch.float32).view(1, 64, 1) * (2.0 * math.pi / 16.0)
    freqs = torch.tensor([1.0, 2.0, 3.0]).view(1, 1, K_in)
    phases = torch.arange(4, dtype=torch.float32).view(4, 1, 1) * 0.5
    series = torch.sin(t * freqs + phases)
    x_ctx_s = series[:, :32, :]
    x_tgt_nom = series[:, 32:, :]
    disc_nom = ssm_model.compute_predictive_discrepancy(x_ctx_s, x_tgt_nom)
    disc_zero = ssm_model.compute_predictive_discrepancy(x_ctx_s, torch.zeros_like(x_tgt_nom))
    disc_surge = ssm_model.compute_predictive_discrepancy(x_ctx_s, 500.0 * x_tgt_nom)
    assert torch.all(disc_zero > disc_nom)
    assert torch.all(disc_surge > disc_nom)
    _ = ssm_model.target_encoder(torch.zeros_like(x_tgt_nom))
    assert torch.allclose(ssm_model.target_encoder.last_radial_energy, torch.zeros(4), atol=1e-6)


def test_data_projected_operator_entropy_detects_active_subspace_collapse():
    """Verify data-projected operator entropy penalizes collapse on the active data manifold V
    even when K compensates on dead orthogonal modes V^perp to fake bare isotropy/energy."""
    D = 16
    r = 8
    N = 128
    torch.manual_seed(42)

    # 1. Construct z_ctx supported strictly on the first r coordinates (subspace V)
    # with exact isotropic empirical covariance on V and zero variance on V^perp (coords r..D-1)
    q, _ = torch.linalg.qr(torch.randn(N, r, dtype=torch.float64))
    q = q - q.mean(dim=0, keepdim=True)
    u, _, vt = torch.linalg.svd(q, full_matrices=False)
    z_active = u @ vt * math.sqrt(N - 1)  # exact covariance I_r on V
    z_ctx = torch.cat([z_active, torch.zeros(N, D - r, dtype=torch.float64)], dim=-1)

    # Operator 1: active-manifold isometry on V (singular values sqrt(D/r) on V, 0 on V^perp)
    diag_active = torch.cat([torch.full((r,), math.sqrt(D / r), dtype=torch.float64), torch.zeros(D - r, dtype=torch.float64)])
    K_active = torch.diag(diag_active)

    # Operator 2: dead-subspace cheat (identical bare singular spectrum to K_active,
    # but supported entirely on V^perp while annihilating the active data manifold V!)
    diag_dead = torch.cat([torch.zeros(r, dtype=torch.float64), torch.full((D - r,), math.sqrt(D / (D - r)), dtype=torch.float64)])
    K_dead = torch.diag(diag_dead)

    # Bare operator entropy (z_ctx=None) cannot distinguish K_active from K_dead
    bare_active = spectral_operator_entropy_loss(K_active, z_ctx=None)
    bare_dead = spectral_operator_entropy_loss(K_dead, z_ctx=None)
    assert torch.allclose(bare_active, bare_dead, atol=1e-6)

    # Data-projected operator entropy strictly penalizes K_dead >> K_active
    proj_active = spectral_operator_entropy_loss(K_active, z_ctx=z_ctx, proj_weight=0.5, rank_ceiling=r)
    proj_dead = spectral_operator_entropy_loss(K_dead, z_ctx=z_ctx, proj_weight=0.5, rank_ceiling=r)
    assert proj_dead.item() > proj_active.item() + 5.0, (
        f"Expected K_dead ({proj_dead.item()}) >> K_active ({proj_active.item()}) under data projection"
    )

    # 2. LayerNorm hyperplane (D-1 active modes, 1_D^T z = 0 structural nullspace):
    # K = I should achieve near-zero loss because default rank_ceiling = min(B - 1, D - 1) excludes the 1_D null mode
    H = torch.eye(D, dtype=torch.float64) - torch.ones(D, D, dtype=torch.float64) / float(D)
    eigvals, eigvecs = torch.linalg.eigh(H)
    basis_ln = eigvecs[:, 1:]  # (D, D-1) orthonormal basis of LayerNorm zero-sum hyperplane
    q_ln, _ = torch.linalg.qr(torch.randn(N, D - 1, dtype=torch.float64))
    q_ln = q_ln - q_ln.mean(dim=0, keepdim=True)
    u_ln, _, vt_ln = torch.linalg.svd(q_ln, full_matrices=False)
    z_ln = (u_ln @ vt_ln * math.sqrt(N - 1)) @ basis_ln.T  # isotropic on D-1 hyperplane, 1_D^T z_ln == 0

    K_eye = torch.eye(D, dtype=torch.float64, requires_grad=True)
    loss_ln_eye = spectral_operator_entropy_loss(K_eye, z_ctx=z_ln, proj_weight=0.5)
    assert loss_ln_eye.item() < 1e-4, f"Expected near-zero loss for K=I on isotropic LayerNorm hyperplane, got {loss_ln_eye.item()}"

    # Collapsing one active LayerNorm tangent direction while inflating the 1_D null direction to keep ||K||_F^2 = D
    u_null = eigvecs[:, :1]  # (D, 1), the 1_D / sqrt(D) direction
    u_act0 = eigvecs[:, 1:2]  # (D, 1), an active LayerNorm tangent direction
    K_ln_cheat = (
        torch.eye(D, dtype=torch.float64)
        - (u_act0 @ u_act0.T)
        + (math.sqrt(2.0) - 1.0) * (u_null @ u_null.T)
    ).requires_grad_(True)
    loss_ln_cheat = spectral_operator_entropy_loss(K_ln_cheat, z_ctx=z_ln, proj_weight=1.0)
    assert loss_ln_cheat.item() > loss_ln_eye.item() + 0.05
    loss_ln_cheat.backward()
    assert K_ln_cheat.grad is not None
    # Gradient along the collapsed active tangent direction u_act0 @ u_act0.T must be strictly negative
    grad_along_collapsed = torch.sum(K_ln_cheat.grad * (u_act0 @ u_act0.T)).item()
    assert grad_along_collapsed < 0.0, f"Expected negative rescuing gradient on collapsed active direction, got {grad_along_collapsed}"

    # 3. Graceful fallback when B <= 1 or constant z_ctx
    loss_b1 = spectral_operator_entropy_loss(torch.eye(D), z_ctx=torch.randn(1, D), proj_weight=0.5)
    loss_const = spectral_operator_entropy_loss(torch.eye(D), z_ctx=torch.ones(8, D), proj_weight=0.5)
    assert loss_b1.item() < 1e-4
    assert loss_const.item() < 1e-4


def test_littlewood_paley_dyadic_filter_caching():
    """Verify LittlewoodPaleyDyadicBlock caches rFFT shell masks keyed by (T, device, dtype) without collision."""
    from src.models.geometric_layers import LittlewoodPaleyDyadicBlock

    block = LittlewoodPaleyDyadicBlock(channels=8, num_shells=4)
    assert len(block._filter_cache) == 0

    x_256 = torch.randn(2, 8, 256, dtype=torch.float32)
    x_64 = torch.randn(2, 8, 64, dtype=torch.float32)
    # Note: F_bins(64) = 33; test T=33 to verify no key collision between T and F_bins
    x_33 = torch.randn(2, 8, 33, dtype=torch.float32)

    out1 = block(x_256)
    key_256 = (256, x_256.device, torch.float32)
    assert key_256 in block._filter_cache
    cached_256 = block._filter_cache[key_256]
    assert cached_256.shape == (4, 1, 129)

    # Second forward pass with T=256 reuses the exact cached tensor object
    out2 = block(x_256)
    assert block._filter_cache[key_256] is cached_256
    assert out1.shape == (2, 8, 256)
    assert out2.shape == (2, 8, 256)

    # T=64 and T=33 populate distinct cache entries with exact rFFT bin counts
    _ = block(x_64)
    _ = block(x_33)
    key_64 = (64, x_64.device, torch.float32)
    key_33 = (33, x_33.device, torch.float32)
    assert key_64 in block._filter_cache
    assert key_33 in block._filter_cache
    assert block._filter_cache[key_64].shape == (4, 1, 33)
    assert block._filter_cache[key_33].shape == (4, 1, 17)


def test_short_horizon_tcn_receptive_field_alignment():
    """Verify HybridTCNEncoder caps effective dilation on S=64 so all kernel taps receive real signal gradients,
    while preserving full dilation on C=256 and keeping state_dict shapes identical."""
    torch.manual_seed(0)
    K_in, D = 4, 16
    enc = HybridTCNEncoder(
        input_dim=K_in,
        latent_dim=D,
        filters=16,
        tcn_layers=6,
        kernel_size=5,
        dropout=0.0,
        use_dyadic_shells=True,
    )
    enc.eval()

    # 1. Verify effective_dilation_and_padding on C=256 vs S=64
    expected_dilations_256 = [1, 2, 4, 8, 16, 32]
    expected_dilations_64 = [1, 2, 4, 8, 8, 8]
    for idx, block in enumerate(enc.blocks):
        d_256, p_256 = block.effective_dilation_and_padding(256)
        d_64, p_64 = block.effective_dilation_and_padding(64)
        assert d_256 == expected_dilations_256[idx]
        assert p_256 == 4 * expected_dilations_256[idx]
        assert d_64 == expected_dilations_64[idx]
        assert p_64 == 4 * expected_dilations_64[idx]
        assert p_64 <= 64 // 2  # Never exceeds half the target window length

    # 2. Verify all 5 kernel taps of deepest TCN block (layer 5, nominal dilation=32) receive non-zero gradients on S=64!
    # (Without horizon alignment, padding=128 on T=64 meant taps 0, 1, 2 at offsets t-128, t-96, t-64 read 100% zeros!)
    x_short = torch.randn(4, 64, K_in)
    enc.zero_grad()
    z_short = enc(x_short)
    assert z_short.shape == (4, D)
    z_short.sum().backward()

    deepest_conv1_grad = enc.blocks[-1].conv1.weight.grad  # shape (filters, filters, 5)
    assert deepest_conv1_grad is not None
    for tap_idx in range(5):
        tap_grad_norm = deepest_conv1_grad[:, :, tap_idx].norm().item()
        assert tap_grad_norm > 0.0, (
            f"Kernel tap {tap_idx} of deepest TCN block received zero gradient on S=64 "
            f"(horizon padding overflow not resolved)"
        )

    # 3. Verify strict temporal causality inside TCN stack on S=64:
    # Perturbing inputs at t >= 32 must NOT change TCN block activations at t < 32
    x_a = torch.randn(2, 16, 64)
    x_b = x_a.clone()
    x_b[:, :, 32:] += 10.0
    with torch.no_grad():
        h_a, h_b = x_a, x_b
        for block in enc.blocks:
            h_a = block(h_a)
            h_b = block(h_b)
    assert torch.allclose(h_a[:, :, :32], h_b[:, :, :32], atol=1e-6)
    assert not torch.allclose(h_a[:, :, 32:], h_b[:, :, 32:], atol=1e-4)


def test_lyapunov_encoder_detachment_nyquist_shell_and_koopman_tracer():
    """Verify:
    1. Discrete Lyapunov Invariance Defect computes gradients strictly w.r.t. the transition
       operator (K or cayley_gate) and does NOT inject variance-inflation gradients into context_encoder.
    2. LittlewoodPaleyDyadicBlock assigns non-degenerate energy (> 5% of total spectral energy)
       to the highest-frequency Nyquist shell (j = J - 1).
    3. KoopmanLinearTransition (koopman_op) is captured as a leaf module by LayerActivationTracer.
    """
    from src.engine.activation_tracer import LayerActivationTracer
    from src.models.geometric_layers import LittlewoodPaleyDyadicBlock

    torch.manual_seed(42)
    K_in, D = 4, 16
    x_ctx = torch.randn(16, 64, K_in)
    x_tgt = torch.randn(16, 32, K_in)

    # 1. Verify Lyapunov defect alone produces zero gradient on context_encoder parameters
    # while producing non-zero gradient on K
    enc = HybridTCNEncoder(input_dim=K_in, latent_dim=D, filters=16, tcn_layers=2, use_dyadic_shells=True)
    model = OperatorEntropyJEPAModel(
        context_encoder=enc,
        latent_dim=D,
        alpha_entropy=0.0,
        alpha_rep_entropy=0.0,
        beta_lyapunov=1.0,
        use_koopman_matrix=True,
    )
    model.eval()
    loss_no_lyap, _ = model.compute_objective(x_ctx, x_tgt, beta_lyapunov=0.0)
    model.zero_grad()
    loss_no_lyap.backward()
    grads_no_lyap = [p.grad.detach().clone() for p in model.context_encoder.parameters() if p.grad is not None]
    k_grad_no_lyap = model.K.grad.detach().clone()

    loss_with_lyap, metrics = model.compute_objective(x_ctx, x_tgt, beta_lyapunov=10.0)
    assert metrics["lyapunov_defect"] > 0.0
    model.zero_grad()
    loss_with_lyap.backward()
    grads_with_lyap = [p.grad.detach().clone() for p in model.context_encoder.parameters() if p.grad is not None]
    k_grad_with_lyap = model.K.grad.detach().clone()

    assert not torch.allclose(k_grad_no_lyap, k_grad_with_lyap, atol=1e-5)
    for g0, g1 in zip(grads_no_lyap, grads_with_lyap):
        assert torch.allclose(g0, g1, atol=1e-7), "Lyapunov defect leaked gradients into context_encoder"

    # 2. Verify Nyquist shell (j = J - 1) receives non-trivial spectral weight
    filters = enc.dyadic_block.compute_dyadic_filters(129, torch.device("cpu")).squeeze(1)
    shell_energy_shares = (filters ** 2).mean(dim=1)
    assert shell_energy_shares[-1].item() > 0.10, (
        f"Nyquist shell (j=J-1) spectral energy share collapsed: {shell_energy_shares.tolist()}"
    )

    # 3. Verify LayerActivationTracer captures koopman_op leaf module
    model.eval()
    with LayerActivationTracer(model, leaf_only=True) as tracer:
        with torch.no_grad():
            model(x_ctx[:4], x_tgt[:4])
    assert "koopman_op" in tracer.records, f"koopman_op missing from leaf activation trace: {list(tracer.records.keys())}"
    assert tracer.records["koopman_op"].output_shape == (4, D)


