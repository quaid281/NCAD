"""Unit tests covering the model & loss audit fixes and behavior validations."""

import io

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.baselines import AnomalyTransformer, TranAD
from src.models.baselines.anomaly_transformer import PositionalEmbedding as AT_PositionalEmbedding
from src.models.baselines.tranad import PositionalEncoding as TranAD_PositionalEncoding
from src.models.encoders.tcn_encoder import HybridTCNEncoder
from src.models.jepa.flow_ts_jepa import FlowTSJEPA
from src.models.jepa.multiscale_ts_jepa import MultiScaleTSJEPA
from src.models.jepa.ncad_jepa import NCADJEPAModel
from src.models.jepa.patch_flow_jepa import PatchFlowJEPA
from src.models.jepa.patch_ts_jepa import PatchSequenceEncoder, PatchTSJEPA
from src.models.jepa.patch_ts_jepa import PositionalEncoding as Patch_PositionalEncoding
from src.models.losses.anomaly_injector import AnomalyInjectionConfig, ContextualAnomalyInjector
from src.models.losses.fei_sigreg import FrequencyMasker, sigreg_loss
from src.models.memory.sindy_scorer import SINDyConfig, SINDyDynamicsScorer


@pytest.fixture
def dummy_batch():
    batch_size = 4
    context_len = 32
    target_len = 16
    n_features = 5
    ctx = torch.randn(batch_size, context_len, n_features)
    tgt = torch.randn(batch_size, target_len, n_features)
    return ctx, tgt


# =========================================================================
# 1. Anomaly Transformer Minimax Direction
# =========================================================================

def test_anomaly_transformer_minimax_direction():
    B, L, C = 4, 32, 3
    x = torch.randn(B, L, C)
    model = AnomalyTransformer(c_in=C, d_model=16, n_heads=2, e_layers=2, d_ff=32, dropout=0.0)
    model.eval()

    rec, series_list, prior_list = model(x)
    rec_loss = torch.mean((rec - x) ** 2)

    series_detached = [s.detach() for s in series_list]
    ass_dis_prior = model.association_discrepancy(prior_list, series_detached).mean()
    loss_prior, loss_series = model.minimax_losses(x, lambda_weight=3.0)

    # Phase 1 minimizes discrepancy (+ lambda * ass_dis)
    expected_prior = rec_loss + 3.0 * ass_dis_prior
    assert torch.allclose(loss_prior, expected_prior, atol=1e-5)

    # Phase 2 maximizes discrepancy (- lambda * ass_dis)
    prior_detached = [p.detach() for p in prior_list]
    ass_dis_series = model.association_discrepancy(prior_detached, series_list).mean()
    expected_series = rec_loss - 3.0 * ass_dis_series
    assert torch.allclose(loss_series, expected_series, atol=1e-5)

    # Both losses can backpropagate through their respective active branches
    loss_prior.backward(retain_graph=True)
    loss_series.backward()


# =========================================================================
# 2. TranAD Loss Boundedness for Epoch > 1
# =========================================================================

def test_tranad_loss_bounded_epoch_greater_than_1():
    B, L, C = 4, 32, 3
    x = torch.randn(B, L, C)
    model = TranAD(c_in=C, d_model=16, n_heads=2, e_layers=1, d_layers=1, d_ff=32)
    rec1, rec2 = model(x)

    mse1 = torch.mean((rec1 - x) ** 2)
    mse2 = torch.mean((rec2 - x) ** 2)
    mse2_vs_rec1 = torch.mean((rec2 - rec1.detach()) ** 2)

    for epoch in [1, 2, 5, 10]:
        n = epoch
        l1, l2 = model.adversarial_loss(rec1, rec2, x, epoch=epoch)
        expected_l1 = (1.0 / n) * mse1
        expected_l2 = (1.0 / n) * mse2 + (1.0 - 1.0 / n) * mse2_vs_rec1
        assert torch.allclose(l1, expected_l1, atol=1e-5)
        assert torch.allclose(l2, expected_l2, atol=1e-5)
        assert l1.item() >= 0.0
        assert l2.item() >= 0.0
        total_loss = l1 + l2
        assert total_loss.item() >= 0.0
        assert torch.isfinite(total_loss)


def test_tranad_adversarial_loss_uses_detached_rec1():
    """Verify that l2 uses rec1.detach() in the consistency term (not rec1 directly).

    We can't check gradient isolation on the full model because rec2 depends on rec1
    through the forward pass (dec2 = embedding(rec1)). Instead, verify the formula
    matches the detached version and that direct rec1 input gets no grad from l2.
    """
    B, L, C = 4, 32, 3
    x = torch.randn(B, L, C)
    rec1 = torch.randn(B, L, C, requires_grad=True)
    rec2 = torch.randn(B, L, C, requires_grad=True)

    _, l2 = TranAD.adversarial_loss(rec1, rec2, x, epoch=5)
    l2.backward()

    # rec1 is a leaf tensor here — it should get NO gradient from l2 because
    # it only appears via rec1.detach() in the consistency term
    assert rec1.grad is None or rec1.grad.abs().sum().item() == 0.0, (
        "rec1 received direct gradients from l2 — detach is missing"
    )
    # rec2 should receive gradients from both terms in l2
    assert rec2.grad is not None and rec2.grad.abs().sum().item() > 0.0, (
        "rec2 received no gradients from l2"
    )


# =========================================================================
# 3. EMA Teacher eval() Mode Persistence
# =========================================================================

def test_ncad_jepa_target_encoder_eval_persistence(dummy_batch):
    ctx, tgt = dummy_batch
    model = NCADJEPAModel(
        input_dim=5,
        latent_dim=16,
        filters=16,
        tcn_layers=2,
        dropout=0.5,  # high dropout to detect training mode
    )

    # Calling model.train() should keep target_encoder in eval mode
    model.train()
    assert not model.target_encoder.training
    assert model.context_encoder.training

    # Evaluate target determinism: two passes of target_encoder with same input should give identical outputs
    with torch.no_grad():
        out1 = model.target_encoder(tgt)
        out2 = model.target_encoder(tgt)
        assert torch.equal(out1, out2)


def test_multiscale_ts_jepa_eval_persistence_and_buffer_reload(dummy_batch):
    ctx, tgt = dummy_batch
    model = MultiScaleTSJEPA(
        input_dim=5,
        latent_dim=16,
        horizons=(4, 8, 16),
        filters=16,
        tcn_layers=2,
        dropout=0.5,
    )

    model.train()
    assert not model.target_encoder.training
    assert model.context_encoder.training

    # Fit Mahalanobis
    assert not bool(model.precision_fitted.item())
    model.fit_mahalanobis_covariance(ctx, tgt)
    assert bool(model.precision_fitted.item())

    # Verify state_dict serialization / deserialization preserves precision_fitted
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)

    model_new = MultiScaleTSJEPA(
        input_dim=5,
        latent_dim=16,
        horizons=(4, 8, 16),
        filters=16,
        tcn_layers=2,
    )
    assert not bool(model_new.precision_fitted.item())
    model_new.load_state_dict(torch.load(buffer))
    assert bool(model_new.precision_fitted.item())


def test_multiscale_mahalanobis_unfitted_raises_error(dummy_batch):
    ctx, tgt = dummy_batch
    model = MultiScaleTSJEPA(
        input_dim=5,
        latent_dim=16,
        horizons=(4, 8, 16),
        filters=16,
        tcn_layers=2,
    )
    with pytest.raises(RuntimeError, match="Mahalanobis scoring requested, but covariance has not been fitted"):
        model.compute_predictive_discrepancy(ctx, tgt, use_mahalanobis=True)


# =========================================================================
# 4. Patch Target Length & Shape Validation
# =========================================================================

def test_patch_jepa_shape_validation():
    B, C = 2, 4
    patch_size = 4
    n_ctx_patches = 4
    n_tgt_patches = 2
    ctx_len = n_ctx_patches * patch_size  # 16
    tgt_len = n_tgt_patches * patch_size  # 8

    ctx = torch.randn(B, ctx_len, C)
    tgt_correct = torch.randn(B, tgt_len, C)
    tgt_wrong = torch.randn(B, tgt_len + patch_size, C)  # 12 != 8

    model = PatchTSJEPA(
        input_dim=C,
        patch_size=patch_size,
        d_model=16,
        n_target_patches=n_tgt_patches,
    )

    # Valid forward pass
    h_ctx, h_tgt_true, h_tgt_pred = model(ctx, tgt_correct)
    assert h_tgt_true.shape == (B, n_tgt_patches, 16)
    assert h_tgt_pred.shape == (B, n_tgt_patches, 16)

    # Wrong target length raises ValueError
    with pytest.raises(ValueError, match="target_windows sequence length"):
        model(ctx, tgt_wrong)

    # Loss mismatch raises ValueError
    with pytest.raises(ValueError, match="Shape mismatch"):
        model.compute_patch_loss(h_tgt_pred, torch.randn(B, 3, 16))


def test_patch_flow_jepa_shape_validation():
    B, C = 2, 4
    patch_size = 4
    n_ctx_patches = 4
    n_tgt_patches = 2
    ctx_len = n_ctx_patches * patch_size  # 16
    tgt_len = n_tgt_patches * patch_size  # 8

    ctx = torch.randn(B, ctx_len, C)
    tgt_correct = torch.randn(B, tgt_len, C)
    tgt_wrong = torch.randn(B, tgt_len + patch_size, C)

    model = PatchFlowJEPA(
        input_dim=C,
        patch_size=patch_size,
        d_model=16,
        n_target_patches=n_tgt_patches,
    )

    h_ctx, z_tgt_true, v_pred, v_target = model(ctx, tgt_correct)
    assert z_tgt_true.shape == (B, n_tgt_patches, 16)

    with pytest.raises(ValueError, match="target_windows sequence length"):
        model(ctx, tgt_wrong)


# =========================================================================
# 5. Flow Models Double Precision (torch.float64)
# =========================================================================

def test_flow_ts_jepa_double_precision():
    B, L_ctx, L_tgt, C = 2, 16, 8, 4
    ctx = torch.randn(B, L_ctx, C, dtype=torch.float64)
    tgt = torch.randn(B, L_tgt, C, dtype=torch.float64)

    encoder = HybridTCNEncoder(input_dim=C, latent_dim=16, filters=16, tcn_layers=2)
    model = FlowTSJEPA(context_encoder=encoder, latent_dim=16).double()

    # Forward
    z_ctx, z_tgt_true, v_pred, v_target = model(ctx, tgt)
    assert z_ctx.dtype == torch.float64
    assert v_pred.dtype == torch.float64
    assert v_target.dtype == torch.float64

    # Sample target
    z_sampled = model.sample_target(ctx, n_steps=2)
    assert z_sampled.dtype == torch.float64

    # Discrepancy
    disc = model.compute_predictive_discrepancy(ctx, tgt)
    assert disc.dtype == torch.float64
    assert disc.shape == (B,)


def test_patch_flow_jepa_double_precision():
    B, C = 2, 4
    patch_size = 4
    n_ctx_patches = 4
    n_tgt_patches = 2
    ctx = torch.randn(B, n_ctx_patches * patch_size, C, dtype=torch.float64)
    tgt = torch.randn(B, n_tgt_patches * patch_size, C, dtype=torch.float64)

    model = PatchFlowJEPA(
        input_dim=C,
        patch_size=patch_size,
        d_model=16,
        n_target_patches=n_tgt_patches,
    ).double()

    h_ctx, z_tgt_true, v_pred, v_target = model(ctx, tgt)
    assert h_ctx.dtype == torch.float64
    assert v_pred.dtype == torch.float64

    disc = model.compute_predictive_discrepancy(ctx, tgt)
    assert disc.dtype == torch.float64
    assert disc.shape == (B,)


# =========================================================================
# 6. Flow Mahalanobis Unfitted Error Check
# =========================================================================

def test_flow_mahalanobis_unfitted_raises_error(dummy_batch):
    ctx, tgt = dummy_batch
    encoder = HybridTCNEncoder(input_dim=5, latent_dim=16, filters=16, tcn_layers=2)
    model = FlowTSJEPA(context_encoder=encoder, latent_dim=16)

    with pytest.raises(RuntimeError, match="Mahalanobis scoring requested, but covariance has not been fitted"):
        model.compute_predictive_discrepancy(ctx, tgt, use_mahalanobis=True)


# =========================================================================
# 7. Positional Encoding with Odd and Even d_model
# =========================================================================

@pytest.mark.parametrize("d_model", [5, 7, 16, 31, 64])
def test_positional_encodings_odd_even_dimensions(d_model):
    B, N = 2, 20
    x = torch.randn(B, N, d_model)

    pe_patch = Patch_PositionalEncoding(d_model=d_model, max_len=100)
    out_patch = pe_patch(x)
    assert out_patch.shape == (B, N, d_model)

    pe_tranad = TranAD_PositionalEncoding(d_model=d_model, max_len=100)
    out_tranad = pe_tranad(x)
    assert out_tranad.shape == (B, N, d_model)

    pe_at = AT_PositionalEmbedding(d_model=d_model, max_len=100)
    out_at = pe_at(x)
    assert out_at.shape == (1, N, d_model)


# =========================================================================
# 8. SINDy Dynamics Scorer Validation
# =========================================================================

def test_sindy_scorer_poly_degrees():
    z = np.random.randn(20, 4)

    # Degree 0 (constant only)
    scorer0 = SINDyDynamicsScorer(SINDyConfig(poly_degree=0, include_constant=True))
    lib0 = scorer0.build_library(z)
    assert lib0.shape == (20, 1)

    # Degree 1
    scorer1 = SINDyDynamicsScorer(SINDyConfig(poly_degree=1, include_constant=True))
    lib1 = scorer1.build_library(z)
    assert lib1.shape == (20, 1 + 4)

    # Degree 2
    scorer2 = SINDyDynamicsScorer(SINDyConfig(poly_degree=2, include_constant=True))
    lib2 = scorer2.build_library(z)
    assert lib2.shape == (20, 1 + 4 + 10)

    # Invalid degree
    scorer_inv = SINDyDynamicsScorer(SINDyConfig(poly_degree=3))
    with pytest.raises(ValueError, match="Unsupported poly_degree"):
        scorer_inv.build_library(z)


def test_sindy_scorer_short_sequence_error():
    z_short = np.random.randn(2, 4)
    scorer = SINDyDynamicsScorer()
    scorer.coefficients = np.ones((5, 4), dtype=np.float32)
    scorer.n_features = 5
    with pytest.raises(ValueError, match="Need at least 3 samples"):
        scorer.score(z_short)


# =========================================================================
# 9. Anomaly Injector Zero Ratio
# =========================================================================

def test_anomaly_injector_zero_ratio():
    windows = np.random.randn(10, 50, 4)
    injector = ContextualAnomalyInjector(AnomalyInjectionConfig(injection_ratio=0.0))
    modified, labels = injector.inject_batch(windows, context_size=30)
    assert np.array_equal(windows, modified)
    assert np.all(labels == 0.0)


# =========================================================================
# 10. Frequency Masker Reproducibility
# =========================================================================

def test_frequency_masker_reproducibility():
    x = torch.randn(4, 32, 4)
    masker1 = FrequencyMasker(mask_ratio=0.3, seed=123)
    masker2 = FrequencyMasker(mask_ratio=0.3, seed=123)

    out1 = masker1.mask_batch(x)
    out2 = masker2.mask_batch(x)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_frequency_masker_type_hints_and_ratio_bounds():
    import typing
    hints = typing.get_type_hints(FrequencyMasker.__init__)
    assert "mask_ratio" in hints
    assert "seed" in hints

    # Invalid ratios
    with pytest.raises(ValueError, match="mask_ratio must be between 0.0 and 1.0"):
        FrequencyMasker(mask_ratio=-0.1)

    with pytest.raises(ValueError, match="mask_ratio must be between 0.0 and 1.0"):
        FrequencyMasker(mask_ratio=1.5)

    # Valid boundary ratios (0.0 and 1.0)
    x = torch.randn(2, 16, 4)
    masker_zero = FrequencyMasker(mask_ratio=0.0)
    out_zero = masker_zero.mask_batch(x)
    assert torch.allclose(out_zero, x, atol=1e-5)

    masker_one = FrequencyMasker(mask_ratio=1.0)
    out_one = masker_one.mask_batch(x)
    assert out_one.shape == x.shape


def test_flow_n_eval_times_multi_point_evaluation(dummy_batch):
    ctx, tgt = dummy_batch
    encoder = HybridTCNEncoder(input_dim=5, latent_dim=16, filters=16, tcn_layers=2)
    model = FlowTSJEPA(context_encoder=encoder, latent_dim=16)

    # Compare 1 point vs 3 points
    score_1 = model.compute_instantaneous_energy_discrepancy(ctx, tgt, n_eval_times=1)
    score_3 = model.compute_instantaneous_energy_discrepancy(ctx, tgt, n_eval_times=3)
    assert score_1.shape == (len(ctx),)
    assert score_3.shape == (len(ctx),)
    assert torch.isfinite(score_1).all()
    assert torch.isfinite(score_3).all()


def test_patch_flow_n_eval_times_multi_point_evaluation():
    B, C = 2, 4
    patch_size = 4
    n_ctx_patches = 4
    n_tgt_patches = 2
    ctx = torch.randn(B, n_ctx_patches * patch_size, C)
    tgt = torch.randn(B, n_tgt_patches * patch_size, C)

    model = PatchFlowJEPA(
        input_dim=C,
        patch_size=patch_size,
        d_model=16,
        n_target_patches=n_tgt_patches,
    )

    patch_s1, win_s1 = model.compute_patch_instantaneous_discrepancy(ctx, tgt, n_eval_times=1)
    patch_s3, win_s3 = model.compute_patch_instantaneous_discrepancy(ctx, tgt, n_eval_times=3)
    assert patch_s1.shape == (B, n_tgt_patches)
    assert patch_s3.shape == (B, n_tgt_patches)
    assert win_s1.shape == (B,)
    assert win_s3.shape == (B,)


def test_multiscale_ts_jepa_invalid_horizons():
    with pytest.raises(ValueError, match="horizons must be a non-empty sequence"):
        MultiScaleTSJEPA(input_dim=5, latent_dim=16, horizons=())

    with pytest.raises(ValueError, match="positive integer"):
        MultiScaleTSJEPA(input_dim=5, latent_dim=16, horizons=(16, -1))


def test_build_encoder_architectures():
    from src.config import CSMConfig
    from src.engine.trainer import build_encoder

    device = torch.device("cpu")

    # relational_gat
    cfg_gat = CSMConfig(encoder_architecture="relational_gat", latent_dim=16, filters=32, tcn_layers=2)
    enc_gat = build_encoder(cfg_gat, input_dim=5, device=device)
    assert enc_gat.architecture == "relational_gat"

    # selective_ssm
    cfg_ssm = CSMConfig(encoder_architecture="selective_ssm", latent_dim=16, filters=32, tcn_layers=2)
    enc_ssm = build_encoder(cfg_ssm, input_dim=5, device=device)
    assert enc_ssm.architecture == "selective_ssm"
