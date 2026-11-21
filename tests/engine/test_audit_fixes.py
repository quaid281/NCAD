"""Unit tests covering model & loss audit fixes and behavior validations."""

import io

import numpy as np
import pytest
import torch

from src.models.baselines import AnomalyTransformer, TranAD
from src.models.baselines.anomaly_transformer import PositionalEmbedding as AT_PositionalEmbedding
from src.models.baselines.tranad import PositionalEncoding as TranAD_PositionalEncoding
from src.models.jepa.multiscale_ts_jepa import MultiScaleTSJEPA
from src.models.losses.anomaly_injector import AnomalyInjectionConfig, ContextualAnomalyInjector
from src.models.losses.fei_sigreg import FrequencyMasker


@pytest.fixture
def dummy_batch():
    batch_size = 4
    context_len = 32
    target_len = 16
    n_features = 5
    ctx = torch.randn(batch_size, context_len, n_features)
    tgt = torch.randn(batch_size, target_len, n_features)
    return ctx, tgt


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

    expected_prior = rec_loss + 3.0 * ass_dis_prior
    assert torch.allclose(loss_prior, expected_prior, atol=1e-5)

    prior_detached = [p.detach() for p in prior_list]
    ass_dis_series = model.association_discrepancy(prior_detached, series_list).mean()
    expected_series = rec_loss - 3.0 * ass_dis_series
    assert torch.allclose(loss_series, expected_series, atol=1e-5)

    loss_prior.backward(retain_graph=True)
    loss_series.backward()


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
    B, L, C = 4, 32, 3
    x = torch.randn(B, L, C)
    rec1 = torch.randn(B, L, C, requires_grad=True)
    rec2 = torch.randn(B, L, C, requires_grad=True)

    _, l2 = TranAD.adversarial_loss(rec1, rec2, x, epoch=5)
    l2.backward()

    assert rec1.grad is None or rec1.grad.abs().sum().item() == 0.0
    assert rec2.grad is not None and rec2.grad.abs().sum().item() > 0.0


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

    assert not bool(model.precision_fitted.item())
    model.fit_mahalanobis_covariance(ctx, tgt)
    assert bool(model.precision_fitted.item())

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


@pytest.mark.parametrize("d_model", [5, 7, 16, 31, 64])
def test_positional_encodings_odd_even_dimensions(d_model):
    B, N = 2, 20
    x = torch.randn(B, N, d_model)

    pe_tranad = TranAD_PositionalEncoding(d_model=d_model, max_len=100)
    out_tranad = pe_tranad(x)
    assert out_tranad.shape == (B, N, d_model)

    pe_at = AT_PositionalEmbedding(d_model=d_model, max_len=100)
    out_at = pe_at(x)
    assert out_at.shape == (1, N, d_model)


def test_anomaly_injector_zero_ratio():
    windows = np.random.randn(10, 50, 4)
    injector = ContextualAnomalyInjector(AnomalyInjectionConfig(injection_ratio=0.0))
    modified, labels = injector.inject_batch(windows, context_size=30)
    assert np.array_equal(windows, modified)
    assert np.all(labels == 0.0)


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

    with pytest.raises(ValueError, match="mask_ratio must be between 0.0 and 1.0"):
        FrequencyMasker(mask_ratio=-0.1)

    with pytest.raises(ValueError, match="mask_ratio must be between 0.0 and 1.0"):
        FrequencyMasker(mask_ratio=1.5)

    x = torch.randn(2, 16, 4)
    masker_zero = FrequencyMasker(mask_ratio=0.0)
    out_zero = masker_zero.mask_batch(x)
    assert torch.allclose(out_zero, x, atol=1e-5)

    masker_one = FrequencyMasker(mask_ratio=1.0)
    out_one = masker_one.mask_batch(x)
    assert out_one.shape == x.shape


def test_multiscale_ts_jepa_invalid_horizons():
    with pytest.raises(ValueError, match="horizons must be a non-empty sequence"):
        MultiScaleTSJEPA(input_dim=5, latent_dim=16, horizons=())

    with pytest.raises(ValueError, match="positive integer"):
        MultiScaleTSJEPA(input_dim=5, latent_dim=16, horizons=(16, -1))


def test_build_encoder_architectures():
    from src.config import CSMConfig
    from src.engine.trainer import build_encoder

    device = torch.device("cpu")

    cfg_gat = CSMConfig(encoder_architecture="relational_gat", latent_dim=16, filters=32, tcn_layers=2)
    enc_gat = build_encoder(cfg_gat, input_dim=5, device=device)
    assert enc_gat.architecture == "relational_gat"

    cfg_ssm = CSMConfig(encoder_architecture="selective_ssm", latent_dim=16, filters=32, tcn_layers=2)
    enc_ssm = build_encoder(cfg_ssm, input_dim=5, device=device)
    assert enc_ssm.architecture == "selective_ssm"
