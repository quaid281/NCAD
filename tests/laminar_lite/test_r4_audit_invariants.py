"""Deterministic mathematical and protocol tests for TNNLS R4 audit revisions."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.tnnls_revision.common_eval import (
    apply_horizon_whitening_operator,
    false_alarm_episodes_per_1k,
    fit_horizon_whitening_operator,
    project_raw_to_eigenbasis,
    stabilize_eigenbasis_residuals,
)
from src.config import CSMConfig
from src.models.laminar_lite import LaminarJEPALiteModel, LaminarLiteEncoder
from src.scoring.laminar_lite_panel import PositionDriftLitePanel


def test_proposition2_scale_homogeneous_vs_additive_floor() -> None:
    """Homogeneous floor *(1+1e-4) is strictly equivariant; additive +1e-4 breaks under scaling."""
    rng = np.random.default_rng(42)
    va_ew = rng.standard_normal((200, 3, 8))
    te_ew = rng.standard_normal((120, 3, 8))
    scale = np.array([100.0, 0.01, 50.0, 0.25, 1.0, 10.0, 0.5, 4.0])[None, None, :]

    panel_hom = PositionDriftLitePanel(context_size=32, suspect_size=16, scale_homogeneous_floor=True)
    b_base = panel_hom.calibrate_and_trip_windows(va_ew, te_ew, n_cal_w=100)
    b_scal = panel_hom.calibrate_and_trip_windows(va_ew * scale, te_ew * scale, n_cal_w=100)
    assert np.array_equal(b_base["bits_pd_va"], b_scal["bits_pd_va"])
    assert np.array_equal(b_base["bits_pd_te"], b_scal["bits_pd_te"])

    d_base = panel_hom.evaluate_stream_decisions(b_base, len_va=247, len_test=167, mapping="trailing")
    d_scal = panel_hom.evaluate_stream_decisions(b_scal, len_va=247, len_test=167, mapping="trailing")
    assert d_base.k_gate_balanced == pytest.approx(d_scal.k_gate_balanced, abs=1e-12)
    assert np.array_equal(d_base.test_pred_balanced, d_scal.test_pred_balanced)

    # Constructive counterexample showing additive +1e-4 flips a near-threshold decision at s=10
    cal_near = np.linspace(0.0, 1.0 / 0.985, 201)[None, :].T
    va_add = np.repeat(cal_near[:, None, :], 3, axis=1).repeat(4, axis=2)
    # Force exact q985 = 1.0 on coordinate 0 by setting 98.5th percentile linearly
    va_add[:, :, :] = np.broadcast_to(np.linspace(0.0, 1.0 / 0.985, 201)[:, None, None], (201, 3, 4))
    q_val = float(np.percentile(np.abs(va_add[:101, 0, 0]), 98.5))
    te_probe = np.full((10, 3, 4), q_val + 0.5e-4)

    panel_add = PositionDriftLitePanel(context_size=32, suspect_size=16, scale_homogeneous_floor=False)
    trip_1 = panel_add.calibrate_and_trip_windows(va_add, te_probe, n_cal_w=101)
    trip_10 = panel_add.calibrate_and_trip_windows(va_add * 10.0, te_probe * 10.0, n_cal_w=101)
    assert float(trip_1["bits_pd_te"].mean()) == 0.0
    assert float(trip_10["bits_pd_te"].mean()) == 1.0


@pytest.mark.parametrize(
    ("s_len", "expected_effective"),
    [
        (1, (1, 1, 1)),
        (4, (2, 4, 4)),
        (16, (4, 8, 16)),
        (32, (8, 16, 32)),
        (64, (16, 32, 64)),
    ],
)
def test_short_horizon_integer_prefixes_and_gradients(
    s_len: int, expected_effective: tuple[int, int, int]
) -> None:
    """Verify exact integer prefix slicing and finite backward gradients for S in {1,4,16,32,64}."""
    torch.manual_seed(42)
    enc = LaminarLiteEncoder(input_dim=25, latent_dim=8, filters=12, tcn_layers=2)
    model = LaminarJEPALiteModel(enc, latent_dim=8, predictor_hidden_dim=16, num_horizons=3)
    cfg = CSMConfig(model_type="laminar_jepa_lite", context_size=32, suspect_size=s_len, latent_dim=8)

    raw_prefixes = model.prefix_lengths(s_len, num_horizons=3)
    eff_prefixes = tuple(min(p, s_len) for p in raw_prefixes)
    assert eff_prefixes == expected_effective

    ctx = torch.randn(4, 32, 25)
    tgt = torch.randn(4, s_len, 25)
    _, _, _, z_sub_pred, z_sub_tgt = model.forward(ctx, tgt)
    assert z_sub_pred.shape == (4, 3, 8)
    assert z_sub_tgt is not None and z_sub_tgt.shape == (4, 3, 8)

    if s_len == 1:
        assert torch.allclose(z_sub_tgt[:, 0], z_sub_tgt[:, 1], atol=1e-6)
        assert torch.allclose(z_sub_tgt[:, 1], z_sub_tgt[:, 2], atol=1e-6)
    elif s_len == 4:
        assert torch.allclose(z_sub_tgt[:, 1], z_sub_tgt[:, 2], atol=1e-6)

    model.train()
    loss, _ = model.compute_objective(ctx, tgt, cfg)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    )


def test_fair_pca_eigenbasis_shrinkage_and_kronecker_pca() -> None:
    """Consistent eigenbasis shrinkage preserves eigen-coordinate correlation and Kronecker alignment."""
    rng = np.random.default_rng(42)
    mix = rng.standard_normal((24, 24)) * 0.4 + np.eye(24)
    fit_raw = (rng.standard_normal((200, 24)) @ mix).reshape(200, 3, 8).astype(np.float32)
    q_raw = (rng.standard_normal((120, 24)) @ mix).reshape(120, 3, 8).astype(np.float32)
    va_raw = (rng.standard_normal((160, 24)) @ mix).reshape(160, 3, 8).astype(np.float32)
    te_raw = (rng.standard_normal((120, 24)) @ mix).reshape(120, 3, 8).astype(np.float32)

    op_pca = fit_horizon_whitening_operator(fit_raw, mode="pca_eigen")
    op_kpca = fit_horizon_whitening_operator(fit_raw, mode="shared_kronecker_pca")

    # Shared Kronecker PCA must have zero cross-horizon leakage
    pert = va_raw.copy()
    pert[:, 0, :] += 10.0
    w0 = apply_horizon_whitening_operator(va_raw, op_kpca)
    wp = apply_horizon_whitening_operator(pert, op_kpca)
    assert np.allclose(wp[:, 1:, :], w0[:, 1:, :], atol=1e-5)

    # Consistent eigenbasis shrinkage achieves near-unit correlation with projected eigen-coordinates
    for op in (op_pca, op_kpca):
        va_s, _ = stabilize_eigenbasis_residuals(op, q_raw, va_raw, te_raw, n_cal_w=80)
        va_proj = project_raw_to_eigenbasis(va_raw, op)
        a = va_s.reshape(len(va_s), -1).astype(np.float64)
        b = va_proj.reshape(len(va_proj), -1).astype(np.float64)
        a -= a.mean(axis=0, keepdims=True)
        b -= b.mean(axis=0, keepdims=True)
        corrs = (a * b).sum(axis=0) / (np.sqrt((a**2).sum(axis=0) * (b**2).sum(axis=0)) + 1e-12)
        assert float(np.mean(corrs)) > 0.99


def test_false_alarm_episodes_per_1k_distinguishes_chatter_from_burst() -> None:
    """Verify false-alarm episode counter distinguishes scattered chatter from one contiguous block."""
    assert false_alarm_episodes_per_1k(np.zeros(100, dtype=int)) == (0.0, 0.0)

    chatter = np.zeros(1000, dtype=int)
    for k in range(25):
        chatter[40 * k : 40 * k + 2] = 1
    ep_chat, len_chat = false_alarm_episodes_per_1k(chatter)
    assert ep_chat == pytest.approx(25.0)
    assert len_chat == pytest.approx(2.0)

    block = np.zeros(1000, dtype=int)
    block[500:550] = 1
    ep_blk, len_blk = false_alarm_episodes_per_1k(block)
    assert ep_blk == pytest.approx(1.0)
    assert len_blk == pytest.approx(50.0)
