"""Unit and mathematical invariant tests for Laminar-JEPA-Lite and PositionDriftLitePanel."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import CSMConfig
from src.engine.activation_tracer import autopsy_model
from src.engine.builder import build_ts_jepa_model
from src.models.laminar_lite import LaminarJEPALiteModel, LaminarLiteEncoder, LiteLatentPredictor
from src.scoring.circuit_breaker_panel import MultiPerspectiveCircuitBreakerPanel, stabilize_whitened_subblocks
from src.scoring.laminar_lite_panel import PositionDriftLitePanel, build_position_drift_dials


def test_laminar_lite_channel_gating_and_parameter_parsimony() -> None:
    """Verify K<=1 stem bypass, K>=25 dyadic gating, zero active Dropout, and parameter reduction."""
    enc_k1 = LaminarLiteEncoder(input_dim=1, latent_dim=16, filters=16, tcn_layers=2)
    enc_k9 = LaminarLiteEncoder(input_dim=9, latent_dim=16, filters=16, tcn_layers=2)
    enc_k25 = LaminarLiteEncoder(input_dim=25, latent_dim=16, filters=16, tcn_layers=2)

    assert enc_k1.channel_stem is None and enc_k1.dyadic_block is None
    assert enc_k9.channel_stem is not None and enc_k9.dyadic_block is None
    assert enc_k25.channel_stem is not None and enc_k25.dyadic_block is not None

    for enc in (enc_k1, enc_k9):
        dropout_leaves = [m for m in enc.modules() if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout1d))]
        assert len(dropout_leaves) == 0
    assert all(m.p == 0.0 for m in enc_k25.modules() if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout1d)))

    device = torch.device("cpu")
    for k_ch in (9, 25):
        cfg_flag = CSMConfig(model_type="soliton_reynolds_jepa", context_size=64, suspect_size=16, latent_dim=32)
        cfg_lite = CSMConfig(model_type="laminar_jepa_lite", context_size=64, suspect_size=16, latent_dim=32)
        m_flag = build_ts_jepa_model(cfg_flag, input_dim=k_ch, device=device)
        m_lite = build_ts_jepa_model(cfg_lite, input_dim=k_ch, device=device)
        p_flag = sum(p.numel() for p in m_flag.parameters() if p.requires_grad)
        p_lite = sum(p.numel() for p in m_lite.parameters() if p.requires_grad)
        max_ratio = 0.50 if k_ch < 25 else 0.76
        assert p_lite < max_ratio * p_flag, f"Expected <{max_ratio:.0%} params at K={k_ch}, got {p_lite} vs {p_flag}"


def test_lite_predictor_single_layernorm_and_parallel_horizons() -> None:
    """Verify LiteLatentPredictor has only one internal LayerNorm and matches per-horizon calls."""
    torch.manual_seed(42)
    pred = LiteLatentPredictor(latent_dim=8, hidden_dim=16)
    ln_modules = [m for m in pred.modules() if isinstance(m, torch.nn.LayerNorm)]
    assert len(ln_modules) == 1 and ln_modules[0].normalized_shape == (16,)

    z_batch = torch.randn(6, 3, 8)
    out_parallel = pred(z_batch)
    out_serial = torch.stack([pred(z_batch[:, m, :]) for m in range(3)], dim=1)
    assert out_parallel.shape == (6, 3, 8)
    assert torch.allclose(out_parallel, out_serial, atol=1e-6)
    norms = torch.linalg.norm(out_parallel, dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, float(np.sqrt(8.0))), atol=1e-3)


def test_laminar_lite_training_ema_barycenter_and_psd_whitening() -> None:
    """Verify finite backward gradients, synchronized EMA centroid, and PSD resolvent precision."""
    torch.manual_seed(42)
    np.random.seed(42)
    cfg = CSMConfig(
        model_type="laminar_jepa_lite",
        context_size=32,
        suspect_size=16,
        latent_dim=8,
        filters=16,
        tcn_layers=2,
    )
    model = build_ts_jepa_model(cfg, input_dim=6, device=torch.device("cpu"))
    assert isinstance(model, LaminarJEPALiteModel)

    windows = np.random.randn(24, 48, 6).astype(np.float32)
    batch = torch.from_numpy(windows[:8])
    model.train()
    loss, stats = model.compute_objective(batch[:, :32], batch[:, 32:], cfg)
    assert torch.isfinite(loss) and stats["loss"] > 0.0
    loss.backward()

    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grad_norms) > 0 and all(np.isfinite(g) for g in grad_norms)

    model.update_target_encoder()
    assert bool(model.target_tap_head.has_centroid.item())
    assert torch.allclose(model.target_tap_head.running_centroid, model.context_tap_head.running_centroid)

    model.fit_resolvent_whitening(windows, context_len=32, batch_size=12)
    assert bool(model.precision_fitted.item())
    eigvals = torch.linalg.eigvalsh(model.precision_matrix)
    assert float(eigvals.min().item()) >= -1e-6

    raw3, white3 = model.extract_raw_and_whitened_residuals(windows, context_len=32, batch_size=12)
    assert raw3.shape == (24, 3, 8) and white3.shape == (24, 3, 8)
    assert np.all(np.isfinite(white3))


def test_laminar_lite_activation_tracer_zero_hook_leak_and_zero_dead_leaves() -> None:
    """Verify autopsy_model leaves zero hooks, zero dormant leaves, and strict context causality."""
    torch.manual_seed(42)
    enc = LaminarLiteEncoder(input_dim=5, latent_dim=8, filters=12, tcn_layers=2)
    model = LaminarJEPALiteModel(context_encoder=enc, latent_dim=8, predictor_hidden_dim=16, num_horizons=3).eval()

    x_ctx = torch.randn(4, 32, 5)
    x_tgt_nom = torch.randn(4, 16, 5)
    x_tgt_anom = x_tgt_nom.clone()
    x_tgt_anom[:, -4:, 0] += 6.0

    report = autopsy_model(
        model,
        sample_input=(x_ctx, x_tgt_nom),
        anomalous_input=(x_ctx, x_tgt_anom),
        model_name="Laminar-JEPA-Lite",
    )
    assert report.global_health != "CRITICAL_UNSTABLE"

    traced_bases = {r.name.split("#")[0] for r in report.records}
    zero_hit = [
        name for name, mod in model.named_modules() if name and len(list(mod.children())) == 0 and name not in traced_bases
    ]
    assert len(zero_hit) == 0, f"Expected zero dormant leaves in LaminarJEPALiteModel, found: {zero_hit}"

    pred_hits = [r.name for r in report.records if r.name.startswith("predictor.fc1")]
    tgt_tap_hits = [r.name for r in report.records if r.name.startswith("target_tap_head.net.0")]
    assert len(pred_hits) == 1
    assert len(tgt_tap_hits) == 3

    for k, div in report.anomaly_divergence.items():
        if k.startswith(("context_encoder.", "context_tap_head.", "predictor.")):
            assert div == pytest.approx(0.0, abs=1e-6)
    last_tap_key = [r.name for r in report.records if r.name.startswith("target_tap_head.net.3")][-1]
    assert report.anomaly_divergence[last_tap_key] > 0.0

    for mod in model.modules():
        assert len(mod._forward_hooks) == 0


def test_position_drift_panel_energy_identity_and_causal_non_leakage() -> None:
    """Verify Reynolds energy decomposition and strict right-edge causal non-leakage."""
    rng = np.random.default_rng(42)
    ew = rng.normal(size=(80, 3, 8)).astype(np.float32)
    dials = build_position_drift_dials(ew)
    assert dials.shape == (80, 32) and np.all(dials >= 0.0)

    pos_energy = np.mean(dials[:, :24].reshape(80, 3, 8) ** 2, axis=1)
    drift_energy = dials[:, 24:] ** 2
    fluct_energy = np.mean((ew - ew.mean(axis=1, keepdims=True)) ** 2, axis=1)
    assert np.allclose(pos_energy, drift_energy + fluct_energy, atol=1e-5)

    va_ew = rng.normal(scale=0.5, size=(120, 3, 8)).astype(np.float32)
    te_clean = rng.normal(scale=0.5, size=(90, 3, 8)).astype(np.float32)
    te_spike = te_clean.copy()
    te_spike[50:, :, 0] += 12.0  # Fault enters window 50 -> causal timestamp 32 + 16 - 1 + 50 = 97

    panel = PositionDriftLitePanel(context_size=32, suspect_size=16, use_hijack_guard=True)
    bits_clean = panel.calibrate_and_trip_windows(va_ew, te_clean, n_cal_w=60)
    bits_spike = panel.calibrate_and_trip_windows(va_ew, te_spike, n_cal_w=60)

    dec_clean = panel.evaluate_stream_decisions(bits_clean, len_va=120 + 47, len_test=90 + 47, mapping="trailing")
    dec_spike = panel.evaluate_stream_decisions(bits_spike, len_va=120 + 47, len_test=90 + 47, mapping="trailing")

    # Prior to window 50, causal decisions must be bitwise identical (zero future leakage).
    assert np.array_equal(dec_clean.test_pred_balanced[:50], dec_spike.test_pred_balanced[:50])
    assert dec_spike.test_pred_balanced[50:].sum() > dec_clean.test_pred_balanced[50:].sum()


def test_disjoint_adaptation_non_leakage_and_score_modes() -> None:
    """Verify X_q adaptation leaves X_cal untouched in Eq. (2) and score_mode preserves binary gates."""
    rng = np.random.default_rng(7)
    q_w = rng.normal(scale=1.4, size=(96, 4, 8)).astype(np.float32)
    q_r = rng.normal(scale=0.6, size=(96, 4, 8)).astype(np.float32)
    va_w = rng.normal(scale=0.9, size=(120, 4, 8)).astype(np.float32)
    va_r = rng.normal(scale=0.4, size=(120, 4, 8)).astype(np.float32)
    te_w = rng.normal(scale=1.0, size=(80, 4, 8)).astype(np.float32)
    te_r = rng.normal(scale=0.5, size=(80, 4, 8)).astype(np.float32)

    va_ew1, te_ew1 = stabilize_whitened_subblocks(va_w, te_w, va_r, te_r, n_cal_w=60, adapt_white4=q_w, adapt_raw4=q_r)
    va_w_perturbed = va_w.copy()
    va_r_perturbed = va_r.copy()
    va_w_perturbed[:60] += 50.0
    va_r_perturbed[:60] += 50.0
    va_ew2, te_ew2 = stabilize_whitened_subblocks(
        va_w_perturbed, te_w, va_r_perturbed, te_r, n_cal_w=60, adapt_white4=q_w, adapt_raw4=q_r
    )
    assert np.array_equal(te_ew1, te_ew2)
    assert np.array_equal(va_ew1[60:], va_ew2[60:])

    with pytest.raises(ValueError, match="must both be provided"):
        stabilize_whitened_subblocks(va_w, te_w, va_r, te_r, n_cal_w=60, adapt_white4=q_w, adapt_raw4=None)

    panel = MultiPerspectiveCircuitBreakerPanel(context_size=32, suspect_size=16)
    bits = panel.calibrate_and_trip_windows(va_ew1, te_ew1, n_cal_w=60)
    d_w = panel.evaluate_stream_decisions(bits, len_va=120 + 47, len_test=80 + 47, mapping="trailing", score_mode="weighted")
    d_u = panel.evaluate_stream_decisions(bits, len_va=120 + 47, len_test=80 + 47, mapping="trailing", score_mode="unweighted_224")
    d_ue = panel.evaluate_stream_decisions(bits, len_va=120 + 47, len_test=80 + 47, mapping="trailing", score_mode="unweighted_224_edge")
    assert np.array_equal(d_w.test_pred_balanced, d_u.test_pred_balanced)
    assert np.array_equal(d_w.test_pred_balanced, d_ue.test_pred_balanced)
    assert np.all(d_w.test_switch_counts >= d_ue.test_switch_counts - 1e-6)
    assert np.all(d_ue.test_switch_counts >= d_u.test_switch_counts - 1e-6)

