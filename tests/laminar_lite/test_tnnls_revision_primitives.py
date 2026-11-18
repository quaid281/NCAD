"""Fast deterministic unit tests for TNNLS revision mathematical and statistical primitives."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.tnnls_revision.common_eval import (
    FrozenLatentDriftMonitor,
    QuarantinedRecalibrationController,
    apply_horizon_whitening_operator,
    compute_rankme,
    effective_sample_size,
    fit_horizon_whitening_operator,
    loco_channel_attribution,
    mbb_quorum_thresholds,
    simulate_hijacker_masking_vs_quorum,
)
from scripts.tnnls_revision.run_wu_keogh_plausible_suite import inject_wu_keogh_anomaly
from src.models.laminar_lite import LaminarJEPALiteModel, LaminarLiteEncoder
from src.scoring.laminar_lite_panel import PositionDriftLitePanel


def test_proposition1_hijacker_masking_and_quorum_scale_invariance() -> None:
    """Verify Proposition 1: exact invariance of per-dial quorum under coordinate variance inflation."""
    res = simulate_hijacker_masking_vs_quorum(
        dim=32, n_hijack=2, eta=5.0, n_fault_coords=10, fault_amp=3.8, n_cal=2000, seed=42
    )
    assert res["quorum_exact_invariance"] is True
    assert res["k_quorum_base"] == pytest.approx(res["k_quorum_drift"], abs=1e-12)
    assert res["scalar_detects_without_hijack"] is True
    assert res["scalar_masked_under_hijack"] is True
    assert res["quorum_detects_under_hijack"] is True


def test_mbb_quorum_thresholds_respect_floor_and_ordering() -> None:
    """Moving-block bootstrap thresholds must satisfy floor bounds and monotone ordering."""
    rng = np.random.default_rng(42)
    # Autocorrelated AR(1)-like nonnegative count series
    raw = np.zeros(600, dtype=np.float64)
    for t in range(1, 600):
        raw[t] = 0.8 * raw[t - 1] + rng.poisson(0.6)
    tau, n_eff = effective_sample_size(raw)
    assert tau > 1.5
    assert 10.0 < n_eff < 600.0

    ths = mbb_quorum_thresholds(raw, floor=3.5, q=0.995, safety=1.12, n_boot=200, seed=42)
    assert ths["deployed"] >= 3.5
    assert ths["mbb_median"] >= 3.5
    assert ths["mbb_p90"] >= ths["mbb_median"]
    assert ths["block_len"] >= 16.0


@pytest.mark.parametrize("latent_dim", [8, 16, 32, 64])
def test_laminar_lite_supports_latent_dim_sweep(latent_dim: int) -> None:
    """Verify Laminar-JEPA-Lite and PositionDriftLitePanel across D in {8, 16, 32, 64}."""
    torch.manual_seed(42)
    enc = LaminarLiteEncoder(input_dim=6, latent_dim=latent_dim, filters=16, tcn_layers=2)
    model = LaminarJEPALiteModel(enc, latent_dim=latent_dim, predictor_hidden_dim=2 * latent_dim, num_horizons=3)

    windows = np.random.default_rng(42).standard_normal((24, 96, 6)).astype(np.float32)
    ctx = torch.from_numpy(windows[:, :64])
    tgt = torch.from_numpy(windows[:, 64:])
    loss, _ = model.compute_objective(ctx, tgt)
    assert torch.isfinite(loss)

    model.fit_resolvent_whitening(windows, context_len=64, batch_size=12)
    raw, white = model.extract_raw_and_whitened_residuals(windows, context_len=64, batch_size=12)
    assert raw.shape == (24, 3, latent_dim)
    assert white.shape == (24, 3, latent_dim)

    panel = PositionDriftLitePanel(context_size=64, suspect_size=32)
    bits = panel.calibrate_and_trip_windows(white, white, n_cal_w=12)
    dec = panel.evaluate_stream_decisions(bits, len_va=120, len_test=120, mapping="trailing")
    assert dec.test_switch_counts.ndim == 1
    assert compute_rankme(white[:, -1, :]) >= 1.0


def test_frozen_latent_drift_monitor_catches_slow_ramp() -> None:
    """FrozenLatentDriftMonitor should stay quiet on stationary hold data and catch slow DC drift."""
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    enc = LaminarLiteEncoder(input_dim=4, latent_dim=8, filters=8, tcn_layers=1)
    model = LaminarJEPALiteModel(enc, latent_dim=8, predictor_hidden_dim=16, num_horizons=3)

    fit_w = rng.standard_normal((80, 96, 4)).astype(np.float32)
    val_w = rng.standard_normal((120, 96, 4)).astype(np.float32)
    # Slow accumulating bias across test windows
    ramp = np.linspace(0.0, 4.0, 100, dtype=np.float32)[:, None, None]
    test_w = rng.standard_normal((100, 96, 4)).astype(np.float32) + ramp

    mon = FrozenLatentDriftMonitor(ewma_alpha=0.1, safety=1.15, q=0.99)
    f_fit = mon.extract_pre_ln_context_summaries(model, fit_w, context_len=64, batch_size=64)
    f_val = mon.extract_pre_ln_context_summaries(model, val_w, context_len=64, batch_size=64)
    f_te = mon.extract_pre_ln_context_summaries(model, test_w, context_len=64, batch_size=64)
    mon.fit_reference(f_fit)
    dec = mon.calibrate_and_score(f_val, f_te, n_cal_w=60, suspect_size=16)

    assert float(dec.hold_pred.mean()) <= 0.10
    assert int(dec.test_pred[-20:].sum()) >= 15


def test_wu_keogh_cycle_swap_and_phase_desync_preserve_marginals() -> None:
    """Wu & Keogh plausible anomalies must preserve marginal min/max bounds."""
    rng = np.random.default_rng(42)
    seg = rng.standard_normal((1200, 6)).astype(np.float32)
    mod_swap, y_swap = inject_wu_keogh_anomaly(seg, "Cycle_Swap", t0=600, dur=200, rng=rng)
    assert y_swap[600:800].sum() == 200
    assert float(np.max(np.abs(mod_swap))) <= float(np.max(np.abs(seg))) + 1e-6

    mod_desync, y_desync = inject_wu_keogh_anomaly(seg, "Phase_Desync", t0=600, dur=200, rng=rng)
    assert y_desync[600:800].sum() == 200
    assert float(np.max(np.abs(mod_desync))) <= float(np.max(np.abs(seg))) + 1e-6


def test_quarantined_recalibration_prevents_unconfirmed_fault_absorption() -> None:
    """Quarantined controller must keep tripping on unconfirmed faults and support rollback."""
    rng = np.random.default_rng(42)
    cal_dials = rng.gamma(shape=1.5, scale=0.4, size=(200, 32)).astype(np.float32)
    tau_init = (np.percentile(cal_dials, 98.5, axis=0) + 1e-4).astype(np.float32)
    k_init = 3.5

    ctrl = QuarantinedRecalibrationController(tau_dials=tau_init, k_quorum=k_init, floor=3.5)
    shifted_dials = cal_dials + 2.5

    # Under unconfirmed distribution shift (ks >= 0.50), active thresholds remain untouched (100% fault trips)
    alarms_unconf, _, state = ctrl.process_batch(shifted_dials, ks_distance=0.80)
    assert state == "QUARANTINE_ALERT"
    assert float(alarms_unconf.mean()) == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(ctrl.tau_dials, tau_init)

    # Confirming a legitimate mode shift updates thresholds and reduces benign post-shift FPR
    assert ctrl.confirm_mode_transition(mode_confirmed=True) is True
    assert ctrl.state == "NORMAL"
    assert not np.allclose(ctrl.tau_dials, tau_init)
    alarms_after, _, _ = ctrl.process_batch(shifted_dials, ks_distance=0.10)
    assert float(alarms_after.mean()) < 0.05

    # Revoking the update restores exact pre-shift thresholds
    assert ctrl.rollback() is True
    assert np.allclose(ctrl.tau_dials, tau_init)
    assert ctrl.k_quorum == pytest.approx(k_init)


def test_loco_channel_attribution_identifies_perturbed_sensor() -> None:
    """LOCO channel attribution must isolate a strongly perturbed channel in top attribution slots."""
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    enc = LaminarLiteEncoder(input_dim=6, latent_dim=8, filters=16, tcn_layers=2)
    model = LaminarJEPALiteModel(enc, latent_dim=8, predictor_hidden_dim=16, num_horizons=3)

    cal_win = (0.25 * rng.standard_normal((24, 96, 6))).astype(np.float32)
    model.fit_resolvent_whitening(cal_win, context_len=64, batch_size=12)
    _, cal_white = model.extract_raw_and_whitened_residuals(cal_win, context_len=64, batch_size=12)
    from src.scoring.laminar_lite_panel import build_position_drift_dials

    cal_pd = build_position_drift_dials(cal_white)
    tau = (np.percentile(cal_pd, 98.5, axis=0) + 1e-4).astype(np.float32)

    x_fault = (0.25 * rng.standard_normal((96, 6))).astype(np.float32)
    x_fault[64:, 3] += 8.0
    attr = loco_channel_attribution(model, tau, x_fault, context_len=64)
    assert attr.shape == (6,)
    assert int(np.argmax(attr)) == 3 or 3 in np.argsort(-attr)[:2].tolist()


def test_horizon_preserving_whitening_operators_and_zca_alignment() -> None:
    """Shared-Kronecker and Block-Diagonal ZCA must have zero cross-horizon leakage; ZCA > PCA alignment."""
    rng = np.random.default_rng(42)
    # Correlated across both horizons (M=3) and coordinates (D=8)
    mix = rng.standard_normal((24, 24)).astype(np.float64) * 0.4 + np.eye(24)
    fit_raw = (rng.standard_normal((256, 24)) @ mix).reshape(256, 3, 8).astype(np.float32)
    eval_raw = (rng.standard_normal((64, 24)) @ mix).reshape(64, 3, 8).astype(np.float32)

    op_shared = fit_horizon_whitening_operator(fit_raw, mode="shared_kronecker_zca")
    op_bdiag = fit_horizon_whitening_operator(fit_raw, mode="block_diag_zca")
    op_full = fit_horizon_whitening_operator(fit_raw, mode="full_zca")
    op_pca = fit_horizon_whitening_operator(fit_raw, mode="pca_eigen")

    # Zero cross-horizon leakage when perturbing horizon m=0 alone:
    pert = eval_raw.copy()
    pert[:, 0, :] += 10.0
    for op in (op_shared, op_bdiag):
        w0 = apply_horizon_whitening_operator(eval_raw, op)
        wp = apply_horizon_whitening_operator(pert, op)
        assert np.allclose(wp[:, 1:, :], w0[:, 1:, :], atol=1e-5)

    # Exact commutation of shared Kronecker ZCA with horizon averaging:
    w_sh = apply_horizon_whitening_operator(eval_raw, op_shared)
    cent_mean = (eval_raw.astype(np.float64) - op_shared["mu"]).mean(axis=1)
    direct_mean_white = (cent_mean @ op_shared["w_d"]).astype(np.float32)
    assert np.allclose(w_sh.mean(axis=1), direct_mean_white, atol=1e-5)

    # ZCA optimality: entrywise alignment with centered residuals strictly exceeds eigenbasis PCA
    w_zca = apply_horizon_whitening_operator(eval_raw, op_full)
    w_eig = apply_horizon_whitening_operator(eval_raw, op_pca)
    cent = (eval_raw.astype(np.float64) - op_full["mu"]).astype(np.float32)
    assert float(np.sum(w_zca * cent)) > float(np.sum(w_eig * cent))


def test_proposition2_requires_matched_calibration_and_operational_scaling() -> None:
    """Verify binary trip invariance holds under matched scaling and fails under post-freeze drift."""
    rng = np.random.default_rng(42)
    u_cal = rng.standard_normal((500, 16))
    u_op = rng.standard_normal((200, 16))
    scale = np.ones((1, 16))
    scale[0, :3] = 5.0

    tau_base = np.quantile(np.abs(u_cal), 0.985, axis=0, keepdims=True)
    b_base = np.abs(u_op) > tau_base

    # Matched calibration + operational scaling -> exact pointwise trip invariance
    tau_matched = np.quantile(np.abs(u_cal * scale), 0.985, axis=0, keepdims=True)
    b_matched = np.abs(u_op * scale) > tau_matched
    assert np.array_equal(b_base, b_matched)

    # Unmatched scaling (operational shift after freezing tau_base) -> trips inflate on drifted coords
    b_post_freeze = np.abs(u_op * scale) > tau_base
    assert int(b_post_freeze[:, :3].sum()) > int(b_base[:, :3].sum())


