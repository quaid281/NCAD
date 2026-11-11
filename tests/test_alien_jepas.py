"""Unit tests for Alien-Space-of-Science Discovered JEPA Variants."""

import numpy as np
import torch

from src.engine.activation_tracer import autopsy_model
from src.models.encoders.tcn_encoder import HybridTCNEncoder
from src.models.jepa.alien_jepas import (
    AlienDualTowerJEPAModel,
    AnchoredStokesJEPAModel,
    DeltaJEPAModel,
    EddyDeltaJEPAModel,
    HelmholtzSolitonJEPAModel,
    SolitonReynoldsJEPAModel,
    SpeculativeJEPAModel,
)


def test_delta_jepa_forward_loss_and_discrepancy():
    torch.manual_seed(42)
    B, C_len, S_len, K, D = 4, 32, 8, 3, 8
    enc = HybridTCNEncoder(input_dim=K, latent_dim=D, filters=8, tcn_layers=1)
    model = DeltaJEPAModel(
        context_encoder=enc,
        latent_dim=D,
        predictor_hidden_dim=16,
        predictor_layers=2,
        num_subblocks=4,
    )

    ctx = torch.randn(B, C_len, K)
    tgt = torch.randn(B, S_len, K)

    loss, metrics = model.compute_objective(ctx, tgt)
    assert torch.isfinite(loss) and loss.item() > 0.0
    assert "loss" in metrics
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad and "target_encoder" not in name:
            assert param.grad is not None, f"Missing gradient on {name}"

    disc = model.compute_predictive_discrepancy(ctx, tgt)
    assert disc.shape == (B,)
    assert torch.all(disc >= 0.0)


def test_speculative_jepa_forward_loss_and_discrepancy():
    torch.manual_seed(42)
    B, C_len, S_len, K, D = 4, 32, 16, 3, 8
    enc = HybridTCNEncoder(input_dim=K, latent_dim=D, filters=8, tcn_layers=1)
    model = SpeculativeJEPAModel(
        context_encoder=enc,
        latent_dim=D,
        predictor_hidden_dim=16,
        coreset_size=6,
        top_k_prototypes=3,
        null_dim=2,
    )

    ctx = torch.randn(B, C_len, K)
    tgt = torch.randn(B, S_len, K)

    loss, metrics = model.compute_objective(ctx, tgt)
    assert torch.isfinite(loss) and loss.item() > 0.0
    assert "loss" in metrics
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad and "target_encoder" not in name:
            assert param.grad is not None, f"Missing gradient on {name}"

    train_wins = np.random.randn(12, C_len + S_len, K).astype(np.float32)
    model.fit_alien_bank(train_wins, context_len=C_len, batch_size=6)
    assert bool(model.bank_fitted.item()) is True

    disc = model.compute_predictive_discrepancy(ctx, tgt, use_alien_bank=True)
    assert disc.shape == (B,)
    assert torch.all(disc >= 0.0)


def test_alien_dual_tower_jepa_forward_and_bank():
    torch.manual_seed(42)
    B, C_len, S_len, K, D = 8, 32, 8, 3, 8
    enc = HybridTCNEncoder(input_dim=K, latent_dim=D, filters=8, tcn_layers=1)
    model = AlienDualTowerJEPAModel(
        context_encoder=enc,
        latent_dim=D,
        predictor_hidden_dim=16,
        coreset_size=6,
        top_k_prototypes=3,
        null_dim=2,
    )

    ctx = torch.randn(B, C_len, K)
    tgt = torch.randn(B, S_len, K)

    loss, _ = model.compute_objective(ctx, tgt)
    assert torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()

    train_wins = np.random.randn(12, C_len + S_len, K).astype(np.float32)
    model.fit_alien_bank(train_wins, context_len=C_len, batch_size=6)
    assert bool(model.bank_fitted.item()) is True

    disc = model.compute_predictive_discrepancy(ctx, tgt)
    assert disc.shape == (B,)
    assert torch.all(disc >= 0.0)


def test_alien_jepas_layer_activation_autopsy():
    torch.manual_seed(42)
    B, C_len, S_len, K, D = 6, 32, 16, 3, 8
    ctx_nom = torch.randn(B, C_len, K)
    tgt_nom = torch.randn(B, S_len, K)
    ctx_anom = ctx_nom.clone()
    tgt_anom = tgt_nom + 3.0 * torch.randn(B, S_len, K)

    for cls in (
        DeltaJEPAModel,
        SpeculativeJEPAModel,
        AlienDualTowerJEPAModel,
        EddyDeltaJEPAModel,
        AnchoredStokesJEPAModel,
        SolitonReynoldsJEPAModel,
        HelmholtzSolitonJEPAModel,
    ):
        enc = HybridTCNEncoder(input_dim=K, latent_dim=D, filters=8, tcn_layers=1)
        m = cls(context_encoder=enc, latent_dim=D, predictor_hidden_dim=16)
        rep = autopsy_model(
            m,
            sample_input=(ctx_nom, tgt_nom),
            anomalous_input=(ctx_anom, tgt_anom),
            model_name=cls.__name__,
        )
        assert rep.global_health == "HEALTHY"
        assert len(rep.records) > 0
        assert len(rep.anomaly_divergence) > 0


def test_eddy_delta_jepa_solenoidal_vorticity_and_scale_homogeneity():
    torch.manual_seed(42)
    B, C_len, S_len, K, D = 4, 32, 8, 3, 8
    enc = HybridTCNEncoder(input_dim=K, latent_dim=D, filters=8, tcn_layers=1)
    model = EddyDeltaJEPAModel(
        context_encoder=enc,
        latent_dim=D,
        predictor_hidden_dim=16,
        predictor_layers=2,
        num_subblocks=4,
    )

    ctx = torch.randn(B, C_len, K)
    tgt = torch.randn(B, S_len, K)

    _, _, _, eddy_unit, omega = model.forward(ctx, tgt)
    # Exact skew-symmetry (\Omega + \Omega^T = 0) and solenoidal zero-trace (Tr(\Omega) = 0)
    skew_err = torch.linalg.norm(omega + omega.transpose(1, 2), dim=(-2, -1)).max()
    assert skew_err.item() < 1e-6
    diag_trace = torch.diagonal(omega, dim1=-2, dim2=-1).sum(dim=-1).abs().max()
    assert diag_trace.item() < 1e-6
    # Scale-preserving mean-1 eddy viscosity unit
    assert torch.allclose(eddy_unit.mean(dim=-1), torch.ones(B), atol=1e-4)

    loss, metrics = model.compute_objective(ctx, tgt)
    assert torch.isfinite(loss) and loss.item() > 0.0
    assert "loss" in metrics
    loss.backward()
    assert model.vorticity_gate.grad is not None and model.vorticity_gate.grad.norm().item() > 1e-4

    disc = model.compute_predictive_discrepancy(ctx, tgt)
    assert disc.shape == (B,)
    assert torch.all(disc >= 0.0)


def test_anchored_stokes_jepa_monotone_operator_and_rollout():
    torch.manual_seed(42)
    B, C_len, S_len, K, D = 4, 32, 8, 3, 8
    enc = HybridTCNEncoder(input_dim=K, latent_dim=D, filters=8, tcn_layers=1)
    model = AnchoredStokesJEPAModel(
        context_encoder=enc,
        latent_dim=D,
        predictor_hidden_dim=16,
        rollout_steps=4,
        nu_0=1e-2,
    )

    # Verify strict monotonicity: <G(z) - G(w), z - w> >= nu_0 * ||z - w||^2
    z = torch.randn(B, D)
    w = torch.randn(B, D)
    z_eq = torch.randn(B, D)
    gz = model.apply_monotone_operator(z, z_eq)
    gw = model.apply_monotone_operator(w, z_eq)
    inner = ((gz - gw) * (z - w)).sum(dim=-1)
    sq_norm = (z - w).pow(2).sum(dim=-1)
    assert torch.all(inner >= (model.nu_0 - 1e-6) * sq_norm)

    ctx = torch.randn(B, C_len, K)
    tgt = torch.randn(B, S_len, K)
    loss, _ = model.compute_objective(ctx, tgt)
    assert torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()
    assert model.stokes_op.diss_gen.grad is not None and model.stokes_op.diss_gen.grad.norm().item() > 1e-3
    assert model.stokes_op.skew_gen.grad is not None and model.stokes_op.skew_gen.grad.norm().item() > 1e-3

    disc = model.compute_predictive_discrepancy(ctx, tgt)
    assert disc.shape == (B,)
    assert torch.all(disc >= 0.0)


def test_soliton_reynolds_jepa_lieb_thirring_equality_and_bounds():
    torch.manual_seed(42)
    B, M, D = 4, 4, 8
    # 1. Rank-1 soliton packet: E = u (x) v^T -> exact Lieb-Thirring equality R_LT = 1.0
    u = torch.randn(B, M, 1)
    v = torch.randn(B, 1, D)
    soliton_packet = torch.bmm(u, v)
    r_soliton = SolitonReynoldsJEPAModel.lieb_thirring_soliton_ratio(soliton_packet, gamma=1.0)
    assert torch.allclose(r_soliton, torch.ones(B), atol=1e-5)

    # 2. Isotropic Equal-Norm Orthogonal Channels -> minimal ratio R_LT = 1 / M
    q, _ = torch.linalg.qr(torch.randn(B, D, M))
    iso_packet = q.transpose(1, 2)  # (B, M, D) with E E^T = I_M
    r_iso = SolitonReynoldsJEPAModel.lieb_thirring_soliton_ratio(iso_packet, gamma=1.0)
    assert torch.allclose(r_iso, torch.full((B,), 1.0 / M), atol=1e-5)

    # 3. Scale invariance: R_LT(c * E) == R_LT(E)
    rand_packet = torch.randn(B, M, D)
    r_1 = SolitonReynoldsJEPAModel.lieb_thirring_soliton_ratio(rand_packet, gamma=1.0)
    r_scaled = SolitonReynoldsJEPAModel.lieb_thirring_soliton_ratio(7.5 * rand_packet, gamma=1.0)
    assert torch.allclose(r_1, r_scaled, atol=1e-5)

    # 4. End-to-end objective & discrepancy
    C_len, S_len, K = 32, 16, 3
    enc = HybridTCNEncoder(input_dim=K, latent_dim=D, filters=8, tcn_layers=1)
    model = SolitonReynoldsJEPAModel(context_encoder=enc, latent_dim=D, predictor_hidden_dim=16)
    ctx = torch.randn(B, C_len, K)
    tgt = torch.randn(B, S_len, K)
    loss, _ = model.compute_objective(ctx, tgt)
    assert torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()
    assert model.subblock_embeddings.grad is not None and model.subblock_embeddings.grad.norm().item() > 1e-4

    disc = model.compute_predictive_discrepancy(ctx, tgt)
    assert disc.shape == (B,)
    assert torch.all(disc >= 0.0)



def test_helmholtz_soliton_jepa_forward_and_discrepancy():
    torch.manual_seed(42)
    B, C_len, S_len, K, D = 4, 32, 16, 3, 8
    enc = HybridTCNEncoder(input_dim=K, latent_dim=D, filters=8, tcn_layers=1)
    model = HelmholtzSolitonJEPAModel(
        context_encoder=enc,
        latent_dim=D,
        predictor_hidden_dim=16,
        predictor_layers=2,
        num_subblocks=4,
        coreset_size=6,
        top_k_prototypes=3,
        null_dim=2,
    )

    ctx = torch.randn(B, C_len, K)
    tgt = torch.randn(B, S_len, K)

    # 1. Forward pass inspection
    zc, zt, zp, eddy_unit, z_sub_pred, z_sub_tgt = model.forward(ctx, tgt)
    assert zc.shape == (B, D)
    assert zt.shape == (B, D)
    assert zp.shape == (B, D)
    assert z_sub_pred.shape == (B, 3, D)
    assert z_sub_tgt.shape == (B, 3, D)

    # Unit scale on eddy multiplier
    assert torch.allclose(eddy_unit.mean(dim=-1), torch.ones(B), atol=1e-4)

    # 2. Loss & backward pass
    loss, metrics = model.compute_objective(ctx, tgt)
    assert torch.isfinite(loss) and loss.item() > 0.0
    assert "loss" in metrics
    loss.backward()
    assert model.vorticity_gate.grad is not None
    assert model.subblock_embeddings.grad is not None

    # 3. Predictive discrepancy without bank
    disc_raw = model.compute_predictive_discrepancy(ctx, tgt, use_alien_bank=False)
    assert disc_raw.shape == (B,)
    assert torch.all(disc_raw >= 0.0)

    # 4. Fit alien bank & compute bank-calibrated discrepancy
    train_wins = np.random.randn(12, C_len + S_len, K).astype(np.float32)
    model.fit_alien_bank(train_wins, context_len=C_len, batch_size=6)
    assert bool(model.bank_fitted.item()) is True

    disc_bank = model.compute_predictive_discrepancy(ctx, tgt, use_alien_bank=True)
    assert disc_bank.shape == (B,)
    assert torch.all(disc_bank >= 0.0)



def test_multi_perspective_224bit_circuit_breaker_panel():
    from src.scoring.circuit_breaker_panel import (
        MultiPerspectiveCircuitBreakerPanel,
        build_224_perspective_dials,
        extract_multihorizon_4subblocks,
        stabilize_whitened_subblocks,
    )

    torch.manual_seed(42)
    np.random.seed(42)
    C_len, S_len, K, D = 32, 16, 3, 8
    enc = HybridTCNEncoder(input_dim=K, latent_dim=D, filters=8, tcn_layers=1)
    model = SpeculativeJEPAModel(
        context_encoder=enc,
        latent_dim=D,
        predictor_hidden_dim=16,
        coreset_size=6,
        top_k_prototypes=3,
        null_dim=2,
    )
    train_wins = np.random.randn(24, C_len + S_len, K).astype(np.float32)
    va_wins = np.random.randn(40, C_len + S_len, K).astype(np.float32)
    te_wins = np.random.randn(40, C_len + S_len, K).astype(np.float32)
    te_wins[20:30] += 6.0  # Inject synthetic fault burst

    model.fit_alien_bank(train_wins, context_len=C_len, batch_size=8)
    va_raw4, va_white4 = extract_multihorizon_4subblocks(model, va_wins, C_len, torch.device("cpu"))
    te_raw4, te_white4 = extract_multihorizon_4subblocks(model, te_wins, C_len, torch.device("cpu"))
    assert va_raw4.shape == (40, 4, D)
    assert te_white4.shape == (40, 4, D)

    va_ew, te_ew = stabilize_whitened_subblocks(va_white4, te_white4, va_raw4, te_raw4, n_cal_w=20)
    dials_va = build_224_perspective_dials(va_ew)
    assert dials_va.shape == (40, 7 * D)
    # Degree-1 scale homogeneity of perspective dial construction: D(c * e) == c * D(e)
    dials_scaled = build_224_perspective_dials(3.5 * va_ew)
    assert np.allclose(dials_scaled, 3.5 * dials_va, atol=1e-5)

    panel = MultiPerspectiveCircuitBreakerPanel(context_size=C_len, suspect_size=S_len)
    bits = panel.calibrate_and_trip_windows(va_ew, te_ew, n_cal_w=20)
    assert set(np.unique(bits["bits224_te"])).issubset({0.0, 1.0})

    for mapping in ("trailing", "smear"):
        dec = panel.evaluate_stream_decisions(bits, len_va=40 + C_len + S_len - 1, len_test=40 + C_len + S_len - 1, mapping=mapping)
        assert dec.test_pred_balanced.shape == dec.test_switch_counts.shape
        assert set(np.unique(dec.test_pred_balanced)).issubset({0, 1})
        assert dec.test_switch_counts[20:30].mean() > dec.cal_switch_counts.mean()


def test_proposition1_fluctuation_gram_dimensions_and_equality_bounds():
    from src.scoring.circuit_breaker_panel import build_224_perspective_dials

    torch.manual_seed(7)
    B, M, D = 5, 4, 8
    e_tilde = torch.randn(B, M, D, dtype=torch.float64)
    e_bar = e_tilde.mean(dim=1, keepdim=True)
    fluct = e_tilde - e_bar  # F(t) in R^{M x D}

    # 1. Gram matrix G = F F^T in S_+^M (not F^T F in R^{D x D}) with null vector 1_M
    ones_m = torch.ones(B, M, 1, dtype=torch.float64)
    assert torch.allclose(torch.bmm(fluct.transpose(1, 2), ones_m), torch.zeros(B, D, 1, dtype=torch.float64), atol=1e-12)
    gram = torch.bmm(fluct, fluct.transpose(1, 2))
    assert gram.shape == (B, M, M)
    assert torch.allclose(torch.bmm(gram, ones_m), torch.zeros(B, M, 1, dtype=torch.float64), atol=1e-12)
    evals = torch.linalg.eigvalsh(gram)
    assert torch.allclose(evals[:, 0], torch.zeros(B, dtype=torch.float64), atol=1e-12)

    # 2. General interior bounds: R_LT(F) in [1 / (M - 1), 1] for gamma > -0.5
    for gamma in (-0.25, 0.5, 1.0):
        r_gen = SolitonReynoldsJEPAModel.lieb_thirring_soliton_ratio(fluct.float(), gamma=gamma)
        assert torch.all(r_gen >= (1.0 / (M - 1)) - 1e-5)
        assert torch.all(r_gen <= 1.0 + 1e-5)

    # 3. Equality at lower bound 1 / (M - 1): isotropic energy across M - 1 zero-sum horizon modes
    helmert = torch.tensor(
        [
            [1.0, -1.0, 0.0, 0.0],
            [1.0, 1.0, -2.0, 0.0],
            [1.0, 1.0, 1.0, -3.0],
        ],
        dtype=torch.float64,
    )  # (M - 1, M), rows orthogonal to 1_M
    helmert = helmert / torch.linalg.norm(helmert, dim=1, keepdim=True)
    q_d, _ = torch.linalg.qr(torch.randn(B, D, M - 1, dtype=torch.float64))  # orthonormal columns in R^D
    f_iso = torch.matmul(helmert.T.unsqueeze(0), q_d.transpose(1, 2))  # (B, M, D)
    assert torch.allclose(f_iso.sum(dim=1), torch.zeros(B, D, dtype=torch.float64), atol=1e-12)
    r_iso = SolitonReynoldsJEPAModel.lieb_thirring_soliton_ratio(f_iso.float(), gamma=1.0)
    assert torch.allclose(r_iso, torch.full((B,), 1.0 / (M - 1)), atol=1e-5)

    # 4. Equality at upper bound 1: rank-one zero-mean fluctuation F = u v^T with u^T 1_M = 0
    u_raw = torch.randn(B, M, 1, dtype=torch.float64)
    u_zero = u_raw - u_raw.mean(dim=1, keepdim=True)
    v_vec = torch.randn(B, 1, D, dtype=torch.float64)
    f_rank1 = torch.bmm(u_zero, v_vec)
    r_rank1 = SolitonReynoldsJEPAModel.lieb_thirring_soliton_ratio(f_rank1.float(), gamma=1.0)
    assert torch.allclose(r_rank1, torch.ones(B), atol=1e-5)

    # 5. Proposition 1(i): coordinate-wise energy orthogonality and endpoint difference bound
    dials = build_224_perspective_dials(e_tilde.numpy().astype(np.float32))
    d_pos = dials[:, : 4 * D].reshape(B, M, D)
    d_drift = dials[:, 4 * D : 5 * D]
    d_fluc = dials[:, 5 * D : 6 * D]
    d_accel = dials[:, 6 * D : 7 * D]
    assert np.allclose(np.mean(d_pos**2, axis=1), d_drift**2 + d_fluc**2, atol=1e-5)
    assert np.all(d_accel <= np.sqrt(2.0 * M) * d_fluc + 1e-5)


def test_disjoint_checkpoint_selection_split_no_calibration_overlap():
    import pytest
    from scripts.run_tier2_exhaustive_benchmark import W, split_train_checkpoint_block

    raw_nominal = np.arange(4000 * 3, dtype=np.float32).reshape(4000, 3)
    n_val = int(len(raw_nominal) * 0.35)
    tr_full, va_full = raw_nominal[:-n_val], raw_nominal[-n_val:]
    tr_fit, select_blk = split_train_checkpoint_block(tr_full, len(raw_nominal))

    assert len(tr_fit) + len(select_blk) == len(tr_full)
    assert len(select_blk) >= W + 64
    assert np.intersect1d(tr_fit[:, 0], select_blk[:, 0]).size == 0
    assert np.intersect1d(select_blk[:, 0], va_full[:, 0]).size == 0

    with pytest.raises(ValueError, match="Insufficient disjoint training history"):
        split_train_checkpoint_block(raw_nominal[: W + 70], 1000)

