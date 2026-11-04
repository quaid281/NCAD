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
