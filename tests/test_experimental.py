"""Tests for isolated novelty research modules in src/experimental."""

import numpy as np
import pytest
import torch

from src.config import CSMConfig
from src.engine.builder import build_ts_jepa_model
from src.experimental.causal_counterfactual import (
    CounterfactualConfig,
    CounterfactualContextResult,
    CounterfactualContextSubstitutor,
)
from src.experimental.hopfield_context import (
    HopfieldContextConfig,
    HopfieldContextMemory,
    HopfieldRetrieval,
)
from src.experimental.mamba_ssm_encoder import (
    ExperimentalSSMContextEncoder,
    SelectiveStateSpaceBlock,
)
from src.experimental.memoryless_context import (
    MemorylessContextConfig,
    MemorylessContextResult,
    MemorylessContextSubstitutor,
)
from src.experimental.weight_space_programming import (
    ORTHOGONAL_LEGENDRE_4BASIS,
    BlindWSPReport,
    LegendreHorizonBasisParametrization,
    PhaseSynthesisResult,
    WSPConfig,
    WSPDiagnosticReport,
    WSPSurgicalResult,
    apply_blind_wsp_preconditioning,
    apply_surgical_wsp_quench,
    apply_unified_wsp_soliton_program,
    bake_structural_legendre_phase_basis,
    bind_structural_legendre_phase_basis,
    blind_reverse_engineer_weight_space,
    evaluate_blind_dial_reconstruction,
    inspect_soliton_reynolds_weight_space,
    synthesize_subblock_phase_embeddings,
)
from src.scoring.circuit_breaker_panel import extract_multihorizon_4subblocks


def test_hopfield_context_memory():
    rng = np.random.default_rng(42)
    normal_context = rng.normal(0.0, 1.0, size=(64, 16)).astype(np.float32)
    memory = HopfieldContextMemory(HopfieldContextConfig(inverse_temperature=10.0)).fit(normal_context)

    query = normal_context[0] + rng.normal(0.0, 0.1, size=(16,)).astype(np.float32)
    result = memory.retrieve(query)

    assert isinstance(result, HopfieldRetrieval)
    assert result.retrieved_embedding.shape == (16,)
    assert result.distance >= 0.0
    assert 0.0 <= result.confidence <= 1.0


def test_causal_counterfactual_substitutor():
    rng = np.random.default_rng(42)
    contexts = rng.normal(0.0, 1.0, size=(64, 16)).astype(np.float32)
    fulls = contexts * 1.5 + rng.normal(0.0, 0.05, size=contexts.shape).astype(np.float32)

    substitutor = CounterfactualContextSubstitutor(CounterfactualConfig(ridge_alpha=1e-3)).fit(contexts, fulls)

    result = substitutor.score(
        full_embedding=fulls[0],
        observed_context_embedding=contexts[0] + 0.5,
        intervened_context_embedding=contexts[0],
    )

    assert isinstance(result, CounterfactualContextResult)
    assert result.final_score >= 0.0
    assert result.expected_full_embedding.shape == (16,)


def test_memoryless_context_substitutor():
    rng = np.random.default_rng(42)
    normal_ctx = rng.normal(0.0, 1.0, size=(32, 16)).astype(np.float32)
    contam_ctx = normal_ctx + rng.normal(0.0, 0.5, size=normal_ctx.shape).astype(np.float32)
    full_emb = normal_ctx + rng.normal(0.0, 0.05, size=normal_ctx.shape).astype(np.float32)

    config = MemorylessContextConfig(epochs=3, hidden_dim=16, batch_size=16, device="cpu")
    substitutor = MemorylessContextSubstitutor(config).fit(normal_ctx, contam_ctx, full_emb)

    result = substitutor.score(full_emb[0], contam_ctx[0])
    assert isinstance(result, MemorylessContextResult)
    assert 0.0 <= result.contamination_probability <= 1.0


def test_experimental_selective_ssm_encoder():
    encoder = ExperimentalSSMContextEncoder(input_dim=4, latent_dim=16, hidden_dim=32, layers=2)
    x = torch.randn(4, 32, 4)
    latent = encoder(x)

    assert latent.shape == (4, 16)
    assert torch.isfinite(latent).all()


def test_wsp_diagnostic_and_surgical_invariants():
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    ctx_len, susp_len = 64, 16
    win_len = ctx_len + susp_len
    latent_dim = 16

    cfg = CSMConfig(
        model_type="soliton_reynolds_jepa",
        context_size=ctx_len,
        suspect_size=susp_len,
        latent_dim=latent_dim,
    )
    model = build_ts_jepa_model(cfg, input_dim=3, device=torch.device("cpu"))

    tr_w = rng.normal(0.0, 1.0, size=(48, win_len, 3)).astype(np.float32)
    model.fit_alien_bank(tr_w, context_len=ctx_len, batch_size=24)

    # Inject a non-stationary regime shift into calibration windows to trigger WSP surgical quench
    cal_w = rng.normal(0.0, 1.0, size=(80, win_len, 3)).astype(np.float32)
    cal_w[40:, :, 0] += np.linspace(0.5, 4.5, 40, dtype=np.float32)[:, None]
    cal_w[40:, :, 1] *= 2.8

    wsp_cfg = WSPConfig(context_len=ctx_len, suspect_len=susp_len, candidate_k_counts=(4, 2))
    report, _ = inspect_soliton_reynolds_weight_space(model, tr_w, cal_w, wsp_cfg)

    assert isinstance(report, WSPDiagnosticReport)
    assert len(report.profiles) == latent_dim
    assert len(report.ranked_coords) == latent_dim
    md_text = report.to_llm_markdown(top_n=4)
    assert "Weight-Space Programming (WSP) Diagnostic Report" in md_text
    assert "| Rank | Coord `c` |" in md_text

    res = apply_surgical_wsp_quench(model, tr_w, cal_w, wsp_cfg)
    assert isinstance(res, WSPSurgicalResult)
    # Verify zero-gradient execution
    assert all(p.grad is None for p in model.parameters())

    # Verify SPSD cone preservation on precision_matrix
    prec = model.precision_matrix.detach().cpu().numpy()
    assert np.allclose(prec, prec.T, atol=1e-5)
    evals = np.linalg.eigvalsh(0.5 * (prec + prec.T))
    assert float(evals.min()) > 0.0

    if res.mode.startswith("surgical_k"):
        assert 0.0 < res.pct_predictor_edited < 1.0
        assert len(res.rogue_coords) in (2, 4)
        assert res.post_k_pref >= res.baseline_k_pref * wsp_cfg.gate_retention_ratio - 1e-5
        assert res.post_bi_fpr <= res.baseline_bi_fpr + 1e-4


def test_blind_wsp_reverse_engineering_and_preconditioning():
    torch.manual_seed(123)
    rng = np.random.default_rng(123)
    ctx_len, susp_len = 64, 16
    win_len = ctx_len + susp_len
    latent_dim = 16

    cfg = CSMConfig(
        model_type="soliton_reynolds_jepa",
        context_size=ctx_len,
        suspect_size=susp_len,
        latent_dim=latent_dim,
    )
    model = build_ts_jepa_model(cfg, input_dim=3, device=torch.device("cpu"))

    tr_w = rng.normal(0.0, 1.0, size=(96, win_len, 3)).astype(np.float32)
    model.fit_alien_bank(tr_w, context_len=ctx_len, batch_size=32)

    # 1. Data-free reverse engineering from weights alone
    report = blind_reverse_engineer_weight_space(model)
    assert isinstance(report, BlindWSPReport)
    assert report.dynamical_regime in ("low_rank_constrained", "rotational_oscillatory", "soliton_relaxation")
    assert report.vorticity_ratio > 0.0
    assert report.effective_rank > 0.0
    assert report.signal_mode_count > 0
    assert len(report.profiles) == latent_dim
    assert report.reconstructed_covariance.shape == (4 * latent_dim, 4 * latent_dim)

    md_text = report.to_llm_markdown(top_n=4)
    assert "Blind Weight-Space Reverse-Engineering Report" in md_text
    assert "| Fragility Rank | Coord `c` |" in md_text

    # 2. Verify Branch-Filtered Exact Resolvent Inverse accurately reconstructs 224-dial coordinate roles
    tr_raw4, _ = extract_multihorizon_4subblocks(model, tr_w, ctx_len, torch.device("cpu"))
    va_w = rng.normal(0.0, 1.0, size=(32, win_len, 3)).astype(np.float32)
    _, va_white4 = extract_multihorizon_4subblocks(model, va_w, ctx_len, torch.device("cpu"))
    eval_rec = evaluate_blind_dial_reconstruction(report, tr_raw4, va_white4)

    assert eval_rec["role_accuracy"] >= 0.65
    assert eval_rec["drift_rank_corr"] >= 0.70
    assert eval_rec["fluct_rank_corr"] >= 0.70
    assert eval_rec["accel_rank_corr"] >= 0.65
    assert -1.0 <= eval_rec["fragility_rank_corr"] <= 1.0

    # 3. Verify 100% data-free surgical pre-conditioning preserves SPSD cone & zero gradients
    res_blind = apply_blind_wsp_preconditioning(model, k_edit=2, min_eff_rank_abstain=1000.0)
    assert res_blind["mode"] == "blind_k2"
    assert len(res_blind["rogue_coords"]) == 2
    assert all(p.grad is None for p in model.parameters())

    prec = model.precision_matrix.detach().cpu().numpy()
    assert np.allclose(prec, prec.T, atol=1e-5)
    evals = np.linalg.eigvalsh(0.5 * (prec + prec.T))
    assert float(evals.min()) > 0.0


def test_direct_legendre_subblock_phase_synthesis():
    # 1. Verify exact O(4) orthonormality of the 4-point Discrete Legendre Horizon Basis
    gram = ORTHOGONAL_LEGENDRE_4BASIS.T @ ORTHOGONAL_LEGENDRE_4BASIS
    assert np.allclose(gram, np.eye(4), atol=1e-12)

    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    ctx_len, susp_len = 64, 16
    win_len = ctx_len + susp_len
    latent_dim = 16

    cfg = CSMConfig(
        model_type="soliton_reynolds_jepa",
        context_size=ctx_len,
        suspect_size=susp_len,
        latent_dim=latent_dim,
    )
    model = build_ts_jepa_model(cfg, input_dim=3, device=torch.device("cpu"))
    tr_w = rng.normal(0.0, 1.0, size=(48, win_len, 3)).astype(np.float32)
    cal_w = rng.normal(0.0, 1.0, size=(48, win_len, 3)).astype(np.float32)
    cal_w[24:, :, 0] += 1.5
    model.fit_alien_bank(tr_w, context_len=ctx_len, batch_size=24)

    # Inject artificial saw-tooth jitter (mode 3) into subblock_embeddings
    phi3 = torch.from_numpy(ORTHOGONAL_LEGENDRE_4BASIS[:, 3]).float()
    model.subblock_embeddings.data.add_(0.25 * torch.outer(phi3, torch.ones(latent_dim)))

    # 2. Test zero-shot scratch synthesis from E_0 = 0
    res_scratch = synthesize_subblock_phase_embeddings(
        model, tr_w, cal_w, context_len=ctx_len, suspect_len=susp_len, from_scratch=True
    )
    assert isinstance(res_scratch, PhaseSynthesisResult)
    assert res_scratch.mode == "scratch_legendre"
    assert res_scratch.sawtooth_energy_pct_before > 50.0
    assert res_scratch.sawtooth_energy_pct_after < 1e-4
    assert res_scratch.embedding_norm_after > 0.0
    assert res_scratch.synthesized_embeddings.shape == (4, latent_dim)
    assert all(p.grad is None for p in model.parameters())

    # 3. Test surgical 128-parameter Legendre refinement
    model.subblock_embeddings.data.add_(0.20 * torch.outer(phi3, torch.ones(latent_dim)))
    res_surg = synthesize_subblock_phase_embeddings(
        model, tr_w, cal_w, context_len=ctx_len, suspect_len=susp_len, from_scratch=False
    )
    assert isinstance(res_surg, PhaseSynthesisResult)
    assert res_surg.mode in ("phase_abstain", "legendre_gentle", "legendre_moderate", "legendre_strong")
    if res_surg.mode != "phase_abstain":
        assert res_surg.sawtooth_energy_pct_after < 1e-4


def test_structural_legendre_phase_basis_and_unified_wsp():
    torch.manual_seed(99)
    rng = np.random.default_rng(99)
    ctx_len, susp_len = 64, 16
    win_len = ctx_len + susp_len
    latent_dim = 16

    cfg = CSMConfig(
        model_type="soliton_reynolds_jepa",
        context_size=ctx_len,
        suspect_size=susp_len,
        latent_dim=latent_dim,
    )
    model = build_ts_jepa_model(cfg, input_dim=3, device=torch.device("cpu"))
    bind_structural_legendre_phase_basis(model)
    assert torch.nn.utils.parametrize.is_parametrized(model, "subblock_embeddings")

    # Run a gradient step and verify saw-tooth mode (phi_3) and row-mean are identically zero
    tr_w = rng.normal(0.0, 1.0, size=(32, win_len, 3)).astype(np.float32)
    b = torch.from_numpy(tr_w[:16])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss, _ = model.compute_objective(b[:, :ctx_len], b[:, ctx_len:], cfg)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    phi3 = torch.from_numpy(ORTHOGONAL_LEGENDRE_4BASIS[:, 3]).float()
    emb_live = model.subblock_embeddings.detach()
    assert float((phi3 @ emb_live).norm().item()) < 1e-6
    assert float(emb_live.mean(dim=-1).abs().max().item()) < 1e-6

    bake_structural_legendre_phase_basis(model)
    assert not torch.nn.utils.parametrize.is_parametrized(model, "subblock_embeddings")
    assert float((phi3 @ model.subblock_embeddings.detach()).norm().item()) < 1e-6

    model.fit_alien_bank(tr_w, context_len=ctx_len, batch_size=16)
    cal_w = rng.normal(0.0, 1.0, size=(48, win_len, 3)).astype(np.float32)
    cal_w[24:, :, 0] += 2.0
    wsp_cfg = WSPConfig(context_len=ctx_len, suspect_len=susp_len, candidate_k_counts=(4, 2))
    res_uni = apply_unified_wsp_soliton_program(model, tr_w, cal_w, wsp_cfg)
    assert isinstance(res_uni, dict)
    assert res_uni["sawtooth_before"] < 1e-4
    assert res_uni["post_bi_fpr"] <= res_uni["base_bi_fpr"] + 1e-4

