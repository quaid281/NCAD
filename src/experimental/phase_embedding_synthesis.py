"""Experiment 3: Direct Weight Synthesis of Sub-Block Phase Embeddings in the Orthogonal Legendre Basis.

Synthesizes the $4 \\times D$ `subblock_embeddings` parameter tensor ($128$ floats, $0.08\\%$ of
`SolitonReynoldsJEPAModel` parameters) directly in closed form without backpropagation:
  1. Decomposes the 4 sequential prefix horizons ($m \\in \\{0, 1, 2, 3\\}$, corresponding to
     $16, 32, 48, 64$ steps) into the exact 4-point Orthogonal Discrete Legendre Basis
     $\\mathbf{\\Phi} = [\\boldsymbol{\\phi}_0, \\boldsymbol{\\phi}_1, \\boldsymbol{\\phi}_2, \\boldsymbol{\\phi}_3] \\in \\mathrm{O}(4)$:
       - $\\boldsymbol{\\phi}_0$: Slow-Drift mode ($[1, 1, 1, 1]^\\top / 2$)
       - $\\boldsymbol{\\phi}_1$: Linear Horizon Velocity / Right-Edge Tilt ($[-3, -1, +1, +3]^\\top / \\sqrt{20}$)
       - $\\boldsymbol{\\phi}_2$: Quadratic Soliton Dispersion ($[+1, -1, -1, +1]^\\top / 2$)
       - $\\boldsymbol{\\phi}_3$: Parasitic Saw-Tooth Jitter ($[-1, +3, -3, +1]^\\top / \\sqrt{20}$)
  2. Annihilates the unphysical saw-tooth mode $\\boldsymbol{\\phi}_3$ ($\\mathbf{\\Phi}_{:,3}^\\top \\mathbf{E}^* = \\mathbf{0}$)
     that SGD inadvertently excites across the architectural seam between `sub_predictor` ($m=0,1,2$)
     and `predictor` ($m=3$).
  3. Solves for the smooth Legendre phase coefficients $\\mathbf{C}^* \\in \\mathbb{R}^{3 \\times D}$ via
     cascaded tangent-space Gauss-Newton inversion on the LayerNorm sphere $\\mathbb{S}^{D-1}$,
     supporting both:
       - `from_scratch=True`: Zero-shot weight synthesis from $\\mathbf{E}_0 = \\mathbf{0}$ with exact
         resolvent Mahalanobis synchronization.
       - `from_scratch=False`: Surgical 128-parameter phase refinement gated by internal bi-directional
         calibration split verification.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import resolvent_precision, whiten_residuals
from src.models.jepa.alien_jepas import _encode_with_multidepth_taps

ORTHOGONAL_LEGENDRE_4BASIS = np.array(
    [
        [0.5, -3.0 / np.sqrt(20.0), 0.5, -1.0 / np.sqrt(20.0)],
        [0.5, -1.0 / np.sqrt(20.0), -0.5, 3.0 / np.sqrt(20.0)],
        [0.5, 1.0 / np.sqrt(20.0), -0.5, -3.0 / np.sqrt(20.0)],
        [0.5, 3.0 / np.sqrt(20.0), 0.5, 1.0 / np.sqrt(20.0)],
    ],
    dtype=np.float64,
)


@dataclass
class PhaseSynthesisResult:
    """Summary of Direct Weight Synthesis of `subblock_embeddings` (Experiment 3)."""

    mode: str
    sawtooth_energy_pct_before: float
    sawtooth_energy_pct_after: float
    embedding_norm_before: float
    embedding_norm_after: float
    mode_energies_before: tuple[float, float, float, float]
    mode_energies_after: tuple[float, float, float, float]
    synthesized_embeddings: np.ndarray


@torch.no_grad()
def _extract_subblock_latents(
    model: nn.Module,
    windows: np.ndarray,
    context_len: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract `(z_ctx [N, D], z_sub_tgt [N, M, D])` once to enable sub-millisecond phase synthesis."""
    model.eval()
    dev = next(model.parameters()).device
    M, D = int(model.num_subblocks), int(model.latent_dim)
    z_ctx_list, z_sub_tgt_list = [], []
    for i in range(0, len(windows), 128):
        ch = torch.from_numpy(np.ascontiguousarray(windows[i : i + 128])).float().to(dev)
        ctx_win, tgt_win = ch[:, :context_len], ch[:, context_len:]
        z_ctx = _encode_with_multidepth_taps(model.context_encoder, ctx_win, model.context_tap_head)
        s_len = tgt_win.shape[1]
        t_list = [
            _encode_with_multidepth_taps(
                model.target_encoder,
                tgt_win[:, : (s_len if m == M - 1 else max(((m + 1) * s_len) // M, 2)), :],
                model.target_tap_head,
            )
            for m in range(M)
        ]
        z_ctx_list.append(z_ctx)
        z_sub_tgt_list.append(torch.stack(t_list, dim=1))
    return torch.cat(z_ctx_list, dim=0), torch.cat(z_sub_tgt_list, dim=0)


@torch.no_grad()
def _forward_predictors_with_phase(
    model: nn.Module,
    z_ctx: torch.Tensor,
    phase_emb: torch.Tensor,
) -> torch.Tensor:
    """Unroll `sub_predictor` (m=0..M-2) and `predictor` (m=M-1) for a candidate `phase_emb`."""
    M, D = int(model.num_subblocks), int(model.latent_dim)
    d = (D,)
    horizons = []
    z_prev = z_ctx
    for m in range(M - 1):
        z_prev = F.layer_norm(model.sub_predictor(z_prev + phase_emb[m].unsqueeze(0)), d)
        horizons.append(z_prev)
    z_pred = F.layer_norm(model.predictor(z_ctx + phase_emb[-1].unsqueeze(0)), d)
    horizons.append(z_pred)
    return torch.stack(horizons, dim=1)


@torch.no_grad()
def _sync_mahalanobis_buffers(
    model: nn.Module,
    z_ctx_tr: torch.Tensor,
    z_tgt_tr: torch.Tensor,
    phase_emb: torch.Tensor,
) -> None:
    """Synchronize `residual_mean`, `precision_matrix`, and `r_lt_med/mad` for `phase_emb`."""
    M, D = int(model.num_subblocks), int(model.latent_dim)
    z_pred_tr = _forward_predictors_with_phase(model, z_ctx_tr, phase_emb)
    res_flat = (z_tgt_tr - z_pred_tr).reshape(len(z_ctx_tr), M * D)
    mean_r = res_flat.mean(dim=0)
    centered = res_flat - mean_r.unsqueeze(0)
    cov = (centered.T @ centered) / max(len(res_flat) - 1, 1)
    prec = resolvent_precision(cov, u=1e-3)
    model.residual_mean.data.copy_(mean_r)
    model.precision_matrix.data.copy_(prec)
    ew = whiten_residuals(centered, prec).reshape(len(z_ctx_tr), M, D)
    r_lts = model.lieb_thirring_soliton_ratio(ew - ew.mean(dim=1, keepdim=True), gamma=model.lt_gamma)
    med = torch.median(r_lts)
    model.r_lt_med.copy_(med)
    model.r_lt_mad.copy_(torch.median(torch.abs(r_lts - med)).clamp(min=1e-3))


@torch.no_grad()
def synthesize_subblock_phase_embeddings(
    model: nn.Module,
    tr_w: np.ndarray,
    cal_w: np.ndarray,
    context_len: int = 256,
    suspect_len: int = 64,
    from_scratch: bool = False,
    tr_latents: tuple[torch.Tensor, torch.Tensor] | None = None,
    cal_latents: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> PhaseSynthesisResult:
    """Directly synthesize `model.subblock_embeddings` in the 4-point Orthogonal Legendre Basis."""
    from src.experimental.weight_space_programming import (
        _eval_cal_gates_and_internal_split,
        _linearized_mlp_jacobian,
    )

    model.eval()
    dev = next(model.parameters()).device
    M, D = int(model.num_subblocks), int(model.latent_dim)
    if M != 4:
        raise ValueError(f"Legendre 4-subblock synthesis requires num_subblocks=4, got {M}.")

    phi_t = torch.from_numpy(ORTHOGONAL_LEGENDRE_4BASIS).to(dev, dtype=torch.float32)
    z_ctx_tr, z_tgt_tr = tr_latents if tr_latents is not None else _extract_subblock_latents(model, tr_w, context_len)
    z_ctx_cal, z_tgt_cal = cal_latents if cal_latents is not None else _extract_subblock_latents(model, cal_w, context_len)

    e_orig = model.subblock_embeddings.detach().clone()
    orig_mean = model.residual_mean.detach().clone()
    orig_prec = model.precision_matrix.detach().clone()
    orig_rlt_med = model.r_lt_med.detach().clone()
    orig_rlt_mad = model.r_lt_mad.detach().clone()

    c_orig = phi_t.T @ e_orig
    modes_before = (c_orig**2).sum(dim=1).cpu().numpy()
    saw_before = float(100.0 * modes_before[3] / (modes_before.sum() + 1e-12))

    sd = {k: v.detach().cpu().float().numpy() for k, v in model.state_dict().items()}
    a_sub = torch.from_numpy(np.eye(D, dtype=np.float64) + _linearized_mlp_jacobian(sd, "sub_predictor")).to(
        dev, dtype=torch.float32
    )
    a_pred = torch.from_numpy(np.eye(D, dtype=np.float64) + _linearized_mlp_jacobian(sd, "predictor")).to(
        dev, dtype=torch.float32
    )

    def _run_gauss_newton_legendre(e_init: torch.Tensor, ref_e: torch.Tensor, trust_r: float, n_iter: int, scratch_mode: bool) -> torch.Tensor:
        e_curr = e_init.clone()
        eye_d = torch.eye(D, device=dev)
        ones_d = torch.ones((D, D), device=dev) / float(D)
        for _ in range(n_iter):
            z_pred_tr = _forward_predictors_with_phase(model, z_ctx_tr, e_curr)
            z_pred_cal = _forward_predictors_with_phase(model, z_ctx_cal, e_curr)
            res_tr = (z_tgt_tr - z_pred_tr).mean(dim=0)
            res_cal = (z_tgt_cal - z_pred_cal).mean(dim=0)
            if scratch_mode:
                target_dz = 0.55 * res_tr + 0.45 * res_cal
                reg_lam = 0.85
            else:
                cal_shift = res_cal - res_tr
                saw_res = torch.outer(phi_t[:, 3], phi_t[:, 3] @ res_cal)
                target_dz = 0.65 * cal_shift + 0.35 * saw_res
                reg_lam = 0.75

            delta_e = torch.zeros_like(e_curr)
            for m in range(M):
                a_m = a_sub if m < M - 1 else a_pred
                z_bar = (z_pred_tr if scratch_mode else z_pred_cal)[:, m, :].mean(dim=0)
                p_tan = eye_d - ones_d - (1.0 / float(D)) * torch.outer(z_bar, z_bar)
                j_m = p_tan @ a_m
                h_m = j_m.T @ j_m + reg_lam * eye_d
                de_m = torch.linalg.solve(h_m, j_m.T @ target_dz[m])
                delta_e[m] = de_m - de_m.mean()

            dc = phi_t.T @ delta_e
            dc[3, :] = 0.0
            e_curr = e_curr + 0.50 * (phi_t @ dc)
            diff_e = (e_curr - ref_e).clamp(-trust_r, trust_r)
            cf = phi_t.T @ (ref_e + diff_e)
            cf[3, :] = 0.0
            e_curr = phi_t @ cf
        return e_curr

    if from_scratch:
        e_zero = torch.zeros_like(e_orig)
        e_syn = _run_gauss_newton_legendre(e_zero, e_zero, trust_r=0.12, n_iter=4, scratch_mode=True)
        model.subblock_embeddings.data.copy_(e_syn)
        _sync_mahalanobis_buffers(model, z_ctx_tr, z_tgt_tr, e_syn)
        c_after = phi_t.T @ e_syn
        modes_after = (c_after**2).sum(dim=1).cpu().numpy()
        saw_after = float(100.0 * modes_after[3] / (modes_after.sum() + 1e-12))
        return PhaseSynthesisResult(
            mode="scratch_legendre",
            sawtooth_energy_pct_before=saw_before,
            sawtooth_energy_pct_after=saw_after,
            embedding_norm_before=float(e_orig.norm().item()),
            embedding_norm_after=float(e_syn.norm().item()),
            mode_energies_before=tuple(float(x) for x in modes_before),
            mode_energies_after=tuple(float(x) for x in modes_after),
            synthesized_embeddings=e_syn.detach().cpu().numpy(),
        )

    base_eval = _eval_cal_gates_and_internal_split(
        model, cal_w, context_len, suspect_len, cal_latents=(z_ctx_cal, z_tgt_cal)
    )
    if base_eval["fpr_f"] == 0.0 and base_eval["bi_fpr"] == 0.0 and base_eval["bi_excess"] < 0.05:
        return PhaseSynthesisResult(
            mode="phase_abstain",
            sawtooth_energy_pct_before=saw_before,
            sawtooth_energy_pct_after=saw_before,
            embedding_norm_before=float(e_orig.norm().item()),
            embedding_norm_after=float(e_orig.norm().item()),
            mode_energies_before=tuple(float(x) for x in modes_before),
            mode_energies_after=tuple(float(x) for x in modes_before),
            synthesized_embeddings=e_orig.detach().cpu().numpy(),
        )

    c_clean = c_orig.clone()
    c_clean[3, :] = 0.0
    e_proj = phi_t @ c_clean

    best_e = e_orig.clone()
    best_eval = base_eval
    best_mode = "phase_abstain"

    for trust_r, mode_tag in [(0.06, "legendre_gentle"), (0.12, "legendre_moderate"), (0.20, "legendre_strong")]:
        e_cand = _run_gauss_newton_legendre(e_proj, e_proj, trust_r=trust_r, n_iter=3, scratch_mode=False)
        model.subblock_embeddings.data.copy_(e_cand)
        cand_eval = _eval_cal_gates_and_internal_split(
            model, cal_w, context_len, suspect_len, cal_latents=(z_ctx_cal, z_tgt_cal)
        )
        gate_safe = (
            cand_eval["k_pref_tr"] >= base_eval["k_pref_tr"] * 0.94
            and cand_eval["k_128_tr"] >= base_eval["k_128_tr"] * 0.94
            and cand_eval["fpr_f"] <= base_eval["fpr_f"] + 1e-4
        )
        better = cand_eval["bi_fpr"] < best_eval["bi_fpr"] - 1e-4 or (
            abs(cand_eval["bi_fpr"] - best_eval["bi_fpr"]) <= 1e-4
            and cand_eval["bi_excess"] < best_eval["bi_excess"] - 0.005
        )
        if gate_safe and better:
            best_e = e_cand.clone()
            best_eval = cand_eval
            best_mode = mode_tag

    model.subblock_embeddings.data.copy_(best_e)
    model.residual_mean.data.copy_(orig_mean)
    model.precision_matrix.data.copy_(orig_prec)
    model.r_lt_med.copy_(orig_rlt_med)
    model.r_lt_mad.copy_(orig_rlt_mad)

    c_after = phi_t.T @ best_e
    modes_after = (c_after**2).sum(dim=1).cpu().numpy()
    saw_after = float(100.0 * modes_after[3] / (modes_after.sum() + 1e-12))

    return PhaseSynthesisResult(
        mode=best_mode,
        sawtooth_energy_pct_before=saw_before,
        sawtooth_energy_pct_after=saw_after,
        embedding_norm_before=float(e_orig.norm().item()),
        embedding_norm_after=float(best_e.norm().item()),
        mode_energies_before=tuple(float(x) for x in modes_before),
        mode_energies_after=tuple(float(x) for x in modes_after),
        synthesized_embeddings=best_e.detach().cpu().numpy(),
    )


class LegendreHorizonBasisParametrization(nn.Module):
    """Experiment 4: Structural Orthogonal Legendre Horizon Basis Parameterization.

    Projects `subblock_embeddings` ($4 \\times D$) onto the 3 physical Discrete Legendre
    Horizon modes $\\mathrm{span}(\\boldsymbol{\\phi}_0, \\boldsymbol{\\phi}_1, \\boldsymbol{\\phi}_2) = \\boldsymbol{\\phi}_3^\\perp$
    and the LayerNorm tangent hyperplane $\\mathbf{1}_D^\\perp$ on every forward pass.
    Enforces $\\mathbf{\\Phi}_{:,3}^\\top \\mathbf{E} \\equiv \\mathbf{0}$ (`0.0%` parasitic saw-tooth
    energy) structurally throughout training with zero auxiliary loss penalties.
    """

    def __init__(self) -> None:
        super().__init__()
        phi = torch.from_numpy(ORTHOGONAL_LEGENDRE_4BASIS[:, :3]).float()
        p_lp = phi @ phi.T
        self.register_buffer("lowpass_projector", p_lp)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        w_centered = weight - weight.mean(dim=-1, keepdim=True)
        return self.lowpass_projector.to(weight.dtype) @ w_centered

    def right_inverse(self, value: torch.Tensor) -> torch.Tensor:
        v_centered = value - value.mean(dim=-1, keepdim=True)
        return self.lowpass_projector.to(value.dtype) @ v_centered


def bind_structural_legendre_phase_basis(model: nn.Module) -> nn.Module:
    """Register `LegendreHorizonBasisParametrization` on `model.subblock_embeddings` prior to training."""
    if int(getattr(model, "num_subblocks", 0)) != 4:
        raise ValueError("Structural 4-point Legendre horizon basis requires num_subblocks=4.")
    if not torch.nn.utils.parametrize.is_parametrized(model, "subblock_embeddings"):
        param = LegendreHorizonBasisParametrization().to(next(model.parameters()).device)
        with torch.no_grad():
            model.subblock_embeddings.data.copy_(param(model.subblock_embeddings.data))
        torch.nn.utils.parametrize.register_parametrization(model, "subblock_embeddings", param)
    return model


def bake_structural_legendre_phase_basis(model: nn.Module) -> nn.Module:
    """Bake the structural Legendre parameterization into a standard `nn.Parameter(4, D)` tensor."""
    if torch.nn.utils.parametrize.is_parametrized(model, "subblock_embeddings"):
        torch.nn.utils.parametrize.remove_parametrizations(
            model, "subblock_embeddings", leave_parametrized=True
        )
    return model


@torch.no_grad()
def apply_unified_wsp_soliton_program(
    model: nn.Module,
    tr_w: np.ndarray,
    cal_w: np.ndarray,
    wsp_cfg=None,
    tr_latents: tuple[torch.Tensor, torch.Tensor] | None = None,
    cal_latents: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict:
    """Unified WSP-Soliton-Reynolds 224-Dial Post-Training Program (Experiments 1 + 2 + 3 + 4).

    Evaluates candidate weight-space programs in closed form on the internal bi-directional
    calibration split (without touching held-out nominal or test windows):
      1. Baseline trained `SolitonReynoldsJEPAModel` (optionally trained with structural Legendre basis).
      2. Surgical 128-parameter Legendre phase refinement (`synthesize_subblock_phase_embeddings`).
      3. Calibration-guided surgical WSP quenching (`apply_surgical_wsp_quench`).
      4. Combined Legendre phase refinement + surgical WSP quenching.
    Restores the candidate with the lowest internal calibration bi-directional FPR and excess
    subject to strict training quorum gate retention.
    """
    import copy
    from src.experimental.weight_space_programming import (
        WSPConfig,
        _eval_cal_gates_and_internal_split,
        apply_surgical_wsp_quench,
    )

    bake_structural_legendre_phase_basis(model)
    cfg = wsp_cfg or WSPConfig()
    C, S = cfg.context_len, cfg.suspect_len
    z_ctx_tr, z_tgt_tr = tr_latents if tr_latents is not None else _extract_subblock_latents(model, tr_w, C)
    z_ctx_cal, z_tgt_cal = cal_latents if cal_latents is not None else _extract_subblock_latents(model, cal_w, C)

    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    base_eval = _eval_cal_gates_and_internal_split(model, cal_w, C, S, cal_latents=(z_ctx_cal, z_tgt_cal))

    best_state = base_state
    best_eval = base_eval
    best_mode = "unified_abstain"

    # Candidate A: Surgical 128-Parameter Legendre Phase Refinement
    m_phase = copy.deepcopy(model)
    phase_res = synthesize_subblock_phase_embeddings(
        m_phase, tr_w, cal_w, C, S, from_scratch=False,
        tr_latents=(z_ctx_tr, z_tgt_tr), cal_latents=(z_ctx_cal, z_tgt_cal),
    )
    eval_phase = _eval_cal_gates_and_internal_split(m_phase, cal_w, C, S, cal_latents=(z_ctx_cal, z_tgt_cal))

    # Candidate B: Calibration-Guided Surgical WSP Quench
    m_surg = copy.deepcopy(model)
    surg_res = apply_surgical_wsp_quench(m_surg, tr_w, cal_w, cfg)
    eval_surg = _eval_cal_gates_and_internal_split(m_surg, cal_w, C, S, cal_latents=(z_ctx_cal, z_tgt_cal))

    # Candidate C: Cascaded Phase Refinement -> Surgical WSP Quench
    m_both = copy.deepcopy(m_phase)
    both_res = apply_surgical_wsp_quench(m_both, tr_w, cal_w, cfg)
    eval_both = _eval_cal_gates_and_internal_split(m_both, cal_w, C, S, cal_latents=(z_ctx_cal, z_tgt_cal))

    candidates = [
        (f"phase_{phase_res.mode}", m_phase, eval_phase),
        (f"wsp_{surg_res.mode}", m_surg, eval_surg),
        (f"cascaded_{phase_res.mode}_{both_res.mode}", m_both, eval_both),
    ]
    for tag, m_cand, ev in candidates:
        gate_ok = (
            ev["k_pref_tr"] >= base_eval["k_pref_tr"] * cfg.gate_retention_ratio
            and ev["k_128_tr"] >= base_eval["k_128_tr"] * cfg.gate_retention_ratio
            and ev["fpr_f"] <= base_eval["fpr_f"] + 1e-4
        )
        better = ev["bi_fpr"] < best_eval["bi_fpr"] - 1e-4 or (
            abs(ev["bi_fpr"] - best_eval["bi_fpr"]) <= 1e-4
            and ev["bi_excess"] < best_eval["bi_excess"] - 0.005
        )
        if gate_ok and better:
            best_eval = ev
            best_mode = tag
            best_state = {k: v.detach().clone() for k, v in m_cand.state_dict().items()}

    model.load_state_dict(best_state)
    return {
        "mode": best_mode,
        "phase_mode": phase_res.mode,
        "surgical_mode": surg_res.mode,
        "sawtooth_before": phase_res.sawtooth_energy_pct_before,
        "base_bi_fpr": base_eval["bi_fpr"],
        "post_bi_fpr": best_eval["bi_fpr"],
        "base_bi_excess": base_eval["bi_excess"],
        "post_bi_excess": best_eval["bi_excess"],
    }

