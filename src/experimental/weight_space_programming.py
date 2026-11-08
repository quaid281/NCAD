"""Weight-Space Programming (WSP) for SolitonReynoldsJEPAModel & 224-Dial Circuit Breakers.

Implements Experiment 1 (Surgical Zero-Gradient False-Positive Quenching) inspired by
Weight-Space Programming (Takehi, Teoh, Zhang, & Khattab, MIT CSAIL):
  1. Zero-gradient weight-space and multi-perspective dial inspection strictly on training
     windows `tr_w` and nominal calibration prefix `cal_w` (`va_w[:n_cal_w]`).
  2. Structured Markdown serialization of predictor row norms, sub-block phase embeddings,
     Mahalanobis precision spectra, and per-coordinate instability profiles for LLM or
     analytical weight-space reasoning.
  3. Surgical weight-space intervention:
     - Predictor MLP output row & LayerNorm gain damping on diagnosed rogue coordinates
       (`predictor.net[4]`, `predictor.net[6]`, `sub_predictor.net[4]`, `sub_predictor.net[6]`).
     - Sub-block horizon phase-offset alignment (`subblock_embeddings[:, c]`).
     - Residual mean re-centering (`residual_mean`) on rogue coordinates across horizons.
     - Cross-coordinate precision off-diagonal decoupling and passband guard programming
       (`precision_matrix` projected onto the SPSD cone $\\mathcal{S}_+^{MD}$ with clean
       coordinates placed in the $(0, 1.10]$ passband and rogue coordinates scaled to unlock
       the circuit-breaker panel's `tr_shift_224 > 1.25` guard).
  4. Internal bi-directional calibration split verification and coincidence-gate
     non-deflation safety (`k_pref_tr >= 0.92 * k_pref_base`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn

from src.scoring.circuit_breaker_panel import (
    MultiPerspectiveCircuitBreakerPanel,
    aggregate_prefix_224_stream,
    aggregate_vector_stream,
    build_224_perspective_dials,
    extract_multihorizon_4subblocks,
    stabilize_whitened_subblocks,
)


@dataclass
class WSPConfig:
    """Configuration for zero-gradient Weight-Space Programming on SolitonReynoldsJEPAModel."""

    context_len: int = 256
    suspect_len: int = 64
    candidate_k_counts: tuple[int, ...] = (4, 2)
    min_damp: float = 0.35
    max_damp: float = 0.85
    phase_align_step: float = 0.35
    residual_mean_blend: float = 0.75
    off_diagonal_retain: float = 0.15
    clean_passband_rms: float = 0.50
    min_guard_factor: float = 1.30
    max_guard_factor: float = 1.44
    gate_retention_ratio: float = 0.92


@dataclass
class CoordinateWSPProfile:
    """Per-coordinate weight-space and calibration residual diagnostic profile."""

    coord_idx: int
    predictor_row_norm: float
    sub_predictor_row_norm: float
    shift_z: float
    var_inflation: float
    intra_mean_drift: float
    intra_tail_drift: float
    autocorr: float
    anchor_weight: float
    rogue_score: float
    rogue_z: float
    rms_white_cal: float
    rms_raw_cal: float


@dataclass
class WSPDiagnosticReport:
    """Complete zero-gradient diagnostic report for SolitonReynoldsJEPAModel + 224-dial panel."""

    latent_dim: int
    num_subblocks: int
    global_drift: float
    bi_fpr: float
    max_fpr: float
    bi_excess: float
    k_pref_tr: float
    k_128_tr: float
    k_edge_tr: float
    ranked_coords: list[int]
    profiles: list[CoordinateWSPProfile]

    def to_llm_markdown(self, top_n: int = 8) -> str:
        """Serialize weight-space and coordinate diagnostics into compact Markdown for LLM inspection."""
        lines = [
            "### Weight-Space Programming (WSP) Diagnostic Report",
            f"- **Architecture**: `SolitonReynoldsJEPAModel` (`M={self.num_subblocks}`, `D={self.latent_dim}`, `224` perspective dials)",
            f"- **Global Drift Index**: `{self.global_drift:.4f}`",
            f"- **Internal Split Calibration**: `bi_fpr={self.bi_fpr:.4f}`, `max_fpr={self.max_fpr:.4f}`, `bi_excess={self.bi_excess:.4f}`",
            f"- **Coincidence Gates (Trailing)**: `k_pref={self.k_pref_tr:.2f}`, `k_128={self.k_128_tr:.2f}`, `k_edge={self.k_edge_tr:.2f}`",
            "",
            "| Rank | Coord `c` | `rogue_z` | `shift_z` | `var_infl` | `intra_drift` | `tail_drift` | `acf` | `anchor_w` | `||W2[c,:]||` | `rms_white` |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        by_coord = {p.coord_idx: p for p in self.profiles}
        for rank, c in enumerate(self.ranked_coords[:top_n], start=1):
            p = by_coord[c]
            lines.append(
                f"| {rank} | `{p.coord_idx}` | {p.rogue_z:+.2f} | {p.shift_z:.3f} | {p.var_inflation:.3f} | "
                f"{p.intra_mean_drift:.3f} | {p.intra_tail_drift:.3f} | {p.autocorr:.3f} | "
                f"{p.anchor_weight:.3f} | {p.predictor_row_norm:.3f} | {p.rms_white_cal:.3f} |"
            )
        return "\n".join(lines)


@dataclass
class WSPSurgicalResult:
    """Summary of a surgical zero-gradient weight-space intervention."""

    mode: str
    rogue_coords: list[int]
    edited_predictor_elements: int
    total_model_params: int
    pct_predictor_edited: float
    baseline_bi_fpr: float
    post_bi_fpr: float
    baseline_bi_excess: float
    post_bi_excess: float
    baseline_k_pref: float
    post_k_pref: float
    damping_factors: dict[int, float] = field(default_factory=dict)
    guard_factors: dict[int, float] = field(default_factory=dict)
    diagnostic_report: WSPDiagnosticReport | None = None


@torch.no_grad()
def _eval_cal_gates_and_internal_split(
    model: nn.Module,
    cal_w: np.ndarray,
    context_len: int = 256,
    suspect_len: int = 64,
    cal_latents: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict:
    """Evaluate full-calibration coincidence gates and bi-directional internal split FPR on `cal_w`."""
    dev = next(model.parameters()).device
    if cal_latents is not None:
        from src.experimental.phase_embedding_synthesis import _forward_predictors_with_phase
        from src.models._jepa_utils import whiten_residuals

        zc, zt = cal_latents
        M, D = int(model.num_subblocks), int(model.latent_dim)
        raw_t = zt - _forward_predictors_with_phase(model, zc, model.subblock_embeddings)
        white_t = whiten_residuals(raw_t.reshape(len(zc), M * D) - model.residual_mean, model.precision_matrix).reshape(len(zc), M, D)
        cal_raw4 = raw_t.cpu().numpy().astype(np.float32)
        cal_white4 = white_t.cpu().numpy().astype(np.float32)
    else:
        cal_raw4, cal_white4 = extract_multihorizon_4subblocks(model, cal_w, context_len, dev)
    n_cal = len(cal_raw4)
    local_panel = MultiPerspectiveCircuitBreakerPanel(context_size=context_len, suspect_size=suspect_len)

    # 1. Full calibration coincidence gates (n_cal_w = n_cal)
    ew_full, _ = stabilize_whitened_subblocks(cal_white4, cal_white4, cal_raw4, cal_raw4, n_cal)
    bits_full = local_panel.calibrate_and_trip_windows(ew_full, ew_full, n_cal)
    len_sim = 2 * n_cal + context_len + suspect_len - 1
    pad_bits = {k: np.concatenate([v, v], axis=0) for k, v in bits_full.items()}

    vp_tr, vm_tr = aggregate_prefix_224_stream(pad_bits["bits224_va"], len_sim, context_len, suspect_len, "trailing")
    vb_tr, _ = aggregate_vector_stream(pad_bits["bits128_va"], len_sim, context_len, suspect_len, "trailing")
    ve_tr, _ = aggregate_vector_stream(pad_bits["fast_edge_va"], len_sim, context_len, suspect_len, "trailing")
    h_tr = max(1, len(vp_tr[vm_tr]) // 2)
    k_pref_tr = max(float(np.percentile(vp_tr[vm_tr][:h_tr].sum(axis=1), 99.5)) * 1.12, 5.0)
    k_128_tr = max(float(np.percentile(vb_tr[vm_tr][:h_tr].sum(axis=1), 99.5)) * 1.12, 3.5)
    k_edge_tr = max(float(np.percentile(ve_tr[vm_tr][:h_tr].sum(axis=1), 99.90)) * 1.25, 2.5)

    # 2. Internal half-split out-of-sample test (n_cal_w = max(2, n_cal // 2))
    half_cal = max(2, n_cal // 2)
    len_pts = n_cal + context_len + suspect_len - 1
    ew_fwd, _ = stabilize_whitened_subblocks(cal_white4, cal_white4, cal_raw4, cal_raw4, half_cal)
    bits_fwd = local_panel.calibrate_and_trip_windows(ew_fwd, ew_fwd, half_cal)
    dec_fwd = local_panel.evaluate_stream_decisions(bits_fwd, len_pts, len_pts, "trailing")

    raw_rev = cal_raw4[::-1].copy()
    white_rev = cal_white4[::-1].copy()
    ew_rev, _ = stabilize_whitened_subblocks(white_rev, white_rev, raw_rev, raw_rev, half_cal)
    bits_rev = local_panel.calibrate_and_trip_windows(ew_rev, ew_rev, half_cal)
    dec_rev = local_panel.evaluate_stream_decisions(bits_rev, len_pts, len_pts, "trailing")

    fpr_f = float(dec_fwd.hold_pred_balanced.mean()) if len(dec_fwd.hold_pred_balanced) > 0 else 0.0
    fpr_r = float(dec_rev.hold_pred_balanced.mean()) if len(dec_rev.hold_pred_balanced) > 0 else 0.0
    exc_f = float(np.maximum(0.0, dec_fwd.hold_switch_counts - 0.85 * dec_fwd.k_gate_balanced).mean()) if len(dec_fwd.hold_switch_counts) > 0 else 0.0
    exc_r = float(np.maximum(0.0, dec_rev.hold_switch_counts - 0.85 * dec_rev.k_gate_balanced).mean()) if len(dec_rev.hold_switch_counts) > 0 else 0.0

    return {
        "cal_raw4": cal_raw4, "cal_white4": cal_white4, "ew_full": ew_full, "bits_full": bits_full,
        "k_pref_tr": k_pref_tr, "k_128_tr": k_128_tr, "k_edge_tr": k_edge_tr,
        "fpr_f": fpr_f, "fpr_r": fpr_r, "exc_f": exc_f, "exc_r": exc_r,
        "bi_fpr": 0.5 * (fpr_f + fpr_r), "max_fpr": max(fpr_f, fpr_r), "bi_excess": 0.5 * (exc_f + exc_r),
    }


@torch.no_grad()
def inspect_soliton_reynolds_weight_space(
    model: nn.Module,
    tr_w: np.ndarray,
    cal_w: np.ndarray,
    config: WSPConfig | None = None,
) -> tuple[WSPDiagnosticReport, dict]:
    """Inspect SolitonReynoldsJEPAModel weights and calibration residuals without gradients."""
    cfg = config or WSPConfig()
    model.eval()
    dev = next(model.parameters()).device
    M, D = int(model.num_subblocks), int(model.latent_dim)

    base_eval = _eval_cal_gates_and_internal_split(model, cal_w, cfg.context_len, cfg.suspect_len)
    tr_raw4, _ = extract_multihorizon_4subblocks(model, tr_w, cfg.context_len, dev)
    cal_raw4 = base_eval["cal_raw4"]
    cal_white4 = base_eval["cal_white4"]
    n_cal = len(cal_raw4)
    q1 = max(1, n_cal // 2)

    tr_mean = tr_raw4.mean(axis=0)
    cal_mean = cal_raw4.mean(axis=0)
    tr_std = tr_raw4.std(axis=0) + 1e-5
    cal_std = cal_raw4.std(axis=0) + 1e-5

    shift_z = (np.abs(cal_mean - tr_mean) / tr_std).mean(axis=0)
    var_inflation = (cal_std / tr_std).mean(axis=0)

    h1_mean = cal_raw4[:q1].mean(axis=0)
    h2_mean = cal_raw4[q1:].mean(axis=0)
    h1_std = cal_raw4[:q1].std(axis=0) + 1e-5
    h2_std = cal_raw4[q1:].std(axis=0) + 1e-5
    intra_mean_drift = (np.abs(h2_mean - h1_mean) / cal_std).mean(axis=0)
    intra_tail_drift = np.maximum(1.0, h2_std / h1_std).mean(axis=0)

    cal_c = cal_raw4.mean(axis=1)
    cal_c_cent = cal_c - cal_c.mean(axis=0, keepdims=True)
    lag = min(32, max(1, n_cal // 8))
    if n_cal > lag:
        acf = np.maximum(
            0.0, (cal_c_cent[:-lag] * cal_c_cent[lag:]).mean(axis=0) / (cal_c_cent.var(axis=0) + 1e-6)
        )
    else:
        acf = np.zeros(D, dtype=np.float64)

    b224_cal = base_eval["bits_full"]["bits224_va"].reshape(n_cal, 7, D)
    tot_cal = b224_cal.sum(axis=(1, 2))
    q99 = float(np.percentile(tot_cal, 99.0))
    peak_mask = tot_cal >= max(q99, 4.0)
    anchor_weight = b224_cal[peak_mask].mean(axis=(0, 1)) if peak_mask.any() else np.zeros(D, dtype=np.float64)

    rogue_score = (
        0.35 * shift_z
        + 0.30 * np.maximum(0.0, var_inflation - 1.0)
        + 0.45 * intra_mean_drift
        + 0.35 * np.maximum(0.0, intra_tail_drift - 1.0)
        + 0.25 * acf
        - 1.50 * anchor_weight
    )
    med_rogue = float(np.median(rogue_score))
    mad_rogue = float(1.4826 * np.median(np.abs(rogue_score - med_rogue))) + 1e-4
    rogue_z = (rogue_score - med_rogue) / mad_rogue

    global_drift = float(
        np.median(intra_mean_drift) + np.median(np.maximum(0.0, var_inflation - 1.0)) + np.median(shift_z)
    )
    ranked_coords = [int(c) for c in np.argsort(rogue_score)[::-1]]

    w_pred_norms = model.predictor.net[4].weight.detach().norm(dim=1).cpu().numpy()
    w_sub_norms = model.sub_predictor.net[4].weight.detach().norm(dim=1).cpu().numpy()
    rms_white_coord = np.sqrt(np.mean(cal_white4 ** 2, axis=(0, 1)))
    rms_raw_coord = np.sqrt(np.mean(cal_raw4 ** 2, axis=(0, 1)))

    profiles = [
        CoordinateWSPProfile(
            coord_idx=c, predictor_row_norm=float(w_pred_norms[c]), sub_predictor_row_norm=float(w_sub_norms[c]),
            shift_z=float(shift_z[c]), var_inflation=float(var_inflation[c]),
            intra_mean_drift=float(intra_mean_drift[c]), intra_tail_drift=float(intra_tail_drift[c]),
            autocorr=float(acf[c]), anchor_weight=float(anchor_weight[c]),
            rogue_score=float(rogue_score[c]), rogue_z=float(rogue_z[c]),
            rms_white_cal=float(rms_white_coord[c]), rms_raw_cal=float(rms_raw_coord[c]),
        )
        for c in range(D)
    ]

    report = WSPDiagnosticReport(
        latent_dim=D, num_subblocks=M, global_drift=global_drift,
        bi_fpr=float(base_eval["bi_fpr"]), max_fpr=float(base_eval["max_fpr"]),
        bi_excess=float(base_eval["bi_excess"]), k_pref_tr=float(base_eval["k_pref_tr"]),
        k_128_tr=float(base_eval["k_128_tr"]), k_edge_tr=float(base_eval["k_edge_tr"]),
        ranked_coords=ranked_coords, profiles=profiles,
    )
    return report, {"base_eval": base_eval, "tr_mean": tr_mean, "cal_mean": cal_mean, "rogue_z": rogue_z}


@torch.no_grad()
def apply_surgical_wsp_quench(
    model: nn.Module,
    tr_w: np.ndarray,
    cal_w: np.ndarray,
    config: WSPConfig | None = None,
) -> WSPSurgicalResult:
    """Perform zero-gradient surgical Weight-Space Programming on SolitonReynoldsJEPAModel."""
    cfg = config or WSPConfig()
    model.eval()
    dev = next(model.parameters()).device
    M, D = int(model.num_subblocks), int(model.latent_dim)
    total_params = sum(p.numel() for p in model.parameters())

    report, aux = inspect_soliton_reynolds_weight_space(model, tr_w, cal_w, cfg)
    base_eval = aux["base_eval"]
    tr_mean = aux["tr_mean"]
    cal_mean = aux["cal_mean"]
    rogue_z = aux["rogue_z"]

    if (base_eval["fpr_f"] == 0.0 and base_eval["exc_f"] < 0.04) or (
        base_eval["max_fpr"] < 0.005 and base_eval["bi_excess"] < 0.18 and report.global_drift < 0.25
    ):
        return WSPSurgicalResult(
            mode="abstain_clean", rogue_coords=[], edited_predictor_elements=0,
            total_model_params=total_params, pct_predictor_edited=0.0,
            baseline_bi_fpr=report.bi_fpr, post_bi_fpr=report.bi_fpr,
            baseline_bi_excess=report.bi_excess, post_bi_excess=report.bi_excess,
            baseline_k_pref=report.k_pref_tr, post_k_pref=report.k_pref_tr,
            diagnostic_report=report,
        )

    from src.experimental.phase_embedding_synthesis import _extract_subblock_latents, _forward_predictors_with_phase

    cal_latents = _extract_subblock_latents(model, cal_w, cfg.context_len)
    sorted_coords = report.ranked_coords
    orig_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def _apply_for_subset(k_count: int) -> tuple[dict, int, dict[int, float], dict[int, float]]:
        model.load_state_dict({k: v.to(dev) for k, v in orig_state.items()})
        rogue_sub = sorted_coords[:k_count]
        edited_elems = 0
        damping_map: dict[int, float] = {}
        guard_map: dict[int, float] = {}

        # 1. Predictor MLP Output Row & Sub-block Embedding Damping on Rogue Coords
        for c in rogue_sub:
            damp = float(np.clip(1.0 / (1.0 + 0.5 * max(0.0, float(rogue_z[c]))), cfg.min_damp, cfg.max_damp))
            damping_map[c] = round(damp, 4)
            model.predictor.net[4].weight.data[c, :].mul_(damp)
            model.predictor.net[4].bias.data[c].mul_(damp)
            model.predictor.net[6].weight.data[c].mul_(damp)
            model.sub_predictor.net[4].weight.data[c, :].mul_(damp)
            model.sub_predictor.net[4].bias.data[c].mul_(damp)
            model.sub_predictor.net[6].weight.data[c].mul_(damp)
            phase_corr = torch.from_numpy((cal_mean[:, c] - tr_mean[:, c]) * cfg.phase_align_step).to(
                dev, dtype=torch.float32
            )
            model.subblock_embeddings.data[:, c].add_(phase_corr)
            edited_elems += 2 * (model.predictor.net[4].weight.shape[1] + 2) + M

        zc_cal, zt_cal = cal_latents
        cal_raw4_post = (zt_cal - _forward_predictors_with_phase(model, zc_cal, model.subblock_embeddings)).cpu().numpy().astype(np.float32)

        # 2. Residual Mean Alignment on Rogue Coordinates
        cal_mean_post = cal_raw4_post.reshape(len(cal_raw4_post), M * D).mean(axis=0)
        res_mean_new = model.residual_mean.detach().cpu().numpy().copy()
        w_cal_blend = float(cfg.residual_mean_blend)
        for c in rogue_sub:
            for m_idx in range(M):
                flat_idx = m_idx * D + c
                res_mean_new[flat_idx] = (
                    (1.0 - w_cal_blend) * res_mean_new[flat_idx] + w_cal_blend * cal_mean_post[flat_idx]
                )
        model.residual_mean.copy_(torch.from_numpy(res_mean_new).to(dev, dtype=model.residual_mean.dtype))
        edited_elems += len(rogue_sub) * M

        # 3. Precision Matrix Off-Diagonal Decoupling & Passband Guard Scaling
        prec = model.precision_matrix.detach().cpu().numpy().copy()
        for c in rogue_sub:
            for m1 in range(M):
                idx1 = m1 * D + c
                for m2 in range(M):
                    for c2 in range(D):
                        idx2 = m2 * D + c2
                        if c2 != c:
                            prec[idx1, idx2] *= cfg.off_diagonal_retain
                            prec[idx2, idx1] *= cfg.off_diagonal_retain

        evals, evecs = np.linalg.eigh(0.5 * (prec + prec.T))
        evals = np.maximum(evals, 1e-6)
        prec_spsd = (evecs * evals[None, :]) @ evecs.T
        omega_sqrt = (evecs * np.sqrt(evals)[None, :]) @ evecs.T

        diff_cal = cal_raw4_post.reshape(len(cal_raw4_post), M * D) - res_mean_new[None, :]
        w_cal = diff_cal @ omega_sqrt
        rms_cal = np.sqrt(np.mean(w_cal ** 2, axis=0)) + 1e-5

        target_rms = np.full(M * D, cfg.clean_passband_rms, dtype=np.float64)
        for c in rogue_sub:
            g_factor = float(
                np.clip(1.32 + 0.04 * max(0.0, float(rogue_z[c]) - 0.5), cfg.min_guard_factor, cfg.max_guard_factor)
            )
            guard_map[c] = round(g_factor, 4)
            for m_idx in range(M):
                target_rms[m_idx * D + c] = cfg.clean_passband_rms * 1.50 * g_factor

        scale_vec = target_rms / rms_cal
        prec_programmed = (prec_spsd * scale_vec[:, None]) * scale_vec[None, :]
        model.precision_matrix.copy_(torch.from_numpy(prec_programmed).to(dev, dtype=model.precision_matrix.dtype))

        cand_eval = _eval_cal_gates_and_internal_split(model, cal_w, cfg.context_len, cfg.suspect_len, cal_latents=cal_latents)
        return cand_eval, edited_elems, damping_map, guard_map

    best_k = 0
    best_eval = base_eval
    best_elems = 0
    best_damping: dict[int, float] = {}
    best_guards: dict[int, float] = {}

    for k_try in cfg.candidate_k_counts:
        if k_try <= 0 or k_try > D:
            continue
        cand_eval, elems, d_map, g_map = _apply_for_subset(k_try)
        gate_safe = (
            cand_eval["k_pref_tr"] >= base_eval["k_pref_tr"] * cfg.gate_retention_ratio
            and cand_eval["k_128_tr"] >= base_eval["k_128_tr"] * cfg.gate_retention_ratio
            and cand_eval["fpr_f"] <= base_eval["fpr_f"] + 1e-4
        )
        if best_k == 0:
            split_better = (
                cand_eval["bi_fpr"] < base_eval["bi_fpr"] - 1e-4
                or (
                    abs(cand_eval["bi_fpr"] - base_eval["bi_fpr"]) <= 1e-4
                    and cand_eval["bi_excess"] < base_eval["bi_excess"] - 0.01
                )
            )
        elif k_try < best_k:
            # Minimum-intervention sparsity: prefer smaller k_try whenever it matches best_k's bi_fpr
            split_better = cand_eval["bi_fpr"] <= best_eval["bi_fpr"] + 1e-4 and (
                cand_eval["bi_fpr"] < base_eval["bi_fpr"] - 1e-4
                or cand_eval["bi_excess"] < base_eval["bi_excess"] - 0.01
            )
        else:
            split_better = cand_eval["bi_fpr"] < best_eval["bi_fpr"] - 0.002

        if gate_safe and split_better:
            best_k = k_try
            best_eval = cand_eval
            best_elems = elems
            best_damping = d_map
            best_guards = g_map

    if best_k == 0:
        model.load_state_dict({k: v.to(dev) for k, v in orig_state.items()})
        return WSPSurgicalResult(
            mode="abstain_gate", rogue_coords=[], edited_predictor_elements=0,
            total_model_params=total_params, pct_predictor_edited=0.0,
            baseline_bi_fpr=report.bi_fpr, post_bi_fpr=report.bi_fpr,
            baseline_bi_excess=report.bi_excess, post_bi_excess=report.bi_excess,
            baseline_k_pref=report.k_pref_tr, post_k_pref=report.k_pref_tr,
            diagnostic_report=report,
        )

    _apply_for_subset(best_k)
    return WSPSurgicalResult(
        mode=f"surgical_k{best_k}", rogue_coords=sorted_coords[:best_k],
        edited_predictor_elements=best_elems, total_model_params=total_params,
        pct_predictor_edited=100.0 * best_elems / max(total_params, 1),
        baseline_bi_fpr=report.bi_fpr, post_bi_fpr=float(best_eval["bi_fpr"]),
        baseline_bi_excess=report.bi_excess, post_bi_excess=float(best_eval["bi_excess"]),
        baseline_k_pref=report.k_pref_tr, post_k_pref=float(best_eval["k_pref_tr"]),
        damping_factors=best_damping, guard_factors=best_guards, diagnostic_report=report,
    )


# ============================================================================
# Experiment 2: Blind Weight-Space Reverse Engineering & Zero-Data Auditing
# ============================================================================

ROLE_NAMES = ("slow_drift", "turbulent_fluct", "edge_accel")


@dataclass
class BlindCoordinateDialProfile:
    """Data-free reconstructed 224-dial physical profile and fragility score for coordinate `c`."""

    coord_idx: int
    rec_drift: float
    rec_fluct: float
    rec_accel: float
    predicted_role: str
    prec_diag: float
    prec_offdiag: float
    horizon_anisotropy: float
    rollout_mismatch: float
    blind_fragility: float
    fragility_z: float


@dataclass
class BlindWSPReport:
    """100% Data-Free Weight-Space Reverse-Engineering Report (Experiment 2)."""

    latent_dim: int
    num_subblocks: int
    dynamical_regime: str
    vorticity_ratio: float
    r_lt_med: float
    effective_rank: float
    dispersion_ratio: float
    signal_mode_count: int
    ranked_fragile_coords: list[int]
    profiles: list[BlindCoordinateDialProfile]
    reconstructed_covariance: np.ndarray = field(repr=False)

    def to_llm_markdown(self, top_n: int = 8) -> str:
        """Serialize blind weight-space reverse-engineering report into compact Markdown."""
        lines = [
            "### Blind Weight-Space Reverse-Engineering Report (Zero-Data Audit)",
            f"- **Classified Dynamical Regime**: `{self.dynamical_regime}`",
            f"- **Sub-Predictor Vorticity Ratio (`rho_vort`)**: `{self.vorticity_ratio:.3f}`",
            f"- **Soliton Lieb-Thirring Concentration (`r_lt_med`)**: `{self.r_lt_med:.3f}`",
            f"- **Precision Effective Rank (`R_eff`)**: `{self.effective_rank:.1f} / {self.num_subblocks * self.latent_dim}` (`{self.signal_mode_count}` upper-branch signal modes)",
            f"- **Sub-Block Horizon Dispersion Ratio**: `{self.dispersion_ratio:.3f}`",
            "",
            "| Fragility Rank | Coord `c` | `frag_z` | Predicted 224-Dial Role | `rec_drift` | `rec_fluct` | `rec_accel` | `prec_diag` | `anisotropy` | `rollout_err` |",
            "|---:|---:|---:|:---|---:|---:|---:|---:|---:|---:|",
        ]
        by_coord = {p.coord_idx: p for p in self.profiles}
        for rank, c in enumerate(self.ranked_fragile_coords[:top_n], start=1):
            p = by_coord[c]
            lines.append(
                f"| {rank} | `{p.coord_idx}` | {p.fragility_z:+.2f} | `{p.predicted_role}` | "
                f"{p.rec_drift:.3f} | {p.rec_fluct:.3f} | {p.rec_accel:.3f} | "
                f"{p.prec_diag:.2f} | {p.horizon_anisotropy:.1f} | {p.rollout_mismatch:.3f} |"
            )
        return "\n".join(lines)


def _linearized_mlp_jacobian(sd: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    """Compute analytical Jacobian of ResidualPredictorMLP around its LayerNorm bias operating point."""
    W1 = sd[f"{prefix}.net.0.weight"].astype(np.float64)
    b1 = sd[f"{prefix}.net.0.bias"].astype(np.float64)
    g1 = sd[f"{prefix}.net.2.weight"].astype(np.float64)
    W2 = sd[f"{prefix}.net.4.weight"].astype(np.float64)
    g2 = sd[f"{prefix}.net.6.weight"].astype(np.float64)
    x = b1 / (np.std(b1) + 1e-5)
    gelu_grad = 0.5 * (1.0 + np.tanh(0.79788456 * (x + 0.044715 * x**3))) + (
        x / np.sqrt(2.0 * np.pi)
    ) * np.exp(-0.5 * x**2)
    mid_scale = g1 * gelu_grad
    return (g2[:, None] * W2) @ (mid_scale[:, None] * W1)


@torch.no_grad()
def blind_reverse_engineer_weight_space(
    model: nn.Module,
    resolvent_u: float = 1e-3,
) -> BlindWSPReport:
    """Reverse-engineer dynamical regime, 224-dial roles, and coordinate fragility purely from weights."""
    model.eval()
    sd = {k: v.detach().cpu().float().numpy() for k, v in model.state_dict().items()}
    M, D = int(model.num_subblocks), int(model.latent_dim)

    J_pred = _linearized_mlp_jacobian(sd, "predictor")
    J_sub = _linearized_mlp_jacobian(sd, "sub_predictor")
    J_sub_skew = 0.5 * (J_sub - J_sub.T)
    J_sub_sym = 0.5 * (J_sub + J_sub.T)
    vorticity_ratio = float(np.linalg.norm(J_sub_skew, "fro") / (np.linalg.norm(J_sub_sym, "fro") + 1e-6))

    A_sub3 = np.linalg.matrix_power(np.eye(D) + 0.25 * J_sub, 3)
    A_pred = np.eye(D) + 0.25 * J_pred
    rollout_mismatch = np.linalg.norm(A_sub3 - A_pred, axis=1)

    emb = sd["subblock_embeddings"].astype(np.float64)
    emb_accel = np.abs(emb[-1] - emb[0])
    emb_fluct = np.std(emb, axis=0)
    dispersion_ratio = float(np.mean(emb_accel) / (np.mean(emb_fluct) + 1e-6))

    tap_lag = np.zeros(D, dtype=np.float64)
    if "context_encoder.fc.weight" in sd and "target_encoder.fc.weight" in sd:
        tap_diff = sd["context_encoder.fc.weight"] - sd["target_encoder.fc.weight"]
        tap_lag = np.linalg.norm(tap_diff, axis=1).astype(np.float64)

    prec = sd["precision_matrix"].astype(np.float64)
    res_mean = sd["residual_mean"].astype(np.float64).reshape(M, D)
    evals, evecs = np.linalg.eigh(0.5 * (prec + prec.T))
    evals_pos = np.maximum(evals, 0.0)
    p_norm = evals_pos / (np.sum(evals_pos) + 1e-12)
    eff_rank = float(np.exp(-np.sum(p_norm * np.log(np.maximum(p_norm, 1e-12)))))

    # Branch-Filtered Exact Resolvent Inverse of omega = lambda / (lambda + u)^2
    res_flat = res_mean.reshape(-1)
    res_norm2 = float(np.sum(res_flat**2)) + 1e-8
    mean_cos2 = ((evecs.T @ res_flat) ** 2) / res_norm2
    horiz_coh = np.zeros(M * D, dtype=np.float64)
    for i in range(M * D):
        V_i = evecs[:, i].reshape(M, D)
        s_vals = np.linalg.svd(V_i, compute_uv=False)
        horiz_coh[i] = float(s_vals[0] ** 2 / (np.sum(s_vals**2) + 1e-12))

    sig_mask = ((evals_pos > 0.35) | ((evals_pos > 0.06) & (horiz_coh > 0.65))) & (mean_cos2 < 0.45)
    if not np.any(sig_mask):
        sig_mask = evals_pos > np.median(evals_pos)
    lam_rec = np.zeros_like(evals_pos)
    om = evals_pos[sig_mask]
    disc = np.maximum(0.0, 1.0 - 4.0 * resolvent_u * om)
    lam_rec[sig_mask] = (1.0 - 2.0 * resolvent_u * om + np.sqrt(disc)) / (2.0 * om)
    cov_rec = (evecs * lam_rec[None, :]) @ evecs.T

    w_drift = np.full(M, 1.0 / M, dtype=np.float64)
    P_fluct = np.eye(M, dtype=np.float64) - np.full((M, M), 1.0 / M, dtype=np.float64)
    w_accel = np.zeros(M, dtype=np.float64)
    w_accel[0], w_accel[-1] = -1.0, 1.0

    rec_drift = np.zeros(D, dtype=np.float64)
    rec_fluct = np.zeros(D, dtype=np.float64)
    rec_accel = np.zeros(D, dtype=np.float64)
    prec_diag = np.zeros(D, dtype=np.float64)
    prec_offdiag = np.zeros(D, dtype=np.float64)
    horizon_anisotropy = np.zeros(D, dtype=np.float64)

    for c in range(D):
        idx = [m_i * D + c for m_i in range(M)]
        not_idx = [i for i in range(M * D) if i not in idx]
        G_c = cov_rec[np.ix_(idx, idx)] + np.outer(res_mean[:, c], res_mean[:, c])
        rec_drift[c] = float(np.sqrt(max(0.0, float(w_drift @ G_c @ w_drift))))
        rec_fluct[c] = float(np.sqrt(max(0.0, float((1.0 / M) * np.trace(P_fluct @ G_c @ P_fluct)))))
        rec_accel[c] = float(np.sqrt(max(0.0, float(w_accel @ G_c @ w_accel))))

        P_c = prec[np.ix_(idx, idx)]
        prec_diag[c] = float(np.mean(np.diag(P_c)))
        prec_offdiag[c] = float(np.linalg.norm(prec[np.ix_(idx, not_idx)], "fro"))
        ev_c = np.maximum(np.linalg.eigvalsh(0.5 * (P_c + P_c.T)), 1e-3)
        horizon_anisotropy[c] = float(ev_c[-1] / ev_c[0])

    rel_roles = np.stack(
        [
            rec_drift / (np.mean(rec_drift) + 1e-8),
            rec_fluct / (np.mean(rec_fluct) + 1e-8),
            rec_accel / (np.mean(rec_accel) + 1e-8),
        ],
        axis=0,
    )
    pred_role_idx = np.argmax(rel_roles, axis=0)

    def _rob_z(v: np.ndarray) -> np.ndarray:
        med = float(np.median(v))
        mad = float(1.4826 * np.median(np.abs(v - med))) + 1e-5
        return (v - med) / mad

    blind_fragility = (
        0.45 * _rob_z(prec_diag)
        + 0.30 * _rob_z(prec_offdiag)
        + 0.20 * _rob_z(horizon_anisotropy)
        + 0.15 * _rob_z(rollout_mismatch)
        + 0.15 * _rob_z(tap_lag)
    )
    frag_z = _rob_z(blind_fragility)
    ranked_coords = [int(c) for c in np.argsort(blind_fragility)[::-1]]

    if eff_rank < 35.0:
        regime = "low_rank_constrained"
    elif vorticity_ratio >= 1.10:
        regime = "rotational_oscillatory"
    else:
        regime = "soliton_relaxation"

    profiles = [
        BlindCoordinateDialProfile(
            coord_idx=c, rec_drift=float(rec_drift[c]), rec_fluct=float(rec_fluct[c]), rec_accel=float(rec_accel[c]),
            predicted_role=ROLE_NAMES[int(pred_role_idx[c])], prec_diag=float(prec_diag[c]),
            prec_offdiag=float(prec_offdiag[c]), horizon_anisotropy=float(horizon_anisotropy[c]),
            rollout_mismatch=float(rollout_mismatch[c]), blind_fragility=float(blind_fragility[c]),
            fragility_z=float(frag_z[c]),
        )
        for c in range(D)
    ]

    return BlindWSPReport(
        latent_dim=D, num_subblocks=M, dynamical_regime=regime, vorticity_ratio=vorticity_ratio,
        r_lt_med=float(sd.get("r_lt_med", np.array(0.0))), effective_rank=eff_rank,
        dispersion_ratio=dispersion_ratio, signal_mode_count=int(np.sum(sig_mask)),
        ranked_fragile_coords=ranked_coords, profiles=profiles, reconstructed_covariance=cov_rec,
    )


def evaluate_blind_dial_reconstruction(
    report: BlindWSPReport,
    tr_raw4: np.ndarray,
    va_white4: np.ndarray | None = None,
) -> dict[str, float]:
    """Compare data-free reconstructed 224-dial coordinate roles against ground-truth window dials."""
    D = report.latent_dim
    d224 = build_224_perspective_dials(tr_raw4)
    true_drift = d224[:, 4 * D : 5 * D].mean(axis=0)
    true_fluct = d224[:, 5 * D : 6 * D].mean(axis=0)
    true_accel = d224[:, 6 * D : 7 * D].mean(axis=0)

    rec_drift = np.array([p.rec_drift for p in report.profiles], dtype=np.float64)
    rec_fluct = np.array([p.rec_fluct for p in report.profiles], dtype=np.float64)
    rec_accel = np.array([p.rec_accel for p in report.profiles], dtype=np.float64)

    def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        sa, sb = ra - ra.mean(), rb - rb.mean()
        denom = float(np.sqrt(np.sum(sa**2) * np.sum(sb**2)))
        return float(np.sum(sa * sb) / denom) if denom > 1e-12 else 0.0

    true_role_idx = np.argmax(
        np.stack([true_drift / (np.mean(true_drift) + 1e-8), true_fluct / (np.mean(true_fluct) + 1e-8), true_accel / (np.mean(true_accel) + 1e-8)], axis=0),
        axis=0,
    )
    pred_role_idx = np.array([ROLE_NAMES.index(p.predicted_role) for p in report.profiles])
    role_acc = float(np.mean(pred_role_idx == true_role_idx))

    frag_corr = 0.0
    if va_white4 is not None:
        va_rms = np.sqrt(np.mean(va_white4**2, axis=(0, 1)))
        frag_scores = np.array([p.blind_fragility for p in report.profiles], dtype=np.float64)
        frag_corr = _rank_corr(frag_scores, va_rms)

    return {
        "role_accuracy": role_acc,
        "drift_rank_corr": _rank_corr(rec_drift, true_drift),
        "fluct_rank_corr": _rank_corr(rec_fluct, true_fluct),
        "accel_rank_corr": _rank_corr(rec_accel, true_accel),
        "fragility_rank_corr": frag_corr,
    }


@torch.no_grad()
def apply_blind_wsp_preconditioning(
    model: nn.Module,
    k_edit: int = 2,
    min_eff_rank_abstain: float = 55.0,
) -> dict:
    """Perform 100% data-free surgical weight pre-conditioning using only `model.state_dict()`."""
    report = blind_reverse_engineer_weight_space(model)
    dev = next(model.parameters()).device
    M, D = report.num_subblocks, report.latent_dim

    if report.effective_rank >= min_eff_rank_abstain or k_edit <= 0:
        return {"mode": "blind_abstain", "rogue_coords": [], "report": report}

    rogue_sub = report.ranked_fragile_coords[:k_edit]
    by_coord = {p.coord_idx: p for p in report.profiles}

    for c in rogue_sub:
        fz = by_coord[c].fragility_z
        damp = float(np.clip(1.0 / (1.0 + 0.25 * max(0.0, fz)), 0.60, 0.85))
        for net_mod in (model.predictor.net, model.sub_predictor.net):
            net_mod[4].weight.data[c, :].mul_(damp)
            net_mod[4].bias.data[c].mul_(damp)
            net_mod[6].weight.data[c].mul_(damp)

    prec = model.precision_matrix.detach().cpu().numpy().astype(np.float64).copy()
    for c in rogue_sub:
        for m1 in range(M):
            idx1 = m1 * D + c
            for m2 in range(M):
                for c2 in range(D):
                    idx2 = m2 * D + c2
                    if c2 != c:
                        prec[idx1, idx2] *= 0.25
                        prec[idx2, idx1] *= 0.25

    ev_e, evec_e = np.linalg.eigh(0.5 * (prec + prec.T))
    ev_e = np.maximum(ev_e, 1e-6)
    ev_tempered = ev_e / (1.0 + ev_e / 120.0)
    prec_spsd = (evec_e * ev_tempered[None, :]) @ evec_e.T

    omega_sqrt = (evec_e * np.sqrt(ev_tempered)[None, :]) @ evec_e.T
    rms_blind = np.sqrt(np.maximum(np.diag(omega_sqrt @ report.reconstructed_covariance @ omega_sqrt), 1e-4))
    target_rms = np.full(M * D, 0.50, dtype=np.float64)
    for c in rogue_sub:
        fz = by_coord[c].fragility_z
        g_factor = float(np.clip(1.32 + 0.03 * max(0.0, fz - 0.5), 1.30, 1.42))
        for m_idx in range(M):
            target_rms[m_idx * D + c] = 0.50 * 1.50 * g_factor

    scale_vec = target_rms / rms_blind
    prec_prog = (prec_spsd * scale_vec[:, None]) * scale_vec[None, :]
    model.precision_matrix.copy_(torch.from_numpy(prec_prog).to(dev, dtype=model.precision_matrix.dtype))
    return {"mode": f"blind_k{k_edit}", "rogue_coords": rogue_sub, "report": report}


from src.experimental.phase_embedding_synthesis import (  # noqa: E402
    ORTHOGONAL_LEGENDRE_4BASIS,
    LegendreHorizonBasisParametrization,
    PhaseSynthesisResult,
    apply_unified_wsp_soliton_program,
    bake_structural_legendre_phase_basis,
    bind_structural_legendre_phase_basis,
    synthesize_subblock_phase_embeddings,
)
