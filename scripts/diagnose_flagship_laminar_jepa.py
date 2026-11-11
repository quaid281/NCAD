"""Comprehensive Layer-to-Layer Activation Trace and Latent-Space Diagnosis for Flagship Laminar-JEPA.

Audits the 4-stage Flagship `Laminar-JEPA (224-Bit)` pipeline (`SolitonReynoldsJEPAModel` +
`apply_surgical_wsp_quench` + `MultiPerspectiveCircuitBreakerPanel`) across multi-domain telemetry
streams and controlled physical fault injections. Saves RFC-compliant JSON diagnostics atomically.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CSMConfig
from src.data.data_loader import DataLoader
from src.engine.activation_tracer import _compute_batch_variance, _compute_stable_rank, autopsy_model
from src.engine.builder import build_ts_jepa_model
from src.experimental.phase_embedding_synthesis import (
    ORTHOGONAL_LEGENDRE_4BASIS,
    _extract_subblock_latents,
    _forward_predictors_with_phase,
)
from src.experimental.weight_space_programming import (
    WSPConfig,
    apply_surgical_wsp_quench,
    blind_reverse_engineer_weight_space,
    evaluate_blind_dial_reconstruction,
)
from src.models._jepa_utils import whiten_residuals
from src.scoring.circuit_breaker_panel import (
    MultiPerspectiveCircuitBreakerPanel,
    build_224_perspective_dials,
    stabilize_whitened_subblocks,
)
from src.scoring.evt_calibrator import compute_excess_kurtosis
from scripts.run_multiseed_heldout_benchmark import batches, compute_metrics, event_stats, load_stream

DEFAULT_STREAMS = ("Daphnet:S01R01E1", "Daphnet:S03R02E0", "SMAP:P-3", "SMAP:G-2", "SMAP:E-1", "MSL:F-8")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def _clean_record(row: dict[str, Any]) -> dict[str, Any]:
    return {k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v) for k, v in row.items()}


def _rankme_and_spectrum(z: np.ndarray) -> dict[str, Any]:
    zf = np.asarray(z, dtype=np.float64)
    if zf.ndim > 2:
        zf = zf.reshape(zf.shape[0], -1)
    n, d = zf.shape
    zc = zf - zf.mean(axis=0, keepdims=True)
    cov = (zc.T @ zc) / max(n - 1, 1)
    evals = np.maximum(np.linalg.eigvalsh(0.5 * (cov + cov.T)), 0.0)[::-1]
    total = float(evals.sum()) + 1e-12
    p = evals / total
    eff_rank = float(math.exp(-float(np.sum(p * np.log(np.maximum(p, 1e-12))))))
    part_ratio = float((total**2) / (float(np.sum(evals**2)) + 1e-12))
    return {
        "dim": int(d), "eff_rank_rankme": round(eff_rank, 3), "eff_rank_ratio": round(eff_rank / max(d, 1), 4),
        "participation_ratio": round(part_ratio, 3), "top1_var_share": round(float(p[0]), 4),
        "top5_var_share": round(float(p[: min(5, d)].sum()), 4), "lambda_max": round(float(evals[0]), 6),
        "lambda_min": round(float(evals[-1]), 6), "dead_dims": int(np.sum(np.var(zf, axis=0) < 1e-6)),
        "mean_l2_norm": round(float(np.mean(np.linalg.norm(zf, axis=-1))), 4),
    }


def _isotropy_and_uniformity(z: np.ndarray, max_samples: int = 512) -> dict[str, float]:
    zf = np.asarray(z, dtype=np.float64)
    if len(zf) > max_samples:
        zf = zf[np.linspace(0, len(zf) - 1, max_samples, dtype=int)]
    zn = zf / (np.linalg.norm(zf, axis=-1, keepdims=True) + 1e-8)
    cos_vals = (zn @ zn.T)[np.triu_indices(len(zn), k=1)]
    unif = float(np.log(np.mean(np.exp(-2.0 * np.maximum(0.0, 2.0 - 2.0 * cos_vals))) + 1e-12))
    return {
        "pairwise_cos_mean": round(float(np.mean(cos_vals)), 4),
        "pairwise_cos_std": round(float(np.std(cos_vals)), 4),
        "wang_isola_uniformity": round(unif, 4),
    }


def _ks_drift_summary(cal_arr: np.ndarray, hold_arr: np.ndarray) -> dict[str, float]:
    c2, h2 = cal_arr.reshape(len(cal_arr), -1), hold_arr.reshape(len(hold_arr), -1)
    ks_vals = np.array([float(ks_2samp(c2[:, j], h2[:, j]).statistic) for j in range(c2.shape[1])])
    return {
        "ks_mean": round(float(ks_vals.mean()), 4), "ks_median": round(float(np.median(ks_vals)), 4),
        "ks_max": round(float(ks_vals.max()), 4), "ks_gt_015_frac": round(float((ks_vals > 0.15).mean()), 4),
    }


def _safe_auroc_prauc(y_bin: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    yb = np.asarray(y_bin, dtype=int)
    sc = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
    if len(np.unique(yb)) < 2:
        return 0.5, 0.0
    return float(roc_auc_score(yb, sc)), float(average_precision_score(yb, sc))


def _whiten_4block(model: nn.Module, zt: torch.Tensor, zp: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    m_cnt, d_dim = int(model.num_subblocks), int(model.latent_dim)
    raw = zt - zp
    white = whiten_residuals(raw.reshape(len(raw), m_cnt * d_dim) - model.residual_mean, model.precision_matrix)
    return raw.cpu().numpy(), white.reshape(len(raw), m_cnt, d_dim).cpu().numpy()


def _train_flagship_on_stream(
    ds: str, ch: str, seed: int = 42, epochs: int = 15, context_len: int = 256, suspect_len: int = 64,
    device: torch.device | None = None,
) -> dict[str, Any]:
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w_len = context_len + suspect_len
    train, test, y_test = load_stream(ds, ch)
    n_val = int(len(train) * 0.35)
    tr, va = train[:-n_val], train[-n_val:]
    mu, sd_arr = tr.mean(0), tr.std(0)
    sc = np.where(sd_arr < 1e-8, 1.0, np.maximum(sd_arr, 0.01))
    tr_s, va_s, te_s = (tr - mu) / sc, (va - mu) / sc, (test - mu) / sc
    tr_w = DataLoader.create_windows(tr_s, w_len, step=16, copy=False)
    va_w = DataLoader.create_windows(va_s, w_len, step=1, copy=False)
    te_w = DataLoader.create_windows(te_s, w_len, step=1, copy=False)

    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = CSMConfig(model_type="soliton_reynolds_jepa", context_size=context_len, suspect_size=suspect_len, latent_dim=32)
    model = build_ts_jepa_model(cfg, input_dim=train.shape[1], device=dev)
    untrained_copy = copy.deepcopy(model).eval()

    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epoch_losses: list[float] = []
    for _ in range(epochs):
        model.train()
        b_losses = []
        for b in batches(tr_w, 16):
            b = b.to(dev)
            loss, _ = model.compute_objective(b[:, :context_len], b[:, context_len:], cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.update_target_encoder(None)
            b_losses.append(float(loss.detach().item()))
        epoch_losses.append(float(np.mean(b_losses)))

    model.eval()
    model.fit_alien_bank(tr_w, context_len=context_len)
    n_cal_w = len(va_w) // 2
    hold_start = min(len(va_w) - 1, n_cal_w + suspect_len)

    model_wsp = copy.deepcopy(model).eval()
    wsp_cfg = WSPConfig(context_len=context_len, suspect_len=suspect_len, candidate_k_counts=(4, 2))
    wsp_res = apply_surgical_wsp_quench(model_wsp, tr_w, va_w[:n_cal_w], wsp_cfg)

    n_te_w = len(te_w)
    win_anom_frac = np.array([float(np.mean(y_test[i + context_len : i + w_len] > 0)) for i in range(n_te_w)], dtype=np.float32)
    win_pure_nom = np.array([not bool(np.any(y_test[i : i + w_len] > 0)) for i in range(n_te_w)], dtype=bool)
    return {
        "stream": f"{ds}:{ch}", "ds": ds, "ch": ch, "K": int(train.shape[1]),
        "len_tr": int(len(tr)), "len_va": int(len(va)), "len_test": int(len(test)),
        "n_cal_w": int(n_cal_w), "hold_start": int(hold_start), "cfg": cfg,
        "tr_w": tr_w, "va_w": va_w, "te_w": te_w, "y_test": y_test,
        "win_anom_frac": win_anom_frac, "win_tgt_anom": win_anom_frac > 0.0, "win_pure_nom": win_pure_nom,
        "epoch_losses": epoch_losses, "model_untrained": untrained_copy, "model_base": model,
        "model_wsp": model_wsp, "wsp_res": wsp_res,
    }


@torch.inference_mode()
def trace_encoder_functional_hierarchy(
    model: nn.Module, ctx_nom: torch.Tensor, tgt_nom: torch.Tensor, ctx_anom: torch.Tensor, tgt_anom: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    enc, tap_head = model.target_encoder, model.target_tap_head

    def _probe(x_in: torch.Tensor) -> dict[str, torch.Tensor]:
        acts: dict[str, torch.Tensor] = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []
        mk_hook = lambda tag: (lambda _m, _i, out: acts.update({tag: out.detach()}) if isinstance(out, torch.Tensor) else None)
        try:
            if enc.channel_stem is not None:
                handles.append(enc.channel_stem.register_forward_hook(mk_hook("01_channel_stem_conv")))
            handles.append(enc.input_projection.register_forward_hook(mk_hook("02_input_proj_1x1")))
            if enc.dyadic_block is not None:
                handles.append(enc.dyadic_block.register_forward_hook(mk_hook("03_dyadic_lp_block")))
            for idx_b, blk in enumerate(enc.blocks):
                handles.append(blk.register_forward_hook(mk_hook(f"04_tcn_block_{idx_b}")))
            handles.append(enc.temporal_dense.register_forward_hook(mk_hook("05_temporal_dense_128")))
            handles.append(enc.stat_dense.register_forward_hook(mk_hook("06_stat_dense_128")))
            handles.append(enc.bottleneck[0].register_forward_hook(mk_hook("07_bottleneck_pre_ln")))
            handles.append(enc.bottleneck[1].register_forward_hook(mk_hook("08_bottleneck_z_deep")))
            handles.append(tap_head.net.register_forward_hook(mk_hook("09_tap_head_net_raw")))
            z_deep = enc(x_in)
            taps = (acts["03_dyadic_lp_block"], acts[f"04_tcn_block_{(len(enc.blocks) - 1) // 3}"], acts[f"04_tcn_block_{len(enc.blocks) - 1}"])
            tap_vec = torch.cat([tap_head._tap_stats(t) for t in taps], dim=1)
            g_val = float((0.5 * torch.sigmoid(tap_head.gate)).item())
            z_tap_ln = F.layer_norm(tap_head.net(tap_vec), (model.latent_dim,))
            fused = (1.0 - g_val) * z_deep + g_val * z_tap_ln
            centered = fused - tap_head.running_centroid.unsqueeze(0)
            acts["10_tap_z_ln"] = z_tap_ln.detach()
            acts["11_tap_fused_pre_center"] = fused.detach()
            acts["12_barycenter_centered"] = centered.detach()
            acts["13_spherical_z_out"] = F.layer_norm(centered, (model.latent_dim,)).detach()
            if enc.dyadic_block is not None and enc.dyadic_block.last_shell_energy is not None:
                acts["shell_energy"] = enc.dyadic_block.last_shell_energy.detach().clone()
        finally:
            for h in handles:
                h.remove()
        return acts

    horizons_out: dict[str, Any] = {}
    for label, x_n, x_a in [("ctx_T256", ctx_nom, ctx_anom), ("tgt_m1_T16", tgt_nom[:, :16, :], tgt_anom[:, :16, :]), ("tgt_m4_T64", tgt_nom, tgt_anom)]:
        an, aa = _probe(x_n), _probe(x_a)
        seq_len = int(x_n.shape[1])
        stage_rows, prev_t = [], None
        for key in sorted(k for k in an if k != "shell_energy"):
            tn, ta = an[key].float(), aa[key].float()
            tn_win = tn.mean(dim=-1) if tn.ndim == 3 else tn
            div = float(torch.linalg.norm(ta - tn) / (torch.linalg.norm(tn) + 1e-8))
            seq_cos = None
            if prev_t is not None and prev_t.shape == tn.shape:
                seq_cos = round(float(F.cosine_similarity(prev_t.reshape(len(tn), -1), tn.reshape(len(tn), -1), dim=-1).mean().item()), 4)
            prev_t = tn
            eff_dil = enc.blocks[int(key.split("_")[-1])].effective_dilation_and_padding(seq_len)[0] if key.startswith("04_tcn_block_") else None
            stage_rows.append({
                "stage": key, "shape": list(tn.shape), "eff_dilation": eff_dil,
                "l2_norm": round(float(torch.linalg.norm(tn, dim=-1).mean().item()), 4),
                "sparsity": round(float((tn.abs() < 1e-6).float().mean().item()), 4),
                "batch_var": round(float(_compute_batch_variance(tn_win)), 5),
                "r_stab_window": round(float(_compute_stable_rank(tn_win)), 3),
                "r_stab_flat": round(float(_compute_stable_rank(tn)), 3),
                "seq_cos_prev": seq_cos, "anom_rel_divergence": round(div, 4),
            })
        sn_t, sa_t = an.get("shell_energy"), aa.get("shell_energy")
        shell_summary = None
        if sn_t is not None and sa_t is not None:
            sn, sa = sn_t.mean(dim=0).cpu().numpy(), sa_t.mean(dim=0).cpu().numpy()
            shell_summary = {
                "nom_frac": [round(float(x), 4) for x in (sn / (sn.sum() + 1e-12))],
                "anom_frac": [round(float(x), 4) for x in (sa / (sa.sum() + 1e-12))],
                "anom_to_nom_ratio": [round(float(a / (n + 1e-12)), 3) for n, a in zip(sn, sa)],
            }
        horizons_out[label] = {"stages": stage_rows, "dyadic_shells": shell_summary}

    return {
        "context_tap_gate_g": round(float((0.5 * torch.sigmoid(model.context_tap_head.gate)).item()), 4),
        "target_tap_gate_g": round(float((0.5 * torch.sigmoid(model.target_tap_head.gate)).item()), 4),
        "context_centroid_l2": round(float(model.context_tap_head.running_centroid.norm().item()), 4),
        "target_centroid_l2": round(float(model.target_tap_head.running_centroid.norm().item()), 4),
        "dyadic_gamma_ctx": round(float(model.context_encoder.dyadic_block.gamma.item()), 4),
        "dyadic_gamma_tgt": round(float(model.target_encoder.dyadic_block.gamma.item()), 4),
        "stem_gate_tanh_ctx": round(float(torch.tanh(model.context_encoder.stem_gate).item()), 4) if model.context_encoder.stem_gate is not None else 0.0,
        "horizons": horizons_out,
    }


@torch.inference_mode()
def run_full_leaf_autopsy(bundle: dict[str, Any], batch_size: int = 64) -> dict[str, Any]:
    dev = next(bundle["model_base"].parameters()).device
    C = bundle["cfg"].context_size
    hold_pool = bundle["va_w"][bundle["hold_start"] :]
    idx_nom = np.linspace(0, len(hold_pool) - 1, min(batch_size, len(hold_pool)), dtype=int)
    anom_idx_all = np.where(bundle["win_anom_frac"] >= 0.25)[0]
    if len(anom_idx_all) < 8:
        anom_idx_all = np.where(bundle["win_tgt_anom"])[0]
    idx_anom = anom_idx_all[np.linspace(0, len(anom_idx_all) - 1, min(batch_size, len(anom_idx_all)), dtype=int)]
    n_pair = min(len(idx_nom), len(idx_anom))
    nom_pair = torch.from_numpy(hold_pool[idx_nom[:n_pair]]).float().to(dev)
    anom_pair = torch.from_numpy(bundle["te_w"][idx_anom[:n_pair]]).float().to(dev)
    sample_nom = (nom_pair[:, :C, :], nom_pair[:, C:, :])
    sample_anom = (anom_pair[:, :C, :], anom_pair[:, C:, :])
    # Also run a pure target-only perturbation probe (same x_ctx, anomalous x_tgt) to verify exact causality
    sample_tgt_only_anom = (nom_pair[:, :C, :], anom_pair[:, C:, :])

    all_leaves = [n for n, m in bundle["model_base"].named_modules() if len(list(m.children())) == 0 and n != ""]
    autopsies: dict[str, Any] = {}
    for tag, mdl in [("untrained_init", bundle["model_untrained"]), ("trained_unquenched", bundle["model_base"]), ("flagship_laminar_jepa", bundle["model_wsp"])]:
        rep = autopsy_model(mdl, sample_input=sample_nom, anomalous_input=sample_anom, model_name=f"{bundle['stream']}::{tag}")
        records_list = [_clean_record(r) for r in rep.to_dataframe().to_dict(orient="records")]
        hit_names = {r["layer_name"].split("#")[0] for r in records_list}
        subsystems: dict[str, list[dict[str, Any]]] = {}
        for r in records_list:
            ln = r["layer_name"]
            grp = (
                "1_context_encoder" if ln.startswith("context_encoder.") else
                "2_context_tap_head" if ln.startswith("context_tap_head.") else
                "3_sub_predictor_m1_m3" if ln.startswith("sub_predictor.") else
                "4_predictor_m4" if ln.startswith("predictor.") else
                "5_target_encoder_4prefixes" if ln.startswith("target_encoder.") else
                "6_target_tap_head_4prefixes" if ln.startswith("target_tap_head.") else "7_other"
            )
            subsystems.setdefault(grp, []).append(r)

        grp_summary = {}
        for grp, rows in sorted(subsystems.items()):
            divs = [float(x["anomaly_divergence"]) for x in rows if x.get("anomaly_divergence") is not None]
            grp_summary[grp] = {
                "leaf_calls": len(rows),
                "mean_r_stab": round(float(np.mean([x["stable_rank"] for x in rows])), 3),
                "min_r_stab": round(float(np.min([x["stable_rank"] for x in rows])), 3),
                "max_sparsity": round(float(np.max([x["sparsity"] for x in rows])), 4),
                "mean_batch_var": round(float(np.mean([x["batch_variance"] for x in rows])), 5),
                "mean_anom_div": round(float(np.mean(divs)), 4) if divs else 0.0,
                "max_anom_div": round(float(np.max(divs)), 4) if divs else 0.0,
                "flagged_layers": [x["layer_name"] for x in rows if x["health_flag"] != "HEALTHY"],
            }
        autopsies[tag] = {
            "global_health": rep.global_health, "total_leaf_invocations": len(records_list),
            "dormant_leaf_modules": sorted(set(all_leaves) - hit_names),
            "collapsed_layers": rep.collapsed_layers, "dead_layers": rep.dead_layers, "unstable_layers": rep.unstable_layers,
            "subsystem_summary": grp_summary, "records": records_list,
        }

    rep_causal = autopsy_model(bundle["model_wsp"], sample_input=sample_nom, anomalous_input=sample_tgt_only_anom, model_name="causality_check")
    max_ctx_leak = max(
        (v for k, v in rep_causal.anomaly_divergence.items() if k.startswith(("context_encoder.", "context_tap_head.", "sub_predictor.", "predictor."))),
        default=0.0,
    )

    _, _, _, zsp_base, _ = bundle["model_base"](sample_nom[0], sample_nom[1])
    _, _, _, zsp_wsp, _ = bundle["model_wsp"](sample_nom[0], sample_nom[1])
    delta_pred = (zsp_wsp - zsp_base).cpu().numpy()
    rogue_coords = list(bundle["wsp_res"].rogue_coords)
    spillover: dict[str, Any] = {"wsp_mode": bundle["wsp_res"].mode, "rogue_coords": rogue_coords, "max_ctx_target_leakage": round(float(max_ctx_leak), 8)}
    if rogue_coords:
        unsel = [c for c in range(32) if c not in rogue_coords]
        rms_r = float(np.sqrt(np.mean(delta_pred[:, :, rogue_coords] ** 2)))
        rms_u = float(np.sqrt(np.mean(delta_pred[:, :, unsel] ** 2)))
        spillover.update({
            "rms_shift_rogue_coords": round(rms_r, 5), "rms_shift_unselected_coords": round(rms_u, 5),
            "unselected_to_rogue_rms_ratio": round(rms_u / (rms_r + 1e-12), 4),
            "max_abs_shift_rogue": round(float(np.max(np.abs(delta_pred[:, :, rogue_coords]))), 5),
            "max_abs_shift_unselected": round(float(np.max(np.abs(delta_pred[:, :, unsel]))), 5),
        })

    total_p = sum(p.numel() for p in bundle["model_base"].parameters())
    train_p = sum(p.numel() for p in bundle["model_base"].parameters() if p.requires_grad)
    dorm_p = sum(p.numel() for n, p in bundle["model_base"].named_parameters() if n.startswith(("saliency_gate.", "joint_saliency_gate.")))
    return {
        "stream": bundle["stream"],
        "param_inventory": {
            "total_parameters": int(total_p), "trainable_parameters": int(train_p),
            "ema_frozen_parameters": int(total_p - train_p), "dormant_1d_gate_parameters": int(dorm_p),
            "registered_buffer_elements": int(sum(b.numel() for b in bundle["model_base"].buffers())),
        },
        "quenching_spillover_and_causality": spillover,
        "functional_hierarchy": trace_encoder_functional_hierarchy(bundle["model_wsp"], sample_nom[0], sample_nom[1], sample_anom[0], sample_anom[1]),
        "autopsies": autopsies,
    }


@torch.inference_mode()
def diagnose_stream_latent_and_panel(bundle: dict[str, Any]) -> dict[str, Any]:
    model_base, model_wsp = bundle["model_base"], bundle["model_wsp"]
    C, S = bundle["cfg"].context_size, bundle["cfg"].suspect_size
    M, D = int(model_base.num_subblocks), int(model_base.latent_dim)
    n_cal_w, hold_start = bundle["n_cal_w"], bundle["hold_start"]
    dev = next(model_base.parameters()).device

    zc_tr, zt_tr = _extract_subblock_latents(model_base, bundle["tr_w"], C)
    zc_va, zt_va = _extract_subblock_latents(model_base, bundle["va_w"], C)
    zc_te, zt_te = _extract_subblock_latents(model_base, bundle["te_w"], C)
    zp_tr_base = _forward_predictors_with_phase(model_base, zc_tr, model_base.subblock_embeddings)
    zp_va_base = _forward_predictors_with_phase(model_base, zc_va, model_base.subblock_embeddings)
    zp_te_base = _forward_predictors_with_phase(model_base, zc_te, model_base.subblock_embeddings)
    zp_va_wsp = _forward_predictors_with_phase(model_wsp, zc_va, model_wsp.subblock_embeddings)
    zp_te_wsp = _forward_predictors_with_phase(model_wsp, zc_te, model_wsp.subblock_embeddings)

    def _enc_deep(wins: np.ndarray) -> np.ndarray:
        return np.concatenate([model_base.target_encoder(torch.from_numpy(wins[i : i + 512, C:, :]).float().to(dev)).cpu().numpy() for i in range(0, len(wins), 512)], axis=0)

    zc_tr_np, zt_tr_np, zp_tr_np = zc_tr.cpu().numpy(), zt_tr.cpu().numpy(), zp_tr_base.cpu().numpy()
    zt_va_np, zt_te_np = zt_va.cpu().numpy(), zt_te.cpu().numpy()
    anom_mask, nom_te_mask = bundle["win_tgt_anom"], bundle["win_pure_nom"]
    eval_mask = anom_mask | nom_te_mask
    y_win_eval = anom_mask[eval_mask].astype(int)

    manifold = {
        "z_ctx_train": {**_rankme_and_spectrum(zc_tr_np), **_isotropy_and_uniformity(zc_tr_np)},
        "z_tgt_horizons_train": {
            f"m{m+1}_S{(m+1)*16}": {
                **_rankme_and_spectrum(zt_tr_np[:, m, :]), **_isotropy_and_uniformity(zt_tr_np[:, m, :]),
                "ctx_alignment_mse": round(float(np.mean(np.sum((zc_tr_np - zt_tr_np[:, m, :]) ** 2, axis=-1))), 4),
                "pred_mse_train": round(float(np.mean(np.sum((zp_tr_np[:, m, :] - zt_tr_np[:, m, :]) ** 2, axis=-1))), 4),
            }
            for m in range(M)
        },
        "z_pred_m4_train": _rankme_and_spectrum(zp_tr_np[:, 3, :]),
        "z_deep_no_tap_no_barycenter_ks": _ks_drift_summary(_enc_deep(bundle["va_w"][:n_cal_w]), _enc_deep(bundle["va_w"][hold_start:])),
        "z_out_with_tap_and_barycenter_ks": _ks_drift_summary(zt_va_np[:n_cal_w, 3, :], zt_va_np[hold_start:, 3, :]),
    }

    raw_va_base, white_va_base = _whiten_4block(model_base, zt_va, zp_va_base)
    raw_te_base, white_te_base = _whiten_4block(model_base, zt_te, zp_te_base)
    stab_va_base, stab_te_base = stabilize_whitened_subblocks(white_va_base, white_te_base, raw_va_base, raw_te_base, n_cal_w)
    raw_va_wsp, white_va_wsp = _whiten_4block(model_wsp, zt_va, zp_va_wsp)
    raw_te_wsp, white_te_wsp = _whiten_4block(model_wsp, zt_te, zp_te_wsp)
    stab_va_wsp, stab_te_wsp = stabilize_whitened_subblocks(white_va_wsp, white_te_wsp, raw_va_wsp, raw_te_wsp, n_cal_w)

    def _res_stats(va_arr: np.ndarray, te_arr: np.ndarray) -> dict[str, Any]:
        cal_blk, hold_blk = va_arr[:n_cal_w], va_arr[hold_start:]
        hijack = (np.sqrt(np.mean(hold_blk**2, axis=0)) + 1e-5) / (np.sqrt(np.mean(cal_blk**2, axis=0)) + 1e-5)
        hold_norms = np.linalg.norm(hold_blk.reshape(len(hold_blk), -1), axis=-1)
        te_norms = np.linalg.norm(te_arr.reshape(len(te_arr), -1), axis=-1)
        auroc, prauc = _safe_auroc_prauc(y_win_eval, te_norms[eval_mask])
        r_lt_hold = model_base.lieb_thirring_soliton_ratio(torch.from_numpy(hold_blk - hold_blk.mean(axis=1, keepdims=True)).float()).numpy()
        r_lt_te = model_base.lieb_thirring_soliton_ratio(torch.from_numpy(te_arr - te_arr.mean(axis=1, keepdims=True)).float()).numpy()
        rlt_au, _ = _safe_auroc_prauc(y_win_eval, r_lt_te[eval_mask])
        return {
            **_rankme_and_spectrum(cal_blk.reshape(len(cal_blk), -1)), **_ks_drift_summary(cal_blk, hold_blk),
            "hijacker_max_hold_over_cal_rms": round(float(np.max(hijack)), 4),
            "hijacker_p95_hold_over_cal_rms": round(float(np.percentile(hijack, 95.0)), 4),
            "hold_norm_excess_kurtosis": round(float(compute_excess_kurtosis(hold_norms)), 3),
            "r_lt_hold_mean": round(float(np.mean(r_lt_hold)), 4), "r_lt_hold_min": round(float(np.min(r_lt_hold)), 4),
            "r_lt_anom_mean": round(float(np.mean(r_lt_te[anom_mask])), 4) if np.any(anom_mask) else 0.0,
            "r_lt_alone_auroc": round(rlt_au, 4), "frob_norm_win_auroc": round(auroc, 4), "frob_norm_win_prauc": round(prauc, 4),
        }

    def _prec_stats(pt: torch.Tensor) -> dict[str, float]:
        pm = pt.detach().cpu().numpy().astype(np.float64)
        ev = np.maximum(np.linalg.eigvalsh(0.5 * (pm + pm.T)), 1e-12)
        p = ev / ev.sum()
        return {
            "eff_rank": round(float(np.exp(-np.sum(p * np.log(np.maximum(p, 1e-12))))), 2),
            "lambda_max": round(float(ev[-1]), 3), "lambda_min": round(float(ev[0]), 6),
            "cond_number": round(float(ev[-1] / max(ev[0], 1e-8)), 1),
        }

    phi = ORTHOGONAL_LEGENDRE_4BASIS
    emb_modes = np.sum((phi.T @ model_base.subblock_embeddings.detach().cpu().numpy()) ** 2, axis=1)
    raw_tr_np = (zt_tr - zp_tr_base).cpu().numpy()
    res_modes = np.mean(np.sum(np.einsum("hm,nmd->nhd", phi.T, raw_tr_np) ** 2, axis=-1), axis=0)
    blind_rep = blind_reverse_engineer_weight_space(model_base)
    blind_acc = evaluate_blind_dial_reconstruction(blind_rep, raw_tr_np, white_va_base)

    zt_tr_m4, zt_te_m4 = zt_tr_np[:, 3, :], zt_te_np[:, 3, :]
    prec_z = np.linalg.inv(np.cov(zt_tr_m4, rowvar=False) + 1e-3 * np.eye(D))
    dz = zt_te_m4 - zt_tr_m4.mean(axis=0, keepdims=True)
    z_mahal = np.sqrt(np.maximum(0.0, np.sum((dz @ prec_z) * dz, axis=-1)))
    ref_z = zt_tr_m4[np.linspace(0, len(zt_tr_m4) - 1, min(1024, len(zt_tr_m4)), dtype=int)]
    dmat = np.sqrt(np.maximum(0.0, np.sum(zt_te_m4**2, axis=1, keepdims=True) + np.sum(ref_z**2, axis=1)[None, :] - 2.0 * (zt_te_m4 @ ref_z.T)))
    z_knn5 = np.mean(np.partition(dmat, kth=min(5, len(ref_z) - 1), axis=1)[:, :5], axis=1)
    s1d_te = np.concatenate([model_base.compute_predictive_discrepancy(torch.from_numpy(bundle["te_w"][i : i + 256, :C, :]).float().to(dev), torch.from_numpy(bundle["te_w"][i : i + 256, C:, :]).float().to(dev)).cpu().numpy() for i in range(0, len(bundle["te_w"]), 256)], axis=0)

    panel = MultiPerspectiveCircuitBreakerPanel(context_size=C, suspect_size=S)
    bits_base = panel.calibrate_and_trip_windows(stab_va_base, stab_te_base, n_cal_w)
    bits_wsp = panel.calibrate_and_trip_windows(stab_va_wsp, stab_te_wsp, n_cal_w)
    ntot_base = bits_base["bits224_te"].sum(axis=1) + bits_base["bits128_te"].sum(axis=1) + 2.0 * bits_base["fast_edge_te"].sum(axis=1)
    ntot_wsp = bits_wsp["bits224_te"].sum(axis=1) + bits_wsp["bits128_te"].sum(axis=1) + 2.0 * bits_wsp["fast_edge_te"].sum(axis=1)

    ladder = {
        k: {"auroc": round(au, 4), "prauc": round(ap, 4)}
        for k, vec in [
            ("1_ztgt_mahalanobis", z_mahal), ("2_ztgt_5nn_distance", z_knn5),
            ("3_raw_4horizon_frob", np.linalg.norm(raw_te_base.reshape(len(raw_te_base), -1), axis=-1)),
            ("4_1d_saliency_envelope", s1d_te), ("5_panel224_unquenched_ntot", ntot_base), ("6_panel224_laminar_jepa_ntot", ntot_wsp),
        ]
        for au, ap in [_safe_auroc_prauc(y_win_eval, vec[eval_mask])]
    }

    dials_cal = build_224_perspective_dials(stab_va_wsp)[:n_cal_w]
    d_norm = (dials_cal - dials_cal.mean(axis=0, keepdims=True)) / (dials_cal.std(axis=0, keepdims=True) + 1e-6)
    ev_224 = np.maximum(np.linalg.eigvalsh(0.5 * ((d_norm.T @ d_norm) / max(len(d_norm) - 1, 1) + ((d_norm.T @ d_norm) / max(len(d_norm) - 1, 1)).T)), 0.0)
    p_224 = ev_224 / (ev_224.sum() + 1e-12)
    dial_eff_rank = float(np.exp(-np.sum(p_224 * np.log(np.maximum(p_224, 1e-12)))))

    b224_hold = bits_wsp["bits224_va"][hold_start:]
    p_j = b224_hold.mean(axis=0)
    var_inf = float(np.var(b224_hold.sum(axis=1))) / max(float(np.sum(p_j * (1.0 - p_j))), 1e-6)

    family_slices = [
        ("pos_m1_16", bits_wsp["bits224_va"][:, 0:32], bits_wsp["bits224_te"][:, 0:32]),
        ("pos_m2_32", bits_wsp["bits224_va"][:, 32:64], bits_wsp["bits224_te"][:, 32:64]),
        ("pos_m3_48", bits_wsp["bits224_va"][:, 64:96], bits_wsp["bits224_te"][:, 64:96]),
        ("pos_m4_64", bits_wsp["bits224_va"][:, 96:128], bits_wsp["bits224_te"][:, 96:128]),
        ("drift_mean", bits_wsp["bits224_va"][:, 128:160], bits_wsp["bits224_te"][:, 128:160]),
        ("fluc_std", bits_wsp["bits224_va"][:, 160:192], bits_wsp["bits224_te"][:, 160:192]),
        ("accel_diff", bits_wsp["bits224_va"][:, 192:224], bits_wsp["bits224_te"][:, 192:224]),
        ("fast_edge_relay", bits_wsp["fast_edge_va"], bits_wsp["fast_edge_te"]),
    ]
    dial_families = {}
    for fname, b_va, b_te in family_slices:
        rh = float(b_va[hold_start:].mean())
        rn = float(b_te[nom_te_mask].mean()) if np.any(nom_te_mask) else 0.0
        ra = float(b_te[anom_mask].mean()) if np.any(anom_mask) else 0.0
        dial_families[fname] = {
            "hold_nom_trip_rate": round(rh, 5), "test_nom_trip_rate": round(rn, 5),
            "test_anom_trip_rate": round(ra, 5), "anom_over_tenom_enrichment": round(ra / max(rn, 1e-4), 2),
        }

    stream_decisions = {}
    for mapping in ("trailing", "smear"):
        for m_tag, b_dict in [("unquenched_224bit", bits_base), ("flagship_laminar_jepa", bits_wsp)]:
            dec = panel.evaluate_stream_decisions(b_dict, bundle["len_va"], bundle["len_test"], mapping)
            tl = bundle["y_test"][dec.valid_mask]
            mt = compute_metrics(tl, dec.test_pred_balanced, scores=dec.test_switch_counts, valid_mask=np.ones_like(tl, dtype=bool), use_pa=False)
            er, delays = event_stats(tl, dec.test_pred_balanced)
            stream_decisions[f"{mapping}::{m_tag}"] = {
                "hold_fpr": round(float(dec.hold_pred_balanced.mean()) if len(dec.hold_pred_balanced) else 0.0, 4),
                "test_fpr": round(float(dec.test_pred_balanced[tl == 0].mean()) if np.any(tl == 0) else 0.0, 4),
                "f1": round(float(mt["f1"]), 4), "precision": round(float(mt["precision"]), 4), "recall": round(float(mt["recall"]), 4),
                "prauc": round(float(mt.get("pr_auc", 0.0)), 4), "event_recall": round(float(er), 4),
                "mean_delay": round(float(np.mean(delays)), 1) if delays else None, "k_gate_224": round(float(dec.k_gate_balanced), 2),
            }

    return {
        "stream": bundle["stream"], "K": bundle["K"],
        "training_loss_start_end": [round(bundle["epoch_losses"][0], 4), round(bundle["epoch_losses"][-1], 4)],
        "wsp_mode": bundle["wsp_res"].mode, "wsp_rogue_coords": list(bundle["wsp_res"].rogue_coords),
        "wsp_pct_params_edited": round(float(bundle["wsp_res"].pct_predictor_edited), 4),
        "blind_weight_audit": {
            "dynamical_regime": blind_rep.dynamical_regime, "vorticity_ratio": round(blind_rep.vorticity_ratio, 3),
            "precision_eff_rank": round(blind_rep.effective_rank, 2),
            "role_reconstruction_acc": round(blind_acc["role_accuracy"], 4), "fragility_rank_corr": round(blind_acc["fragility_rank_corr"], 4),
        },
        "legendre_horizon_modes": {
            "subblock_embedding_pct_phi0_to_phi3": [round(float(100.0 * x / (emb_modes.sum() + 1e-12)), 2) for x in emb_modes],
            "train_raw_residual_pct_phi0_to_phi3": [round(float(100.0 * x / (res_modes.sum() + 1e-12)), 2) for x in res_modes],
        },
        "precision_spectrum": {"unquenched": _prec_stats(model_base.precision_matrix), "laminar_jepa": _prec_stats(model_wsp.precision_matrix)},
        "spherical_manifold": manifold,
        "residual_stages": {
            "S0_raw_unquenched": _res_stats(raw_va_base, raw_te_base), "S1_whitened_unquenched": _res_stats(white_va_base, white_te_base),
            "S2_stabilized_unquenched": _res_stats(stab_va_base, stab_te_base), "S3_flagship_laminar_jepa": _res_stats(stab_va_wsp, stab_te_wsp),
        },
        "separability_ladder_win": ladder,
        "dial_space_224": {"nominal_corr_eff_rank_of_224": round(dial_eff_rank, 2), "hold_trip_variance_inflation_vs_bernoulli": round(var_inf, 2), "families": dial_families},
        "timeline_metrics": stream_decisions,
    }


@torch.inference_mode()
def run_controlled_modality_probe(bundle: dict[str, Any]) -> dict[str, Any]:
    model_wsp = bundle["model_wsp"]
    C, S = bundle["cfg"].context_size, bundle["cfg"].suspect_size
    n_cal_w, hold_start, va_w = bundle["n_cal_w"], bundle["hold_start"], bundle["va_w"]
    zc_va, zt_va = _extract_subblock_latents(model_wsp, va_w, C)
    raw_va, white_va = _whiten_4block(model_wsp, zt_va, _forward_predictors_with_phase(model_wsp, zc_va, model_wsp.subblock_embeddings))

    hold_pool = va_w[hold_start:]
    clean_w = hold_pool[np.linspace(0, len(hold_pool) - 1, min(128, len(hold_pool)), dtype=int)].copy()
    k_act = min(4, clean_w.shape[2])

    w_spike = clean_w.copy()
    w_spike[:, C + 48 :, :k_act] += 5.0
    w_ctx = clean_w.copy()
    carrier = np.where((np.arange(S, dtype=np.float32) % 4) < 2, 1.0, -1.0)[None, :, None]
    seg = w_ctx[:, C:, :k_act]
    w_ctx[:, C:, :k_act] = seg.mean(axis=1, keepdims=True) + np.maximum(seg.std(axis=1, keepdims=True), 0.5) * carrier * 1.5
    w_freeze = clean_w.copy()
    w_freeze[:, C:, :k_act] = w_freeze[:, C - 1 : C, :k_act]
    w_ramp = clean_w.copy()
    w_ramp[:, C:, :k_act] += (0.02 * np.arange(1, S + 1, dtype=np.float32))[None, :, None]

    panel = MultiPerspectiveCircuitBreakerPanel(context_size=C, suspect_size=S)
    dev = next(model_wsp.parameters()).device
    nom_t = torch.from_numpy(clean_w[:64]).float().to(dev)
    stab_va_only, _ = stabilize_whitened_subblocks(white_va, white_va[:8], raw_va, raw_va[:8], n_cal_w)
    results = {}
    for tag, pert_w in [("clean_heldout_nominal", clean_w), ("right_edge_spike_5sigma", w_spike), ("contextual_harmonic_flip", w_ctx), ("silent_sensor_freeze", w_freeze), ("slow_incipient_ramp_002", w_ramp)]:
        zc_p, zt_p = _extract_subblock_latents(model_wsp, pert_w, C)
        raw_p, white_p = _whiten_4block(model_wsp, zt_p, _forward_predictors_with_phase(model_wsp, zc_p, model_wsp.subblock_embeddings))
        _, stab_p = stabilize_whitened_subblocks(white_va, white_p, raw_va, raw_p, n_cal_w)
        bits = panel.calibrate_and_trip_windows(stab_va_only, stab_p, n_cal_w)
        pert_t = torch.from_numpy(pert_w[:64]).float().to(dev)
        hier = trace_encoder_functional_hierarchy(model_wsp, nom_t[:, :C, :], nom_t[:, C:, :], pert_t[:, :C, :], pert_t[:, C:, :])
        st_map = {r["stage"]: r["anom_rel_divergence"] for r in hier["horizons"]["tgt_m4_T64"]["stages"]}
        b224, bedge = bits["bits224_te"], bits["fast_edge_te"]
        ntot = b224.sum(axis=1) + bits["bits128_te"].sum(axis=1) + 2.0 * bedge.sum(axis=1)
        cal_ntot = bits["bits224_va"][:n_cal_w].sum(axis=1) + bits["bits128_va"][:n_cal_w].sum(axis=1) + 2.0 * bits["fast_edge_va"][:n_cal_w].sum(axis=1)
        k_q = max(1.12 * float(np.percentile(cal_ntot, 99.5)), 5.0)
        results[tag] = {
            "window_alarm_rate": round(float(np.mean(ntot > k_q)), 4), "mean_ntot_switches": round(float(np.mean(ntot)), 2),
            "encoder_divergence": {k: st_map.get(v, 0.0) for k, v in [("dyadic_lp", "03_dyadic_lp_block"), ("tcn_block_0", "04_tcn_block_0"), ("tcn_block_5", "04_tcn_block_5"), ("temporal_dense", "05_temporal_dense_128"), ("stat_dense", "06_stat_dense_128"), ("spherical_z_out", "13_spherical_z_out")]},
            "dial_trip_rates": {"pos_m1": round(float(b224[:, 0:32].mean()), 4), "pos_m4": round(float(b224[:, 96:128].mean()), 4), "drift": round(float(b224[:, 128:160].mean()), 4), "fluc": round(float(b224[:, 160:192].mean()), 4), "accel": round(float(b224[:, 192:224].mean()), 4), "fast_edge": round(float(bedge.mean()), 4)},
        }
    return results


def summarize_archive_162_csv(csv_path: Path) -> dict[str, Any] | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    tr, sm = df[df["mapping"] == "trailing"].copy(), df[df["mapping"] == "smear"].copy()
    act = tr[tr["exp1_mode"].str.startswith("surgical_")]
    return {
        "n_streams": int(len(tr)), "n_datasets": int(tr["ds"].nunique()),
        "wsp_mode_counts": {str(k): int(v) for k, v in tr["exp1_mode"].value_counts().to_dict().items()},
        "active_quench_fraction": round(float(len(act) / max(len(tr), 1)), 4),
        "regime_counts": {str(k): int(v) for k, v in tr["exp2_regime"].value_counts().to_dict().items()},
        "mean_role_reconstruction_acc": round(float(tr["exp2_role_acc"].mean()), 4),
        "mean_fragility_rank_corr": round(float(tr["exp2_frag_corr"].mean()), 4),
        "mean_sawtooth_phi3_pct": round(float(tr["exp3_sawtooth_before"].mean()), 2),
        "all_162_trailing": {m: round(float(tr[m].mean()), 4) for m in ("base_hold_fpr", "wsp_hold_fpr", "base_test_fpr", "wsp_test_fpr", "base_f1", "wsp_f1", "base_event_recall", "wsp_event_recall")},
        "active_48_streams_trailing": {m: round(float(act[m].mean()), 4) for m in ("base_hold_fpr", "wsp_hold_fpr", "base_test_fpr", "wsp_test_fpr", "base_f1", "wsp_f1")},
        "all_162_smear": {m: round(float(sm[m].mean()), 4) for m in ("base_f1", "wsp_f1", "base_event_recall", "wsp_event_recall")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Flagship Laminar-JEPA layer activations and latent space.")
    parser.add_argument("--streams", nargs="*", default=list(DEFAULT_STREAMS))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "reports" / "flagship_diagnosis"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Flagship Laminar-JEPA Autopsy on {dev} across {len(args.streams)} streams...")

    stream_diagnoses, stream_traces, modality_probe = [], {}, {}
    for idx_s, s_spec in enumerate(args.streams, start=1):
        ds, ch = s_spec.split(":")
        print(f"[{idx_s}/{len(args.streams)}] Training & tracing {s_spec} (epochs={args.epochs})...")
        bundle = _train_flagship_on_stream(ds, ch, seed=args.seed, epochs=args.epochs, device=dev)
        trace_out = run_full_leaf_autopsy(bundle)
        diag_out = diagnose_stream_latent_and_panel(bundle)
        stream_diagnoses.append(diag_out)
        if s_spec == "SMAP:P-3" or idx_s == 1:
            stream_traces[s_spec] = trace_out
        else:
            stream_traces[s_spec] = {**trace_out, "autopsies": {k: {kk: vv for kk, vv in v.items() if kk != "records"} for k, v in trace_out["autopsies"].items()}}
        if s_spec == "SMAP:P-3" or (not modality_probe and idx_s == len(args.streams)):
            modality_probe = {"reference_stream": s_spec, "probes": run_controlled_modality_probe(bundle)}

        s3, lad = diag_out["residual_stages"]["S3_flagship_laminar_jepa"], diag_out["separability_ladder_win"]
        print(
            f"    WSP={diag_out['wsp_mode']:13s} | RankMe(z_tgt)={diag_out['spherical_manifold']['z_tgt_horizons_train']['m4_S64']['eff_rank_rankme']:5.2f}/32 "
            f"| HijackRMS: Whitened={diag_out['residual_stages']['S1_whitened_unquenched']['hijacker_max_hold_over_cal_rms']:5.2f} -> WSP={s3['hijacker_max_hold_over_cal_rms']:5.2f} "
            f"| WinAUROC: 1D={lad['4_1d_saliency_envelope']['auroc']:.3f} -> WSP224={lad['6_panel224_laminar_jepa_ntot']['auroc']:.3f}"
        )

    json_path = out_dir / "laminar_jepa_autopsy.json"
    _atomic_write_json(json_path, {
        "device": str(dev), "seed": args.seed, "epochs": args.epochs,
        "archive_162_summary": summarize_archive_162_csv(ROOT / "reports" / "wsp_soliton_166streams.csv"),
        "controlled_modality_probe": modality_probe,
        "stream_latent_diagnoses": stream_diagnoses,
        "stream_layer_traces": stream_traces,
    })
    print(f"\nSaved complete Flagship Laminar-JEPA diagnostic payload to {json_path}")


if __name__ == "__main__":
    main()
