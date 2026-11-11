"""Quenching cross-coordinate LayerNorm spillover and held-out archive component ablation.

Supports two reproducible protocols:
  - ``--cohort dev19``: Evaluates LayerNorm cross-coordinate coupling and before/after fault
    metrics on the cached 19-stream development checkpoints (seed 42).
  - ``--cohort untouched118``: Runs the full held-out archive component-level ablation across
    all 118 untouched streams of ``reports/streams_141_tier2.txt`` (excluding all 23 development
    streams), isolating:
      0_unquenched          : Unquenched SolitonReynoldsJEPAModel + stabilized 224-dial panel
      1_predictor_only_k4   : Predictor MLP row/bias + output LayerNorm gain damping only
      2_embedding_only_k4   : Additive horizon-embedding shift (+0.35 * (mu_cal - mu_tr)) only
      3_buffers_only_k4     : Residual-mean blend + precision decoupling/SPSD/RMS scaling only
      4_complete_ungated_k4 : Complete k=4 parameter + buffer sequence without split gating
      5_complete_gated      : Complete nominal-split-gated intervention (k in {4, 2}, Laminar-JEPA)
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.circuit_breaker_19stream.common import C, S, VAL_FRAC, W, cache_path, load_streams_19
from scripts.experiment_wsp_soliton import extract_raw4_white4_from_latents, select_cohort_streams
from scripts.run_multiseed_heldout_benchmark import batches, compute_metrics, event_stats, load_stream
from scripts.run_tier2_exhaustive_benchmark import atomic_save_csv
from src.config import CSMConfig
from src.data.data_loader import DataLoader
from src.engine.builder import build_ts_jepa_model
from src.experimental.phase_embedding_synthesis import (
    _extract_subblock_latents,
    _forward_predictors_with_phase,
)
from src.experimental.weight_space_programming import (
    WSPConfig,
    apply_surgical_wsp_quench,
    inspect_soliton_reynolds_weight_space,
)
from src.scoring.circuit_breaker_panel import (
    MultiPerspectiveCircuitBreakerPanel,
    stabilize_whitened_subblocks,
)


def relative_coordinate_changes(
    before: np.ndarray, after: np.ndarray, selected: list[int]
) -> dict[str, float]:
    """Quantify RMS residual change on selected vs. unselected coordinates (LayerNorm coupling)."""
    if before.shape != after.shape or before.ndim != 3 or before.shape[0] == 0:
        raise ValueError("Expected nonempty matched residual tensors [windows, horizons, coordinates]")
    if not selected:
        return {"selected_median": 0.0, "other_median": 0.0, "other_above_10pct": 0.0}
    rms = np.sqrt(np.mean(before.astype(np.float64) ** 2, axis=(0, 1)))
    change = np.sqrt(np.mean((after.astype(np.float64) - before.astype(np.float64)) ** 2, axis=(0, 1)))
    relative = change / np.maximum(rms, 1e-4)
    others = np.delete(relative, selected)
    selected_values = relative[selected]
    return {
        "selected_median": float(np.median(selected_values)) if len(selected_values) else 0.0,
        "other_median": float(np.median(others)) if len(others) else 0.0,
        "other_above_10pct": float(np.mean(others > 0.10)) if len(others) else 0.0,
    }


def score_panel_both_mappings(
    va_raw: np.ndarray,
    va_white: np.ndarray,
    te_raw: np.ndarray,
    te_white: np.ndarray,
    len_va: int,
    len_te: int,
    y: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Evaluate 224-dial panel under causal trailing and buffered smear attribution."""
    n_cal = len(va_raw) // 2
    va, te = stabilize_whitened_subblocks(va_white, te_white, va_raw, te_raw, n_cal)
    panel = MultiPerspectiveCircuitBreakerPanel(context_size=C, suspect_size=S)
    decisions = panel.calibrate_and_trip_windows(va, te, n_cal)
    out: dict[str, dict[str, float]] = {}
    for mapping in ("trailing", "smear"):
        dec = panel.evaluate_stream_decisions(decisions, len_va, len_te, mapping)
        labels = y[dec.valid_mask]
        mt = compute_metrics(
            labels,
            dec.test_pred_balanced,
            scores=dec.test_switch_counts,
            valid_mask=np.ones_like(labels, dtype=bool),
            use_pa=False,
        )
        er, dl = event_stats(labels, dec.test_pred_balanced)
        pr = float(average_precision_score(labels, dec.test_switch_counts)) if labels.sum() > 0 else float("nan")
        hfpr = float(dec.hold_pred_balanced.mean()) if len(dec.hold_pred_balanced) > 0 else 0.0
        tfpr = float(dec.test_pred_balanced[labels == 0].mean()) if np.any(labels == 0) else 0.0
        out[mapping] = {
            "f1": float(mt["f1"]),
            "precision": float(mt["precision"]),
            "recall": float(mt["recall"]),
            "event_recall": float(er),
            "pr_auc": pr,
            "hold_fpr": hfpr,
            "test_fpr": tfpr,
            "delay": float(dl),
        }
    return out


@torch.no_grad()
def apply_quench_component_subset(
    model: torch.nn.Module,
    rogue_sub: list[int],
    rogue_z: np.ndarray,
    tr_mean: np.ndarray,
    cal_mean: np.ndarray,
    cal_latents: tuple[torch.Tensor, torch.Tensor],
    cfg: WSPConfig,
    edit_predictor: bool,
    edit_embedding: bool,
    edit_buffers: bool,
) -> np.ndarray:
    """Apply an isolated subset of the surgical quenching edits in exact implementation order."""
    dev = next(model.parameters()).device
    M, D = int(model.num_subblocks), int(model.latent_dim)
    for c in rogue_sub:
        if edit_predictor:
            damp = float(np.clip(1.0 / (1.0 + 0.5 * max(0.0, float(rogue_z[c]))), cfg.min_damp, cfg.max_damp))
            for net_mod in (model.predictor.net, model.sub_predictor.net):
                net_mod[4].weight.data[c, :].mul_(damp)
                net_mod[4].bias.data[c].mul_(damp)
                net_mod[6].weight.data[c].mul_(damp)
        if edit_embedding:
            phase_corr = torch.from_numpy((cal_mean[:, c] - tr_mean[:, c]) * cfg.phase_align_step).to(
                dev, dtype=torch.float32
            )
            model.subblock_embeddings.data[:, c].add_(phase_corr)

    zc_cal, zt_cal = cal_latents
    cal_raw4_post = (
        (zt_cal - _forward_predictors_with_phase(model, zc_cal, model.subblock_embeddings))
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    if not edit_buffers:
        return cal_raw4_post

    cal_mean_post = cal_raw4_post.reshape(len(cal_raw4_post), M * D).mean(axis=0)
    res_mean_new = model.residual_mean.detach().cpu().numpy().copy()
    w_blend = float(cfg.residual_mean_blend)
    for c in rogue_sub:
        for m_idx in range(M):
            flat_idx = m_idx * D + c
            res_mean_new[flat_idx] = (1.0 - w_blend) * res_mean_new[flat_idx] + w_blend * cal_mean_post[flat_idx]
    model.residual_mean.copy_(torch.from_numpy(res_mean_new).to(dev, dtype=model.residual_mean.dtype))

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
    rms_cal = np.sqrt(np.mean(w_cal**2, axis=0)) + 1e-5

    target_rms = np.full(M * D, cfg.clean_passband_rms, dtype=np.float64)
    for c in rogue_sub:
        g_factor = float(
            np.clip(1.32 + 0.04 * max(0.0, float(rogue_z[c]) - 0.5), cfg.min_guard_factor, cfg.max_guard_factor)
        )
        for m_idx in range(M):
            target_rms[m_idx * D + c] = cfg.clean_passband_rms * 1.50 * g_factor
    scale_vec = target_rms / rms_cal
    prec_prog = (prec_spsd * scale_vec[:, None]) * scale_vec[None, :]
    model.precision_matrix.copy_(torch.from_numpy(prec_prog).to(dev, dtype=model.precision_matrix.dtype))
    return cal_raw4_post


def evaluate_dev19_stream(ds: str, ch: str) -> dict:
    train, test, labels = load_stream(ds, ch)
    n_val = int(len(train) * VAL_FRAC)
    tr, va = train[:-n_val], train[-n_val:]
    mu = tr.mean(axis=0, dtype=np.float64)
    sd = tr.std(axis=0, dtype=np.float64)
    scale = np.where(sd < 1e-8, 1.0, np.maximum(sd, 0.01))
    tr_w = DataLoader.create_windows(((tr - mu) / scale).astype(np.float32), W, step=16, copy=False)
    va_w = DataLoader.create_windows(((va - mu) / scale).astype(np.float32), W, step=1, copy=False)
    pt_path = cache_path(ds, ch, 42).with_suffix(".pt")
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    cfg = CSMConfig(model_type="soliton_reynolds_jepa", context_size=C, suspect_size=S, latent_dim=32)
    model = build_ts_jepa_model(cfg, input_dim=train.shape[1], device=torch.device("cpu"))
    model.load_state_dict(ckpt["state_dict"])
    zc_va, zt_va = ckpt["z_ctx_va"], ckpt["z_tgt_va"]
    zc_te, zt_te = ckpt["z_ctx_te"], ckpt["z_tgt_te"]
    if len(test) > 15000:
        anomaly = np.flatnonzero(labels == 1)
        start = max(0, int(anomaly[0]) - 2000) if len(anomaly) else 0
        end = min(len(test), start + 15000)
        start = max(0, end - 15000)
        labels = labels[start:end]

    pre_va, pre_va_white = extract_raw4_white4_from_latents(model, zc_va, zt_va)
    pre_te, pre_te_white = extract_raw4_white4_from_latents(model, zc_te, zt_te)
    pre = score_panel_both_mappings(pre_va, pre_va_white, pre_te, pre_te_white, len(va), len(labels), labels)["trailing"]
    result = apply_surgical_wsp_quench(model, tr_w, va_w[: len(va_w) // 2])
    if result.rogue_coords:
        post_va, post_va_white = extract_raw4_white4_from_latents(model, zc_va, zt_va)
        post_te, post_te_white = extract_raw4_white4_from_latents(model, zc_te, zt_te)
        post = score_panel_both_mappings(post_va, post_va_white, post_te, post_te_white, len(va), len(labels), labels)["trailing"]
    else:
        post_va, post_te, post = pre_va, pre_te, pre

    n_cal = len(pre_va) // 2
    hold = relative_coordinate_changes(pre_va[n_cal + S :], post_va[n_cal + S :], result.rogue_coords)
    nominal_endpoints = labels[W - 1 :] == 0
    nominal = relative_coordinate_changes(pre_te[nominal_endpoints], post_te[nominal_endpoints], result.rogue_coords)
    return dict(
        stream=f"{ds}:{ch}",
        seed=42,
        protocol="dev19_quench_spillover_v1",
        mode=result.mode,
        n_selected=len(result.rogue_coords),
        selected="|".join(map(str, result.rogue_coords)),
        **{f"hold_{k}": v for k, v in hold.items()},
        **{f"nominal_test_{k}": v for k, v in nominal.items()},
        **{f"pre_{k}": v for k, v in pre.items()},
        **{f"post_{k}": v for k, v in post.items()},
    )


def evaluate_untouched_stream_components(
    s_name: str, seed: int, epochs: int, device: torch.device, wsp_cfg: WSPConfig
) -> list[dict]:
    ds, ch = s_name.split(":", 1)
    train, test, y = load_stream(ds, ch)
    n_val = int(len(train) * 0.35)
    tr, va = train[:-n_val], train[-n_val:]
    if len(tr) <= W + 16 or len(va) <= W + 200 or len(test) <= W + 64:
        return []
    mu = tr.mean(axis=0, dtype=np.float64)
    sd = tr.std(axis=0, dtype=np.float64)
    sc = np.where(sd < 1e-8, 1.0, np.maximum(sd, 0.01))
    tr_w = DataLoader.create_windows(((tr - mu) / sc).astype(np.float32), W, step=16, copy=False)
    va_w = DataLoader.create_windows(((va - mu) / sc).astype(np.float32), W, step=1, copy=False)
    te_w = DataLoader.create_windows(((test - mu) / sc).astype(np.float32), W, step=1, copy=False)
    n_cal_w = len(va_w) // 2

    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = CSMConfig(model_type="soliton_reynolds_jepa", context_size=C, suspect_size=S, latent_dim=32)
    model = build_ts_jepa_model(cfg, input_dim=train.shape[1], device=device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for b in batches(tr_w, 16):
            b = b.to(device)
            loss, _ = model.compute_objective(b[:, :C], b[:, C:], cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.update_target_encoder(None)
    model.fit_alien_bank(tr_w, context_len=C)

    zc_va, zt_va = _extract_subblock_latents(model, va_w, C)
    zc_te, zt_te = _extract_subblock_latents(model, te_w, C)
    cal_lat = (zc_va[:n_cal_w], zt_va[:n_cal_w])

    pre_va, pre_va_w = extract_raw4_white4_from_latents(model, zc_va, zt_va)
    pre_te, pre_te_w = extract_raw4_white4_from_latents(model, zc_te, zt_te)
    m_unq = score_panel_both_mappings(pre_va, pre_va_w, pre_te, pre_te_w, len(va), len(test), y)

    report, aux = inspect_soliton_reynolds_weight_space(model, tr_w, va_w[:n_cal_w], wsp_cfg)
    top4 = report.ranked_coords[:4]
    tr_mean, cal_mean, rogue_z = aux["tr_mean"], aux["cal_mean"], aux["rogue_z"]
    cal_rms_pre = np.sqrt(np.mean(pre_va[:n_cal_w].astype(np.float64) ** 2, axis=(0, 1)))

    arms: list[tuple[str, torch.nn.Module, list[int], float, str]] = [
        ("0_unquenched", model, [], float("nan"), "none"),
    ]
    specs = [
        ("1_predictor_only_k4", True, False, False),
        ("2_embedding_only_k4", False, True, False),
        ("3_buffers_only_k4", False, False, True),
        ("4_complete_ungated_k4", True, True, True),
    ]
    for arm_name, ed_p, ed_e, ed_b in specs:
        m_arm = copy.deepcopy(model)
        cal_post = apply_quench_component_subset(
            m_arm, top4, rogue_z, tr_mean, cal_mean, cal_lat, wsp_cfg, ed_p, ed_e, ed_b
        )
        cal_rms_post = np.sqrt(np.mean(cal_post.astype(np.float64) ** 2, axis=(0, 1)))
        red_frac = float(np.mean(cal_rms_post[top4] < cal_rms_pre[top4]))
        arms.append((arm_name, m_arm, top4, red_frac, "ungated_k4"))

    m_gated = copy.deepcopy(model)
    res_gated = apply_surgical_wsp_quench(m_gated, tr_w, va_w[:n_cal_w], wsp_cfg)
    arms.append(("5_complete_gated", m_gated, res_gated.rogue_coords, float("nan"), res_gated.mode))

    out_rows: list[dict] = []
    for arm_name, m_obj, sel_coords, red_frac, gate_mode in arms:
        if arm_name == "0_unquenched" or (arm_name == "5_complete_gated" and not sel_coords):
            va_r, te_r, sc = pre_va, pre_te, m_unq
        else:
            va_r, va_w_post = extract_raw4_white4_from_latents(m_obj, zc_va, zt_va)
            te_r, te_w_post = extract_raw4_white4_from_latents(m_obj, zc_te, zt_te)
            sc = score_panel_both_mappings(va_r, va_w_post, te_r, te_w_post, len(va), len(test), y)
        hold_sp = relative_coordinate_changes(pre_va[n_cal_w + S :], va_r[n_cal_w + S :], sel_coords)
        for mapping in ("trailing", "smear"):
            out_rows.append(
                {
                    "stream": s_name,
                    "dataset": ds.lower(),
                    "seed": seed,
                    "protocol": "untouched118_component_ablation_v1",
                    "arm": arm_name,
                    "gate_mode": res_gated.mode if arm_name == "5_complete_gated" else gate_mode,
                    "gated_accepted": bool(res_gated.rogue_coords),
                    "n_selected": len(sel_coords),
                    "emb_cal_rms_reduced_frac": red_frac,
                    "hold_selected_median": hold_sp["selected_median"],
                    "hold_other_median": hold_sp["other_median"],
                    "hold_other_above_10pct": hold_sp["other_above_10pct"],
                    "mapping": mapping,
                    **sc[mapping],
                }
            )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=("dev19", "untouched118"), default="dev19")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--max-streams", type=int, default=0)
    args = ap.parse_args()

    if args.cohort == "dev19":
        out_path = args.out or (ROOT / "reports" / "quenching_spillover_dev19.csv")
        torch.set_num_threads(2)
        rows: list[dict] = []
        if out_path.exists():
            prev = pd.read_csv(out_path)
            rows = prev.to_dict("records")
        completed = {r["stream"] for r in rows}
        streams = load_streams_19()
        if args.max_streams:
            streams = streams[: args.max_streams]
        for ds, ch in streams:
            if f"{ds}:{ch}" in completed:
                continue
            row = evaluate_dev19_stream(ds, ch)
            rows.append(row)
            atomic_save_csv(pd.DataFrame(rows), out_path)
            print(f"{row['stream']}: {row['mode']}, selected={row['n_selected']}", flush=True)
        return

    out_path = args.out or (ROOT / "reports" / "quenching_archive_component_ablation_118stream.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wsp_cfg = WSPConfig(context_len=C, suspect_len=S, candidate_k_counts=(4, 2))
    rows = pd.read_csv(out_path).to_dict("records") if out_path.exists() else []
    completed = {r["stream"] for r in rows}
    streams_118 = select_cohort_streams("untouched118", "", args.max_streams)
    t0 = time.time()
    for idx, s_name in enumerate(streams_118, 1):
        if s_name in completed:
            continue
        s_rows = evaluate_untouched_stream_components(s_name, args.seed, args.epochs, device, wsp_cfg)
        rows.extend(s_rows)
        atomic_save_csv(pd.DataFrame(rows), out_path)
        tr_g = next(r for r in s_rows if r["arm"] == "5_complete_gated" and r["mapping"] == "trailing")
        tr_u = next(r for r in s_rows if r["arm"] == "0_unquenched" and r["mapping"] == "trailing")
        print(
            f"[{idx:03d}/{len(streams_118):03d} | {time.time() - t0:5.1f}s] {s_name:22s} "
            f"mode={tr_g['gate_mode']:13s} hFPR {tr_u['hold_fpr']:.4f}->{tr_g['hold_fpr']:.4f} "
            f"F1 {tr_u['f1']:.4f}->{tr_g['f1']:.4f}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    for mp in ("trailing", "smear"):
        sub = df[df["mapping"] == mp]
        print(f"\n=== UNTOUCHED HELD-OUT ARCHIVE QUENCHING COMPONENT ABLATION ({mp.upper()}, n={sub['stream'].nunique()}) ===")
        cols = ["hold_fpr", "test_fpr", "f1", "event_recall", "pr_auc", "hold_other_median", "hold_other_above_10pct"]
        print(sub.groupby("arm")[cols].mean().round(4).to_string())
        acc = sub[sub["gated_accepted"]]
        if not acc.empty:
            print(f"\n--- Subset where acceptance gate fired (n={acc['stream'].nunique()} streams) ---")
            print(acc.groupby("arm")[cols].mean().round(4).to_string())
    emb_sub = df[(df["arm"] == "2_embedding_only_k4") & (df["mapping"] == "trailing")]
    print(f"\nEmbedding-only raw cal-RMS reduction fraction across top-4 coords: {emb_sub['emb_cal_rms_reduced_frac'].mean():.4f}")


if __name__ == "__main__":
    main()
