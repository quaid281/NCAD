#!/usr/bin/env python3
"""Comprehensive R2 IEEE TNNLS Audit & External Baseline Verification Suite.

Executes the four Stage-1 empirical audits requested in the second-round review:
  - Part A (Item 9): PCA-LinPred-LN, TranAD, and TimesNet on 118 untouched streams (3 seeds, 5-block, SPOT).
  - Part B (Item 2): Unseen vs. tuned cohorts, 9-fold untouched exclusion, and Nested LODO (seed=42).
  - Part C (Items 3, 5, 6): Delay CIs, N_cal >= 500 cluster CIs/sign-flip p, and 118-stream eligibility funnel.
  - Part D (Items 4, 7): Fault contamination hazard and LOCO physical-channel attribution.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reviewer_analyses.ramp_severity_sweep import longest_nominal_run
from scripts.run_laminar_paired_benchmarks import (
    C, CSMConfig, S, W, _build_schedule, _fast_val_checkpoint_train, _record_1d_spot_rows,
    _split_nominal_5block, _split_nominal_dev_4block, _standardize_arrays, _train_fixed_epochs,
    compute_metrics, event_stats, load_stream,
)
from scripts.run_multiseed_heldout_benchmark import train_model
from scripts.tnnls_revision.common_eval import (
    QuarantinedRecalibrationController, atomic_write_csv, atomic_write_json, loco_channel_attribution,
)
from src.data.data_loader import DataLoader
from src.engine.trainer import build_ts_jepa_model
from src.models.baselines import TimesNet, TranAD
from src.models.laminar_lite import LaminarJEPALiteModel
from src.scoring.circuit_breaker_panel import aggregate_vector_stream, stabilize_whitened_subblocks
from src.scoring.laminar_lite_panel import build_position_drift_dials

ARCHIVE_CSV = ROOT / "reports" / "laminar_tier2_141stream_5block.csv"
FAIL_ATTR_CSV = ROOT / "reports" / "archive_118_failure_attribution.csv"
EXT_CSV = ROOT / "reports" / "archive_118_external_baselines.csv"
OUT_JSON = ROOT / "reports" / "r2_reviewer_audit_summary.json"
SEEDS_3 = (42, 123, 456)


def _boot_ci(vals: np.ndarray, n_boot: int = 4000, seed: int = 42) -> list[float]:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [float("nan"), float("nan")]
    idx = np.random.default_rng(seed).integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _cluster_ci(df_sub: pd.DataFrame, col_a: str, col_b: str | None = None, n_boot: int = 4000, seed: int = 42) -> list[float]:
    diff_s = df_sub[col_a] if col_b is None else (df_sub[col_a] - df_sub[col_b])
    work = pd.DataFrame({"ds": df_sub["dataset"].to_numpy(), "v": diff_s.to_numpy(dtype=np.float64)}).dropna()
    if len(work) == 0:
        return [float("nan"), float("nan")]
    groups = [g["v"].to_numpy() for _, g in work.groupby("ds", sort=True)]
    g_sums = np.array([g.sum() for g in groups], dtype=np.float64)
    g_cnts = np.array([g.size for g in groups], dtype=np.float64)
    draw = np.random.default_rng(seed).integers(0, len(groups), size=(n_boot, len(groups)))
    means = g_sums[draw].sum(axis=1) / np.maximum(g_cnts[draw].sum(axis=1), 1.0)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _sign_flip_p(df_sub: pd.DataFrame, col_a: str, col_b: str) -> float:
    ds_diffs = (df_sub[col_a] - df_sub[col_b]).groupby(df_sub["dataset"]).mean().dropna().to_numpy()
    d_cnt = int(ds_diffs.size)
    if d_cnt == 0:
        return 1.0
    obs = abs(float(ds_diffs.mean()))
    signs = np.array(np.meshgrid(*[[-1.0, 1.0]] * d_cnt)).T.reshape(-1, d_cnt)
    return float(np.mean(np.abs((signs * ds_diffs[None, :]).mean(axis=1)) >= obs - 1e-12))


def _wilcoxon_p(diffs: np.ndarray) -> float:
    arr = np.asarray(diffs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return 1.0 if (arr.size < 2 or np.allclose(arr, 0.0)) else float(stats.wilcoxon(arr).pvalue)


def _train_eval_reconstruction_baseline(
    model_name: str, tr_w: np.ndarray, val_mon_w: np.ndarray, va_w: np.ndarray, te_w: np.ndarray, k_ch: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray, int]:
    """Train TranAD or TimesNet under 5-block early stopping (`val_mon_w`) and return `(va_s, te_s, params)`."""
    tr_sub = tr_w[np.linspace(0, len(tr_w) - 1, 1536, dtype=int)] if len(tr_w) > 1536 else tr_w
    model: torch.nn.Module = (
        TranAD(c_in=k_ch, d_model=48, n_heads=4, e_layers=2, d_layers=2, d_ff=96)
        if model_name == "TranAD" else TimesNet(c_in=k_ch, d_model=48, d_ff=96, e_layers=2)
    ).to(device)
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-5)
    tr_gpu = torch.from_numpy(np.ascontiguousarray(tr_sub)).float().to(device)
    val_gpu = torch.from_numpy(np.ascontiguousarray(val_mon_w)).float().to(device)
    bs = 64 if len(tr_gpu) >= 256 else 16

    best_loss, no_imp = float("inf"), 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for ep in range(1, 11):
        model.train()
        perm = torch.randperm(len(tr_gpu), device=device)
        for i in range(0, len(perm), bs):
            b = tr_gpu[perm[i : i + bs]]
            if model_name == "TranAD":
                r1, r2 = model(b)
                l1, l2 = model.adversarial_loss(r1, r2, b, epoch=ep)
                loss = l1 + l2
            else:
                loss = F.mse_loss(model(b), b)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); opt.step()
        sched.step()
        model.eval()
        tot, n_v = 0.0, 0
        with torch.inference_mode():
            for i in range(0, len(val_gpu), 512):
                vb = val_gpu[i : i + 512]
                vl = 0.5 * sum(F.mse_loss(r, vb) for r in model(vb)) if model_name == "TranAD" else F.mse_loss(model(vb), vb)
                tot += float(vl.item()) * len(vb); n_v += len(vb)
        v_loss = tot / max(n_v, 1)
        if ep == 1 or v_loss < best_loss - 1e-4:
            best_loss, no_imp = v_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
        if ep >= 4 and no_imp >= 3:
            break

    model.load_state_dict(best_state); model.eval()
    def _score_arr(arr: np.ndarray) -> np.ndarray:
        chunks = []
        with torch.inference_mode():
            for i in range(0, len(arr), 1024):
                xb = torch.from_numpy(np.ascontiguousarray(arr[i : i + 1024])).float().to(device)
                chunks.append(model.compute_anomaly_scores(xb)[:, C:].mean(dim=-1).detach().cpu().numpy())
        return np.concatenate(chunks)

    va_s, te_s = _score_arr(va_w), _score_arr(te_w)
    del model
    return va_s, te_s, n_params


def run_part_a_external_baselines(df_arch: pd.DataFrame, device: torch.device) -> dict[str, Any]:
    """Run PCA-LinPred-LN, TranAD, and TimesNet across all 118 untouched archive streams."""
    untouched = (
        df_arch[df_arch["heldout"] == True][["dataset", "channel", "k_channels", "t_nom"]]
        .drop_duplicates().sort_values(["dataset", "channel"]).to_dict(orient="records")
    )
    rows: list[dict[str, Any]] = pd.read_csv(EXT_CSV).to_dict(orient="records") if EXT_CSV.exists() else []
    done = {(str(r["dataset"]), str(r["channel"]), int(r["seed"]), str(r["model"])) for r in rows}

    t0 = time.perf_counter()
    for idx, info in enumerate(untouched, start=1):
        ds, ch = str(info["dataset"]), str(info["channel"])
        needed = [(sd, m) for sd in SEEDS_3 for m in ("PCA-LinPred-LN", "TranAD", "TimesNet") if (ds, ch, sd, m) not in done]
        if not needed:
            continue
        train, test, y = load_stream(ds, ch)
        tr_w, val_mon_w, _, va_w, te_w, len_va = _split_nominal_5block(train, test)
        k_ch, t_nom = int(train.shape[1]), int(len(train))

        if any(m == "PCA-LinPred-LN" for _, m in needed):
            tr_pca = tr_w[np.linspace(0, len(tr_w) - 1, min(len(tr_w), 1536), dtype=int)]
            sc_fn = train_model("PCA-LinPred-LN", tr_pca, k_ch, device, 1)
            va_pca = np.concatenate([sc_fn(va_w[i : i + 2048]) for i in range(0, len(va_w), 2048)])
            te_pca = np.concatenate([sc_fn(te_w[i : i + 2048]) for i in range(0, len(te_w), 2048)])
            for sd in SEEDS_3:
                if (ds, ch, sd, "PCA-LinPred-LN") not in done:
                    meta = {"dataset": ds, "channel": ch, "heldout": True, "k_channels": k_ch, "t_nom": t_nom, "seed": sd, "suite": "archive_118_ext"}
                    _record_1d_spot_rows(rows, meta, "PCA-LinPred-LN", va_pca, te_pca, len_va, len(test), y, 0)
                    done.add((ds, ch, sd, "PCA-LinPred-LN"))

        for sd in SEEDS_3:
            meta = {"dataset": ds, "channel": ch, "heldout": True, "k_channels": k_ch, "t_nom": t_nom, "seed": sd, "suite": "archive_118_ext"}
            for mod in ("TranAD", "TimesNet"):
                if (ds, ch, sd, mod) not in done:
                    torch.manual_seed(sd); np.random.seed(sd)
                    va_s, te_s, np_cnt = _train_eval_reconstruction_baseline(mod, tr_w, val_mon_w, va_w, te_w, k_ch, device)
                    _record_1d_spot_rows(rows, meta, mod, va_s, te_s, len_va, len(test), y, np_cnt)
                    done.add((ds, ch, sd, mod))

        if idx % 10 == 0 or idx == len(untouched):
            atomic_write_csv(pd.DataFrame(rows), EXT_CSV)
            print(f"[Part A {idx:3d}/{len(untouched)} | {time.perf_counter() - t0:5.1f}s] {ds}:{ch}", flush=True)

    atomic_write_csv(pd.DataFrame(rows), EXT_CSV)
    ext_df = pd.read_csv(EXT_CSV)
    lite_df = df_arch[(df_arch["heldout"] == True) & (df_arch["model"] == "Laminar-JEPA-Lite")]
    summary: dict[str, Any] = {}
    for mapping in ("trailing", "smear"):
        l_sub = lite_df[lite_df["mapping"] == mapping].groupby(["dataset", "channel"], as_index=False)[["f1", "pr_auc", "event_recall", "mean_delay", "hold_fpr"]].mean()
        e_sub = ext_df[ext_df["mapping"] == mapping]
        m_map: dict[str, Any] = {}
        for mod in ("PCA-LinPred-LN", "TranAD", "TimesNet"):
            b_sub = e_sub[e_sub["model"] == mod].groupby(["dataset", "channel"], as_index=False)[["f1", "pr_auc", "event_recall", "mean_delay", "hold_fpr"]].mean()
            mg = l_sub.merge(b_sub, on=["dataset", "channel"], suffixes=("_l", "_b"))
            hf = b_sub["hold_fpr"].to_numpy()
            m_map[mod] = {
                "f1_mean": float(b_sub["f1"].mean()), "pr_auc_mean": float(b_sub["pr_auc"].mean()),
                "er_mean": float(b_sub["event_recall"].mean()), "delay_mean": float(b_sub["mean_delay"].mean()),
                "hold_fpr_mean": float(hf.mean()), "hold_fpr_med": float(np.median(hf)),
                "hold_fpr_p90": float(np.percentile(hf, 90)), "hold_fpr_p95": float(np.percentile(hf, 95)),
                "pct_le_5pct": float((hf <= 0.05).mean() * 100.0),
                "lite_minus_base_er": float((mg["event_recall_l"] - mg["event_recall_b"]).mean()),
                "lite_minus_base_er_cluster_ci": _cluster_ci(mg, "event_recall_l", "event_recall_b"),
                "lite_minus_base_er_wilcoxon_p": _wilcoxon_p(mg["event_recall_l"] - mg["event_recall_b"]),
                "lite_minus_base_f1": float((mg["f1_l"] - mg["f1_b"]).mean()),
                "lite_minus_base_f1_cluster_ci": _cluster_ci(mg, "f1_l", "f1_b"),
                "lite_minus_base_f1_wilcoxon_p": _wilcoxon_p(mg["f1_l"] - mg["f1_b"]),
            }
        summary[mapping] = m_map
    return summary


def run_part_b_lodo_and_sensitivity(df_arch: pd.DataFrame, device: torch.device) -> dict[str, Any]:
    """Compute unseen vs. seen dataset splits, 9-fold untouched exclusion, and Nested LODO selection."""
    u = df_arch[(df_arch["heldout"] == True) & (df_arch["mapping"] == "trailing")]
    lite = u[u["model"] == "Laminar-JEPA-Lite"].groupby(["dataset", "channel"], as_index=False)[["f1", "event_recall", "mean_delay", "hold_fpr"]].mean()
    m1d = u[u["model"] == "Multi-Horizon-JEPA-1D"].groupby(["dataset", "channel"], as_index=False)[["f1", "event_recall", "mean_delay", "hold_fpr"]].mean()
    mg = lite.merge(m1d, on=["dataset", "channel"], suffixes=("_l", "_1"))
    tuned_ds = set(df_arch[df_arch["heldout"] == False]["dataset"].unique())
    cohorts = {
        "zero_tuning_4_unseen_datasets": mg[~mg["dataset"].isin(tuned_ds)],
        "zero_tuning_with_gecco": mg[mg["dataset"].isin(["SMD", "Exathlon", "PSM", "creditcard", "GECCO"])],
        "unseen_streams_from_tuned_datasets": mg[mg["dataset"].isin(tuned_ds)],
    }
    cohort_stats = {
        k: {
            "n_streams": int(len(sub)), "datasets": sorted(sub["dataset"].unique().tolist()),
            "lite_er": float(sub["event_recall_l"].mean()), "1d_er": float(sub["event_recall_1"].mean()), "er_p": _wilcoxon_p(sub["event_recall_l"] - sub["event_recall_1"]),
            "lite_f1": float(sub["f1_l"].mean()), "1d_f1": float(sub["f1_1"].mean()), "f1_p": _wilcoxon_p(sub["f1_l"] - sub["f1_1"]),
            "lite_hold_fpr": float(sub["hold_fpr_l"].mean()), "1d_hold_fpr": float(sub["hold_fpr_1"].mean()),
        } for k, sub in cohorts.items()
    }
    excl_118 = {
        ds: {
            "n_rem": int(len(mg[mg["dataset"] != ds])),
            "delta_er_pct": float((mg[mg["dataset"] != ds]["event_recall_l"] - mg[mg["dataset"] != ds]["event_recall_1"]).mean() * 100.0),
            "p_er": _wilcoxon_p(mg[mg["dataset"] != ds]["event_recall_l"] - mg[mg["dataset"] != ds]["event_recall_1"]),
            "delta_f1": float((mg[mg["dataset"] != ds]["f1_l"] - mg[mg["dataset"] != ds]["f1_1"]).mean()),
            "p_f1": _wilcoxon_p(mg[mg["dataset"] != ds]["f1_l"] - mg[mg["dataset"] != ds]["f1_1"]),
            "delta_fpr": float((mg[mg["dataset"] != ds]["hold_fpr_l"] - mg[mg["dataset"] != ds]["hold_fpr_1"]).mean()),
        } for ds in sorted(mg["dataset"].unique())
    }

    streams_141 = df_arch[["dataset", "channel", "heldout"]].drop_duplicates().sort_values(["dataset", "channel"]).to_dict(orient="records")
    grid_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for idx, info in enumerate(streams_141, start=1):
        ds, ch, is_h = str(info["dataset"]), str(info["channel"]), bool(info["heldout"])
        train, test, y = load_stream(ds, ch)
        tr_w, val_mon_w, q_w, va_w, te_w, len_va = _split_nominal_5block(train, test)
        n_cal_w = len(va_w) // 2
        torch.manual_seed(42); np.random.seed(42)
        cfg = CSMConfig(model_type="laminar_jepa_lite", context_size=C, suspect_size=S, latent_dim=32)
        model = build_ts_jepa_model(cfg, input_dim=int(train.shape[1]), device=device)
        assert isinstance(model, LaminarJEPALiteModel)
        _fast_val_checkpoint_train(model, tr_w, val_mon_w, cfg, device, epochs=12, patience=4)
        model.fit_resolvent_whitening(tr_w, context_len=C, batch_size=512)
        va_r, va_wr = model.extract_raw_and_whitened_residuals(va_w, context_len=C, batch_size=1024)
        te_r, te_wr = model.extract_raw_and_whitened_residuals(te_w, context_len=C, batch_size=1024)
        q_r, q_wr = model.extract_raw_and_whitened_residuals(q_w, context_len=C, batch_size=1024)
        del model
        va_ew, te_ew = stabilize_whitened_subblocks(va_wr, te_wr, va_r, te_r, n_cal_w, q_wr, q_r)
        va_pd, te_pd = build_position_drift_dials(va_ew), build_position_drift_dials(te_ew)
        floor_bal = max(2.0, 3.5 * (va_pd.shape[1] / 128.0))
        for qd in (0.980, 0.985, 0.990):
            tau = np.percentile(va_pd[:n_cal_w], 100.0 * qd, axis=0, keepdims=True) + 1e-4
            vp, vm = aggregate_vector_stream((va_pd > tau).sum(axis=1, keepdims=True).astype(np.float32), len_va, C, S, "trailing")
            tp, tm = aggregate_vector_stream((te_pd > tau).sum(axis=1, keepdims=True).astype(np.float32), len(test), C, S, "trailing")
            vc, tc, tl = vp[vm, 0], tp[tm, 0], y[tm]
            half = len(vc) // 2
            q995, hold_cnt = float(np.percentile(vc[:half], 99.5)), vc[half + S :]
            for kap in (1.06, 1.12, 1.18):
                k_th = max(kap * q995, floor_bal)
                h_pred, t_pred = (hold_cnt > k_th).astype(int), (tc > k_th).astype(int)
                mt = compute_metrics(tl, t_pred, scores=tc, valid_mask=np.ones_like(tl, dtype=bool), use_pa=False)
                er, dl = event_stats(tl, t_pred)
                grid_rows.append({"dataset": ds, "channel": ch, "heldout": is_h, "q_dial": qd, "kappa": kap, "f1": float(mt.get("f1", 0.0)), "er": float(er) if er == er else float("nan"), "hold_fpr": float(h_pred.mean()) if len(h_pred) else 0.0, "delay": float(dl) if dl == dl else float("nan")})
        if idx % 35 == 0 or idx == len(streams_141):
            print(f"[Part B Nested-LODO {idx:3d}/{len(streams_141)} | {time.perf_counter() - t0:5.1f}s]", flush=True)

    gdf = pd.DataFrame(grid_rows)
    tune_df, untouch_df = gdf[gdf["heldout"] == False], gdf[gdf["heldout"] == True]
    nested_folds, sel_frames = [], []
    for ds in sorted(untouch_df["dataset"].unique()):
        agg_t = tune_df[tune_df["dataset"] != ds].groupby(["q_dial", "kappa"], as_index=False)[["f1", "er", "hold_fpr"]].mean()
        feas = agg_t[agg_t["hold_fpr"] <= 0.025]
        pool = feas if len(feas) > 0 else agg_t
        best = pool.loc[int((pool["f1"] + 0.25 * pool["er"]).idxmax())]
        q_sel, k_sel = float(best["q_dial"]), float(best["kappa"])
        ev = untouch_df[(untouch_df["dataset"] == ds) & (untouch_df["q_dial"] == q_sel) & (untouch_df["kappa"] == k_sel)]
        sel_frames.append(ev)
        nested_folds.append({"dataset": ds, "n_streams": int(len(ev)), "selected_q_dial": q_sel, "selected_kappa": k_sel, "er": float(ev["er"].mean()), "f1": float(ev["f1"].mean()), "hold_fpr": float(ev["hold_fpr"].mean())})
    m1d_s42 = df_arch[(df_arch["heldout"] == True) & (df_arch["mapping"] == "trailing") & (df_arch["seed"] == 42) & (df_arch["model"] == "Multi-Horizon-JEPA-1D")]
    n_mg = pd.concat(sel_frames, ignore_index=True).merge(m1d_s42[["dataset", "channel", "f1", "event_recall", "hold_fpr"]], on=["dataset", "channel"], suffixes=("_nest", "_1d"))
    return {
        "cohort_split": cohort_stats, "exclusion_118": excl_118,
        "nested_lodo_seed42": {
            "per_dataset": nested_folds, "pooled_lite_er": float(n_mg["er"].mean()), "pooled_1d_er": float(n_mg["event_recall"].mean()), "pooled_er_wilcoxon_p": _wilcoxon_p(n_mg["er"] - n_mg["event_recall"]),
            "pooled_lite_f1": float(n_mg["f1_nest"].mean()), "pooled_1d_f1": float(n_mg["f1_1d"].mean()), "pooled_f1_wilcoxon_p": _wilcoxon_p(n_mg["f1_nest"] - n_mg["f1_1d"]), "pooled_lite_hold_fpr": float(n_mg["hold_fpr_nest"].mean()),
        },
    }


def _subgroup_stats(sub: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_streams": int(len(sub)),
        "er_diff": float((sub["event_recall_l"] - sub["event_recall_1"]).mean()), "er_stream_ci": _boot_ci(sub["event_recall_l"] - sub["event_recall_1"]), "er_cluster_ci": _cluster_ci(sub, "event_recall_l", "event_recall_1"), "er_wilcoxon_p": _wilcoxon_p(sub["event_recall_l"] - sub["event_recall_1"]), "er_sign_flip_p": _sign_flip_p(sub, "event_recall_l", "event_recall_1"),
        "f1_diff": float((sub["f1_l"] - sub["f1_1"]).mean()), "f1_stream_ci": _boot_ci(sub["f1_l"] - sub["f1_1"]), "f1_cluster_ci": _cluster_ci(sub, "f1_l", "f1_1"), "f1_wilcoxon_p": _wilcoxon_p(sub["f1_l"] - sub["f1_1"]), "f1_sign_flip_p": _sign_flip_p(sub, "f1_l", "f1_1"),
    }


def run_part_c_funnel_and_uncertainty(df_arch: pd.DataFrame) -> dict[str, Any]:
    """Compute delay CIs, subgroup cluster CIs + sign-flip p, and 118-stream eligibility funnel."""
    u = df_arch[(df_arch["heldout"] == True) & (df_arch["mapping"] == "trailing")]
    agg_c = {"n_cal": "first", "hold_fpr": "mean", "test_fpr": "mean", "f1": "mean", "event_recall": "mean", "mean_delay": "mean"}
    lite = u[u["model"] == "Laminar-JEPA-Lite"].groupby(["dataset", "channel"], as_index=False).agg(agg_c)
    m1d = u[u["model"] == "Multi-Horizon-JEPA-1D"].groupby(["dataset", "channel"], as_index=False).agg(agg_c)
    m = lite.merge(m1d, on=["dataset", "channel"], suffixes=("_l", "_1"))
    fa = pd.read_csv(FAIL_ATTR_CSV)
    if "stream" in fa.columns and "channel" not in fa.columns:
        fa[["dataset", "channel"]] = fa["stream"].astype(str).str.split(":", n=1, expand=True)
    m = m.merge(fa.groupby(["dataset", "channel"], as_index=False)["n_eff"].mean(), on=["dataset", "channel"], how="left")

    both = m.dropna(subset=["mean_delay_l", "mean_delay_1"])
    s1, s1_lt = m[m["n_cal_l"] >= 500], m[m["n_cal_l"] < 500]
    s2_pass, s2_fail = s1[s1["hold_fpr_l"] <= 0.05], s1[s1["hold_fpr_l"] > 0.05]
    s3_neff = s2_pass[s2_pass["n_eff"] >= 50.0]
    use_lite = (m["n_cal_l"] >= 500) & (m["hold_fpr_l"] <= 0.05)

    return {
        "delay_uncertainty": {
            "lite_mean": float(m["mean_delay_l"].mean()), "lite_stream_ci": _boot_ci(m["mean_delay_l"]), "lite_cluster_ci": _cluster_ci(m, "mean_delay_l"),
            "1d_mean": float(m["mean_delay_1"].mean()), "1d_stream_ci": _boot_ci(m["mean_delay_1"]), "1d_cluster_ci": _cluster_ci(m, "mean_delay_1"),
            "codetected_n": int(len(both)), "codetected_diff": float((both["mean_delay_l"] - both["mean_delay_1"]).mean()), "codetected_stream_ci": _boot_ci(both["mean_delay_l"] - both["mean_delay_1"]), "codetected_cluster_ci": _cluster_ci(both, "mean_delay_l", "mean_delay_1"), "codetected_wilcoxon_p": _wilcoxon_p(both["mean_delay_l"] - both["mean_delay_1"]), "end_to_end_lag_offset_steps": int(S - 1),
        },
        "ncal_ge_500_subgroup": _subgroup_stats(s1), "ncal_lt_500_subgroup": _subgroup_stats(s1_lt),
        "eligibility_funnel": {
            "total_archive": int(len(m)), "pass_ncal_ge_500": int(len(s1)), "routed_short_ncal_lt_500": int(len(s1_lt)),
            "pass_ncal_and_hold_fpr_le_5pct": int(len(s2_pass)), "fail_hold_fpr_gt_5pct": int(len(s2_fail)), "total_routed_to_1d_fallback": int(len(m) - len(s2_pass)),
            "accepted_lite_f1": float(s2_pass["f1_l"].mean()), "accepted_1d_f1": float(s2_pass["f1_1"].mean()), "accepted_f1_p": _wilcoxon_p(s2_pass["f1_l"] - s2_pass["f1_1"]),
            "accepted_lite_er": float(s2_pass["event_recall_l"].mean()), "accepted_1d_er": float(s2_pass["event_recall_1"].mean()), "accepted_lite_hold_fpr": float(s2_pass["hold_fpr_l"].mean()),
            "optional_neff_ge_50_n": int(len(s3_neff)), "optional_neff_lite_f1": float(s3_neff["f1_l"].mean()), "optional_neff_lite_er": float(s3_neff["event_recall_l"].mean()),
            "routed_118_system_f1": float(np.nanmean(np.where(use_lite, m["f1_l"], m["f1_1"]))), "routed_118_system_er": float(np.nanmean(np.where(use_lite, m["event_recall_l"], m["event_recall_1"]))), "routed_118_system_hold_fpr": float(np.nanmean(np.where(use_lite, m["hold_fpr_l"], m["hold_fpr_1"]))),
        },
    }


def _stab_weights(q_wr: np.ndarray, q_r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rms_w = np.sqrt(np.mean(np.clip(np.nan_to_num(q_wr.astype(np.float64), nan=0.0), -100.0, 100.0) ** 2, axis=0, keepdims=True)) + 1e-4
    rms_raw = np.sqrt(np.mean(np.clip(np.nan_to_num(q_r.astype(np.float64), nan=0.0), -100.0, 100.0) ** 2, axis=0, keepdims=True)) + 1e-4
    return 1.0 / (1.0 + 2.0 * np.maximum(0.0, rms_w - 1.10) ** 2), rms_raw


def run_part_d_safety_and_loco(device: torch.device) -> dict[str, Any]:
    """Quantify unconfirmed fault absorption hazard and LOCO physical-channel attribution on dev19."""
    unconf_recalls, quar_recalls, top1_hits, top3_hits, rand1_rates, rand3_rates = [], [], [], [], [], []
    for ds, ch, _ in _build_schedule("dev19_4block"):
        train, test, y = load_stream(ds, ch)
        tr_w, q_w, va_w, _, _ = _split_nominal_dev_4block(train, test)
        n_fit = len(train) - int(len(train) * 0.35) - max(256, int(len(train) * 0.15))
        _, te_std = _standardize_arrays(train[:n_fit], test)
        r0, seg_len = longest_nominal_run(y)
        if seg_len < W + 700:
            continue
        seg, k_ch, n_cal_w = te_std[r0 : r0 + seg_len], int(train.shape[1]), len(va_w) // 2

        torch.manual_seed(42); np.random.seed(42)
        cfg = CSMConfig(model_type="laminar_jepa_lite", context_size=C, suspect_size=S, latent_dim=32)
        model = build_ts_jepa_model(cfg, input_dim=k_ch, device=device)
        _train_fixed_epochs(model, tr_w, cfg, epochs=15, device=device)
        model.fit_resolvent_whitening(tr_w, context_len=C, batch_size=512)
        va_r, va_wr = model.extract_raw_and_whitened_residuals(va_w, context_len=C, batch_size=1024)
        q_r, q_wr = model.extract_raw_and_whitened_residuals(q_w, context_len=C, batch_size=1024)
        w_white, rms_raw_cal = _stab_weights(q_wr, q_r)
        va_pd = build_position_drift_dials(w_white * va_wr + (1.0 - w_white) * (va_r / rms_raw_cal))
        tau = (np.percentile(va_pd[:n_cal_w], 98.5, axis=0) + 1e-4).astype(np.float32)
        k_th = max(1.12 * float(np.percentile((va_pd[:n_cal_w] > tau.reshape(1, -1)).sum(axis=1), 99.5)), 3.5)

        seg_fault = seg[: min(len(seg), W + 1100)].copy()
        t_sh = min(256, len(seg_fault) - 600)
        seg_fault[t_sh:, : max(1, k_ch // 3)] += 2.5
        fr, fwr = model.extract_raw_and_whitened_residuals(DataLoader.create_windows(seg_fault, W, step=1, copy=False), context_len=C, batch_size=512)
        post_pd = build_position_drift_dials(w_white * fwr + (1.0 - w_white) * (fr / rms_raw_cal))[max(0, t_sh - C) :]
        if len(post_pd) >= 480:
            buf = post_pd[:400]
            tau_bad = (np.percentile(buf, 98.5, axis=0) + 1e-4).astype(np.float32)
            k_bad = max(1.12 * float(np.percentile((buf > tau_bad.reshape(1, -1)).sum(axis=1), 99.5)), 3.5)
            unconf_recalls.append(float(((post_pd[400:] > tau_bad.reshape(1, -1)).sum(axis=1) > k_bad).mean()))
            ctrl = QuarantinedRecalibrationController(tau_dials=tau, k_quorum=k_th)
            ctrl.process_batch(buf, ks_distance=0.95)
            unconf_q, _, _ = ctrl.process_batch(post_pd[400:], ks_distance=0.95)
            quar_recalls.append(float(unconf_q.mean()))

        if k_ch >= 5 and len(seg) >= W + 240:
            m_ch = min(3, max(1, k_ch // 4))
            act = np.flatnonzero(np.std(seg[:W], axis=0) > 1e-4)
            f_chs = np.random.default_rng(42).choice(act if len(act) >= m_ch else np.arange(k_ch), size=m_ch, replace=False)
            seg_loc = seg[: W + 240].copy(); seg_loc[W + 20 : W + 220, f_chs] += 2.5
            lw = DataLoader.create_windows(seg_loc, W, step=1, copy=False)
            lr, lwr = model.extract_raw_and_whitened_residuals(lw, context_len=C, batch_size=512)
            l_pd = build_position_drift_dials(w_white * lwr + (1.0 - w_white) * (lr / rms_raw_cal))
            alm_idx = np.flatnonzero(((l_pd > tau.reshape(1, -1)).sum(axis=1) > k_th) & (np.arange(len(lw)) >= 21))
            if len(alm_idx) > 0:
                sel = alm_idx[np.linspace(0, len(alm_idx) - 1, min(len(alm_idx), 15), dtype=int)]
                order = np.argsort(-np.mean([loco_channel_attribution(model, tau, lw[i], context_len=C, w_white=w_white, rms_raw_cal=rms_raw_cal) for i in sel], axis=0))
                top1_hits.append(float(int(order[0]) in set(f_chs.tolist())))
                top3_hits.append(float(len(set(order[:3].tolist()) & set(f_chs.tolist())) > 0))
                rand1_rates.append(float(m_ch / k_ch))
                rand3_rates.append(float(1.0 - math.comb(k_ch - m_ch, 3) / math.comb(k_ch, 3)) if k_ch - m_ch >= 3 else 1.0)
        del model

    return {
        "fault_contamination_hazard": {"n_streams": int(len(unconf_recalls)), "unconfirmed_auto_recal_post400_recall": float(np.mean(unconf_recalls)), "quarantined_controller_post400_recall": float(np.mean(quar_recalls))},
        "loco_physical_attribution": {"n_streams": int(len(top1_hits)), "top1_hit_rate": float(np.mean(top1_hits)), "random_top1_rate": float(np.mean(rand1_rates)), "top3_hit_rate": float(np.mean(top3_hits)), "random_top3_rate": float(np.mean(rand3_rates))},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="R2 TNNLS audit & external baselines.")
    ap.add_argument("--smoke-cd", action="store_true", help="Run fast Parts C+D sanity check only.")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_arch = pd.read_csv(ARCHIVE_CSV)
    payload: dict[str, Any] = {"part_c_funnel_and_ci": run_part_c_funnel_and_uncertainty(df_arch), "part_d_safety_and_loco": run_part_d_safety_and_loco(device)}
    if not args.smoke_cd:
        payload["part_b_lodo_and_sensitivity"] = run_part_b_lodo_and_sensitivity(df_arch, device)
        payload["part_a_external_118"] = run_part_a_external_baselines(df_arch, device)
    atomic_write_json(payload, OUT_JSON)
    print(f"Saved R2 audit summary to {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
