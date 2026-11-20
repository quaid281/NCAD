#!/usr/bin/env python3
"""Consolidated R4 IEEE TNNLS Audit Experiments (Audits 1-6).

Runs across all 118 untouched archive streams (3 seeds {42, 123, 456}, disjoint 5-block
protocol) plus the 19 non-GHL development streams (3 seeds, 4-block protocol for Audit 5):
  1. Matched calibration-budget & post-hoc Precision/Recall/F1/ER/FPR (Point 1).
  2. Additive +1e-4 vs. scale-homogeneous *(1+1e-4) tripwire floor (Point 2).
  3. Prospective false-alarm episodes per 1k healthy steps & 3-stage drift screening (Point 3).
  4. Exact segment-level paired co-detected latency vs. detection-conditioned delay (Point 4).
  5. Fair 8-arm PCA vs. ZCA whitening & Kronecker 95% CIs (Point 5).
  6. Widened W=320-step full-sample-disjoint gap sensitivity on ER, F1, Prec, FPR (Point 11).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.circuit_breaker_19stream.common import DEVELOPMENT_STREAMS
from scripts.reviewer_analyses.calibration_quantile_reliability import tau_sokal
from scripts.reviewer_analyses.run_matched_budget_archive import _rate_threshold, _series, lite_bank_scores
from scripts.tnnls_revision.common_eval import (
    apply_horizon_whitening_operator,
    atomic_write_csv,
    atomic_write_json,
    false_alarm_episodes_per_1k,
    fit_horizon_whitening_operator,
    stabilize_eigenbasis_residuals,
)
from scripts.run_laminar_paired_benchmarks import (
    C, S, W, _build_schedule, _fast_val_checkpoint_train, _split_nominal_dev_4block,
    _score_1d_windows, _standardize_arrays, _train_fixed_epochs, load_stream,
)
from scripts.run_multiseed_heldout_benchmark import thresholds
from src.config import CSMConfig
from src.data.data_loader import DataLoader
from src.engine.trainer import build_encoder, build_ts_jepa_model
from src.models.jepa.alien_jepas import SpeculativeJEPAModel
from src.models.laminar_lite import LaminarJEPALiteModel
from src.scoring.circuit_breaker_panel import aggregate_vector_stream, stabilize_whitened_subblocks
from src.scoring.laminar_lite_panel import PositionDriftLitePanel, aggregate_prefix_pd_stream, build_position_drift_dials

SEEDS_3 = (42, 123, 456)
CAL_RATES = (0.005, 0.010, 0.020, 0.050)
TEST_FPRS = (0.01, 0.05)


def _fast_segs(y: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(y > 0.5)
    if idx.size == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > 1) + 1
    return [(int(g[0]), int(g[-1]) + 1) for g in np.split(idx, splits)]


def _fast_event_stats(segs: list[tuple[int, int]], pred: np.ndarray) -> tuple[float, float]:
    if not segs:
        return float("nan"), float("nan")
    delays = []
    for gs, ge in segs:
        hit = np.flatnonzero(pred[gs:ge] > 0.5)
        if hit.size:
            delays.append(float(hit[0]))
    return len(delays) / len(segs), (float(np.mean(delays)) if delays else float("nan"))


def _eval_series(
    cal: np.ndarray, hold: np.ndarray, test: np.ndarray, y_tl: np.ndarray, th: float,
    segs: list[tuple[int, int]] | None = None, include_pr: bool = False,
) -> dict[str, float]:
    """Compute vectorized step, event, and false-alarm-episode metrics for one threshold."""
    ph = hold > th
    pt = test > th
    pos = y_tl > 0.5
    tp = int(np.sum(pt & pos))
    fp = int(np.sum(pt & (~pos)))
    fn = int(np.sum((~pt) & pos))
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    seg_list = _fast_segs(y_tl) if segs is None else segs
    er, dl = _fast_event_stats(seg_list, pt)
    has_both = bool(np.any(pos) and np.any(~pos))
    ep_h, br_h = false_alarm_episodes_per_1k(ph)
    nom = ~pos
    ep_t, br_t = false_alarm_episodes_per_1k(pt[nom]) if np.any(nom) else (0.0, 0.0)
    return {
        "cal_rate": float(np.mean(cal > th)) if len(cal) else float("nan"),
        "hold_fpr": float(np.mean(ph)) if len(ph) else 0.0,
        "test_fpr": float(np.mean(pt[nom])) if np.any(nom) else 0.0,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "pr_auc": float(average_precision_score(y_tl, test)) if (include_pr and has_both) else float("nan"),
        "event_recall": float(er) if er == er else float("nan"),
        "mean_delay": float(dl) if dl == dl else float("nan"),
        "fa_ep_1k_hold": ep_h,
        "fa_burst_hold": br_h,
        "fa_ep_1k_test": ep_t,
        "fa_burst_test": br_t,
    }


def _eval_dials(
    ew_va: np.ndarray, ew_te: np.ndarray, n_cal_w: int, len_va: int, len_test: int, y_test: np.ndarray,
    use_drift: bool = True, scale_homog: bool = False, gap_steps: int = S,
) -> dict[str, dict[str, Any]]:
    """Calibrate and evaluate Position(+Drift) panel under arbitrary gap and dial count."""
    n_win_va, m_cnt, dim = ew_va.shape
    if use_drift:
        va_d, te_d = build_position_drift_dials(ew_va), build_position_drift_dials(ew_te)
        n_groups = m_cnt + 1
    else:
        va_d = np.abs(ew_va).reshape(n_win_va, m_cnt * dim)
        te_d = np.abs(ew_te).reshape(len(ew_te), m_cnt * dim)
        n_groups = m_cnt

    q_cal = np.percentile(va_d[:n_cal_w], 98.5, axis=0, keepdims=True)
    tau = q_cal * (1.0 + 1e-4) if scale_homog else q_cal + 1e-4
    b_va, b_te = (va_d > tau).astype(np.float32), (te_d > tau).astype(np.float32)
    floor = max(2.0, 3.5 * (float(va_d.shape[1]) / 128.0))

    out: dict[str, dict[str, Any]] = {}
    for mp in ("trailing", "smear"):
        if mp == "smear" and use_drift:
            va_g = b_va.reshape(len(b_va), n_groups, dim).sum(axis=2)
            te_g = b_te.reshape(len(b_te), n_groups, dim).sum(axis=2)
            vp, vm = aggregate_prefix_pd_stream(va_g, len_va, C, S, m_cnt, "smear")
            tp, tm = aggregate_prefix_pd_stream(te_g, len_test, C, S, m_cnt, "smear")
            vc, tc = vp[vm].sum(axis=1), tp[tm].sum(axis=1)
        elif mp == "smear":
            va_g = b_va.reshape(len(b_va), m_cnt, dim).sum(axis=2)
            te_g = b_te.reshape(len(b_te), m_cnt, dim).sum(axis=2)
            pad_va = np.concatenate([va_g, np.zeros((len(va_g), 1), np.float32)], axis=1)
            pad_te = np.concatenate([te_g, np.zeros((len(te_g), 1), np.float32)], axis=1)
            vp, vm = aggregate_prefix_pd_stream(pad_va, len_va, C, S, m_cnt, "smear")
            tp, tm = aggregate_prefix_pd_stream(pad_te, len_test, C, S, m_cnt, "smear")
            vc, tc = vp[vm, :-1].sum(axis=1), tp[tm, :-1].sum(axis=1)
        else:
            vp, vm = aggregate_vector_stream(b_va.sum(axis=1, keepdims=True), len_va, C, S, "trailing")
            tp, tm = aggregate_vector_stream(b_te.sum(axis=1, keepdims=True), len_test, C, S, "trailing")
            vc, tc = vp[vm, 0], tp[tm, 0]

        cal = vc[:n_cal_w]
        hold = vc[n_cal_w + gap_steps :]
        k_th = max(float(np.percentile(cal, 99.5)) * 1.12, floor)
        y_tl = y_test[tm]
        res = _eval_series(cal, hold, tc, y_tl, k_th, include_pr=True)
        res.update({"cal": cal, "hold": hold, "test": tc, "y_tl": y_tl, "th": k_th, "pred_test": (tc > k_th).astype(int), "pred_hold": (hold > k_th).astype(int)})
        out[mp] = res
    return out


def _codetected_event_stats(segs: list[tuple[int, int]], pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    """Compute exact segment-level paired co-detected and detection-conditioned latencies."""
    dl_a, dl_b, co_a, co_b = [], [], [], []
    for gs, ge in segs:
        ha = np.flatnonzero(pred_a[gs:ge] > 0.5)
        hb = np.flatnonzero(pred_b[gs:ge] > 0.5)
        if ha.size:
            dl_a.append(float(ha[0]))
        if hb.size:
            dl_b.append(float(hb[0]))
        if ha.size and hb.size:
            co_a.append(float(ha[0]))
            co_b.append(float(hb[0]))
    return {
        "n_events_total": float(len(segs)),
        "n_events_lite": float(len(dl_a)),
        "n_events_1d": float(len(dl_b)),
        "n_events_codet": float(len(co_a)),
        "delay_lite_cond": float(np.mean(dl_a)) if dl_a else float("nan"),
        "delay_1d_cond": float(np.mean(dl_b)) if dl_b else float("nan"),
        "delay_lite_codet": float(np.mean(co_a)) if co_a else float("nan"),
        "delay_1d_codet": float(np.mean(co_b)) if co_b else float("nan"),
        "delay_codet_diff": float(np.mean(np.asarray(co_a) - np.asarray(co_b))) if co_a else float("nan"),
    }


@torch.no_grad()
def _extract_raw_3h(model: torch.nn.Module, windows: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    dev = next(model.parameters()).device
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for i in range(0, len(windows), batch_size):
            b = torch.from_numpy(np.ascontiguousarray(windows[i : i + batch_size])).float().to(dev)
            _, _, _, zp, zt = model.forward(b[:, :C], b[:, C:])
            assert zt is not None
            chunks.append((zt - zp).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def _split_nominal_5block_post(train: np.ndarray, test: np.ndarray) -> dict[str, Any]:
    """Return 5-block windows plus contiguous `post_w` for W=320 gap verification."""
    n_tot = len(train)
    n_val = int(n_tot * 0.35)
    n_tr = n_tot - n_val
    offset_va = max(416, int(n_tot * 0.22))
    n_fit = n_tr - offset_va
    tr_s, post_s, va_s, te_s = _standardize_arrays(train[:n_fit], train[n_fit:], train[-n_val:], test)
    tr_w = DataLoader.create_windows(tr_s, W, step=16, copy=False)
    post_w = DataLoader.create_windows(post_s, W, step=1, copy=False)
    avail = offset_va - 2 * S
    n_sel_w = max(96, int(0.38 * avail))
    sel_step = max(1, min(8, n_sel_w // 32))
    val_mon_w = post_w[:n_sel_w:sel_step]
    q_end = offset_va - S
    q_start = max(n_sel_w + S, q_end - 2048)
    return {
        "tr_w": tr_w, "val_mon_w": val_mon_w, "post_w": post_w,
        "q_start": q_start, "q_end": q_end, "n_sel_w": n_sel_w, "offset_va": offset_va,
        "va_w": DataLoader.create_windows(va_s, W, step=1, copy=False),
        "te_w": DataLoader.create_windows(te_s, W, step=1, copy=False),
        "len_va": int(n_val), "len_test": int(len(test)),
    }


def _eval_whitening_8arms(
    fit_raw: np.ndarray, q_raw: np.ndarray, va_raw: np.ndarray, te_raw: np.ndarray,
    len_va: int, len_test: int, y: np.ndarray, meta: dict[str, Any], out_rows: list[dict[str, Any]],
) -> None:
    """Audit 5: Fair 8-arm PCA vs. ZCA comparison on identical checkpoints."""
    n_cal_w = len(va_raw) // 2
    ops = {
        m: fit_horizon_whitening_operator(fit_raw, mode=m)
        for m in ("full_zca", "shared_kronecker_zca", "pca_eigen", "shared_kronecker_pca")
    }
    configs: list[tuple[str, np.ndarray, np.ndarray, bool]] = []
    for arm, op_k, shrink, use_d in [
        ("Full_MD_ZCA_Eq3", "full_zca", "raw", True),
        ("Pure_Full_MD_ZCA", "full_zca", "none", True),
        ("Pure_PCA_Eigen_128", "pca_eigen", "none", True),
        ("Pure_PCA_Eigen_96", "pca_eigen", "none", False),
        ("PCA_Eigen_ConsistentShrink", "pca_eigen", "eigen", True),
        ("Shared_Kronecker_ZCA_Eq3", "shared_kronecker_zca", "raw", True),
        ("Shared_Kronecker_PCA_Consistent", "shared_kronecker_pca", "eigen", True),
        ("PCA_Eigen_MismatchedEq3", "pca_eigen", "raw", True),
    ]:
        op = ops[op_k]
        if shrink == "none":
            va_ew = apply_horizon_whitening_operator(va_raw, op)
            te_ew = apply_horizon_whitening_operator(te_raw, op)
        elif shrink == "eigen":
            va_ew, te_ew = stabilize_eigenbasis_residuals(op, q_raw, va_raw, te_raw, n_cal_w)
        else:
            q_w = apply_horizon_whitening_operator(q_raw, op)
            va_w = apply_horizon_whitening_operator(va_raw, op)
            te_w = apply_horizon_whitening_operator(te_raw, op)
            va_ew, te_ew = stabilize_whitened_subblocks(va_w, te_w, va_raw, te_raw, n_cal_w, q_w, q_raw)
        configs.append((arm, va_ew, te_ew, use_d))

    for arm, va_ew, te_ew, use_d in configs:
        res_mp = _eval_dials(va_ew, te_ew, n_cal_w, len_va, len_test, y, use_drift=use_d)
        for mp, vals in res_mp.items():
            out_rows.append({
                **meta, "model": arm, "rule": "whitening", "mapping": mp,
                **{k: v for k, v in vals.items() if isinstance(v, float)},
            })


def _run_118_stream_seed(
    ds: str, ch: str, seed: int, sp: dict[str, Any], y: np.ndarray, k_ch: int, t_nom: int,
    device: torch.device, rows: list[dict[str, Any]], white_rows: list[dict[str, Any]],
) -> None:
    meta = {"dataset": ds, "channel": ch, "k_channels": k_ch, "t_nom": t_nom, "seed": seed, "suite": "archive118_5block"}
    tr_w, val_mon_w, va_w, te_w = sp["tr_w"], sp["val_mon_w"], sp["va_w"], sp["te_w"]
    len_va, len_test = sp["len_va"], sp["len_test"]
    n_cal_w = len(va_w) // 2

    # 1. Multi-Horizon 1D JEPA
    torch.manual_seed(seed); np.random.seed(seed)
    cfg_sp = CSMConfig(model_type="ts_jepa", context_size=C, suspect_size=S, latent_dim=32)
    m_sp = SpeculativeJEPAModel(
        context_encoder=build_encoder(cfg_sp, k_ch, device), latent_dim=32,
        predictor_hidden_dim=64, ema_decay=0.996, dropout=cfg_sp.dropout,
    ).to(device)
    _fast_val_checkpoint_train(m_sp, tr_w, val_mon_w, cfg_sp, device, epochs=12, patience=4)
    m_sp.fit_alien_bank(tr_w, context_len=C, batch_size=512)
    ws_1d = (_score_1d_windows(m_sp, va_w, device, use_bank=True), _score_1d_windows(m_sp, te_w, device, use_bank=True))
    del m_sp

    # 2. Laminar-JEPA-Lite
    torch.manual_seed(seed); np.random.seed(seed)
    cfg_l = CSMConfig(model_type="laminar_jepa_lite", context_size=C, suspect_size=S, latent_dim=32)
    m_l = build_ts_jepa_model(cfg_l, input_dim=k_ch, device=device)
    assert isinstance(m_l, LaminarJEPALiteModel)
    _fast_val_checkpoint_train(m_l, tr_w, val_mon_w, cfg_l, device, epochs=12, patience=4)
    m_l.fit_resolvent_whitening(tr_w, context_len=C, batch_size=512)
    fit_raw = _extract_raw_3h(m_l, tr_w, batch_size=1024)
    post_raw, post_white = m_l.extract_raw_and_whitened_residuals(sp["post_w"], context_len=C, batch_size=1024)
    te_raw, te_white = m_l.extract_raw_and_whitened_residuals(te_w, context_len=C, batch_size=1024)
    del m_l

    q_raw, q_white = post_raw[sp["q_start"] : sp["q_end"]], post_white[sp["q_start"] : sp["q_end"]]
    va_raw, va_white = post_raw[sp["offset_va"] :], post_white[sp["offset_va"] :]
    va_ew, te_ew = stabilize_whitened_subblocks(va_white, te_white, va_raw, te_raw, n_cal_w, q_white, q_raw)

    # Deployed Lite (additive +1e-4 vs homogeneous *(1+1e-4))
    res_lite = _eval_dials(va_ew, te_ew, n_cal_w, len_va, len_test, y, True, False, S)["trailing"]
    res_hom = _eval_dials(va_ew, te_ew, n_cal_w, len_va, len_test, y, True, True, S)["trailing"]
    agree_hold = float(np.mean(res_lite["pred_hold"] == res_hom["pred_hold"]))
    agree_test = float(np.mean(res_lite["pred_test"] == res_hom["pred_test"]))

    # Deployed MH1D
    v1, t1, tm1 = _series(ws_1d, len_va, len_test, "trailing")
    h1 = len(v1) // 2
    cal_1d, hold_1d, y_tl = v1[:h1], v1[h1 + S :], y[tm1]
    segs = _fast_segs(y_tl)
    th_spot = float(thresholds(cal_1d)[0]["SPOT"])
    res_1d = _eval_series(cal_1d, hold_1d, t1, y_tl, th_spot, segs=segs, include_pr=True)
    pred_1d = (t1 > th_spot).astype(int)

    # KS drift, ESS, split-holdout, and co-detected latencies
    win_norm = np.sqrt(np.mean(va_ew ** 2, axis=(1, 2)))
    ks_norm = float(stats.ks_2samp(win_norm[:n_cal_w], win_norm[n_cal_w + S :]).statistic) if len(win_norm) > n_cal_w + S + 10 else float("nan")
    ks_quorum = float(stats.ks_2samp(res_lite["cal"], res_lite["hold"]).statistic) if len(res_lite["hold"]) > 10 else float("nan")
    _, n_eff = (lambda a: (tau_sokal(a) if len(a) > 8 else 1.0, len(a) / max(1.0, tau_sokal(a) if len(a) > 8 else 1.0)))(res_lite["cal"])
    ph = res_lite["pred_hold"]
    hh = len(ph) // 2
    hold_h1, hold_h2 = (float(ph[:hh].mean()), float(ph[hh:].mean())) if hh > 0 else (0.0, 0.0)
    codet = _codetected_event_stats(segs, res_lite["pred_test"], pred_1d)

    lite_scalars = {k: v for k, v in res_lite.items() if isinstance(v, float)}
    hom_scalars = {k: v for k, v in res_hom.items() if isinstance(v, float)}
    rows.append({**meta, "model": "Lite", "rule": "deployed", "n_cal": float(n_cal_w), "n_eff": float(n_eff),
                 "ks_norm": ks_norm, "ks_quorum": ks_quorum, "hold_h1_fpr": hold_h1, "hold_h2_fpr": hold_h2,
                 "agree_hom_hold": agree_hold, "agree_hom_test": agree_test, **lite_scalars, **codet})
    rows.append({**meta, "model": "Lite_HomogFloor", "rule": "deployed", "n_cal": float(n_cal_w), **hom_scalars})
    rows.append({**meta, "model": "MH1D", "rule": "deployed", "n_cal": float(h1), **res_1d})

    # Cross rate-matched + matched calibration & post-hoc budgets (Audit 1)
    series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        "Lite": (res_lite["cal"], res_lite["hold"], res_lite["test"]),
        "MH1D": (cal_1d, hold_1d, t1),
    }
    for bname, s_va, s_te in lite_bank_scores(va_ew, te_ew):
        vb, tb, _ = _series((s_va, s_te), len_va, len_test, "trailing")
        hb = len(vb) // 2
        series[bname] = (vb[:hb], vb[hb + S :], tb)

    rows.append({**meta, "model": "MH1D", "rule": "match_lite_cal", **_eval_series(cal_1d, hold_1d, t1, y_tl, _rate_threshold(cal_1d, res_lite["cal_rate"]), segs=segs)})
    rows.append({**meta, "model": "Lite", "rule": "match_1d_cal", **_eval_series(res_lite["cal"], res_lite["hold"], res_lite["test"], y_tl, _rate_threshold(res_lite["cal"], res_1d["cal_rate"]), segs=segs)})
    for mname, (c_, h_, t_) in series.items():
        for r in CAL_RATES:
            rows.append({**meta, "model": mname, "rule": f"cal{r:.3f}", **_eval_series(c_, h_, t_, y_tl, _rate_threshold(c_, r), segs=segs)})
        if np.any(y_tl == 0) and np.any(y_tl == 1):
            nom = t_[y_tl == 0]
            for f in TEST_FPRS:
                rows.append({**meta, "model": mname, "rule": f"posthoc_f{int(f * 100):02d}", **_eval_series(c_, h_, t_, y_tl, float(np.quantile(nom, 1.0 - f, method="higher")), segs=segs)})

    # Audit 6: Widened W=320 full-sample-disjoint gap on qualifying streams
    n_cal_320 = (len(va_raw) - W) // 2
    q_end_320 = sp["offset_va"] - W
    if n_cal_320 >= 100 and len(va_raw) - n_cal_320 - W >= 100 and q_end_320 - (sp["n_sel_w"] + S) >= 160:
        q_s320 = max(sp["n_sel_w"] + W if q_end_320 - (sp["n_sel_w"] + W) >= 160 else sp["n_sel_w"] + S, q_end_320 - 2048)
        all_320 = int(q_s320 >= sp["n_sel_w"] + W)
        qr_3, qw_3 = post_raw[q_s320:q_end_320], post_white[q_s320:q_end_320]
        va_ew_3, te_ew_3 = stabilize_whitened_subblocks(va_white, te_white, va_raw, te_raw, n_cal_320, qw_3, qr_3)
        res_g320 = _eval_dials(va_ew_3, te_ew_3, n_cal_320, len_va, len_test, y, True, False, W)["trailing"]
        rows.append({**meta, "model": "Lite", "rule": "gap320", "all_blocks_320": float(all_320), **{k: v for k, v in res_g320.items() if isinstance(v, float)}})
        rows.append({**meta, "model": "Lite", "rule": "gap64_paired", "all_blocks_320": float(all_320), **lite_scalars})

    # Audit 5: 8-arm fair PCA vs. ZCA whitening
    _eval_whitening_8arms(fit_raw, q_raw, va_raw, te_raw, len_va, len_test, y, meta, white_rows)


def _boot_ci(vals: np.ndarray, ds: np.ndarray | None = None, n_boot: int = 4000, seed: int = 42) -> list[float]:
    arr = np.asarray(vals, dtype=np.float64)
    mask = np.isfinite(arr)
    arr = arr[mask]
    if arr.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    if ds is None:
        idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
        means = arr[idx].mean(axis=1)
    else:
        ds_m = np.asarray(ds)[mask]
        u = np.unique(ds_m)
        groups = [arr[ds_m == g] for g in u]
        g_sums = np.array([g.sum() for g in groups], dtype=np.float64)
        g_cnts = np.array([g.size for g in groups], dtype=np.float64)
        draw = rng.integers(0, len(groups), size=(n_boot, len(groups)))
        means = g_sums[draw].sum(axis=1) / np.maximum(g_cnts[draw].sum(axis=1), 1.0)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _wilcoxon_p(diffs: np.ndarray) -> float:
    arr = np.asarray(diffs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return 1.0 if (arr.size < 2 or np.allclose(arr, 0.0)) else float(stats.wilcoxon(arr).pvalue)


def _summarize(df_m: pd.DataFrame, df_w: pd.DataFrame) -> dict[str, Any]:
    """Compute publication-ready statistics for Audits 1-6."""
    cols = ["cal_rate", "hold_fpr", "test_fpr", "f1", "precision", "recall", "pr_auc",
            "event_recall", "mean_delay", "fa_ep_1k_hold", "fa_burst_hold", "fa_ep_1k_test", "fa_burst_test"]
    sm = lambda mod, rule: df_m[(df_m["model"] == mod) & (df_m["rule"] == rule)].groupby(["dataset", "channel"], as_index=False).mean(numeric_only=True)

    # Audit 1: Matched budgets + deployed + rate-matched
    a1: dict[str, Any] = {}
    for rule in ("deployed", "match_lite_cal", "match_1d_cal", "cal0.005", "cal0.010", "cal0.020", "cal0.050", "posthoc_f01", "posthoc_f05"):
        a1[rule] = {}
        for mod in ("Lite", "MH1D", "Fisher", "TopK", "MaxZ"):
            sub = sm(mod, rule)
            if not len(sub):
                continue
            a1[rule][mod] = {c: float(sub[c].mean()) for c in cols if c in sub.columns}
            a1[rule][mod].update({
                "hold_fpr_med": float(sub["hold_fpr"].median()),
                "test_fpr_med": float(sub["test_fpr"].median()),
                "fa_ep_1k_test_med": float(sub["fa_ep_1k_test"].median()),
            })

    # Audit 2: Scale-homogeneous floor vs additive
    l_add, l_hom = sm("Lite", "deployed"), sm("Lite_HomogFloor", "deployed")
    mg2 = l_add.merge(l_hom, on=["dataset", "channel"], suffixes=("_add", "_hom"))
    a2 = {
        "n_streams": int(len(mg2)),
        "mean_abs_delta_er": float(np.abs(mg2["event_recall_hom"] - mg2["event_recall_add"]).mean()),
        "max_abs_delta_er": float(np.abs(mg2["event_recall_hom"] - mg2["event_recall_add"]).max()),
        "mean_abs_delta_f1": float(np.abs(mg2["f1_hom"] - mg2["f1_add"]).mean()),
        "max_abs_delta_f1": float(np.abs(mg2["f1_hom"] - mg2["f1_add"]).max()),
        "mean_abs_delta_hold_fpr": float(np.abs(mg2["hold_fpr_hom"] - mg2["hold_fpr_add"]).mean()),
        "mean_agree_hold": float(l_add["agree_hom_hold"].mean()),
        "mean_agree_test": float(l_add["agree_hom_test"].mean()),
    }

    # Audit 3: Screening funnels + prospective episodes/1k
    m1 = sm("MH1D", "deployed")
    mg3 = l_add.merge(m1, on=["dataset", "channel"], suffixes=("_l", "_1"))
    funnels: dict[str, Any] = {}
    for fname, cond in [
        ("2stage_ncal500_hold5", (mg3["n_cal_l"] >= 500) & (mg3["hold_fpr_l"] <= 0.05)),
        ("3stage_ks_norm_050", (mg3["n_cal_l"] >= 500) & (mg3["hold_fpr_l"] <= 0.05) & (mg3["ks_norm_l"] <= 0.50)),
        ("3stage_ks_quorum_050", (mg3["n_cal_l"] >= 500) & (mg3["hold_fpr_l"] <= 0.05) & (mg3["ks_quorum_l"] <= 0.50)),
        ("3stage_split_hold_5_ks050", (mg3["n_cal_l"] >= 500) & (mg3["hold_fpr_l"] <= 0.05) & (mg3["hold_h1_fpr_l"] <= 0.05) & (mg3["hold_h2_fpr_l"] <= 0.05) & (mg3["ks_norm_l"] <= 0.50)),
    ]:
        acc = mg3[cond]
        hyb = {c: np.where(cond, mg3[f"{c}_l"], mg3[f"{c}_1"]) for c in ("hold_fpr", "test_fpr", "f1", "precision", "recall", "event_recall", "fa_ep_1k_test", "fa_ep_1k_hold")}
        funnels[fname] = {
            "n_accepted": int(len(acc)), "pct_118": float(100.0 * len(acc) / len(mg3)),
            "acc_hold_fpr_mean": float(acc["hold_fpr_l"].mean()), "acc_hold_fpr_med": float(acc["hold_fpr_l"].median()),
            "acc_test_fpr_mean": float(acc["test_fpr_l"].mean()), "acc_test_fpr_med": float(acc["test_fpr_l"].median()),
            "acc_test_fpr_le5_pct": float(100.0 * (acc["test_fpr_l"] <= 0.05).mean()),
            "acc_test_fpr_le10_pct": float(100.0 * (acc["test_fpr_l"] <= 0.10).mean()),
            "acc_fa_ep_1k_test_mean": float(acc["fa_ep_1k_test_l"].mean()), "acc_fa_ep_1k_test_med": float(acc["fa_ep_1k_test_l"].median()),
            "acc_er": float(acc["event_recall_l"].mean()), "acc_f1": float(acc["f1_l"].mean()), "acc_prec": float(acc["precision_l"].mean()),
            "hyb_hold_fpr_mean": float(np.mean(hyb["hold_fpr"])), "hyb_test_fpr_mean": float(np.mean(hyb["test_fpr"])),
            "hyb_test_fpr_med": float(np.median(hyb["test_fpr"])), "hyb_test_fpr_le5_pct": float(100.0 * np.mean(hyb["test_fpr"] <= 0.05)),
            "hyb_fa_ep_1k_test_mean": float(np.mean(hyb["fa_ep_1k_test"])), "hyb_fa_ep_1k_test_med": float(np.median(hyb["fa_ep_1k_test"])),
            "hyb_er": float(np.nanmean(hyb["event_recall"])), "hyb_f1": float(np.mean(hyb["f1"])), "hyb_prec": float(np.mean(hyb["precision"])),
        }

    # Audit 4: Segment-paired vs stream-paired vs detection-conditioned latency
    co_ev = l_add.dropna(subset=["delay_codet_diff"])
    co_st = mg3.dropna(subset=["mean_delay_l", "mean_delay_1"])
    a4 = {
        "cond_lite_delay": float(l_add["mean_delay"].mean()), "cond_lite_n": int(l_add["mean_delay"].notna().sum()),
        "cond_1d_delay": float(m1["mean_delay"].mean()), "cond_1d_n": int(m1["mean_delay"].notna().sum()),
        "stream_codet_n": int(len(co_st)),
        "stream_codet_diff": float((co_st["mean_delay_l"] - co_st["mean_delay_1"]).mean()),
        "stream_codet_p": _wilcoxon_p(co_st["mean_delay_l"] - co_st["mean_delay_1"]),
        "event_codet_streams_n": int(len(co_ev)),
        "event_codet_lite": float(co_ev["delay_lite_codet"].mean()),
        "event_codet_1d": float(co_ev["delay_1d_codet"].mean()),
        "event_codet_diff": float(co_ev["delay_codet_diff"].mean()),
        "event_codet_stream_ci": _boot_ci(co_ev["delay_codet_diff"].to_numpy()),
        "event_codet_cluster_ci": _boot_ci(co_ev["delay_codet_diff"].to_numpy(), co_ev["dataset"].to_numpy()),
        "event_codet_p": _wilcoxon_p(co_ev["delay_codet_diff"].to_numpy()),
    }

    # Audit 5: 8-arm whitening summary & pairwise contrasts
    a5: dict[str, Any] = {}
    for (suite, mp), g_w in df_w.groupby(["suite", "mapping"]):
        sw = g_w.groupby(["model", "dataset", "channel"], as_index=False).mean(numeric_only=True)
        means = sw.groupby("model")[["f1", "precision", "recall", "pr_auc", "event_recall", "hold_fpr", "test_fpr", "mean_delay"]].mean().to_dict(orient="index")
        pairs = {}
        for ma, mb in [
            ("Shared_Kronecker_ZCA_Eq3", "Full_MD_ZCA_Eq3"),
            ("Pure_PCA_Eigen_128", "Pure_Full_MD_ZCA"),
            ("Pure_PCA_Eigen_96", "Pure_Full_MD_ZCA"),
            ("PCA_Eigen_ConsistentShrink", "Full_MD_ZCA_Eq3"),
            ("PCA_Eigen_ConsistentShrink", "PCA_Eigen_MismatchedEq3"),
            ("Shared_Kronecker_PCA_Consistent", "Shared_Kronecker_ZCA_Eq3"),
            ("PCA_Eigen_MismatchedEq3", "Full_MD_ZCA_Eq3"),
        ]:
            sa = sw[sw["model"] == ma].set_index(["dataset", "channel"])
            sb = sw[sw["model"] == mb].set_index(["dataset", "channel"]).loc[sa.index]
            pairs[f"{ma}__minus__{mb}"] = {
                c: {
                    "diff": float((sa[c] - sb[c]).mean()),
                    "stream_ci": _boot_ci((sa[c] - sb[c]).to_numpy()),
                    "cluster_ci": _boot_ci((sa[c] - sb[c]).to_numpy(), sa.index.get_level_values(0).to_numpy()),
                    "p": _wilcoxon_p((sa[c] - sb[c]).to_numpy()),
                }
                for c in ("f1", "event_recall", "hold_fpr")
            }
        a5[f"{suite}_{mp}"] = {"means": means, "pairs": pairs}

    # Audit 6: W=320 gap comparison
    g320 = sm("Lite", "gap320")
    g64 = sm("Lite", "gap64_paired")
    a6: dict[str, Any] = {}
    if len(g320) and len(g64):
        mg6 = g320.merge(g64, on=["dataset", "channel"], suffixes=("_320", "_64"))
        mg6_all = mg6[mg6["all_blocks_320_320"] >= 0.99]
        for lbl, sub in [("q_cal_hold_320", mg6), ("all_postfit_320", mg6_all)]:
            a6[lbl] = {"n_streams": int(len(sub))}
            if not len(sub):
                continue
            for c in ("event_recall", "f1", "precision", "hold_fpr", "test_fpr"):
                diff = (sub[f"{c}_320"] - sub[f"{c}_64"]).to_numpy()
                a6[lbl][c] = {
                    "mean_64": float(sub[f"{c}_64"].mean()), "mean_320": float(sub[f"{c}_320"].mean()),
                    "diff": float(diff.mean()), "ci": _boot_ci(diff), "p": _wilcoxon_p(diff),
                }

    # Dataset-level Lite vs MH1D gain distributions (Point 7)
    ds_rows = []
    for ds_name, g in mg3.groupby("dataset", sort=True):
        ds_rows.append({
            "dataset": str(ds_name), "n": int(len(g)),
            "lite_er": float(g["event_recall_l"].mean()), "1d_er": float(g["event_recall_1"].mean()),
            "lite_f1": float(g["f1_l"].mean()), "1d_f1": float(g["f1_1"].mean()),
            "lite_prec": float(g["precision_l"].mean()), "1d_prec": float(g["precision_1"].mean()),
            "lite_hold_fpr": float(g["hold_fpr_l"].mean()), "1d_hold_fpr": float(g["hold_fpr_1"].mean()),
            "lite_test_fpr": float(g["test_fpr_l"].mean()), "1d_test_fpr": float(g["test_fpr_1"].mean()),
        })

    return {
        "audit1_matched_budgets": a1, "audit2_scale_floor": a2, "audit3_prospective_screening": funnels,
        "audit4_codetected_latency": a4, "audit5_fair_whitening": a5, "audit6_gap320": a6,
        "per_dataset_118": ds_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-streams", type=int, default=0)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    rep = ROOT / "reports"
    sum_json = rep / "r4_reviewer_audit_summary.json"
    if args.num_shards > 1:
        main_csv = rep / f"r4_audit_118_shard{args.shard_id}.csv"
        white_csv = rep / f"r4_whitening_8arm_shard{args.shard_id}.csv"
    else:
        main_csv = rep / "r4_audit_118_cells.csv"
        white_csv = rep / "r4_whitening_8arm.csv"

    if args.analyze_only:
        m_shards = sorted(rep.glob("r4_audit_118_shard*.csv"))
        w_shards = sorted(rep.glob("r4_whitening_8arm_shard*.csv"))
        df_m = pd.concat([pd.read_csv(p) for p in m_shards], ignore_index=True) if m_shards else pd.read_csv(rep / "r4_audit_118_cells.csv")
        df_w = pd.concat([pd.read_csv(p) for p in w_shards], ignore_index=True) if w_shards else pd.read_csv(rep / "r4_whitening_8arm.csv")
        df_m = df_m.drop_duplicates(subset=["suite", "dataset", "channel", "seed", "model", "rule"], keep="last")
        df_w = df_w.drop_duplicates(subset=["suite", "dataset", "channel", "seed", "model", "rule", "mapping"], keep="last")
        summary = _summarize(df_m, df_w)
        atomic_write_json(summary, sum_json)
        print(f"Saved R4 summary -> {sum_json} (main rows={len(df_m)}, white rows={len(df_w)})")
        return

    rows: list[dict[str, Any]] = pd.read_csv(main_csv).to_dict("records") if main_csv.exists() else []
    white_rows: list[dict[str, Any]] = pd.read_csv(white_csv).to_dict("records") if white_csv.exists() else []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    done_118 = {(str(r["dataset"]), str(r["channel"]), int(r["seed"])) for r in rows if r["model"] == "Lite" and r["rule"] == "deployed"}
    done_dev = {(str(r["dataset"]), str(r["channel"]), int(r["seed"])) for r in white_rows if r["suite"] == "dev19_4block"}

    sched_118 = [it for it in _build_schedule("tier2_141_5block") if (it[0].lower(), str(it[1])) not in DEVELOPMENT_STREAMS]
    sched_dev = _build_schedule("dev19_4block")
    if args.max_streams > 0:
        sched_118, sched_dev = sched_118[: args.max_streams], sched_dev[: min(args.max_streams, len(sched_dev))]
    if args.num_shards > 1:
        sched_118 = [it for i, it in enumerate(sched_118) if i % args.num_shards == args.shard_id]
        sched_dev = [it for i, it in enumerate(sched_dev) if i % args.num_shards == args.shard_id]

    t0 = time.perf_counter()
    for idx, (ds, ch, _) in enumerate(sched_118, start=1):
        need = [sd for sd in SEEDS_3 if (ds, str(ch), sd) not in done_118]
        if not need:
            continue
        train, test, y = load_stream(ds, ch)
        sp = _split_nominal_5block_post(train, test)
        for sd in need:
            _run_118_stream_seed(ds, str(ch), sd, sp, y, int(train.shape[1]), int(len(train)), device, rows, white_rows)
            done_118.add((ds, str(ch), sd))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        atomic_write_csv(pd.DataFrame(rows), main_csv)
        atomic_write_csv(pd.DataFrame(white_rows), white_csv)
        print(f"[S{args.shard_id} 118-Arch {idx:3d}/{len(sched_118)}] {ds}:{ch} ({time.perf_counter() - t0:.1f}s)", flush=True)

    for idx, (ds, ch, _) in enumerate(sched_dev, start=1):
        need = [sd for sd in SEEDS_3 if (ds, str(ch), sd) not in done_dev]
        if not need:
            continue
        train, test, y = load_stream(ds, ch)
        tr_w, q_w, va_w, te_w, len_va = _split_nominal_dev_4block(train, test)
        for sd in need:
            torch.manual_seed(sd); np.random.seed(sd)
            cfg = CSMConfig(model_type="laminar_jepa_lite", context_size=C, suspect_size=S, latent_dim=32)
            m_l = build_ts_jepa_model(cfg, input_dim=int(train.shape[1]), device=device)
            _train_fixed_epochs(m_l, tr_w, cfg, epochs=15, device=device)
            meta = {"dataset": ds, "channel": str(ch), "k_channels": int(train.shape[1]), "t_nom": int(len(train)), "seed": sd, "suite": "dev19_4block"}
            _eval_whitening_8arms(_extract_raw_3h(m_l, tr_w), _extract_raw_3h(m_l, q_w), _extract_raw_3h(m_l, va_w), _extract_raw_3h(m_l, te_w), len_va, len(test), y, meta, white_rows)
            del m_l
            done_dev.add((ds, str(ch), sd))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        atomic_write_csv(pd.DataFrame(white_rows), white_csv)
        print(f"[S{args.shard_id} Dev19    {idx:3d}/{len(sched_dev)}] {ds}:{ch} ({time.perf_counter() - t0:.1f}s)", flush=True)

    if args.num_shards <= 1:
        summary = _summarize(pd.DataFrame(rows), pd.DataFrame(white_rows))
        atomic_write_json(summary, sum_json)
        print(f"Saved R4 summary -> {sum_json}")


if __name__ == "__main__":
    main()
