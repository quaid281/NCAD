#!/usr/bin/env python3
"""Paired Multi-Seed Benchmark Suite for Laminar-JEPA and Laminar-JEPA-Lite.

Supports canonical manuscript evaluation suites with atomic checkpointing and resume:
  1. `dev33`: 33 primary development streams (`19` non-GHL x 5 seeds + `14` GHL x 3 seeds),
     fixed 15 epochs on 65% nominal training block.
  2. `dev19_4block`: 19 non-GHL development streams (`3` seeds `{42,123,456}`), fixed 15 epochs
     with strictly disjoint adaptation (`X_fit -> X_q -> X_cal -> X_hold`, 64-step target gaps).
  3. `tier2_141`: 141 archive streams under 4-block disjoint checkpoint selection (`X_sel _|_ X_cal`).
  4. `tier2_141_5block`: All 141 qualifying Tier-2 multivariate streams (`118` untouched held-out +
     `23` development) across 3 seeds `{42,123,456}` under the strictly disjoint 5-block protocol
     (`X_fit -> X_sel -> X_q -> X_cal -> X_hold`, separated by 64-step isolation gaps), evaluating
     TS-JEPA (1D), Multi-Horizon JEPA (1D), Laminar-JEPA (w/o Quench), Laminar-JEPA, Laminar-JEPA-Lite,
     system-matched coordinate detector banks, and 2-fold chronological cross-fitting.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score
import torch
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CSMConfig
from src.data.data_loader import DataLoader
from src.engine.trainer import build_encoder, build_ts_jepa_model
from src.experimental.phase_embedding_synthesis import _forward_predictors_with_phase
from src.experimental.weight_space_programming import WSPConfig, apply_surgical_wsp_quench
from src.models._jepa_utils import resolvent_precision
from src.models.jepa.alien_jepas import (
    SpeculativeJEPAModel,
    _encode_with_multidepth_taps,
    _whitening_defect_loss,
)
from src.models.laminar_lite import LaminarJEPALiteModel
from src.scoring.circuit_breaker_panel import (
    CircuitBreakerStreamDecision,
    MultiPerspectiveCircuitBreakerPanel,
    aggregate_vector_stream,
    stabilize_whitened_subblocks,
)
from src.scoring.evt_calibrator import compute_excess_kurtosis
from src.scoring.laminar_lite_panel import PositionDriftLitePanel
from scripts.circuit_breaker_19stream.common import DEVELOPMENT_STREAMS
from scripts.experiment_wsp_soliton import extract_raw4_white4_from_latents
from scripts.run_multiseed_heldout_benchmark import (
    C,
    S,
    W,
    compute_metrics,
    event_stats,
    load_stream,
    thresholds,
)
from scripts.run_tier2_exhaustive_benchmark import atomic_save_csv, split_train_checkpoint_block


def _standardize_arrays(tr: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """Standardize arrays using float64 training moments of `tr` alone."""
    tr64 = tr.astype(np.float64, copy=False)
    mu = tr64.mean(0)
    sd_arr = tr64.std(0)
    sc = np.where(sd_arr < 1e-8, 1.0, np.maximum(sd_arr, 0.01))
    out = [((tr64 - mu) / sc).astype(np.float32)]
    for arr in others:
        out.append(((arr.astype(np.float64, copy=False) - mu) / sc).astype(np.float32))
    return tuple(out)


def _split_nominal_5block(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Partition `train` into 5 strictly disjoint, S=64 target-isolated blocks."""
    n_tot = len(train)
    n_val = int(n_tot * 0.35)
    n_tr = n_tot - n_val
    offset_va = max(416, int(n_tot * 0.22))
    n_fit = n_tr - offset_va
    if n_fit < W + 256:
        raise ValueError(f"Insufficient nominal length {n_tot} for 5-block partition (n_fit={n_fit}).")

    tr_fit = train[:n_fit]
    post_fit = train[n_fit:]
    va = train[-n_val:]
    tr_s, post_s, va_s, te_s = _standardize_arrays(tr_fit, post_fit, va, test)

    tr_w = DataLoader.create_windows(tr_s, W, step=16, copy=False)
    post_w = DataLoader.create_windows(post_s, W, step=1, copy=False)
    avail = offset_va - 2 * S
    n_sel_w = max(96, int(0.38 * avail))
    sel_step = max(1, min(8, n_sel_w // 32))
    val_mon_w = post_w[:n_sel_w:sel_step]
    q_end = offset_va - S
    q_start = max(n_sel_w + S, q_end - 2048)
    q_w = post_w[q_start:q_end]
    va_w = DataLoader.create_windows(va_s, W, step=1, copy=False)
    te_w = DataLoader.create_windows(te_s, W, step=1, copy=False)
    assert len(tr_w) >= 16 and len(val_mon_w) >= 12 and len(q_w) >= 160 and len(va_w) // 2 >= 100
    return tr_w, val_mon_w, q_w, va_w, te_w, len(va)


def _split_nominal_dev_4block(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Partition `train` into 4 disjoint blocks (`X_fit -> X_q -> X_cal -> X_hold`) for fixed-epoch dev."""
    n_tot = len(train)
    n_val = int(n_tot * 0.35)
    n_tr = n_tot - n_val
    offset_va = max(256, int(n_tot * 0.15))
    n_fit = n_tr - offset_va
    tr_s, post_s, va_s, te_s = _standardize_arrays(train[:n_fit], train[n_fit:], train[-n_val:], test)
    tr_w = DataLoader.create_windows(tr_s, W, step=16, copy=False)
    post_w = DataLoader.create_windows(post_s, W, step=1, copy=False)
    q_end = offset_va - S
    q_w = post_w[max(0, q_end - 2048) : q_end]
    va_w = DataLoader.create_windows(va_s, W, step=1, copy=False)
    te_w = DataLoader.create_windows(te_s, W, step=1, copy=False)
    assert len(tr_w) >= 16 and len(q_w) >= 160
    return tr_w, q_w, va_w, te_w, n_val


@torch.no_grad()
def _fast_extract_4h_latents(
    model: torch.nn.Module, windows: np.ndarray, batch_size: int = 1024
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched GPU extraction of `(z_ctx [N, D], z_sub_tgt [N, 4, D])` at `batch_size=1024`."""
    model.eval()
    dev = next(model.parameters()).device
    m_cnt = int(model.num_subblocks)
    z_ctx_list, z_tgt_list = [], []
    with torch.inference_mode():
        for i in range(0, len(windows), batch_size):
            ch = torch.from_numpy(np.ascontiguousarray(windows[i : i + batch_size])).float().to(dev)
            ctx_win, tgt_win = ch[:, :C], ch[:, C:]
            z_ctx = _encode_with_multidepth_taps(model.context_encoder, ctx_win, model.context_tap_head)
            s_len = tgt_win.shape[1]
            t_list = [
                _encode_with_multidepth_taps(
                    model.target_encoder,
                    tgt_win[:, : (s_len if m == m_cnt - 1 else max(((m + 1) * s_len) // m_cnt, 2)), :],
                    model.target_tap_head,
                )
                for m in range(m_cnt)
            ]
            z_ctx_list.append(z_ctx)
            z_tgt_list.append(torch.stack(t_list, dim=1))
    return torch.cat(z_ctx_list, dim=0), torch.cat(z_tgt_list, dim=0)


@torch.no_grad()
def _single_pass_val_metric(
    model: torch.nn.Module, val_gpu: torch.Tensor, cfg: CSMConfig
) -> float:
    """Compute exact `val_cos_err + 0.25 * val_loss` in a single batched GPU pass."""
    model.eval()
    tot_loss, tot_cos, tot_n = 0.0, 0.0, 0
    with torch.inference_mode():
        for i in range(0, len(val_gpu), 512):
            b = val_gpu[i : i + 512]
            ctx, tgt = b[:, :C], b[:, C:]
            if isinstance(model, SpeculativeJEPAModel):
                loss, _ = model.compute_objective(ctx, tgt, cfg)
                _, z_targets, z_drafts = model.forward(ctx, tgt)
                cos_err = 1.0 - F.cosine_similarity(z_drafts, z_targets, dim=-1).mean()
            elif hasattr(model, "num_subblocks") or isinstance(model, LaminarJEPALiteModel):
                _, z_tgt, z_pred, z_sub_pred, z_sub_tgt = model.forward(ctx, tgt)
                assert z_tgt is not None and z_sub_tgt is not None
                m_cnt = z_sub_pred.shape[1]
                weights = [0.50 / max(m_cnt - 1, 1)] * (m_cnt - 1) + [0.50]
                loss = sum(
                    w_m * (F.mse_loss(z_sub_pred[:, m, :], z_sub_tgt[:, m, :]) + _whitening_defect_loss(z_sub_pred[:, m, :], gamma=0.75))
                    for m, w_m in enumerate(weights)
                )
                cos_err = 1.0 - F.cosine_similarity(z_pred, z_tgt, dim=-1).mean()
            else:
                loss, _ = model.compute_objective(ctx, tgt, cfg)
                _, z_tgt, z_pred = model.forward(ctx, tgt)[:3]
                cos_err = 1.0 - F.cosine_similarity(z_pred, z_tgt, dim=-1).mean()
            n = len(b)
            tot_loss += float(loss.item()) * n
            tot_cos += float(cos_err.item()) * n
            tot_n += n
    denom = max(tot_n, 1)
    return (tot_cos / denom) + 0.25 * (tot_loss / denom)


def _fast_val_checkpoint_train(
    model: torch.nn.Module,
    tr_w: np.ndarray,
    val_w: np.ndarray,
    cfg: CSMConfig,
    device: torch.device,
    epochs: int = 12,
    patience: int = 4,
) -> None:
    """Train with CosineAnnealingLR and early stopping on disjoint `X_sel` (`val_w`)."""
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=1e-5)
    best_score = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    no_improve = 0
    tr_gpu = torch.from_numpy(np.ascontiguousarray(tr_w)).float().to(device)
    val_gpu = torch.from_numpy(np.ascontiguousarray(val_w)).float().to(device)
    bs = 64 if len(tr_gpu) >= 256 else 16

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.from_numpy(np.random.permutation(len(tr_gpu))).to(device)
        for i in range(0, len(perm), bs):
            b = tr_gpu[perm[i : i + bs]]
            loss, _ = model.compute_objective(b[:, :C], b[:, C:], cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            model.update_target_encoder(None)
        sched.step()

        sel_metric = _single_pass_val_metric(model, val_gpu, cfg)
        if ep == 1 or sel_metric < best_score - 1e-4:
            best_score = sel_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if patience > 0 and ep >= 6 and no_improve >= patience:
            break

    model.load_state_dict(best_state)


@torch.no_grad()
def _score_1d_windows(model: torch.nn.Module, w: np.ndarray, device: torch.device, use_bank: bool = False) -> np.ndarray:
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(w), 512):
            ch = torch.from_numpy(np.ascontiguousarray(w[i : i + 512])).float().to(device)
            if isinstance(model, SpeculativeJEPAModel):
                s = model.compute_predictive_discrepancy(ch[:, :C], ch[:, C:], use_alien_bank=use_bank)
            else:
                s = model.compute_predictive_discrepancy(ch[:, :C], ch[:, C:])
            out.append(s.cpu().numpy())
    return np.concatenate(out)


def _diagnostic_recalls(scores: np.ndarray, labels: np.ndarray, cal_scores: np.ndarray) -> dict[str, float]:
    if labels.sum() == 0 or np.all(labels == 1):
        return {"er_q05": float("nan"), "er_f01": float("nan"), "er_f05": float("nan")}
    s_clean = np.where(np.isfinite(scores), scores, np.inf)
    nom = s_clean[labels == 0]
    th_q05 = float(np.quantile(cal_scores, 0.95)) if len(cal_scores) > 0 else float("inf")
    th_f01 = float(np.quantile(nom, 0.99))
    th_f05 = float(np.quantile(nom, 0.95))
    er_q05, _ = event_stats(labels, (s_clean > th_q05).astype(int))
    er_f01, _ = event_stats(labels, (s_clean > th_f01).astype(int))
    er_f05, _ = event_stats(labels, (s_clean > th_f05).astype(int))
    return {"er_q05": float(er_q05), "er_f01": float(er_f01), "er_f05": float(er_f05)}


def _record_1d_spot_rows(
    out_rows: list[dict[str, Any]],
    meta: dict[str, Any],
    model_name: str,
    vws: np.ndarray,
    tws: np.ndarray,
    len_va: int,
    len_test: int,
    y_test: np.ndarray,
    n_params: int,
) -> None:
    for mapping in ("trailing", "smear"):
        vp, vm = aggregate_vector_stream(vws[:, None], len_va, C, S, mapping)
        tp, tm = aggregate_vector_stream(tws[:, None], len_test, C, S, mapping)
        v = vp[vm, 0]
        half = len(v) // 2
        cal, hold = v[:half], v[half + S :]
        ts, tl = tp[tm, 0], y_test[tm]
        ths, kurt = thresholds(cal)
        th = float(ths["SPOT"])
        ks_val = float(stats.ks_2samp(cal, hold).statistic) if len(hold) > 10 else float("nan")
        pred_hold = (hold > th).astype(int)
        pred_test = (ts > th).astype(int)
        mt = compute_metrics(tl, pred_test, scores=ts, valid_mask=np.ones_like(tl, dtype=bool), use_pa=False)
        er, dl = event_stats(tl, pred_test)
        pr = float(average_precision_score(tl, ts)) if (np.any(tl == 1) and np.any(tl == 0)) else 0.0
        diag = _diagnostic_recalls(ts, tl, cal)
        out_rows.append({
            **meta,
            "model": model_name,
            "mapping": mapping,
            "calibrator": "SPOT",
            "trainable_params": int(n_params),
            "threshold": th,
            "cal_kurtosis": float(kurt),
            "ks_drift": ks_val,
            "n_cal": int(len(cal)),
            "n_hold": int(len(hold)),
            "hold_fpr": float(pred_hold.mean()) if len(pred_hold) > 0 else 0.0,
            "test_fpr": float(pred_test[tl == 0].mean()) if np.any(tl == 0) else 0.0,
            "f1": float(mt.get("f1", 0.0)),
            "precision": float(mt.get("precision", 0.0)),
            "recall": float(mt.get("recall", 0.0)),
            "pr_auc": pr,
            "pr_auc_unw224": pr,
            "pr_auc_unw224_edge": pr,
            "event_recall": float(er) if er == er else float("nan"),
            "mean_delay": float(dl) if dl == dl else float("nan"),
            **diag,
        })


def _record_panel_rows(
    out_rows: list[dict[str, Any]],
    meta: dict[str, Any],
    model_name: str,
    calibrator_name: str,
    panel: MultiPerspectiveCircuitBreakerPanel | PositionDriftLitePanel,
    va_raw: np.ndarray,
    va_white: np.ndarray,
    te_raw: np.ndarray,
    te_white: np.ndarray,
    len_va: int,
    len_test: int,
    y_test: np.ndarray,
    n_params: int,
    q_raw: np.ndarray | None = None,
    q_white: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_cal_w = len(va_raw) // 2
    va_ew, te_ew = stabilize_whitened_subblocks(
        va_white, te_white, va_raw, te_raw, n_cal_w, adapt_white4=q_white, adapt_raw4=q_raw
    )
    bits = panel.calibrate_and_trip_windows(va_ew, te_ew, n_cal_w)

    win_norm_va = np.sqrt(np.mean(va_ew ** 2, axis=(1, 2)))
    cal_norm, hold_norm = win_norm_va[:n_cal_w], win_norm_va[n_cal_w + S :]
    kurt = float(compute_excess_kurtosis(cal_norm)) if len(cal_norm) > 10 else 0.0
    ks_val = float(stats.ks_2samp(cal_norm, hold_norm).statistic) if len(hold_norm) > 10 else float("nan")

    for mapping in ("trailing", "smear"):
        dec: CircuitBreakerStreamDecision = panel.evaluate_stream_decisions(bits, len_va, len_test, mapping)
        tl = y_test[dec.valid_mask]
        ts = dec.test_switch_counts
        mt = compute_metrics(
            tl, dec.test_pred_balanced, scores=ts, valid_mask=np.ones_like(tl, dtype=bool), use_pa=False
        )
        er, dl = event_stats(tl, dec.test_pred_balanced)
        has_both = bool(np.any(tl == 1) and np.any(tl == 0))
        pr = float(average_precision_score(tl, ts)) if has_both else 0.0
        if dec.test_unw224_counts is not None and dec.test_unw224_edge_counts is not None and has_both:
            pr_u = float(average_precision_score(tl, dec.test_unw224_counts))
            pr_ue = float(average_precision_score(tl, dec.test_unw224_edge_counts))
        else:
            pr_u, pr_ue = pr, pr
        hfpr = float(dec.hold_pred_balanced.mean()) if len(dec.hold_pred_balanced) > 0 else 0.0
        tfpr = float(dec.test_pred_balanced[tl == 0].mean()) if np.any(tl == 0) else 0.0
        diag = _diagnostic_recalls(ts, tl, dec.cal_switch_counts)

        out_rows.append({
            **meta,
            "model": model_name,
            "mapping": mapping,
            "calibrator": calibrator_name,
            "trainable_params": int(n_params),
            "threshold": float(dec.k_gate_balanced),
            "cal_kurtosis": kurt,
            "ks_drift": ks_val,
            "n_cal": int(len(dec.cal_switch_counts)),
            "n_hold": int(len(dec.hold_pred_balanced)),
            "hold_fpr": hfpr,
            "test_fpr": tfpr,
            "f1": float(mt.get("f1", 0.0)),
            "precision": float(mt.get("precision", 0.0)),
            "recall": float(mt.get("recall", 0.0)),
            "pr_auc": pr,
            "pr_auc_unw224": pr_u,
            "pr_auc_unw224_edge": pr_ue,
            "event_recall": float(er) if er == er else float("nan"),
            "mean_delay": float(dl) if dl == dl else float("nan"),
            **diag,
        })
    return va_ew, te_ew


def _record_lite_banks_and_crossfit(
    out_rows: list[dict[str, Any]],
    meta: dict[str, Any],
    panel_lite: PositionDriftLitePanel,
    va_raw: np.ndarray,
    va_white: np.ndarray,
    te_raw: np.ndarray,
    te_white: np.ndarray,
    va_ew: np.ndarray,
    te_ew: np.ndarray,
    len_va: int,
    len_test: int,
    y_test: np.ndarray,
    n_params: int,
) -> None:
    """Record held-out coordinate detector-bank controls & 2-fold disjoint cross-fit (`X_q _|_ X_cal`)."""
    n_cal_w = len(va_ew) // 2
    bits = panel_lite.calibrate_and_trip_windows(va_ew, te_ew, n_cal_w)
    dec_ref = panel_lite.evaluate_stream_decisions(bits, len_va, len_test, "trailing")
    tl = y_test[dec_ref.valid_mask]
    has_both = bool(np.any(tl == 1) and np.any(tl == 0))
    cal_rate = float(np.mean(dec_ref.cal_switch_counts > dec_ref.k_gate_balanced))

    abs_va = np.abs(va_ew.reshape(len(va_ew), -1))
    abs_te = np.abs(te_ew.reshape(len(te_ew), -1))
    med = np.median(abs_va[:n_cal_w], axis=0, keepdims=True)
    mad = 1.4826 * np.median(np.abs(abs_va[:n_cal_w] - med), axis=0, keepdims=True) + 1e-4
    z_va = np.maximum(0.0, (abs_va - med) / mad)
    z_te = np.maximum(0.0, (abs_te - med) / mad)
    q985 = np.percentile(z_va[:n_cal_w], 98.5, axis=0, keepdims=True) + 1e-4

    bank_stats = [
        ("Lite-MaxCoordZ", z_va.max(axis=1), z_te.max(axis=1)),
        ("Lite-RawTopK", (z_va > q985).sum(axis=1).astype(np.float64), (z_te > q985).sum(axis=1).astype(np.float64)),
        ("Lite-ConformalL2", np.sum(abs_va ** 2, axis=1), np.sum(abs_te ** 2, axis=1)),
    ]
    for bname, s_va_w, s_te_w in bank_stats:
        vp, vm = aggregate_vector_stream(s_va_w[:, None], len_va, C, S, "trailing")
        tp, tm = aggregate_vector_stream(s_te_w[:, None], len_test, C, S, "trailing")
        v, ts = vp[vm, 0], tp[tm, 0]
        half = len(v) // 2
        th = float(np.quantile(v[:half], 1.0 - cal_rate))
        ph = (v[half + S :] > th).astype(int)
        pt = (ts > th).astype(int)
        mt = compute_metrics(tl, pt, scores=ts, valid_mask=np.ones_like(tl, dtype=bool), use_pa=False)
        er, dl = event_stats(tl, pt)
        pr = float(average_precision_score(tl, ts)) if has_both else 0.0
        out_rows.append({
            **meta, "model": bname, "mapping": "trailing", "calibrator": "SystemMatched",
            "trainable_params": int(n_params), "threshold": th, "cal_kurtosis": 0.0, "ks_drift": float("nan"),
            "n_cal": int(half), "n_hold": int(len(ph)),
            "hold_fpr": float(ph.mean()) if len(ph) > 0 else 0.0,
            "test_fpr": float(pt[tl == 0].mean()) if np.any(tl == 0) else 0.0,
            "f1": float(mt.get("f1", 0.0)), "precision": float(mt.get("precision", 0.0)),
            "recall": float(mt.get("recall", 0.0)), "pr_auc": pr, "pr_auc_unw224": pr, "pr_auc_unw224_edge": pr,
            "event_recall": float(er) if er == er else float("nan"),
            "mean_delay": float(dl) if dl == dl else float("nan"),
            **_diagnostic_recalls(ts, tl, v[:half]),
        })

    # 2-fold chronological cross-fit across va halves: Fold 1 adapts on B1, calibrates on B2; Fold 2 swaps.
    b1_r, b1_w = va_raw[:n_cal_w], va_white[:n_cal_w]
    b2_r, b2_w = va_raw[n_cal_w + S :], va_white[n_cal_w + S :]
    if len(b2_r) >= 80:
        f1_cal_ew, f1_te_ew = stabilize_whitened_subblocks(b2_w, te_white, b2_r, te_raw, len(b2_r), b1_w, b1_r)
        f2_cal_ew, f2_te_ew = stabilize_whitened_subblocks(b1_w, te_white, b1_r, te_raw, len(b1_r), b2_w, b2_r)
        bits1 = panel_lite.calibrate_and_trip_windows(f1_cal_ew, f1_te_ew, len(b2_r))
        bits2 = panel_lite.calibrate_and_trip_windows(f2_cal_ew, f2_te_ew, len(b1_r))
        tp1, _ = aggregate_vector_stream(np.asarray(bits1["bits_pd_te"], np.float32).sum(axis=1, keepdims=True), len_test, C, S, "trailing")
        tp2, tm2 = aggregate_vector_stream(np.asarray(bits2["bits_pd_te"], np.float32).sum(axis=1, keepdims=True), len_test, C, S, "trailing")
        c1, c2 = tp1[tm2, 0], tp2[tm2, 0]
        k1 = max(float(np.percentile(np.asarray(bits1["bits_pd_va"], np.float32).sum(axis=1), 99.5)) * 1.12, 3.5)
        k2 = max(float(np.percentile(np.asarray(bits2["bits_pd_va"], np.float32).sum(axis=1), 99.5)) * 1.12, 3.5)
        pt_and = ((c1 > k1) & (c2 > k2)).astype(int)
        sc_cf = 0.5 * (c1 + c2)
        mt_cf = compute_metrics(tl, pt_and, scores=sc_cf, valid_mask=np.ones_like(tl, dtype=bool), use_pa=False)
        er_cf, dl_cf = event_stats(tl, pt_and)
        pr_cf = float(average_precision_score(tl, sc_cf)) if has_both else 0.0
        out_rows.append({
            **meta, "model": "Laminar-JEPA-Lite-CF", "mapping": "trailing", "calibrator": "CrossFit2Fold",
            "trainable_params": int(n_params), "threshold": 0.5 * (k1 + k2), "cal_kurtosis": 0.0,
            "ks_drift": float("nan"), "n_cal": int(len(b1_r) + len(b2_r)), "n_hold": 0,
            "hold_fpr": float("nan"), "test_fpr": float(pt_and[tl == 0].mean()) if np.any(tl == 0) else 0.0,
            "f1": float(mt_cf.get("f1", 0.0)), "precision": float(mt_cf.get("precision", 0.0)),
            "recall": float(mt_cf.get("recall", 0.0)), "pr_auc": pr_cf, "pr_auc_unw224": pr_cf,
            "pr_auc_unw224_edge": pr_cf, "event_recall": float(er_cf) if er_cf == er_cf else float("nan"),
            "mean_delay": float(dl_cf) if dl_cf == dl_cf else float("nan"),
            "er_q05": float("nan"), "er_f01": float("nan"), "er_f05": float("nan"),
        })


def _train_fixed_epochs(model: torch.nn.Module, tr_w: np.ndarray, cfg: CSMConfig, epochs: int, device: torch.device) -> None:
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tr_gpu = torch.from_numpy(np.ascontiguousarray(tr_w)).float().to(device)
    bs = 64 if len(tr_gpu) >= 256 else 16
    for _ in range(epochs):
        model.train()
        perm = torch.from_numpy(np.random.permutation(len(tr_gpu))).to(device)
        for i in range(0, len(perm), bs):
            b = tr_gpu[perm[i : i + bs]]
            loss, _ = model.compute_objective(b[:, :C], b[:, C:], cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.update_target_encoder(None)


def _build_schedule(suite: str) -> list[tuple[str, str, list[int]]]:
    if suite == "dev33":
        s19 = [tuple(ln.strip().split(":", 1)) for ln in (ROOT / "reports" / "streams_19_nonghl.txt").read_text().splitlines() if ln.strip()]
        s14 = [tuple(ln.strip().split(":", 1)) for ln in (ROOT / "reports" / "streams_14_ghl.txt").read_text().splitlines() if ln.strip()]
        return [(ds, ch, [42, 123, 456, 7, 2024]) for ds, ch in s19] + [(ds, ch, [42, 123, 456]) for ds, ch in s14]
    if suite == "dev19_4block":
        s19 = [tuple(ln.strip().split(":", 1)) for ln in (ROOT / "reports" / "streams_19_nonghl.txt").read_text().splitlines() if ln.strip()]
        return [(ds, ch, [42, 123, 456]) for ds, ch in s19]
    if suite in ("tier2_141", "tier2_141_5block"):
        s141 = [tuple(ln.strip().split(":", 1)) for ln in (ROOT / "reports" / "streams_141_tier2.txt").read_text().splitlines() if ln.strip()]
        return [(ds, ch, [42, 123, 456]) for ds, ch in s141]
    raise ValueError(f"Unknown suite '{suite}'.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run paired multi-seed benchmarks for Laminar-JEPA and Laminar-JEPA-Lite.")
    ap.add_argument("--suite", choices=("dev33", "dev19_4block", "tier2_141", "tier2_141_5block"), required=True)
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--lite-only", action="store_true", help="Evaluate Laminar-JEPA-Lite only")
    ap.add_argument("--max-streams", type=int, default=0)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    include_1d = args.suite == "tier2_141_5block" and not args.lite_only
    if args.lite_only:
        target_models = ("Laminar-JEPA-Lite",)
    elif include_1d:
        target_models = ("TS-JEPA-1D", "Multi-Horizon-JEPA-1D", "Laminar-JEPA", "Laminar-JEPA-Unq", "Laminar-JEPA-Lite")
    else:
        target_models = ("Laminar-JEPA", "Laminar-JEPA-Unq", "Laminar-JEPA-Lite")

    out_path = Path(args.out)
    rows: list[dict[str, Any]] = []
    if out_path.exists():
        rows = pd.read_csv(out_path).to_dict("records")
    done_keys = {(str(r["dataset"]), str(r["channel"]), int(r["seed"]), str(r["model"])) for r in rows}

    schedule = _build_schedule(args.suite)
    if args.max_streams > 0:
        schedule = schedule[: args.max_streams]
    if args.num_shards > 1:
        schedule = [item for i, item in enumerate(schedule) if i % args.num_shards == args.shard_id]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wsp_cfg = WSPConfig(context_len=C, suspect_len=S, candidate_k_counts=(4, 2))
    panel_224 = MultiPerspectiveCircuitBreakerPanel(context_size=C, suspect_size=S)
    panel_lite = PositionDriftLitePanel(context_size=C, suspect_size=S, use_hijack_guard=True)

    print(f"=== Starting suite={args.suite} ({len(schedule)} streams, models={target_models}) -> {out_path} ===", flush=True)

    for idx, (ds, ch, seeds) in enumerate(schedule, start=1):
        needed = [sd for sd in seeds if not all((ds, str(ch), sd, m) in done_keys for m in target_models)]
        if not needed:
            continue

        train, test, y = load_stream(ds, ch)
        q_w: np.ndarray | None = None
        val_mon_w: np.ndarray | None = None
        if args.suite == "tier2_141_5block":
            tr_w, val_mon_w, q_w, va_w, te_w, len_va = _split_nominal_5block(train, test)
        elif args.suite == "dev19_4block":
            tr_w, q_w, va_w, te_w, len_va = _split_nominal_dev_4block(train, test)
        elif args.suite == "tier2_141":
            n_val = int(len(train) * 0.35)
            tr_full, va = train[:-n_val], train[-n_val:]
            tr_fit, select = split_train_checkpoint_block(tr_full, len(train))
            tr_s, select_s, va_s, te_s = _standardize_arrays(tr_fit, select, va, test)
            tr_w = DataLoader.create_windows(tr_s, W, step=16, copy=False)
            val_mon_w = DataLoader.create_windows(select_s, W, step=max(1, min(8, (len(select_s) - W) // 32)), copy=False)
            va_w = DataLoader.create_windows(va_s, W, step=1, copy=False)
            te_w = DataLoader.create_windows(te_s, W, step=1, copy=False)
            len_va = len(va)
        else:
            n_val = int(len(train) * 0.35)
            tr_full, va = train[:-n_val], train[-n_val:]
            tr_s, va_s, te_s = _standardize_arrays(tr_full, va, test)
            tr_w = DataLoader.create_windows(tr_s, W, step=16, copy=False)
            va_w = DataLoader.create_windows(va_s, W, step=1, copy=False)
            te_w = DataLoader.create_windows(te_s, W, step=1, copy=False)
            len_va = len(va)

        n_cal_w = len(va_w) // 2
        k_ch = int(train.shape[1])
        is_heldout = (ds.lower(), str(ch)) not in DEVELOPMENT_STREAMS
        t_nom = int(len(train))

        t_s0 = time.perf_counter()
        for seed in needed:
            meta = {
                "dataset": ds, "channel": str(ch), "heldout": bool(is_heldout),
                "k_channels": k_ch, "t_nom": t_nom, "seed": int(seed), "suite": args.suite,
            }

            if include_1d and (ds, str(ch), seed, "TS-JEPA-1D") not in done_keys:
                torch.manual_seed(seed)
                np.random.seed(seed)
                cfg_ts = CSMConfig(model_type="ts_jepa", context_size=C, suspect_size=S, latent_dim=32)
                m_ts = build_ts_jepa_model(cfg_ts, input_dim=k_ch, device=device)
                p_ts = sum(p.numel() for p in m_ts.parameters() if p.requires_grad)
                assert val_mon_w is not None
                _fast_val_checkpoint_train(m_ts, tr_w, val_mon_w, cfg_ts, device, epochs=12, patience=4)
                _record_1d_spot_rows(
                    rows, meta, "TS-JEPA-1D", _score_1d_windows(m_ts, va_w, device),
                    _score_1d_windows(m_ts, te_w, device), len_va, len(test), y, p_ts,
                )
                del m_ts

            if include_1d and (ds, str(ch), seed, "Multi-Horizon-JEPA-1D") not in done_keys:
                torch.manual_seed(seed)
                np.random.seed(seed)
                cfg_sp = CSMConfig(model_type="ts_jepa", context_size=C, suspect_size=S, latent_dim=32)
                m_sp = SpeculativeJEPAModel(
                    context_encoder=build_encoder(cfg_sp, k_ch, device),
                    latent_dim=32, predictor_hidden_dim=64, ema_decay=0.996, dropout=cfg_sp.dropout,
                ).to(device)
                p_sp = sum(p.numel() for p in m_sp.parameters() if p.requires_grad)
                assert val_mon_w is not None
                _fast_val_checkpoint_train(m_sp, tr_w, val_mon_w, cfg_sp, device, epochs=12, patience=4)
                m_sp.fit_alien_bank(tr_w, context_len=C, batch_size=512)
                _record_1d_spot_rows(
                    rows, meta, "Multi-Horizon-JEPA-1D",
                    _score_1d_windows(m_sp, va_w, device, use_bank=True),
                    _score_1d_windows(m_sp, te_w, device, use_bank=True),
                    len_va, len(test), y, p_sp,
                )
                del m_sp

            if not args.lite_only and any((ds, str(ch), seed, m) not in done_keys for m in ("Laminar-JEPA", "Laminar-JEPA-Unq")):
                torch.manual_seed(seed)
                np.random.seed(seed)
                cfg_full = CSMConfig(model_type="soliton_reynolds_jepa", context_size=C, suspect_size=S, latent_dim=32)
                m_full = build_ts_jepa_model(cfg_full, input_dim=k_ch, device=device)
                p_full = sum(p.numel() for p in m_full.parameters() if p.requires_grad)

                if val_mon_w is not None:
                    _fast_val_checkpoint_train(m_full, tr_w, val_mon_w, cfg_full, device, epochs=12, patience=4)
                else:
                    _train_fixed_epochs(m_full, tr_w, cfg_full, epochs=15, device=device)

                with torch.inference_mode():
                    z_ctx_tr, z_tgt_tr = _fast_extract_4h_latents(m_full, tr_w, batch_size=512)
                    raw_tr_t = z_tgt_tr - _forward_predictors_with_phase(m_full, z_ctx_tr, m_full.subblock_embeddings)
                    res_flat = raw_tr_t.reshape(len(tr_w), -1)
                    mean_res = res_flat.mean(dim=0)
                    diff_tr = res_flat - mean_res
                    cov_tr = (diff_tr.T @ diff_tr) / max(len(res_flat) - 1, 1)
                    m_full.residual_mean.copy_(mean_res)
                    m_full.precision_matrix.copy_(resolvent_precision(cov_tr, u=1e-3))
                    m_full.precision_fitted.fill_(True)
                    tr_raw4_u = raw_tr_t.cpu().numpy().astype(np.float32)
                    del z_ctx_tr, z_tgt_tr, raw_tr_t, res_flat, diff_tr, cov_tr

                z_ctx_va, z_tgt_va = _fast_extract_4h_latents(m_full, va_w, batch_size=512)
                z_ctx_te, z_tgt_te = _fast_extract_4h_latents(m_full, te_w, batch_size=512)
                va_raw4_u, va_white4_u = extract_raw4_white4_from_latents(m_full, z_ctx_va, z_tgt_va)
                te_raw4_u, te_white4_u = extract_raw4_white4_from_latents(m_full, z_ctx_te, z_tgt_te)
                q_latents: tuple[torch.Tensor, torch.Tensor] | None = None
                if q_w is not None:
                    q_latents = _fast_extract_4h_latents(m_full, q_w, batch_size=512)
                    q_raw4_u, q_white4_u = extract_raw4_white4_from_latents(m_full, q_latents[0], q_latents[1])
                else:
                    q_raw4_u, q_white4_u = None, None
                _record_panel_rows(
                    rows, meta, "Laminar-JEPA-Unq", "CircuitBreaker", panel_224,
                    va_raw4_u, va_white4_u, te_raw4_u, te_white4_u, len_va, len(test), y, p_full,
                    q_raw=q_raw4_u, q_white=q_white4_u,
                )

                m_q = copy.deepcopy(m_full)
                quench_src = q_w if q_w is not None else va_w[:n_cal_w]
                q_lat_src = q_latents if q_latents is not None else (z_ctx_va[:n_cal_w], z_tgt_va[:n_cal_w])
                apply_surgical_wsp_quench(m_q, tr_w, quench_src, wsp_cfg, cal_latents=q_lat_src, tr_raw4=tr_raw4_u)
                va_raw4_q, va_white4_q = extract_raw4_white4_from_latents(m_q, z_ctx_va, z_tgt_va)
                te_raw4_q, te_white4_q = extract_raw4_white4_from_latents(m_q, z_ctx_te, z_tgt_te)
                if q_latents is not None:
                    q_raw4_q, q_white4_q = extract_raw4_white4_from_latents(m_q, q_latents[0], q_latents[1])
                else:
                    q_raw4_q, q_white4_q = None, None
                _record_panel_rows(
                    rows, meta, "Laminar-JEPA", "CircuitBreaker", panel_224,
                    va_raw4_q, va_white4_q, te_raw4_q, te_white4_q, len_va, len(test), y, p_full,
                    q_raw=q_raw4_q, q_white=q_white4_q,
                )
                del m_full, m_q, z_ctx_va, z_tgt_va, z_ctx_te, z_tgt_te, q_latents

            if (ds, str(ch), seed, "Laminar-JEPA-Lite") not in done_keys:
                torch.manual_seed(seed)
                np.random.seed(seed)
                cfg_lite = CSMConfig(model_type="laminar_jepa_lite", context_size=C, suspect_size=S, latent_dim=32)
                m_lite = build_ts_jepa_model(cfg_lite, input_dim=k_ch, device=device)
                assert isinstance(m_lite, LaminarJEPALiteModel)
                p_lite = sum(p.numel() for p in m_lite.parameters() if p.requires_grad)

                if val_mon_w is not None:
                    _fast_val_checkpoint_train(m_lite, tr_w, val_mon_w, cfg_lite, device, epochs=12, patience=4)
                else:
                    _train_fixed_epochs(m_lite, tr_w, cfg_lite, epochs=15, device=device)
                m_lite.fit_resolvent_whitening(tr_w, context_len=C, batch_size=512)

                va_raw3, va_white3 = m_lite.extract_raw_and_whitened_residuals(va_w, context_len=C, batch_size=512)
                te_raw3, te_white3 = m_lite.extract_raw_and_whitened_residuals(te_w, context_len=C, batch_size=512)
                if q_w is not None:
                    q_raw3, q_white3 = m_lite.extract_raw_and_whitened_residuals(q_w, context_len=C, batch_size=512)
                else:
                    q_raw3, q_white3 = None, None
                va_ew3, te_ew3 = _record_panel_rows(
                    rows, meta, "Laminar-JEPA-Lite", "PositionDrift128", panel_lite,
                    va_raw3, va_white3, te_raw3, te_white3, len_va, len(test), y, p_lite,
                    q_raw=q_raw3, q_white=q_white3,
                )
                if args.suite == "tier2_141_5block":
                    _record_lite_banks_and_crossfit(
                        rows, meta, panel_lite, va_raw3, va_white3, te_raw3, te_white3,
                        va_ew3, te_ew3, len_va, len(test), y, p_lite,
                    )
                del m_lite

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for m_done in target_models:
                done_keys.add((ds, str(ch), seed, m_done))
            atomic_save_csv(pd.DataFrame(rows), out_path)

        dt = time.perf_counter() - t_s0
        sub_s = [r for r in rows if r["dataset"] == ds and str(r["channel"]) == str(ch) and r["mapping"] == "trailing"]
        f_lite = float(np.mean([r["f1"] for r in sub_s if r["model"] == "Laminar-JEPA-Lite"]))
        f_full = float(np.mean([r["f1"] for r in sub_s if r["model"] == "Laminar-JEPA"])) if not args.lite_only else float("nan")
        print(
            f"[{idx:3d}/{len(schedule):3d}] {ds}:{ch} (K={k_ch:2d}, T={t_nom}, {dt:.1f}s) | "
            f"CausF1: Lam={f_full:.4f} Lite={f_lite:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
