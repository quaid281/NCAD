#!/usr/bin/env python3
"""Paired Multi-Seed Benchmark Suite for Laminar-JEPA and Laminar-JEPA-Lite.

Supports two canonical manuscript evaluation suites with atomic checkpointing and resume:
  1. `dev33`: 33 primary development streams (`19` non-GHL across 5 seeds `{42,123,456,7,2024}`
     + `14` GHL across 3 seeds `{42,123,456}`), fixed 15 epochs on 65% nominal training block,
     64-step isolation gap between 17.5% nominal calibration and 17.5% nominal holdout.
  2. `tier2_141`: All 141 qualifying Tier-2 multivariate streams (`118` untouched held-out +
     `23` development) across 3 seeds `{42,123,456}` under strictly disjoint checkpoint selection
     (`X_fit` 50% / `X_sel` 15% disjoint from `X_cal` 17.5% / 64-step gap / `X_hold` 17.5%),
     `epochs=12, patience=4`, matching `heldout_tier2_141stream_independent_select.csv`.
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
from src.engine.builder import build_ts_jepa_model
from src.experimental.weight_space_programming import WSPConfig, apply_surgical_wsp_quench
from src.models.jepa.alien_jepas import _encode_with_multidepth_taps, _whitening_defect_loss
from src.models.laminar_lite import LaminarJEPALiteModel
from src.scoring.circuit_breaker_panel import (
    CircuitBreakerStreamDecision,
    MultiPerspectiveCircuitBreakerPanel,
    stabilize_whitened_subblocks,
)
from src.scoring.evt_calibrator import compute_excess_kurtosis
from src.scoring.laminar_lite_panel import PositionDriftLitePanel
from scripts.circuit_breaker_19stream.common import DEVELOPMENT_STREAMS
from scripts.experiment_wsp_soliton import extract_raw4_white4_from_latents
from scripts.run_multiseed_heldout_benchmark import C, S, W, batches, compute_metrics, event_stats, load_stream
from scripts.run_tier2_exhaustive_benchmark import atomic_save_csv, split_train_checkpoint_block


def _standardize_splits(
    tr: np.ndarray, va: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize using float64 training moments to avoid float32 overflow on solar/industrial units."""
    tr64 = tr.astype(np.float64, copy=False)
    mu = tr64.mean(0)
    sd_arr = tr64.std(0)
    sc = np.where(sd_arr < 1e-8, 1.0, np.maximum(sd_arr, 0.01))
    tr_s = ((tr64 - mu) / sc).astype(np.float32)
    va_s = ((va.astype(np.float64, copy=False) - mu) / sc).astype(np.float32)
    te_s = ((test.astype(np.float64, copy=False) - mu) / sc).astype(np.float32)
    return tr_s, va_s, te_s, mu, sc


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
def _single_pass_val_metric(model: torch.nn.Module, val_w: np.ndarray, device: torch.device) -> float:
    """Compute exact `val_cos_err + 0.25 * val_loss` in a single `forward` pass (`batch_size=512`)."""
    model.eval()
    tot_loss, tot_cos, tot_n = 0.0, 0.0, 0
    with torch.inference_mode():
        for i in range(0, len(val_w), 512):
            b = torch.from_numpy(np.ascontiguousarray(val_w[i : i + 512])).float().to(device)
            _, z_tgt, z_pred, z_sub_pred, z_sub_tgt = model.forward(b[:, :C], b[:, C:])
            assert z_tgt is not None and z_sub_tgt is not None
            m_cnt = z_sub_pred.shape[1]
            weights = [0.50 / max(m_cnt - 1, 1)] * (m_cnt - 1) + [0.50]
            loss = sum(
                w_m * (F.mse_loss(z_sub_pred[:, m, :], z_sub_tgt[:, m, :]) + _whitening_defect_loss(z_sub_pred[:, m, :], gamma=0.75))
                for m, w_m in enumerate(weights)
            )
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
    """Identical optimizer/schedule/early-stop rule to `run_tier2_exhaustive_benchmark`, single-pass val."""
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=1e-5)
    best_score = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        for b in batches(tr_w, 16):
            b = b.to(device)
            loss, _ = model.compute_objective(b[:, :C], b[:, C:], cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            model.update_target_encoder(None)
        sched.step()

        sel_metric = _single_pass_val_metric(model, val_w, device)
        if ep == 1 or sel_metric < best_score - 1e-4:
            best_score = sel_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if patience > 0 and ep >= 6 and no_improve >= patience:
            break

    model.load_state_dict(best_state)


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


def _record_panel_rows(
    out_rows: list[dict[str, Any]],
    ds: str,
    ch: str,
    is_heldout: bool,
    k_ch: int,
    seed: int,
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
) -> None:
    n_cal_w = len(va_raw) // 2
    va_ew, te_ew = stabilize_whitened_subblocks(va_white, te_white, va_raw, te_raw, n_cal_w)
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
        pr = float(average_precision_score(tl, ts)) if (np.any(tl == 1) and np.any(tl == 0)) else 0.0
        hfpr = float(dec.hold_pred_balanced.mean()) if len(dec.hold_pred_balanced) > 0 else 0.0
        tfpr = float(dec.test_pred_balanced[tl == 0].mean()) if np.any(tl == 0) else 0.0
        diag = _diagnostic_recalls(ts, tl, dec.cal_switch_counts)

        out_rows.append(
            {
                "dataset": ds,
                "channel": ch,
                "heldout": bool(is_heldout),
                "k_channels": int(k_ch),
                "seed": int(seed),
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
                "event_recall": float(er) if er == er else float("nan"),
                "mean_delay": float(dl) if dl == dl else float("nan"),
                **diag,
            }
        )


def _train_fixed_epochs(model: torch.nn.Module, tr_w: np.ndarray, cfg: CSMConfig, epochs: int, device: torch.device) -> None:
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


def _build_schedule(suite: str) -> list[tuple[str, str, list[int]]]:
    if suite == "dev33":
        s19 = [tuple(ln.strip().split(":", 1)) for ln in (ROOT / "reports" / "streams_19_nonghl.txt").read_text().splitlines() if ln.strip()]
        s14 = [tuple(ln.strip().split(":", 1)) for ln in (ROOT / "reports" / "streams_14_ghl.txt").read_text().splitlines() if ln.strip()]
        return [(ds, ch, [42, 123, 456, 7, 2024]) for ds, ch in s19] + [(ds, ch, [42, 123, 456]) for ds, ch in s14]
    if suite == "tier2_141":
        s141 = [tuple(ln.strip().split(":", 1)) for ln in (ROOT / "reports" / "streams_141_tier2.txt").read_text().splitlines() if ln.strip()]
        return [(ds, ch, [42, 123, 456]) for ds, ch in s141]
    raise ValueError(f"Unknown suite '{suite}'.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run paired multi-seed benchmarks for Laminar-JEPA and Laminar-JEPA-Lite.")
    ap.add_argument("--suite", choices=("dev33", "tier2_141"), required=True)
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--lite-only", action="store_true", help="Evaluate Laminar-JEPA-Lite only (when full Laminar-JEPA is already frozen)")
    ap.add_argument("--max-streams", type=int, default=0)
    args = ap.parse_args()

    target_models = ("Laminar-JEPA-Lite",) if args.lite_only else ("Laminar-JEPA", "Laminar-JEPA-Unq", "Laminar-JEPA-Lite")
    out_path = Path(args.out)
    rows: list[dict[str, Any]] = []
    if out_path.exists():
        rows = pd.read_csv(out_path).to_dict("records")
    done_keys = {(str(r["dataset"]), str(r["channel"]), int(r["seed"]), str(r["model"])) for r in rows}

    schedule = _build_schedule(args.suite)
    if args.max_streams > 0:
        schedule = schedule[: args.max_streams]

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
        n_val = int(len(train) * 0.35)
        tr_full, va = train[:-n_val], train[-n_val:]
        if args.suite == "tier2_141":
            tr_fit, select = split_train_checkpoint_block(tr_full, len(train))
            tr_s, va_s, te_s, mu, sc = _standardize_splits(tr_fit, va, test)
            select_s = ((select.astype(np.float64, copy=False) - mu) / sc).astype(np.float32)
            val_mon_w = DataLoader.create_windows(
                select_s, W, step=max(1, min(8, (len(select_s) - W) // 32)), copy=False
            )
        else:
            tr_s, va_s, te_s, _, _ = _standardize_splits(tr_full, va, test)
            val_mon_w = None

        tr_w = DataLoader.create_windows(tr_s, W, step=16, copy=False)
        va_w = DataLoader.create_windows(va_s, W, step=1, copy=False)
        te_w = DataLoader.create_windows(te_s, W, step=1, copy=False)
        n_cal_w = len(va_w) // 2
        k_ch = int(train.shape[1])
        is_heldout = (ds.lower(), str(ch)) not in DEVELOPMENT_STREAMS

        t_s0 = time.perf_counter()
        for seed in needed:
            if not args.lite_only and any((ds, str(ch), seed, m) not in done_keys for m in ("Laminar-JEPA", "Laminar-JEPA-Unq")):
                torch.manual_seed(seed)
                np.random.seed(seed)
                cfg_full = CSMConfig(model_type="soliton_reynolds_jepa", context_size=C, suspect_size=S, latent_dim=32)
                m_full = build_ts_jepa_model(cfg_full, input_dim=k_ch, device=device)
                p_full = sum(p.numel() for p in m_full.parameters() if p.requires_grad)

                if args.suite == "tier2_141":
                    assert val_mon_w is not None
                    _fast_val_checkpoint_train(m_full, tr_w, val_mon_w, cfg_full, device, epochs=12, patience=4)
                else:
                    _train_fixed_epochs(m_full, tr_w, cfg_full, epochs=15, device=device)
                m_full.fit_alien_bank(tr_w, context_len=C)

                z_ctx_va, z_tgt_va = _fast_extract_4h_latents(m_full, va_w, batch_size=1024)
                z_ctx_te, z_tgt_te = _fast_extract_4h_latents(m_full, te_w, batch_size=1024)
                va_raw4_u, va_white4_u = extract_raw4_white4_from_latents(m_full, z_ctx_va, z_tgt_va)
                te_raw4_u, te_white4_u = extract_raw4_white4_from_latents(m_full, z_ctx_te, z_tgt_te)
                _record_panel_rows(
                    rows, ds, str(ch), is_heldout, k_ch, seed, "Laminar-JEPA-Unq", "CircuitBreaker",
                    panel_224, va_raw4_u, va_white4_u, te_raw4_u, te_white4_u, len(va), len(test), y, p_full,
                )

                m_q = copy.deepcopy(m_full)
                apply_surgical_wsp_quench(m_q, tr_w, va_w[:n_cal_w], wsp_cfg)
                va_raw4_q, va_white4_q = extract_raw4_white4_from_latents(m_q, z_ctx_va, z_tgt_va)
                te_raw4_q, te_white4_q = extract_raw4_white4_from_latents(m_q, z_ctx_te, z_tgt_te)
                _record_panel_rows(
                    rows, ds, str(ch), is_heldout, k_ch, seed, "Laminar-JEPA", "CircuitBreaker",
                    panel_224, va_raw4_q, va_white4_q, te_raw4_q, te_white4_q, len(va), len(test), y, p_full,
                )

            if (ds, str(ch), seed, "Laminar-JEPA-Lite") not in done_keys:
                torch.manual_seed(seed)
                np.random.seed(seed)
                cfg_lite = CSMConfig(model_type="laminar_jepa_lite", context_size=C, suspect_size=S, latent_dim=32)
                m_lite = build_ts_jepa_model(cfg_lite, input_dim=k_ch, device=device)
                assert isinstance(m_lite, LaminarJEPALiteModel)
                p_lite = sum(p.numel() for p in m_lite.parameters() if p.requires_grad)

                if args.suite == "tier2_141":
                    assert val_mon_w is not None
                    _fast_val_checkpoint_train(m_lite, tr_w, val_mon_w, cfg_lite, device, epochs=12, patience=4)
                else:
                    _train_fixed_epochs(m_lite, tr_w, cfg_lite, epochs=15, device=device)
                m_lite.fit_resolvent_whitening(tr_w, context_len=C, batch_size=512)

                va_raw3, va_white3 = m_lite.extract_raw_and_whitened_residuals(va_w, context_len=C, batch_size=1024)
                te_raw3, te_white3 = m_lite.extract_raw_and_whitened_residuals(te_w, context_len=C, batch_size=1024)
                _record_panel_rows(
                    rows, ds, str(ch), is_heldout, k_ch, seed, "Laminar-JEPA-Lite", "PositionDrift128",
                    panel_lite, va_raw3, va_white3, te_raw3, te_white3, len(va), len(test), y, p_lite,
                )

            for m_done in target_models:
                done_keys.add((ds, str(ch), seed, m_done))
            atomic_save_csv(pd.DataFrame(rows), out_path)

        dt = time.perf_counter() - t_s0
        sub_s = [r for r in rows if r["dataset"] == ds and str(r["channel"]) == str(ch) and r["mapping"] == "trailing"]
        f_lite = float(np.mean([r["f1"] for r in sub_s if r["model"] == "Laminar-JEPA-Lite"]))
        if args.lite_only:
            print(f"[{idx:3d}/{len(schedule):3d}] {ds}:{ch} (K={k_ch:2d}, seeds={len(needed)}, {dt:.1f}s) | CausF1 Lite={f_lite:.4f}", flush=True)
        else:
            f_full = float(np.mean([r["f1"] for r in sub_s if r["model"] == "Laminar-JEPA"]))
            print(f"[{idx:3d}/{len(schedule):3d}] {ds}:{ch} (K={k_ch:2d}, seeds={len(needed)}, {dt:.1f}s) | CausF1: Laminar={f_full:.4f} Lite={f_lite:.4f}", flush=True)


if __name__ == "__main__":
    main()
