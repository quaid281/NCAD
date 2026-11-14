#!/usr/bin/env python3
"""Matched-budget archive rerun for the TNNLS review (operating-point and same-run dial comparisons).

For every archive stream x seed under the disjoint 5-block protocol, this script retrains three
backbones and records, on identical partitions:

1. Multi-Horizon JEPA (1D) with SPOT, plus a label-free *rate-matched* variant whose threshold is set
   so that its calibration alarm rate equals Laminar-JEPA-Lite's calibration alarm rate.
2. Laminar-JEPA-Lite (128-dial panel), plus the reverse rate-matched variant (Lite count thresholded at
   the 1D SPOT calibration alarm rate), and system-matched coordinate aggregators on the same frozen
   residuals: max-z, raw top-k vote, l2^2 conformal, Fisher and Stouffer conformal p-value pooling,
   and a Benjamini-Hochberg multiplicity-controlled conformal bank.
3. Laminar-JEPA (unquenched) with the 224-dial panel and the 160-dial position-plus-drift panel computed
   from the *same* stabilized residual tensor.

All thresholds use calibration data only; `er_f01` / `er_f05` remain post-hoc diagnostics.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import special
from sklearn.metrics import average_precision_score
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CSMConfig
from src.engine.trainer import build_encoder, build_ts_jepa_model
from src.experimental.phase_embedding_synthesis import _forward_predictors_with_phase
from src.models._jepa_utils import resolvent_precision
from src.models.jepa.alien_jepas import SpeculativeJEPAModel
from src.models.laminar_lite import LaminarJEPALiteModel
from src.scoring.circuit_breaker_panel import (
    MultiPerspectiveCircuitBreakerPanel,
    aggregate_vector_stream,
    stabilize_whitened_subblocks,
)
from src.scoring.laminar_lite_panel import PositionDriftLitePanel
from scripts.circuit_breaker_19stream.common import DEVELOPMENT_STREAMS
from scripts.experiment_wsp_soliton import extract_raw4_white4_from_latents
from scripts.run_laminar_paired_benchmarks import (
    _build_schedule,
    _diagnostic_recalls,
    _fast_extract_4h_latents,
    _fast_val_checkpoint_train,
    _record_panel_rows,
    _score_1d_windows,
    _split_nominal_5block,
)
from scripts.run_multiseed_heldout_benchmark import C, S, compute_metrics, event_stats, load_stream, thresholds
from scripts.run_tier2_exhaustive_benchmark import atomic_save_csv

MODELS_DONE = ("MH1D", "Lite", "Full")


def _row(meta: dict[str, Any], model: str, mapping: str, calibrator: str, n_params: int, th: float,
         cal: np.ndarray, hold: np.ndarray, test: np.ndarray, tl: np.ndarray) -> dict[str, Any]:
    """Build one result row from calibration / hold / test score series and a fixed threshold."""
    ph = (hold > th).astype(int)
    pt = (test > th).astype(int)
    mt = compute_metrics(tl, pt, scores=test, valid_mask=np.ones_like(tl, dtype=bool), use_pa=False)
    er, dl = event_stats(tl, pt)
    has_both = bool(np.any(tl == 1) and np.any(tl == 0))
    return {
        **meta, "model": model, "mapping": mapping, "calibrator": calibrator,
        "trainable_params": int(n_params), "threshold": float(th),
        "n_cal": int(len(cal)), "n_hold": int(len(ph)),
        "cal_rate": float(np.mean(cal > th)) if len(cal) else float("nan"),
        "hold_fpr": float(ph.mean()) if len(ph) else 0.0,
        "test_fpr": float(pt[tl == 0].mean()) if np.any(tl == 0) else 0.0,
        "f1": float(mt.get("f1", 0.0)), "precision": float(mt.get("precision", 0.0)),
        "recall": float(mt.get("recall", 0.0)),
        "pr_auc": float(average_precision_score(tl, test)) if has_both else 0.0,
        "event_recall": float(er) if er == er else float("nan"),
        "mean_delay": float(dl) if dl == dl else float("nan"),
        **_diagnostic_recalls(test, tl, cal),
    }


def _rate_threshold(cal: np.ndarray, rate: float) -> float:
    """Threshold so that the empirical calibration alarm rate (strict `>`) is at most `rate`."""
    if rate <= 0.0 or len(cal) == 0:
        return float(np.max(cal)) + 1e-9 if len(cal) else float("inf")
    return float(np.quantile(cal, 1.0 - rate, method="higher"))


def _series(win_scores: np.ndarray, len_va: int, len_te: int, mapping: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vp, vm = aggregate_vector_stream(win_scores[0][:, None], len_va, C, S, mapping)
    tp, tm = aggregate_vector_stream(win_scores[1][:, None], len_te, C, S, mapping)
    return vp[vm, 0], tp[tm, 0], tm


def _conformal_pvals(z_cal: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Per-dial split-conformal upper-tail p-values `(1 + #{cal >= z}) / (n_cal + 1)`."""
    n = len(z_cal)
    srt = np.sort(z_cal, axis=0)
    out = np.empty_like(z, dtype=np.float64)
    for j in range(z.shape[1]):
        ge = n - np.searchsorted(srt[:, j], z[:, j], side="left")
        out[:, j] = (1.0 + ge) / (n + 1.0)
    return out


def lite_coordinate_z(va_ew: np.ndarray, te_ew: np.ndarray) -> dict[str, np.ndarray]:
    """Robust per-coordinate position+drift z-scores and conformal p-values on frozen Lite residuals."""
    n_cal_w = len(va_ew) // 2
    abs_va = np.abs(va_ew.reshape(len(va_ew), -1))
    abs_te = np.abs(te_ew.reshape(len(te_ew), -1))
    pd_va = np.concatenate([abs_va, np.abs(va_ew.mean(axis=1))], axis=1)
    pd_te = np.concatenate([abs_te, np.abs(te_ew.mean(axis=1))], axis=1)
    med = np.median(pd_va[:n_cal_w], axis=0, keepdims=True)
    mad = 1.4826 * np.median(np.abs(pd_va[:n_cal_w] - med), axis=0, keepdims=True) + 1e-4
    z_va, z_te = np.maximum(0.0, (pd_va - med) / mad), np.maximum(0.0, (pd_te - med) / mad)
    return {
        "abs_va": abs_va, "abs_te": abs_te, "z_va": z_va, "z_te": z_te,
        "q985": np.percentile(z_va[:n_cal_w], 98.5, axis=0, keepdims=True) + 1e-4,
        "p_va": _conformal_pvals(z_va[:n_cal_w], z_va), "p_te": _conformal_pvals(z_va[:n_cal_w], z_te),
    }


def lite_bank_scores(va_ew: np.ndarray, te_ew: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Window-level max-z, raw top-k, and Fisher scores on frozen Lite residuals."""
    zz = lite_coordinate_z(va_ew, te_ew)
    z_va, z_te, q985 = zz["z_va"], zz["z_te"], zz["q985"]
    return [
        ("MaxZ", z_va.max(1), z_te.max(1)),
        ("TopK", (z_va > q985).sum(1).astype(np.float64), (z_te > q985).sum(1).astype(np.float64)),
        ("Fisher", -2.0 * np.log(zz["p_va"]).sum(1), -2.0 * np.log(zz["p_te"]).sum(1)),
    ]


def _lite_aggregators(meta, rows, va_ew, te_ew, len_va, len_te, y, n_params, lite_cal_rate) -> None:
    """System-level aggregators on the identical frozen Lite residuals (trailing attribution)."""
    zz = lite_coordinate_z(va_ew, te_ew)
    abs_va, abs_te, z_va, z_te, q985 = zz["abs_va"], zz["abs_te"], zz["z_va"], zz["z_te"], zz["q985"]
    p_va, p_te = zz["p_va"], zz["p_te"]
    fisher_va, fisher_te = -2.0 * np.log(p_va).sum(1), -2.0 * np.log(p_te).sum(1)
    stouf_va = -special.ndtri(np.clip(p_va, 1e-12, 1 - 1e-12)).sum(1) / np.sqrt(p_va.shape[1])
    stouf_te = -special.ndtri(np.clip(p_te, 1e-12, 1 - 1e-12)).sum(1) / np.sqrt(p_te.shape[1])

    def bh_reject(p: np.ndarray, q: float = 0.005) -> np.ndarray:
        m = p.shape[1]
        ps = np.sort(p, axis=1)
        crit = q * np.arange(1, m + 1) / m
        return np.any(ps <= crit[None, :], axis=1).astype(np.float64)

    banks = [
        ("Lite-MaxCoordZ", z_va.max(1), z_te.max(1)),
        ("Lite-RawTopK", (z_va > q985).sum(1).astype(np.float64), (z_te > q985).sum(1).astype(np.float64)),
        ("Lite-ConformalL2", (abs_va ** 2).sum(1), (abs_te ** 2).sum(1)),
        ("Lite-Fisher", fisher_va, fisher_te),
        ("Lite-Stouffer", stouf_va, stouf_te),
    ]
    for name, s_va, s_te in banks:
        v, t, tm = _series((s_va, s_te), len_va, len_te, "trailing")
        half = len(v) // 2
        th = _rate_threshold(v[:half], lite_cal_rate)
        rows.append(_row(meta, name, "trailing", "SystemMatched", n_params, th, v[:half], v[half + S:], t, y[tm]))
    v, t, tm = _series((bh_reject(p_va), bh_reject(p_te)), len_va, len_te, "trailing")
    half = len(v) // 2
    rows.append(_row(meta, "Lite-BH-Conformal", "trailing", "BH-FDR", n_params, 0.5, v[:half], v[half + S:], t, y[tm]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-streams", type=int, default=0)
    args = ap.parse_args()

    out_path = Path(args.out)
    rows: list[dict[str, Any]] = pd.read_csv(out_path).to_dict("records") if out_path.exists() else []
    done = {(str(r["dataset"]), str(r["channel"]), int(r["seed"])) for r in rows if r["model"] == "Laminar-JEPA-160"}

    schedule = _build_schedule("tier2_141_5block")
    if args.max_streams > 0:
        schedule = schedule[: args.max_streams]
    if args.num_shards > 1:
        schedule = [it for i, it in enumerate(schedule) if i % args.num_shards == args.shard_id]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panel_224 = MultiPerspectiveCircuitBreakerPanel(context_size=C, suspect_size=S)
    panel_pd = PositionDriftLitePanel(context_size=C, suspect_size=S, use_hijack_guard=True)
    print(f"=== matched-budget rerun: {len(schedule)} streams -> {out_path} ===", flush=True)

    for idx, (ds, ch, seeds) in enumerate(schedule, start=1):
        need = [sd for sd in seeds if (ds, str(ch), sd) not in done]
        if not need:
            continue
        train, test, y = load_stream(ds, ch)
        tr_w, sel_w, q_w, va_w, te_w, len_va = _split_nominal_5block(train, test)
        len_te, k_ch = len(test), int(train.shape[1])
        t0 = time.perf_counter()
        for seed in need:
            meta = {"dataset": ds, "channel": str(ch), "heldout": (ds.lower(), str(ch)) not in DEVELOPMENT_STREAMS,
                    "k_channels": k_ch, "t_nom": int(len(train)), "seed": int(seed), "suite": "matched_budget_5block"}

            # --- (1) Multi-Horizon JEPA (1D) ---
            torch.manual_seed(seed); np.random.seed(seed)
            cfg_sp = CSMConfig(model_type="ts_jepa", context_size=C, suspect_size=S, latent_dim=32)
            m_sp = SpeculativeJEPAModel(context_encoder=build_encoder(cfg_sp, k_ch, device), latent_dim=32,
                                        predictor_hidden_dim=64, ema_decay=0.996, dropout=cfg_sp.dropout).to(device)
            p_sp = sum(p.numel() for p in m_sp.parameters() if p.requires_grad)
            _fast_val_checkpoint_train(m_sp, tr_w, sel_w, cfg_sp, device, epochs=12, patience=4)
            m_sp.fit_alien_bank(tr_w, context_len=C, batch_size=512)
            ws_1d = (_score_1d_windows(m_sp, va_w, device, use_bank=True), _score_1d_windows(m_sp, te_w, device, use_bank=True))
            del m_sp

            # --- (2) Laminar-JEPA-Lite ---
            torch.manual_seed(seed); np.random.seed(seed)
            cfg_l = CSMConfig(model_type="laminar_jepa_lite", context_size=C, suspect_size=S, latent_dim=32)
            m_l = build_ts_jepa_model(cfg_l, input_dim=k_ch, device=device)
            assert isinstance(m_l, LaminarJEPALiteModel)
            p_l = sum(p.numel() for p in m_l.parameters() if p.requires_grad)
            _fast_val_checkpoint_train(m_l, tr_w, sel_w, cfg_l, device, epochs=12, patience=4)
            m_l.fit_resolvent_whitening(tr_w, context_len=C, batch_size=512)
            va_r3, va_w3 = m_l.extract_raw_and_whitened_residuals(va_w, context_len=C, batch_size=512)
            te_r3, te_w3 = m_l.extract_raw_and_whitened_residuals(te_w, context_len=C, batch_size=512)
            q_r3, q_w3 = m_l.extract_raw_and_whitened_residuals(q_w, context_len=C, batch_size=512)
            del m_l
            n_cal_w = len(va_w) // 2
            va_ew3, te_ew3 = stabilize_whitened_subblocks(va_w3, te_w3, va_r3, te_r3, n_cal_w, adapt_white4=q_w3, adapt_raw4=q_r3)
            bits_l = panel_pd.calibrate_and_trip_windows(va_ew3, te_ew3, n_cal_w)

            lite_rate: dict[str, float] = {}
            for mapping in ("trailing", "smear"):
                dec = panel_pd.evaluate_stream_decisions(bits_l, len_va, len_te, mapping)
                tl = y[dec.valid_mask]
                k = float(dec.k_gate_balanced)
                rows.append(_row(meta, "Laminar-JEPA-Lite", mapping, "PositionDrift128", p_l, k,
                                 dec.cal_switch_counts, dec.hold_switch_counts, dec.test_switch_counts, tl))
                lite_rate[mapping] = float(np.mean(dec.cal_switch_counts > k))
                # 1D SPOT and both rate-matched directions on this mapping.
                v1, t1, tm1 = _series(ws_1d, len_va, len_te, mapping)
                half = len(v1) // 2
                th_spot = float(thresholds(v1[:half])[0]["SPOT"])
                rows.append(_row(meta, "Multi-Horizon-JEPA-1D", mapping, "SPOT", p_sp, th_spot, v1[:half], v1[half + S:], t1, y[tm1]))
                th_m = _rate_threshold(v1[:half], lite_rate[mapping])
                rows.append(_row(meta, "MH1D-RateMatched-Lite", mapping, "RateMatched", p_sp, th_m, v1[:half], v1[half + S:], t1, y[tm1]))
                r1 = float(np.mean(v1[:half] > th_spot))
                th_l = _rate_threshold(dec.cal_switch_counts, r1)
                rows.append(_row(meta, "Lite-RateMatched-1D", mapping, "RateMatched", p_l, th_l,
                                 dec.cal_switch_counts, dec.hold_switch_counts, dec.test_switch_counts, tl))
            _lite_aggregators(meta, rows, va_ew3, te_ew3, len_va, len_te, y, p_l, lite_rate["trailing"])

            # --- (3) Laminar-JEPA (unquenched): 224-dial and 160-dial panels on the same residuals ---
            torch.manual_seed(seed); np.random.seed(seed)
            cfg_f = CSMConfig(model_type="soliton_reynolds_jepa", context_size=C, suspect_size=S, latent_dim=32)
            m_f = build_ts_jepa_model(cfg_f, input_dim=k_ch, device=device)
            p_f = sum(p.numel() for p in m_f.parameters() if p.requires_grad)
            _fast_val_checkpoint_train(m_f, tr_w, sel_w, cfg_f, device, epochs=12, patience=4)
            with torch.inference_mode():
                z_c, z_t = _fast_extract_4h_latents(m_f, tr_w, batch_size=512)
                res = (z_t - _forward_predictors_with_phase(m_f, z_c, m_f.subblock_embeddings)).reshape(len(tr_w), -1)
                mu = res.mean(0)
                cov = ((res - mu).T @ (res - mu)) / max(len(res) - 1, 1)
                m_f.residual_mean.copy_(mu)
                m_f.precision_matrix.copy_(resolvent_precision(cov, u=1e-3))
                m_f.precision_fitted.fill_(True)
                del z_c, z_t, res, cov
            zc_va, zt_va = _fast_extract_4h_latents(m_f, va_w, batch_size=512)
            zc_te, zt_te = _fast_extract_4h_latents(m_f, te_w, batch_size=512)
            zc_q, zt_q = _fast_extract_4h_latents(m_f, q_w, batch_size=512)
            va_r4, va_w4 = extract_raw4_white4_from_latents(m_f, zc_va, zt_va)
            te_r4, te_w4 = extract_raw4_white4_from_latents(m_f, zc_te, zt_te)
            q_r4, q_w4 = extract_raw4_white4_from_latents(m_f, zc_q, zt_q)
            del m_f, zc_va, zt_va, zc_te, zt_te, zc_q, zt_q
            _record_panel_rows(rows, meta, "Laminar-JEPA-Unq", "CircuitBreaker", panel_224,
                               va_r4, va_w4, te_r4, te_w4, len_va, len_te, y, p_f, q_raw=q_r4, q_white=q_w4)
            _record_panel_rows(rows, meta, "Laminar-JEPA-160", "PositionDrift160", panel_pd,
                               va_r4, va_w4, te_r4, te_w4, len_va, len_te, y, p_f, q_raw=q_r4, q_white=q_w4)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            done.add((ds, str(ch), seed))
            atomic_save_csv(pd.DataFrame(rows), out_path)
        print(f"[{idx:3d}/{len(schedule):3d}] {ds}:{ch} K={k_ch} T={len(train)} ({time.perf_counter() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
