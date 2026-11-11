#!/usr/bin/env python3
"""Supplementary Paired Evaluations for Laminar-JEPA and Laminar-JEPA-Lite.

Produces two focused reviewer-facing outputs with atomic CSV persistence:
  1. `reports/horizon_laminar_lite_sweep.csv`: Evaluates `Laminar-JEPA-Lite` across target horizons
     `S in {1, 4, 16, 32, 64}` on the 5 horizon-sweep streams x 3 seeds `{42, 123, 456}`, matching
     `scripts/reviewer_analyses/run_224bit_horizon_sweep.py` (`EPOCHS=10`, `TRAIN_BS=64`).
  2. `reports/laminar_dev19_contextual_and_diag.csv` + `reports/laminar_contextual_summary.json`:
     Evaluates `Laminar-JEPA` (quenched 224-Bit) and `Laminar-JEPA-Lite` (128-Bit P+D) across the
     19 non-GHL development streams x 5 seeds `{42, 123, 456, 7, 2024}`, recording:
       - Per-stream / per-dataset buffered & causal metrics and diagnostic recalls (`q*=0.05`, `f*=0.01, 0.05`)
       - NASA SMAP/MSL Contextual vs. Point ground-truth taxonomy (`tab:point_vs_contextual` Panel a)
       - 19-Stream in-range Contextual (`max|z| <= 5`) vs. out-of-range Point (`max|z| > 5`) (`Panel b`)
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CSMConfig
from src.data.data_loader import DataLoader
from src.engine.builder import build_ts_jepa_model
from src.experimental.weight_space_programming import WSPConfig, apply_surgical_wsp_quench
from src.models.laminar_lite import LaminarJEPALiteModel
from src.scoring.circuit_breaker_panel import MultiPerspectiveCircuitBreakerPanel, stabilize_whitened_subblocks
from src.scoring.laminar_lite_panel import PositionDriftLitePanel
from scripts.experiment_wsp_soliton import extract_raw4_white4_from_latents
from scripts.reviewer_analyses.run_224bit_horizon_sweep import HORIZONS, STREAMS_5
from scripts.run_laminar_paired_benchmarks import (
    _fast_extract_4h_latents,
    _record_panel_rows,
    _standardize_splits,
    _train_fixed_epochs,
)
from scripts.run_multiseed_heldout_benchmark import C, S, W, batches, compute_metrics, event_stats, load_stream
from scripts.run_tier2_exhaustive_benchmark import atomic_save_csv

NASA_CONTEXTUAL_SEQS = [
    ("SMAP", "E-1", (5550, 6150)),
    ("SMAP", "A-2", None),
    ("SMAP", "A-7", None),
    ("MSL", "M-1", None),
    ("MSL", "C-1", (2000, 2260)),
]
NASA_POINT_SEQS = [
    ("SMAP", "E-1", (4950, 5080)),
    ("SMAP", "A-1", None),
    ("SMAP", "P-3", None),
    ("SMAP", "D-3", None),
    ("MSL", "D-14", (1600, 1700)),
    ("MSL", "D-14", (1750, 2050)),
    ("MSL", "D-15", None),
    ("MSL", "C-1", (500, 800)),
]


def _contiguous_segments(y: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(y == 1)
    if len(idx) == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    return [(int(g[0]), int(g[-1]) + 1) for g in np.split(idx, splits)]


def _atomic_write_json(payload: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(f".tmp.{os.getpid()}.json")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, out_path)


def run_horizon_lite_sweep(device: torch.device) -> None:
    out_path = ROOT / "reports" / "horizon_laminar_lite_sweep.csv"
    if out_path.exists() and len(pd.read_csv(out_path)) >= len(STREAMS_5) * len(HORIZONS) * 3 * 2:
        print(f"Horizon sweep already complete at {out_path}")
        return

    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for ds, ch in STREAMS_5:
        train, test, y_test = load_stream(ds, ch)
        n_val = int(len(train) * 0.35)
        tr_s, va_s, te_s, _, _ = _standardize_splits(train[:-n_val], train[-n_val:], test)

        for s_len in HORIZONS:
            w_len = C + s_len
            tr_w = DataLoader.create_windows(tr_s, w_len, step=16, copy=False)
            va_w = DataLoader.create_windows(va_s, w_len, step=1, copy=False)
            te_w = DataLoader.create_windows(te_s, w_len, step=1, copy=False)
            n_cal_w = len(va_w) // 2
            panel_lite = PositionDriftLitePanel(context_size=C, suspect_size=s_len, use_hijack_guard=True)

            for seed in (42, 123, 456):
                torch.manual_seed(seed)
                np.random.seed(seed)
                cfg = CSMConfig(model_type="laminar_jepa_lite", context_size=C, suspect_size=s_len, latent_dim=32)
                m_lite = build_ts_jepa_model(cfg, input_dim=train.shape[1], device=device)
                assert isinstance(m_lite, LaminarJEPALiteModel)
                opt = optim.AdamW(m_lite.parameters(), lr=1e-3, weight_decay=1e-4)
                for _ in range(10):
                    m_lite.train()
                    for b in batches(tr_w, 64):
                        b = b.to(device)
                        loss, _ = m_lite.compute_objective(b[:, :C], b[:, C:], cfg)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                        m_lite.update_target_encoder(None)
                m_lite.fit_resolvent_whitening(tr_w, context_len=C, batch_size=512)

                va_raw, va_white = m_lite.extract_raw_and_whitened_residuals(va_w, context_len=C, batch_size=1024)
                te_raw, te_white = m_lite.extract_raw_and_whitened_residuals(te_w, context_len=C, batch_size=1024)
                va_ew, te_ew = stabilize_whitened_subblocks(va_white, te_white, va_raw, te_raw, n_cal_w)
                bits = panel_lite.calibrate_and_trip_windows(va_ew, te_ew, n_cal_w)

                for mapping in ("trailing", "smear"):
                    dec = panel_lite.evaluate_stream_decisions(bits, len(va_s), len(te_s), mapping)
                    tl = y_test[dec.valid_mask]
                    mt = compute_metrics(
                        tl, dec.test_pred_balanced, scores=dec.test_switch_counts,
                        valid_mask=np.ones_like(tl, dtype=bool), use_pa=False,
                    )
                    er, dl = event_stats(tl, dec.test_pred_balanced)
                    tfpr = float(dec.test_pred_balanced[tl == 0].mean()) if np.any(tl == 0) else 0.0
                    hfpr = float(dec.hold_pred_balanced.mean()) if len(dec.hold_pred_balanced) > 0 else 0.0
                    rows.append({
                        "dataset": ds,
                        "stream": f"{ds}:{ch}",
                        "horizon_S": s_len,
                        "structural_delay": s_len - 1 if mapping == "trailing" else 0,
                        "seed": seed,
                        "variant": "Laminar-JEPA-Lite",
                        "mapping": mapping,
                        "f1": float(mt["f1"]),
                        "precision": float(mt["precision"]),
                        "recall": float(mt["recall"]),
                        "event_recall": float(er),
                        "fpr": tfpr,
                        "hold_fpr": hfpr,
                        "pr_auc": float(mt["pr_auc"]),
                        "delay": float(dl) if dl == dl else float("nan"),
                    })
            atomic_save_csv(pd.DataFrame(rows), out_path)
            print(f"[Horizon {time.perf_counter() - t0:5.1f}s] {ds}:{ch} S={s_len:2d} done", flush=True)


def _eval_event_subset(
    scores: np.ndarray,
    auto_pred: np.ndarray,
    cal_scores: np.ndarray,
    labels: np.ndarray,
    segs: list[tuple[int, int]],
) -> dict[str, float]:
    if not segs or np.all(labels == 1):
        return {"auto": float("nan"), "q05": float("nan"), "f01": float("nan"), "f05": float("nan")}
    s_clean = np.where(np.isfinite(scores), scores, np.inf)
    nom = s_clean[labels == 0]
    th_q05 = float(np.quantile(cal_scores, 0.95)) if len(cal_scores) > 0 else float("inf")
    th_f01 = float(np.quantile(nom, 0.99))
    th_f05 = float(np.quantile(nom, 0.95))

    def _seg_er(pred_bin: np.ndarray) -> float:
        hits = [1.0 if np.any(pred_bin[a:b] > 0) else 0.0 for a, b in segs if b > a]
        return float(np.mean(hits)) if hits else float("nan")

    return {
        "auto": _seg_er(auto_pred),
        "q05": _seg_er((s_clean > th_q05).astype(int)),
        "f01": _seg_er((s_clean > th_f01).astype(int)),
        "f05": _seg_er((s_clean > th_f05).astype(int)),
    }


def run_dev19_laminar_and_contextual(device: torch.device) -> None:
    out_csv = ROOT / "reports" / "laminar_dev19_full_multiseed.csv"
    out_ctx_csv = ROOT / "reports" / "laminar_contextual_events.csv"
    s19 = [
        tuple(ln.strip().split(":", 1))
        for ln in (ROOT / "reports" / "streams_19_nonghl.txt").read_text().splitlines()
        if ln.strip()
    ]
    seeds = [42, 123, 456, 7, 2024]
    wsp_cfg = WSPConfig(context_len=C, suspect_len=S, candidate_k_counts=(4, 2))
    panel_224 = MultiPerspectiveCircuitBreakerPanel(context_size=C, suspect_size=S)
    panel_lite = PositionDriftLitePanel(context_size=C, suspect_size=S, use_hijack_guard=True)

    rows: list[dict[str, Any]] = pd.read_csv(out_csv).to_dict("records") if out_csv.exists() else []
    ctx_rows: list[dict[str, Any]] = pd.read_csv(out_ctx_csv).to_dict("records") if out_ctx_csv.exists() else []
    done = {(str(r["dataset"]), str(r["channel"]), int(r["seed"]), str(r["model"])) for r in rows}

    for idx, (ds, ch) in enumerate(s19, start=1):
        needed = [sd for sd in seeds if (ds, ch, sd, "Laminar-JEPA") not in done]
        if not needed and any(r["dataset"] == ds and str(r["channel"]) == ch for r in ctx_rows):
            continue

        train, test, y = load_stream(ds, ch)
        n_val = int(len(train) * 0.35)
        tr_s, va_s, te_s, _, _ = _standardize_splits(train[:-n_val], train[-n_val:], test)
        tr_w = DataLoader.create_windows(tr_s, W, step=16, copy=False)
        va_w = DataLoader.create_windows(va_s, W, step=1, copy=False)
        te_w = DataLoader.create_windows(te_s, W, step=1, copy=False)
        n_cal_w = len(va_w) // 2
        k_ch = int(train.shape[1])

        t0 = time.perf_counter()
        for seed in seeds:
            if (ds, ch, seed, "Laminar-JEPA") in done and any(
                r["dataset"] == ds and str(r["channel"]) == ch and int(r["seed"]) == seed for r in ctx_rows
            ):
                continue

            # 1. Full Laminar-JEPA (4H + gated surgical quenching) + Unquenched control
            torch.manual_seed(seed)
            np.random.seed(seed)
            cfg_full = CSMConfig(model_type="soliton_reynolds_jepa", context_size=C, suspect_size=S, latent_dim=32)
            m_full = build_ts_jepa_model(cfg_full, input_dim=k_ch, device=device)
            p_full = sum(p.numel() for p in m_full.parameters() if p.requires_grad)
            _train_fixed_epochs(m_full, tr_w, cfg_full, epochs=15, device=device)
            m_full.fit_alien_bank(tr_w, context_len=C)
            z_ctx_va, z_tgt_va = _fast_extract_4h_latents(m_full, va_w, batch_size=1024)
            z_ctx_te, z_tgt_te = _fast_extract_4h_latents(m_full, te_w, batch_size=1024)

            va_raw_u, va_w_u = extract_raw4_white4_from_latents(m_full, z_ctx_va, z_tgt_va)
            te_raw_u, te_w_u = extract_raw4_white4_from_latents(m_full, z_ctx_te, z_tgt_te)
            _record_panel_rows(
                rows, ds, ch, False, k_ch, seed, "Laminar-JEPA-Unq", "CircuitBreaker",
                panel_224, va_raw_u, va_w_u, te_raw_u, te_w_u, len(va_s), len(te_s), y, p_full,
            )

            m_q = copy.deepcopy(m_full)
            apply_surgical_wsp_quench(m_q, tr_w, va_w[:n_cal_w], wsp_cfg)
            va_raw_q, va_w_q = extract_raw4_white4_from_latents(m_q, z_ctx_va, z_tgt_va)
            te_raw_q, te_w_q = extract_raw4_white4_from_latents(m_q, z_ctx_te, z_tgt_te)
            _record_panel_rows(
                rows, ds, ch, False, k_ch, seed, "Laminar-JEPA", "CircuitBreaker",
                panel_224, va_raw_q, va_w_q, te_raw_q, te_w_q, len(va_s), len(te_s), y, p_full,
            )

            # 2. Laminar-JEPA-Lite for contextual vs point stratification
            torch.manual_seed(seed)
            np.random.seed(seed)
            cfg_lite = CSMConfig(model_type="laminar_jepa_lite", context_size=C, suspect_size=S, latent_dim=32)
            m_lite = build_ts_jepa_model(cfg_lite, input_dim=k_ch, device=device)
            assert isinstance(m_lite, LaminarJEPALiteModel)
            _train_fixed_epochs(m_lite, tr_w, cfg_lite, epochs=15, device=device)
            m_lite.fit_resolvent_whitening(tr_w, context_len=C, batch_size=512)
            va_raw_l, va_w_l = m_lite.extract_raw_and_whitened_residuals(va_w, context_len=C, batch_size=1024)
            te_raw_l, te_w_l = m_lite.extract_raw_and_whitened_residuals(te_w, context_len=C, batch_size=1024)

            for mname, pnl, vr, vw, tr_r, tw in [
                ("Laminar-JEPA", panel_224, va_raw_q, va_w_q, te_raw_q, te_w_q),
                ("Laminar-JEPA-Lite", panel_lite, va_raw_l, va_w_l, te_raw_l, te_w_l),
            ]:
                va_ew, te_ew = stabilize_whitened_subblocks(vw, tw, vr, tr_r, n_cal_w)
                bits = pnl.calibrate_and_trip_windows(va_ew, te_ew, n_cal_w)
                dec_tr = pnl.evaluate_stream_decisions(bits, len(va_s), len(te_s), "trailing")
                dec_sm = pnl.evaluate_stream_decisions(bits, len(va_s), len(te_s), "smear")

                tl_tr = y[dec_tr.valid_mask]
                te_valid = te_s[dec_tr.valid_mask]
                all_segs = _contiguous_segments(tl_tr)
                ctx_5s = [sg for sg in all_segs if float(np.max(np.abs(te_valid[sg[0] : sg[1]]))) <= 5.0]
                pt_5s = [sg for sg in all_segs if float(np.max(np.abs(te_valid[sg[0] : sg[1]]))) > 5.0]

                r_ctx5 = _eval_event_subset(dec_tr.test_switch_counts, dec_tr.test_pred_balanced, dec_tr.cal_switch_counts, tl_tr, ctx_5s)
                r_pt5 = _eval_event_subset(dec_tr.test_switch_counts, dec_tr.test_pred_balanced, dec_tr.cal_switch_counts, tl_tr, pt_5s)

                offset = int(np.flatnonzero(dec_tr.valid_mask)[0])
                for tax_name, seq_list in [("nasa_contextual", NASA_CONTEXTUAL_SEQS), ("nasa_point", NASA_POINT_SEQS)]:
                    for s_ds, s_ch, bnd in seq_list:
                        if (s_ds, s_ch) != (ds, ch):
                            continue
                        segs_cand = all_segs if bnd is None else [
                            sg for sg in all_segs if sg[0] + offset >= bnd[0] and sg[1] + offset <= bnd[1]
                        ]
                        if segs_cand:
                            seq_id = f"{s_ds}:{s_ch}:{bnd[0]}" if bnd is not None else f"{s_ds}:{s_ch}:all"
                            ctx_rows.append({
                                "dataset": ds, "channel": seq_id, "seed": seed, "model": mname, "taxonomy": tax_name,
                                "buf_pr": float(average_precision_score(y[dec_sm.valid_mask], dec_sm.test_switch_counts)),
                                "hold_fpr": float(dec_tr.hold_pred_balanced.mean()),
                                **_eval_event_subset(dec_tr.test_switch_counts, dec_tr.test_pred_balanced, dec_tr.cal_switch_counts, tl_tr, segs_cand),
                            })
                if ctx_5s:
                    ctx_rows.append({"dataset": ds, "channel": ch, "seed": seed, "model": mname, "taxonomy": "19stream_ctx_le5", "buf_pr": np.nan, "hold_fpr": np.nan, **r_ctx5})
                if pt_5s:
                    ctx_rows.append({"dataset": ds, "channel": ch, "seed": seed, "model": mname, "taxonomy": "19stream_pt_gt5", "buf_pr": np.nan, "hold_fpr": np.nan, **r_pt5})

            done.add((ds, ch, seed, "Laminar-JEPA"))
            atomic_save_csv(pd.DataFrame(rows), out_csv)
            atomic_save_csv(pd.DataFrame(ctx_rows), out_ctx_csv)
        print(f"[Dev19 {idx:2d}/19] {ds}:{ch} ({time.perf_counter() - t0:.1f}s) done", flush=True)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_horizon_lite_sweep(device)
    run_dev19_laminar_and_contextual(device)
    df_ctx = pd.read_csv(ROOT / "reports" / "laminar_contextual_events.csv")
    sm = df_ctx.groupby(["taxonomy", "model", "dataset", "channel"], as_index=False).mean(numeric_only=True)
    summary = sm.groupby(["taxonomy", "model"]).mean(numeric_only=True).round(4).to_dict("index")
    _atomic_write_json({f"{k[0]}::{k[1]}": v for k, v in summary.items()}, ROOT / "reports" / "laminar_contextual_summary.json")
    print("Saved contextual summary -> reports/laminar_contextual_summary.json")


if __name__ == "__main__":
    main()
