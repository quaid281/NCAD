"""Paired Head-to-Head Benchmark: Laminar-JEPA-Lite vs. Flagship Laminar-JEPA.

Trains and evaluates both architectures from identical seeds and chronological splits
(`65% train / 17.5% nominal cal / 64-step isolation gap / 17.5% nominal hold` + labeled test):
  1. `Flagship_LaminarJEPA`: `SolitonReynoldsJEPAModel` (6 TCN blocks, always-on rFFT dyadic,
     split predictors, 4H) + `apply_surgical_wsp_quench` + `MultiPerspectiveCircuitBreakerPanel` (224+32).
  2. `Reynolds_224Bit_Unq`: Same 4H backbone without WSP quenching + 224-Bit panel.
  3. `LaminarJEPA_Lite`: `LaminarJEPALiteModel` (4 TCN blocks, `K>=25` gated rFFT dyadic,
     single parallel single-LN predictor, 3H, 96x96 resolvent whitening, zero WSP) +
     `PositionDriftLitePanel` (128 P+D dials, single quorum, hijack guard).
  4. `LaminarJEPA_Lite_NoGuard`: Same `LaminarJEPALiteModel` residuals + unguarded 128 P+D panel.

Reports parameter counts, training/inference wall-clock speed, and strictly causal (`trailing`)
and retrospective buffered (`smear`) accuracy/FPR metrics with paired Wilcoxon tests and 95%
stream-bootstrap confidence intervals. Uses PID-scoped atomic CSV/JSON persistence.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CSMConfig
from src.data.data_loader import DataLoader
from src.engine.builder import build_ts_jepa_model
from src.experimental.phase_embedding_synthesis import _extract_subblock_latents
from src.experimental.weight_space_programming import WSPConfig, apply_surgical_wsp_quench
from src.models.laminar_lite import LaminarJEPALiteModel
from src.scoring.circuit_breaker_panel import (
    CircuitBreakerStreamDecision,
    MultiPerspectiveCircuitBreakerPanel,
    stabilize_whitened_subblocks,
)
from src.scoring.laminar_lite_panel import PositionDriftLitePanel
from scripts.circuit_breaker_19stream.common import DEVELOPMENT_STREAMS
from scripts.experiment_wsp_soliton import (
    extract_raw4_white4_from_latents,
    select_cohort_streams,
)
from scripts.reviewer_analyses.benchmark_141_uncertainty import boot_ci
from scripts.run_multiseed_heldout_benchmark import batches, compute_metrics, event_stats, load_stream


UNTOUCHED_STRATIFIED_20: tuple[str, ...] = (
    "Daphnet:S05R01E0",
    "Daphnet:S05R02E0",
    "Daphnet:S06R01E0",
    "MSL:F-5",
    "MSL:F-8",
    "MSL:M-2",
    "SMAP:A-3",
    "SMAP:E-2",
    "SMAP:G-2",
    "SMD:machine-1-2",
    "SMD:machine-2-1",
    "SMD:machine-3-2",
    "OPPORTUNITY:S1-ADL2",
    "OPPORTUNITY:S2-ADL1",
    "OPPORTUNITY:S3-ADL3",
    "Exathlon:2_1_100000_60",
    "Exathlon:5_1_100000_64",
    "Exathlon:6_3_200000_76",
    "GECCO:water_quality",
    "PSM:default",
)


def _atomic_write_csv(df: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        df.to_csv(fh, index=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)


def _atomic_write_json(payload: dict[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _score_decision(
    dec: CircuitBreakerStreamDecision, y_test: np.ndarray
) -> dict[str, float]:
    tl = y_test[dec.valid_mask]
    mt = compute_metrics(
        tl,
        dec.test_pred_balanced,
        scores=dec.test_switch_counts,
        valid_mask=np.ones_like(tl, dtype=bool),
        use_pa=False,
    )
    er, dl = event_stats(tl, dec.test_pred_balanced)
    pr_auc = (
        float(average_precision_score(tl, dec.test_switch_counts))
        if (np.any(tl == 1) and np.any(tl == 0))
        else 0.0
    )
    hfpr = float(dec.hold_pred_balanced.mean()) if len(dec.hold_pred_balanced) > 0 else 0.0
    tfpr = float(dec.test_pred_balanced[tl == 0].mean()) if np.any(tl == 0) else 0.0
    return {
        "hold_fpr": hfpr,
        "test_fpr": tfpr,
        "f1": float(mt["f1"]),
        "pr_auc": pr_auc,
        "precision": float(mt["precision"]),
        "recall": float(mt["recall"]),
        "event_recall": float(er) if er == er else float("nan"),
        "delay": float(dl) if dl == dl else float("nan"),
        "k_gate": float(dec.k_gate_balanced),
    }


def _evaluate_panel(
    panel: MultiPerspectiveCircuitBreakerPanel | PositionDriftLitePanel,
    va_raw: np.ndarray,
    va_white: np.ndarray,
    te_raw: np.ndarray,
    te_white: np.ndarray,
    len_va: int,
    len_test: int,
    y_test: np.ndarray,
) -> dict[str, dict[str, float]]:
    n_cal_w = len(va_raw) // 2
    va_ew, te_ew = stabilize_whitened_subblocks(va_white, te_white, va_raw, te_raw, n_cal_w)
    bits = panel.calibrate_and_trip_windows(va_ew, te_ew, n_cal_w)
    return {
        mapping: _score_decision(panel.evaluate_stream_decisions(bits, len_va, len_test, mapping), y_test)
        for mapping in ("trailing", "smear")
    }


def _resolve_streams(cohort: str, streams_file: str, max_streams: int) -> list[str]:
    if cohort == "untouched_stratified20":
        streams = list(UNTOUCHED_STRATIFIED_20)
    elif cohort == "combined39":
        dev19 = select_cohort_streams("dev19", streams_file, 0)
        streams = list(dict.fromkeys(dev19 + list(UNTOUCHED_STRATIFIED_20)))
    else:
        streams = select_cohort_streams(cohort, streams_file, 0)
    return streams[:max_streams] if max_streams > 0 else streams


def _train_jepa_stream(
    model: torch.nn.Module,
    cfg: CSMConfig,
    tr_w: np.ndarray,
    context_size: int,
    epochs: int,
    device: torch.device,
) -> float:
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    _sync_cuda(device)
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for b in batches(tr_w, 16):
            b = b.to(device)
            loss, _ = model.compute_objective(b[:, :context_size], b[:, context_size:], cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.update_target_encoder(None)
    model.fit_alien_bank(tr_w, context_len=context_size)
    _sync_cuda(device)
    return time.perf_counter() - t0


def _summarize_cohort(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    rng = np.random.default_rng(42)
    stream_means = df.groupby(["cohort_split", "mapping", "model", "stream"], as_index=False).mean(numeric_only=True)

    for split_label in ["dev", "untouched", "ALL"]:
        sub = stream_means if split_label == "ALL" else stream_means[stream_means["cohort_split"] == split_label]
        if sub.empty:
            continue
        summary[split_label] = {}
        for mapping in ("trailing", "smear"):
            m_df = sub[sub["mapping"] == mapping]
            arm_stats: dict[str, Any] = {}
            for model_name, grp in m_df.groupby("model"):
                arm_stats[str(model_name)] = {
                    "n_streams": int(len(grp)),
                    "mean_f1": float(grp["f1"].mean()),
                    "mean_pr_auc": float(grp["pr_auc"].mean()),
                    "mean_event_recall": float(grp["event_recall"].dropna().mean()),
                    "mean_delay": float(grp["delay"].dropna().mean()),
                    "mean_hold_fpr": float(grp["hold_fpr"].mean()),
                    "median_hold_fpr": float(grp["hold_fpr"].median()),
                    "mean_test_fpr": float(grp["test_fpr"].mean()),
                    "mean_params": float(grp["trainable_params"].mean()),
                    "mean_train_sec": float(grp["train_sec"].mean()),
                    "mean_infer_ms_per_1k_win": float(grp["infer_ms_per_1k_win"].mean()),
                }

            piv_f1 = m_df.pivot(index="stream", columns="model", values="f1").dropna()
            piv_pr = m_df.pivot(index="stream", columns="model", values="pr_auc").dropna()
            piv_fpr = m_df.pivot(index="stream", columns="model", values="hold_fpr").dropna()
            paired: dict[str, Any] = {}
            if "Flagship_LaminarJEPA" in piv_f1.columns and "LaminarJEPA_Lite" in piv_f1.columns:
                diff_f1 = (piv_f1["LaminarJEPA_Lite"] - piv_f1["Flagship_LaminarJEPA"]).to_numpy()
                diff_pr = (piv_pr["LaminarJEPA_Lite"] - piv_pr["Flagship_LaminarJEPA"]).to_numpy()
                diff_fpr = (piv_fpr["LaminarJEPA_Lite"] - piv_fpr["Flagship_LaminarJEPA"]).to_numpy()
                nz = diff_f1[np.abs(diff_f1) > 1e-12]
                p_f1 = float(wilcoxon(nz)[1]) if len(nz) >= 5 else 1.0
                nz_fpr = diff_fpr[np.abs(diff_fpr) > 1e-12]
                p_fpr = float(wilcoxon(nz_fpr)[1]) if len(nz_fpr) >= 5 else 1.0
                ci_lo, ci_hi = boot_ci(diff_f1, None, rng)
                pr_lo, pr_hi = boot_ci(diff_pr, None, rng)
                fpr_lo, fpr_hi = boot_ci(diff_fpr, None, rng)
                paired["Lite_minus_Flagship"] = {
                    "delta_f1_mean": float(diff_f1.mean()),
                    "delta_f1_ci95": [float(ci_lo), float(ci_hi)],
                    "wilcoxon_p_f1": p_f1,
                    "wins_losses_ties_f1": [
                        int(np.sum(diff_f1 > 1e-4)),
                        int(np.sum(diff_f1 < -1e-4)),
                        int(np.sum(np.abs(diff_f1) <= 1e-4)),
                    ],
                    "delta_pr_auc_mean": float(diff_pr.mean()),
                    "delta_pr_auc_ci95": [float(pr_lo), float(pr_hi)],
                    "delta_hold_fpr_mean": float(diff_fpr.mean()),
                    "delta_hold_fpr_ci95": [float(fpr_lo), float(fpr_hi)],
                    "wilcoxon_p_hold_fpr": p_fpr,
                }
            summary[split_label][mapping] = {"arms": arm_stats, "paired": paired}
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark Laminar-JEPA-Lite vs Flagship Laminar-JEPA.")
    ap.add_argument("--cohort", default="combined39", help="dev19 | untouched_stratified20 | combined39 | untouched118 | all166")
    ap.add_argument("--streams-file", default=str(ROOT / "reports" / "streams_19_nonghl.txt"))
    ap.add_argument("--max-streams", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out-csv", default=str(ROOT / "reports" / "laminar_lite_vs_flagship.csv"))
    ap.add_argument("--out-json", default=str(ROOT / "reports" / "laminar_lite_vs_flagship_summary.json"))
    args = ap.parse_args()

    c_len, s_len, w_len = 256, 64, 320
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wsp_cfg = WSPConfig(context_len=c_len, suspect_len=s_len, candidate_k_counts=(4, 2))
    streams = _resolve_streams(args.cohort, args.streams_file, args.max_streams)

    out_csv = Path(args.out_csv)
    rows: list[dict[str, Any]] = []
    done_keys: set[tuple[str, int]] = set()
    if out_csv.exists():
        prev_df = pd.read_csv(out_csv)
        rows = prev_df.to_dict("records")
        done_keys = {(str(r["stream"]), int(r["seed"])) for r in rows}
        print(f"Resuming {out_csv}: {len(done_keys)} (stream, seed) pairs already evaluated.")

    panel_224 = MultiPerspectiveCircuitBreakerPanel(context_size=c_len, suspect_size=s_len)
    panel_lite_guard = PositionDriftLitePanel(context_size=c_len, suspect_size=s_len, use_hijack_guard=True)
    panel_lite_noguard = PositionDriftLitePanel(context_size=c_len, suspect_size=s_len, use_hijack_guard=False)

    for idx_s, s_name in enumerate(streams, start=1):
        ds, ch = s_name.split(":")
        cohort_split = "dev" if (ds.lower(), ch) in DEVELOPMENT_STREAMS else "untouched"
        try:
            train, test, y = load_stream(ds, ch)
        except Exception as exc:
            print(f"[{idx_s:3d}/{len(streams):3d}] {s_name:24s} | SKIPPED load error: {exc}")
            continue

        n_val = int(len(train) * 0.35)
        tr, va = train[:-n_val], train[-n_val:]
        if len(tr) <= w_len + 16 or len(va) <= w_len + 200 or len(test) <= w_len + 64:
            print(f"[{idx_s:3d}/{len(streams):3d}] {s_name:24s} | SKIPPED short stream")
            continue

        tr64 = tr.astype(np.float64, copy=False)
        mu = tr64.mean(0)
        sd_arr = tr64.std(0)
        sc = np.where(sd_arr < 1e-8, 1.0, np.maximum(sd_arr, 0.01))
        tr_s = ((tr64 - mu) / sc).astype(np.float32)
        va_s = ((va.astype(np.float64, copy=False) - mu) / sc).astype(np.float32)
        te_s = ((test.astype(np.float64, copy=False) - mu) / sc).astype(np.float32)
        tr_w = DataLoader.create_windows(tr_s, w_len, step=16, copy=False)
        va_w = DataLoader.create_windows(va_s, w_len, step=1, copy=False)
        te_w = DataLoader.create_windows(te_s, w_len, step=1, copy=False)
        n_cal_w = len(va_w) // 2
        n_eval_win = len(va_w) + len(te_w)

        for seed in args.seeds:
            if (s_name, seed) in done_keys:
                continue

            # --- 1. Flagship 4H SolitonReynoldsJEPA + WSP Quench + 224-Bit Panel ---
            torch.manual_seed(seed)
            np.random.seed(seed)
            cfg_flag = CSMConfig(model_type="soliton_reynolds_jepa", context_size=c_len, suspect_size=s_len, latent_dim=32)
            m_flag = build_ts_jepa_model(cfg_flag, input_dim=train.shape[1], device=device)
            params_flag = sum(p.numel() for p in m_flag.parameters() if p.requires_grad)
            train_sec_flag = _train_jepa_stream(m_flag, cfg_flag, tr_w, c_len, args.epochs, device)

            _sync_cuda(device)
            t_inf0 = time.perf_counter()
            z_ctx_va, z_tgt_va = _extract_subblock_latents(m_flag, va_w, c_len)
            z_ctx_te, z_tgt_te = _extract_subblock_latents(m_flag, te_w, c_len)
            va_raw4_unq, va_white4_unq = extract_raw4_white4_from_latents(m_flag, z_ctx_va, z_tgt_va)
            te_raw4_unq, te_white4_unq = extract_raw4_white4_from_latents(m_flag, z_ctx_te, z_tgt_te)
            _sync_cuda(device)
            inf_sec_unq = time.perf_counter() - t_inf0

            m_wsp = copy.deepcopy(m_flag)
            _sync_cuda(device)
            t_wsp0 = time.perf_counter()
            apply_surgical_wsp_quench(m_wsp, tr_w, va_w[:n_cal_w], wsp_cfg)
            va_raw4_wsp, va_white4_wsp = extract_raw4_white4_from_latents(m_wsp, z_ctx_va, z_tgt_va)
            te_raw4_wsp, te_white4_wsp = extract_raw4_white4_from_latents(m_wsp, z_ctx_te, z_tgt_te)
            _sync_cuda(device)
            wsp_extra_sec = time.perf_counter() - t_wsp0

            res_unq = _evaluate_panel(panel_224, va_raw4_unq, va_white4_unq, te_raw4_unq, te_white4_unq, len(va), len(test), y)
            res_flag = _evaluate_panel(panel_224, va_raw4_wsp, va_white4_wsp, te_raw4_wsp, te_white4_wsp, len(va), len(test), y)

            # --- 2. Laminar-JEPA-Lite 3H Unified + 128-Bit P+D Panel ---
            torch.manual_seed(seed)
            np.random.seed(seed)
            cfg_lite = CSMConfig(model_type="laminar_jepa_lite", context_size=c_len, suspect_size=s_len, latent_dim=32)
            m_lite = build_ts_jepa_model(cfg_lite, input_dim=train.shape[1], device=device)
            assert isinstance(m_lite, LaminarJEPALiteModel)
            params_lite = sum(p.numel() for p in m_lite.parameters() if p.requires_grad)
            train_sec_lite = _train_jepa_stream(m_lite, cfg_lite, tr_w, c_len, args.epochs, device)

            _sync_cuda(device)
            t_lite0 = time.perf_counter()
            va_raw3, va_white3 = m_lite.extract_raw_and_whitened_residuals(va_w, context_len=c_len)
            te_raw3, te_white3 = m_lite.extract_raw_and_whitened_residuals(te_w, context_len=c_len)
            _sync_cuda(device)
            inf_sec_lite = time.perf_counter() - t_lite0

            res_lite = _evaluate_panel(panel_lite_guard, va_raw3, va_white3, te_raw3, te_white3, len(va), len(test), y)
            res_lite_ng = _evaluate_panel(panel_lite_noguard, va_raw3, va_white3, te_raw3, te_white3, len(va), len(test), y)

            arms = [
                ("Flagship_LaminarJEPA", res_flag, params_flag, train_sec_flag + wsp_extra_sec, inf_sec_unq),
                ("Reynolds_224Bit_Unq", res_unq, params_flag, train_sec_flag, inf_sec_unq),
                ("LaminarJEPA_Lite", res_lite, params_lite, train_sec_lite, inf_sec_lite),
                ("LaminarJEPA_Lite_NoGuard", res_lite_ng, params_lite, train_sec_lite, inf_sec_lite),
            ]
            for arm_name, arm_res, n_p, tr_sec, in_sec in arms:
                for mapping in ("trailing", "smear"):
                    rows.append(
                        {
                            "stream": s_name,
                            "dataset": ds.lower(),
                            "cohort_split": cohort_split,
                            "k_channels": int(train.shape[1]),
                            "seed": int(seed),
                            "model": arm_name,
                            "mapping": mapping,
                            "trainable_params": int(n_p),
                            "train_sec": float(tr_sec),
                            "infer_ms_per_1k_win": float(1000.0 * in_sec / max(n_eval_win, 1)),
                            **arm_res[mapping],
                        }
                    )

            _atomic_write_csv(pd.DataFrame(rows), out_csv)
            cf_flag = res_flag["trailing"]["f1"]
            cf_lite = res_lite["trailing"]["f1"]
            fp_flag = res_flag["trailing"]["hold_fpr"]
            fp_lite = res_lite["trailing"]["hold_fpr"]
            spd = (train_sec_flag + wsp_extra_sec + inf_sec_unq) / max(train_sec_lite + inf_sec_lite, 1e-6)
            print(
                f"[{idx_s:2d}/{len(streams):2d}] {s_name:24s} ({cohort_split:9s} K={train.shape[1]:2d}) | "
                f"CausF1: Flag={cf_flag:.4f} Lite={cf_lite:.4f} | "
                f"hFPR: Flag={fp_flag:.4f} Lite={fp_lite:.4f} | "
                f"Params: {params_lite // 1000}k/{params_flag // 1000}k | {spd:.2f}x faster",
                flush=True,
            )

    final_df = pd.DataFrame(rows)
    summary = _summarize_cohort(final_df)
    _atomic_write_json(summary, Path(args.out_json))
    print("\n=== PAIRED COHORT SUMMARY SAVED TO", args.out_json, "===")
    for split_key, split_val in summary.items():
        for mapping in ("trailing", "smear"):
            if mapping not in split_val:
                continue
            print(f"\n[{split_key.upper()} | {mapping.upper()}]")
            for m_name, st in split_val[mapping]["arms"].items():
                print(
                    f"  {m_name:24s} | F1={st['mean_f1']:.4f} PR={st['mean_pr_auc']:.4f} "
                    f"ER={st['mean_event_recall']:.3f} Del={st['mean_delay']:.1f} "
                    f"hFPR={st['mean_hold_fpr']:.4f} (med {st['median_hold_fpr']:.4f}) | "
                    f"Params={st['mean_params']/1e3:.1f}k Train={st['mean_train_sec']:.2f}s "
                    f"Inf={st['mean_infer_ms_per_1k_win']:.2f}ms/1k"
                )
            if "Lite_minus_Flagship" in split_val[mapping]["paired"]:
                pr = split_val[mapping]["paired"]["Lite_minus_Flagship"]
                print(
                    f"  -> Lite - Flagship: dF1={pr['delta_f1_mean']:+.4f} "
                    f"CI95=[{pr['delta_f1_ci95'][0]:+.4f}, {pr['delta_f1_ci95'][1]:+.4f}] "
                    f"p={pr['wilcoxon_p_f1']:.4g} W/L/T={pr['wins_losses_ties_f1']} "
                    f"dhFPR={pr['delta_hold_fpr_mean']:+.4f}"
                )


if __name__ == "__main__":
    main()
