#!/usr/bin/env python3
"""Experiment 2A & 2B: Latent Dimensionality (D in {8, 16, 32, 64}) & Calibration/Damping Sensitivity.

Evaluates on the 19 non-GHL development streams x 3 seeds `{42, 123, 456}` under the strictly
disjoint 4-block protocol (`X_fit -> X_q -> X_cal -> X_hold`):
  - Part 2A: Retrains `Laminar-JEPA-Lite` and `Multi-Horizon JEPA (1D)` across `D in {8, 16, 32, 64}`
    (predictor hidden width `2*D`, panel size `(3+1)*D in {32, 64, 128, 256}`), recording causal
    and buffered Point-F1, PR-AUC, Event Recall, Detection Delay, Held-Out/Test FPR, RankMe,
    trainable parameters, and inference latency (`ms/window`).
  - Part 2B: On the `D = 32` checkpoints, sweeps whitening damping `reg in {1e-4, 1e-3, 1e-2, 1e-1}`,
    per-dial tripwire quantile `q_dial in {0.975, 0.980, 0.985, 0.990}`, and quorum safety factor
    `safety in {1.00, 1.06, 1.12, 1.20}`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_laminar_paired_benchmarks import (
    C,
    CSMConfig,
    S,
    _build_schedule,
    _score_1d_windows,
    _split_nominal_dev_4block,
    _train_fixed_epochs,
    compute_metrics,
    event_stats,
    load_stream,
    thresholds,
)
from scripts.tnnls_revision.common_eval import atomic_write_csv, atomic_write_json, compute_rankme
from src.engine.builder import build_encoder
from src.models._jepa_utils import fit_covariance_batched
from src.models.jepa.alien_jepas import SpeculativeJEPAModel
from src.models.laminar_lite import LaminarJEPALiteModel, LaminarLiteEncoder
from src.scoring.circuit_breaker_panel import aggregate_vector_stream, stabilize_whitened_subblocks
from src.scoring.laminar_lite_panel import PositionDriftLitePanel, build_position_drift_dials

LATENT_DIMS = (8, 16, 32, 64)
DAMPING_VALS = (1e-4, 1e-3, 1e-2, 1e-1)
DIAL_QUANTILES = (0.975, 0.980, 0.985, 0.990)
SAFETY_FACTORS = (1.00, 1.06, 1.12, 1.20)


def _eval_custom_lite_panel(
    va_ew: np.ndarray,
    te_ew: np.ndarray,
    n_cal_w: int,
    len_va: int,
    len_te: int,
    y_test: np.ndarray,
    dial_q: float = 0.985,
    safety: float = 1.12,
    quorum_q: float = 0.995,
) -> dict[str, dict[str, float]]:
    """Score `PositionDriftLitePanel` with configurable `dial_q`, `safety`, and `quorum_q`."""
    va_pd = build_position_drift_dials(va_ew)
    te_pd = build_position_drift_dials(te_ew)
    tau = np.percentile(va_pd[:n_cal_w], 100.0 * dial_q, axis=0, keepdims=True) + 1e-4
    bits = {
        "bits_pd_va": (va_pd > tau).astype(np.float32),
        "bits_pd_te": (te_pd > tau).astype(np.float32),
        "num_horizons": int(va_ew.shape[1]),
    }
    panel = PositionDriftLitePanel(context_size=C, suspect_size=S)
    n_dials = float(va_pd.shape[1])
    floor_bal = max(2.0, 3.5 * (n_dials / 128.0))
    out: dict[str, dict[str, float]] = {}

    for mapping in ("trailing", "smear"):
        dec = panel.evaluate_stream_decisions(bits, len_va, len_te, mapping)
        v_cnt = np.concatenate([dec.cal_switch_counts, np.zeros(S, dtype=np.float32), dec.hold_switch_counts])
        cal_cnt = dec.cal_switch_counts
        hold_cnt = dec.hold_switch_counts
        test_cnt = dec.test_switch_counts
        k_th = max(float(np.percentile(cal_cnt, 100.0 * quorum_q)) * safety, floor_bal)
        h_pred = (hold_cnt > k_th).astype(int)
        t_pred = (test_cnt > k_th).astype(int)
        tl = y_test[dec.valid_mask]
        has_both = bool(np.any(tl == 1) and np.any(tl == 0))
        mt = compute_metrics(tl, t_pred, scores=test_cnt, valid_mask=np.ones_like(tl, dtype=bool), use_pa=False)
        er, dl = event_stats(tl, t_pred)
        pr = float(average_precision_score(tl, test_cnt)) if has_both else 0.0
        _ = v_cnt
        out[mapping] = {
            "f1": float(mt.get("f1", 0.0)),
            "pr_auc": pr,
            "event_recall": float(er) if er == er else float("nan"),
            "delay": float(dl) if dl == dl else float("nan"),
            "hold_fpr": float(h_pred.mean()) if len(h_pred) > 0 else 0.0,
            "test_fpr": float(t_pred[tl == 0].mean()) if np.any(tl == 0) else 0.0,
            "k_quorum": float(k_th),
        }
    return out


def _eval_1d_spot(
    va_win_scores: np.ndarray, te_win_scores: np.ndarray, len_va: int, len_te: int, y_test: np.ndarray
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for mapping in ("trailing", "smear"):
        vp, vm = aggregate_vector_stream(va_win_scores[:, None], len_va, C, S, mapping)
        tp, tm = aggregate_vector_stream(te_win_scores[:, None], len_te, C, S, mapping)
        vs, ts = vp[vm, 0], tp[tm, 0]
        half = len(vs) // 2
        cal_s, hold_s = vs[:half], vs[half + S :]
        th = float(thresholds(cal_s)[0].get("SPOT", np.quantile(cal_s, 0.995)))
        h_pred = (hold_s > th).astype(int)
        t_pred = (ts > th).astype(int)
        tl = y_test[tm]
        has_both = bool(np.any(tl == 1) and np.any(tl == 0))
        mt = compute_metrics(tl, t_pred, scores=ts, valid_mask=np.ones_like(tl, dtype=bool), use_pa=False)
        er, dl = event_stats(tl, t_pred)
        pr = float(average_precision_score(tl, ts)) if has_both else 0.0
        out[mapping] = {
            "f1": float(mt.get("f1", 0.0)),
            "pr_auc": pr,
            "event_recall": float(er) if er == er else float("nan"),
            "delay": float(dl) if dl == dl else float("nan"),
            "hold_fpr": float(h_pred.mean()) if len(h_pred) > 0 else 0.0,
            "test_fpr": float(t_pred[tl == 0].mean()) if np.any(tl == 0) else 0.0,
            "k_quorum": float(th),
        }
    return out


@torch.no_grad()
def _extract_target_rankme(model: LaminarJEPALiteModel, windows: np.ndarray, batch_size: int = 512) -> float:
    model.eval()
    dev = next(model.parameters()).device
    zs: list[np.ndarray] = []
    with torch.inference_mode():
        for i in range(0, min(len(windows), 2048), batch_size):
            tgt = torch.from_numpy(np.ascontiguousarray(windows[i : i + batch_size, C:])).float().to(dev)
            z_tgt = model.target_encoder(tgt).detach().cpu().numpy()
            zs.append(z_tgt)
    return compute_rankme(np.concatenate(zs, axis=0))


def main() -> None:
    dim_csv = ROOT / "reports" / "latent_dim_sweep_dev19.csv"
    sens_csv = ROOT / "reports" / "hyperparam_sensitivity_dev19.csv"
    sum_json = ROOT / "reports" / "latent_and_sensitivity_summary.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = _build_schedule("dev19_4block")

    dim_rows: list[dict[str, Any]] = pd.read_csv(dim_csv).to_dict("records") if dim_csv.exists() else []
    sens_rows: list[dict[str, Any]] = pd.read_csv(sens_csv).to_dict("records") if sens_csv.exists() else []
    done_dim = {(str(r["stream"]), int(r["seed"]), int(r["latent_dim"]), str(r["model"])) for r in dim_rows}
    done_sens = {(str(r["stream"]), int(r["seed"])) for r in sens_rows}

    for idx, (ds, ch, seeds) in enumerate(schedule, start=1):
        s_key = f"{ds}:{ch}"
        need_any = any(
            (s_key, sd, d, m) not in done_dim
            for sd in seeds
            for d in LATENT_DIMS
            for m in ("Laminar-JEPA-Lite", "Multi-Horizon-1D")
        ) or any((s_key, sd) not in done_sens for sd in seeds)
        if not need_any:
            continue

        train, test, y = load_stream(ds, ch)
        tr_w, q_w, va_w, te_w, len_va = _split_nominal_dev_4block(train, test)
        k_ch = int(train.shape[1])
        n_cal_w = len(va_w) // 2

        for sd in seeds:
            for d_lat in LATENT_DIMS:
                # 1. Laminar-JEPA-Lite at dimension d_lat
                if (s_key, sd, d_lat, "Laminar-JEPA-Lite") not in done_dim or (
                    d_lat == 32 and (s_key, sd) not in done_sens
                ):
                    torch.manual_seed(sd)
                    np.random.seed(sd)
                    cfg = CSMConfig(model_type="laminar_jepa_lite", context_size=C, suspect_size=S, latent_dim=d_lat)
                    enc = LaminarLiteEncoder(k_ch, latent_dim=d_lat, filters=48, tcn_layers=4, proj_dim=max(32, 2 * d_lat))
                    m_lite = LaminarJEPALiteModel(enc, latent_dim=d_lat, predictor_hidden_dim=2 * d_lat, num_horizons=3).to(device)
                    n_params = sum(p.numel() for p in m_lite.parameters() if p.requires_grad)
                    _train_fixed_epochs(m_lite, tr_w, cfg, epochs=15, device=device)
                    m_lite.fit_resolvent_whitening(tr_w, context_len=C, batch_size=512)
                    rank_me = _extract_target_rankme(m_lite, va_w)

                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    t0 = time.perf_counter()
                    va_raw, va_white = m_lite.extract_raw_and_whitened_residuals(va_w, context_len=C, batch_size=512)
                    te_raw, te_white = m_lite.extract_raw_and_whitened_residuals(te_w, context_len=C, batch_size=512)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    infer_ms = 1000.0 * (time.perf_counter() - t0) / max(len(va_w) + len(te_w), 1)
                    q_raw, q_white = m_lite.extract_raw_and_whitened_residuals(q_w, context_len=C, batch_size=512)

                    if (s_key, sd, d_lat, "Laminar-JEPA-Lite") not in done_dim:
                        va_ew, te_ew = stabilize_whitened_subblocks(va_white, te_white, va_raw, te_raw, n_cal_w, q_white, q_raw)
                        res = _eval_custom_lite_panel(va_ew, te_ew, n_cal_w, len_va, len(test), y)
                        for mp, mv in res.items():
                            dim_rows.append(
                                {
                                    "stream": s_key,
                                    "k_channels": k_ch,
                                    "seed": sd,
                                    "latent_dim": d_lat,
                                    "n_dials": 4 * d_lat,
                                    "model": "Laminar-JEPA-Lite",
                                    "mapping": mp,
                                    "params": n_params,
                                    "infer_ms": infer_ms,
                                    "rankme": rank_me,
                                    **mv,
                                }
                            )

                    # Part 2B sensitivity sweeps at D = 32
                    if d_lat == 32 and (s_key, sd) not in done_sens:
                        va_ew0, te_ew0 = stabilize_whitened_subblocks(va_white, te_white, va_raw, te_raw, n_cal_w, q_white, q_raw)
                        for dq in DIAL_QUANTILES:
                            r_dq = _eval_custom_lite_panel(va_ew0, te_ew0, n_cal_w, len_va, len(test), y, dial_q=dq, safety=1.12)
                            sens_rows.append({"stream": s_key, "seed": sd, "axis": "dial_q", "value": dq, **r_dq["trailing"]})
                        for sf in SAFETY_FACTORS:
                            r_sf = _eval_custom_lite_panel(va_ew0, te_ew0, n_cal_w, len_va, len(test), y, dial_q=0.985, safety=sf)
                            sens_rows.append({"stream": s_key, "seed": sd, "axis": "safety", "value": sf, **r_sf["trailing"]})
                        for reg in DAMPING_VALS:
                            fit_covariance_batched(
                                m_lite,
                                tr_w[:, :C],
                                tr_w[:, C:],
                                residual_fn=m_lite._subblock_residuals_flat,
                                dim=3 * d_lat,
                                batch_size=512,
                                reg=reg,
                                precision_buffer=m_lite.precision_matrix,
                                residual_mean_buffer=m_lite.residual_mean,
                                fitted_buffer=m_lite.precision_fitted,
                            )
                            vr_d, vw_d = m_lite.extract_raw_and_whitened_residuals(va_w, context_len=C, batch_size=512)
                            tr_d, tw_d = m_lite.extract_raw_and_whitened_residuals(te_w, context_len=C, batch_size=512)
                            qr_d, qw_d = m_lite.extract_raw_and_whitened_residuals(q_w, context_len=C, batch_size=512)
                            vew_d, tew_d = stabilize_whitened_subblocks(vw_d, tw_d, vr_d, tr_d, n_cal_w, qw_d, qr_d)
                            r_rg = _eval_custom_lite_panel(vew_d, tew_d, n_cal_w, len_va, len(test), y)
                            sens_rows.append({"stream": s_key, "seed": sd, "axis": "damping", "value": reg, **r_rg["trailing"]})
                    del m_lite

                # 2. Multi-Horizon JEPA (1D) comparator at dimension d_lat
                if (s_key, sd, d_lat, "Multi-Horizon-1D") not in done_dim:
                    torch.manual_seed(sd)
                    np.random.seed(sd)
                    cfg1 = CSMConfig(model_type="ts_jepa", context_size=C, suspect_size=S, latent_dim=d_lat)
                    m_1d = SpeculativeJEPAModel(
                        context_encoder=build_encoder(cfg1, k_ch, device),
                        latent_dim=d_lat,
                        predictor_hidden_dim=max(32, 2 * d_lat),
                        ema_decay=0.995,
                        dropout=cfg1.dropout,
                    ).to(device)
                    n_p1 = sum(p.numel() for p in m_1d.parameters() if p.requires_grad)
                    _train_fixed_epochs(m_1d, tr_w, cfg1, epochs=15, device=device)
                    m_1d.fit_alien_bank(tr_w, context_len=C, batch_size=512)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    t1 = time.perf_counter()
                    va_s1 = _score_1d_windows(m_1d, va_w, device, use_bank=True)
                    te_s1 = _score_1d_windows(m_1d, te_w, device, use_bank=True)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    ms1 = 1000.0 * (time.perf_counter() - t1) / max(len(va_w) + len(te_w), 1)
                    res1 = _eval_1d_spot(va_s1, te_s1, len_va, len(test), y)
                    for mp, mv in res1.items():
                        dim_rows.append(
                            {
                                "stream": s_key,
                                "k_channels": k_ch,
                                "seed": sd,
                                "latent_dim": d_lat,
                                "n_dials": 1,
                                "model": "Multi-Horizon-1D",
                                "mapping": mp,
                                "params": n_p1,
                                "infer_ms": ms1,
                                "rankme": float("nan"),
                                **mv,
                            }
                        )
                    del m_1d

            atomic_write_csv(pd.DataFrame(dim_rows), dim_csv)
            atomic_write_csv(pd.DataFrame(sens_rows), sens_csv)
        print(f"[Exp2 {idx:2d}/{len(schedule)}] {s_key} completed", flush=True)

    df_dim = pd.DataFrame(dim_rows)
    df_sens = pd.DataFrame(sens_rows)
    s_dim = df_dim.groupby(["model", "mapping", "latent_dim", "stream"], as_index=False).mean(numeric_only=True)
    dim_sum = (
        s_dim.groupby(["model", "mapping", "latent_dim"], as_index=False)[
            ["f1", "pr_auc", "event_recall", "delay", "hold_fpr", "test_fpr", "params", "infer_ms", "rankme"]
        ]
        .mean()
        .round(4)
        .to_dict("records")
    )
    s_sens = df_sens.groupby(["axis", "value", "stream"], as_index=False).mean(numeric_only=True)
    sens_sum = (
        s_sens.groupby(["axis", "value"], as_index=False)[["f1", "event_recall", "delay", "hold_fpr", "test_fpr", "k_quorum"]]
        .mean()
        .round(4)
        .to_dict("records")
    )
    summary = {"latent_dim_sweep": dim_sum, "hyperparam_sensitivity": sens_sum}
    atomic_write_json(summary, sum_json)
    print("Exp 2 summary saved to", sum_json, flush=True)


if __name__ == "__main__":
    main()
