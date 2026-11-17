#!/usr/bin/env python3
"""Experiment 1A: Healthy Regime-Shift Stress Test on 19 Multivariate Development Streams.

Deliberately introduces fault-free healthy operating shifts halfway through clean nominal test
segments across 19 streams x 3 seeds `{42, 123, 456}` (disjoint 4-block protocol):
  0. Clean_Nominal         : Unperturbed fault-free segment
  1. Mean_Step_1sigma      : +1.0*sigma step on 25% of active channels
  2. Mean_Step_2sigma      : +2.0*sigma step on 25% of active channels
  3. Var_Scale_1.5x        : 1.5x amplitude scaling around channel mean on 25% of channels
  4. Var_Scale_2.0x        : 2.0x amplitude scaling around channel mean on 25% of channels
  5. Corr_Rotation         : 45-deg orthogonal Givens mixing of top-4 correlated channels (preserving mean & variance)
  6. Gradual_Drift_1.5sig  : Linear 1.5*sigma healthy baseline ramp over 2,000 steps
  7. Abrupt_Duty_Cycle     : Alternating +1.0*sigma nominal duty-cycle switch every 600 steps

Records overall post-shift causal FPR (`Lite` vs. `Multi-Horizon 1D`), transient boundary FPR
(`[t_shift, t_shift + C + S)`), steady-state post-adaptation FPR (`[t_shift + C + S, end)`),
regime settling steps, quorum `D_KS` shift, and Step-(5) live `D_KS`-triggered recalibration FPR.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reviewer_analyses.ramp_severity_sweep import longest_nominal_run
from scripts.run_laminar_paired_benchmarks import (
    C,
    CSMConfig,
    S,
    W,
    _build_schedule,
    _score_1d_windows,
    _split_nominal_dev_4block,
    _standardize_arrays,
    _train_fixed_epochs,
    load_stream,
    thresholds,
)
from scripts.tnnls_revision.common_eval import atomic_write_csv, atomic_write_json
from src.data.data_loader import DataLoader
from src.engine.builder import build_encoder
from src.models.jepa.alien_jepas import SpeculativeJEPAModel
from src.models.laminar_lite import LaminarJEPALiteModel, LaminarLiteEncoder
from src.scoring.circuit_breaker_panel import aggregate_vector_stream, stabilize_whitened_subblocks
from src.scoring.laminar_lite_panel import PositionDriftLitePanel


def _apply_healthy_perturbation(
    seg_std: np.ndarray, mode: str, t_shift: int, rng: np.random.Generator
) -> np.ndarray:
    """Apply a controlled healthy regime shift starting at `t_shift` on standardized `[T, K]` data."""
    out = seg_std.copy()
    t_len, k_ch = out.shape
    if mode == "Clean_Nominal":
        return out

    act = np.flatnonzero(np.std(out[:t_shift], axis=0) > 1e-4)
    if len(act) == 0:
        act = np.arange(k_ch)
    n_sub = max(1, int(np.ceil(0.25 * len(act))))
    ch_sub = rng.choice(act, size=n_sub, replace=False)

    if mode == "Mean_Step_1sigma":
        out[t_shift:, ch_sub] += 1.0
    elif mode == "Mean_Step_2sigma":
        out[t_shift:, ch_sub] += 2.0
    elif mode == "Var_Scale_1.5x":
        mu = out[:t_shift, ch_sub].mean(axis=0, keepdims=True)
        out[t_shift:, ch_sub] = mu + 1.5 * (out[t_shift:, ch_sub] - mu)
    elif mode == "Var_Scale_2.0x":
        mu = out[:t_shift, ch_sub].mean(axis=0, keepdims=True)
        out[t_shift:, ch_sub] = mu + 2.0 * (out[t_shift:, ch_sub] - mu)
    elif mode == "Corr_Rotation":
        if len(act) >= 2:
            # Orthogonal Givens rotation preserves marginal zero mean & unit variance when uncorrelated,
            # while altering pairwise cross-channel correlation structure.
            pairs = act[: min(4, 2 * (len(act) // 2))]
            theta = np.deg2rad(45.0)
            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
            for p in range(0, len(pairs), 2):
                c0, c1 = pairs[p], pairs[p + 1]
                # Flip sign of cross-correlation via orthogonal reflection + rotation
                pair_post = out[t_shift:, [c0, c1]].copy()
                pair_post[:, 1] = -pair_post[:, 1]
                out[t_shift:, [c0, c1]] = pair_post @ rot.T
        else:
            out[t_shift:, ch_sub] *= -1.0
    elif mode == "Gradual_Drift_1.5sig":
        span = min(2000, t_len - t_shift)
        ramp = np.linspace(0.0, 1.5, span, dtype=np.float32)[:, None]
        out[t_shift : t_shift + span, ch_sub] += ramp
        if t_shift + span < t_len:
            out[t_shift + span :, ch_sub] += 1.5
    elif mode == "Abrupt_Duty_Cycle":
        period = 600
        for s0 in range(t_shift, t_len, 2 * period):
            out[s0 : min(s0 + period, t_len), ch_sub] += 1.0
    else:
        raise ValueError(f"Unknown healthy perturbation mode: {mode}")
    return out


def _settling_steps(pred_post: np.ndarray, clean_fpr: float, win: int = 64) -> float:
    """Steps after `t_shift` until rolling `win`-step alarm rate drops to `<= max(0.02, 2*clean_fpr)`."""
    target = max(0.02, 2.0 * clean_fpr)
    if len(pred_post) < win:
        return 0.0 if float(pred_post.mean()) <= target else float(len(pred_post))
    csum = np.concatenate([[0.0], np.cumsum(pred_post, dtype=np.float64)])
    roll = (csum[win:] - csum[:-win]) / float(win)
    ok = np.flatnonzero(roll <= target)
    return float(ok[0]) if len(ok) > 0 else float(len(pred_post))


def main() -> None:
    out_csv = ROOT / "reports" / "regime_shift_stress_dev19.csv"
    out_json = ROOT / "reports" / "regime_shift_stress_summary.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = _build_schedule("dev19_4block")
    panel = PositionDriftLitePanel(context_size=C, suspect_size=S)

    modes = (
        "Clean_Nominal",
        "Mean_Step_1sigma",
        "Mean_Step_2sigma",
        "Var_Scale_1.5x",
        "Var_Scale_2.0x",
        "Corr_Rotation",
        "Gradual_Drift_1.5sig",
        "Abrupt_Duty_Cycle",
    )

    rows: list[dict[str, Any]] = pd.read_csv(out_csv).to_dict("records") if out_csv.exists() else []
    done = {(str(r["stream"]), int(r["seed"])) for r in rows}

    for idx, (ds, ch, seeds) in enumerate(schedule, start=1):
        s_key = f"{ds}:{ch}"
        need = [sd for sd in seeds if (s_key, sd) not in done]
        if not need:
            continue

        train, test, y = load_stream(ds, ch)
        tr_w, q_w, va_w, _, len_va = _split_nominal_dev_4block(train, test)
        k_ch = int(train.shape[1])
        n_cal_w = len(va_w) // 2

        # Extract longest nominal test segment (fallback to nominal validation suffix if too short)
        s0, n_run = longest_nominal_run(y)
        n_fit = len(train) - int(len(train) * 0.35) - max(256, int(len(train) * 0.15))
        _, seg_all = _standardize_arrays(train[:n_fit], test[s0 : s0 + min(n_run, 6000)] if n_run >= 1200 else train[-min(len(train), 3000) :])
        if len(seg_all) < 2 * W + 200:
            continue
        t_shift = max(W + 100, len(seg_all) // 3)

        for sd in need:
            torch.manual_seed(sd)
            np.random.seed(sd)
            cfg_l = CSMConfig(model_type="laminar_jepa_lite", context_size=C, suspect_size=S, latent_dim=32)
            enc = LaminarLiteEncoder(k_ch, latent_dim=32, filters=48, tcn_layers=4, proj_dim=64)
            m_lite = LaminarJEPALiteModel(enc, latent_dim=32, predictor_hidden_dim=64, num_horizons=3).to(device)
            _train_fixed_epochs(m_lite, tr_w, cfg_l, epochs=15, device=device)
            m_lite.fit_resolvent_whitening(tr_w, context_len=C, batch_size=512)
            va_raw, va_white = m_lite.extract_raw_and_whitened_residuals(va_w, context_len=C, batch_size=512)
            q_raw, q_white = m_lite.extract_raw_and_whitened_residuals(q_w, context_len=C, batch_size=512)

            torch.manual_seed(sd)
            np.random.seed(sd)
            cfg_1 = CSMConfig(model_type="ts_jepa", context_size=C, suspect_size=S, latent_dim=32)
            m_1d = SpeculativeJEPAModel(
                context_encoder=build_encoder(cfg_1, k_ch, device),
                latent_dim=32,
                predictor_hidden_dim=64,
                ema_decay=0.995,
                dropout=cfg_1.dropout,
            ).to(device)
            _train_fixed_epochs(m_1d, tr_w, cfg_1, epochs=15, device=device)
            m_1d.fit_alien_bank(tr_w, context_len=C, batch_size=512)
            va_s1 = _score_1d_windows(m_1d, va_w, device, use_bank=True)
            vp1, vm1 = aggregate_vector_stream(va_s1[:, None], len_va, C, S, "trailing")
            cal_s1 = vp1[vm1, 0][: len(vp1[vm1, 0]) // 2]
            th_1d = float(thresholds(cal_s1)[0].get("SPOT", np.quantile(cal_s1, 0.995)))

            clean_fpr_lite = 0.01
            for m_idx, mode in enumerate(modes):
                rng = np.random.default_rng([sd, idx, m_idx])
                seg_p = _apply_healthy_perturbation(seg_all, mode, t_shift, rng)
                seg_w = DataLoader.create_windows(seg_p, W, step=1, copy=False)

                sr, sw = m_lite.extract_raw_and_whitened_residuals(seg_w, context_len=C, batch_size=512)
                va_ew, se_ew = stabilize_whitened_subblocks(va_white, sw, va_raw, sr, n_cal_w, q_white, q_raw)
                dec = panel.evaluate_stream_decisions(
                    panel.calibrate_and_trip_windows(va_ew, se_ew, n_cal_w), len_va, len(seg_p), "trailing"
                )
                valid_idx = np.flatnonzero(dec.valid_mask)
                post_mask = valid_idx >= t_shift
                trans_mask = (valid_idx >= t_shift) & (valid_idx < t_shift + W)
                steady_mask = valid_idx >= (t_shift + W)

                pred_l = dec.test_pred_balanced
                cnt_l = dec.test_switch_counts
                if mode == "Clean_Nominal":
                    clean_fpr_lite = float(pred_l[post_mask].mean()) if np.any(post_mask) else 0.0

                ks_shift = float(stats.ks_2samp(dec.cal_switch_counts, cnt_l[post_mask]).statistic) if np.any(post_mask) else 0.0

                # Step-(5) live D_KS >= 0.50 recalibration on first 400 post-shift windows
                recal_fpr = float(pred_l[steady_mask].mean()) if np.any(steady_mask) else 0.0
                post_w_start = max(0, t_shift - C)
                if ks_shift >= 0.50 and len(se_ew) > post_w_start + 500:
                    recal_n = 400
                    bits_rc = panel.calibrate_and_trip_windows(
                        se_ew[post_w_start : post_w_start + 2 * recal_n],
                        se_ew[post_w_start + recal_n + S :],
                        recal_n,
                    )
                    rc_cnt = bits_rc["bits_pd_te"].sum(axis=1)
                    rc_cal = bits_rc["bits_pd_va"][:recal_n].sum(axis=1)
                    k_rc = max(float(np.percentile(rc_cal, 99.5)) * 1.12, 3.5)
                    recal_fpr = float((rc_cnt > k_rc).mean()) if len(rc_cnt) > 0 else 0.0

                # 1D comparator
                ss1 = _score_1d_windows(m_1d, seg_w, device, use_bank=True)
                sp1, sm1 = aggregate_vector_stream(ss1[:, None], len(seg_p), C, S, "trailing")
                pred_1d = (sp1[sm1, 0] > th_1d).astype(int)

                rows.append(
                    {
                        "stream": s_key,
                        "k_channels": k_ch,
                        "seed": sd,
                        "mode": mode,
                        "lite_post_fpr": float(pred_l[post_mask].mean()) if np.any(post_mask) else 0.0,
                        "lite_transient_fpr": float(pred_l[trans_mask].mean()) if np.any(trans_mask) else 0.0,
                        "lite_steady_fpr": float(pred_l[steady_mask].mean()) if np.any(steady_mask) else 0.0,
                        "lite_settling_steps": _settling_steps(pred_l[post_mask], clean_fpr_lite),
                        "lite_ks_shift": ks_shift,
                        "lite_step5_recal_fpr": recal_fpr,
                        "mh1d_post_fpr": float(pred_1d[post_mask].mean()) if np.any(post_mask) else 0.0,
                        "mh1d_steady_fpr": float(pred_1d[steady_mask].mean()) if np.any(steady_mask) else 0.0,
                    }
                )
            del m_lite, m_1d
            atomic_write_csv(pd.DataFrame(rows), out_csv)
        print(f"[Exp1A {idx:2d}/{len(schedule)}] {s_key} completed", flush=True)

    df = pd.DataFrame(rows)
    s_avg = df.groupby(["mode", "stream"], as_index=False).mean(numeric_only=True)
    summary_rows = []
    for mode, g in s_avg.groupby("mode"):
        summary_rows.append(
            {
                "mode": mode,
                "lite_post_fpr_mean": float(g["lite_post_fpr"].mean()),
                "lite_post_fpr_med": float(g["lite_post_fpr"].median()),
                "lite_transient_fpr_mean": float(g["lite_transient_fpr"].mean()),
                "lite_steady_fpr_mean": float(g["lite_steady_fpr"].mean()),
                "lite_settling_med": float(g["lite_settling_steps"].median()),
                "lite_ks_mean": float(g["lite_ks_shift"].mean()),
                "lite_step5_recal_fpr_mean": float(g["lite_step5_recal_fpr"].mean()),
                "mh1d_post_fpr_mean": float(g["mh1d_post_fpr"].mean()),
                "mh1d_steady_fpr_mean": float(g["mh1d_steady_fpr"].mean()),
            }
        )
    atomic_write_json({"stress_modes": summary_rows}, out_json)
    print("Exp 1A summary saved to", out_json, flush=True)


if __name__ == "__main__":
    main()
