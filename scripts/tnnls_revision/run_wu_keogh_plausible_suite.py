#!/usr/bin/env python3
"""Wu & Keogh (2021) Plausible One-Liner-Resistant & Noise-Invariance Suite.

Implements multivariate analogues of Wu & Keogh (`arXiv:2009.13807`) Section 3.2--3.3 on clean
nominal segments of the 19 multivariate development streams x 3 seeds `{42, 123, 456}`:
  1. `Cycle_Swap`    : Replaces a 200-step segment at a random interior location with a reversed-time
                       copy of an earlier nominal segment from the same stream (preserving exact
                       per-channel min/max/mean/std so amplitude one-liners cannot separate it).
  2. `Phase_Desync`  : Shifts half of the active channels by +16 steps over a 200-step interior span
                       (preserving every channel's marginal univariate distribution identically).
  3. `Noise_0.15` & `Noise_0.30` : Injects Gaussian observation noise (`sigma = 0.15, 0.30`) on top
                       of `Phase_Desync` to test noise invariance (Wu & Keogh Fig. 13).

Compares `Laminar-JEPA-Lite`, `Multi-Horizon JEPA (1D)` (SPOT), and a Wu & Keogh amplitude
one-liner (`max_c |x_c(t)|` calibrated at 99.5th nominal percentile) under live causal reporting.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
    compute_metrics,
    event_stats,
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


def inject_wu_keogh_anomaly(
    seg_std: np.ndarray, mode: str, t0: int, dur: int = 200, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Inject a marginal-preserving multivariate anomaly at interior `t0` (zero run-to-failure bias)."""
    out = seg_std.copy()
    y_ep = np.zeros(len(out), dtype=int)
    t1 = min(len(out) - 64, t0 + dur)
    y_ep[t0:t1] = 1
    span = t1 - t0
    act = np.flatnonzero(np.std(out[:t0], axis=0) > 1e-4)
    if len(act) == 0:
        act = np.arange(out.shape[1])

    if mode == "Cycle_Swap":
        # Time-reversed segment of identical length from an earlier nominal region:
        # preserves exact per-channel value multiset (min, max, mean, quantiles unchanged).
        src_start = max(0, t0 - span - 64)
        ch_sub = act[: max(1, len(act) // 2)]
        out[t0:t1, ch_sub] = out[src_start : src_start + span, ch_sub][::-1]
    elif mode.startswith(("Phase_Desync", "Noise_")):
        # Shift half the active channels by 16 steps: marginal distributions unchanged,
        # cross-channel phase coupling broken cleanly.
        ch_shift = act[::2]
        shift = 16
        out[t0:t1, ch_shift] = seg_std[t0 - shift : t1 - shift, ch_shift]
        if mode == "Noise_0.15":
            assert rng is not None
            out += rng.normal(0.0, 0.15, size=out.shape).astype(np.float32)
        elif mode == "Noise_0.30":
            assert rng is not None
            out += rng.normal(0.0, 0.30, size=out.shape).astype(np.float32)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return out, y_ep


def main() -> None:
    out_csv = ROOT / "reports" / "wu_keogh_plausible_dev19.csv"
    out_json = ROOT / "reports" / "wu_keogh_plausible_summary.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = _build_schedule("dev19_4block")
    panel = PositionDriftLitePanel(context_size=C, suspect_size=S)
    modes = ("Cycle_Swap", "Phase_Desync", "Noise_0.15", "Noise_0.30")

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

        s0, n_run = longest_nominal_run(y)
        n_fit = len(train) - int(len(train) * 0.35) - max(256, int(len(train) * 0.15))
        _, seg_all = _standardize_arrays(
            train[:n_fit], test[s0 : s0 + min(n_run, 4000)] if n_run >= 1400 else train[-min(len(train), 2500) :]
        )
        if len(seg_all) < W + 700:
            continue
        # Calibration threshold for Wu & Keogh amplitude one-liner `max_c |x_c(t)|` on X_val
        _, va_std = _standardize_arrays(train[:n_fit], train[-len_va:])
        ol_cal = np.max(np.abs(va_std[: len_va // 2]), axis=1)
        th_oneliner = float(np.quantile(ol_cal, 0.995))

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

            rng = np.random.default_rng([sd, idx])
            # Random interior onset (never at end of stream -> zero run-to-failure bias)
            t0 = int(rng.integers(W + 250, max(W + 251, len(seg_all) - 350)))

            for mode in modes:
                seg_mod, y_ep = inject_wu_keogh_anomaly(seg_all, mode, t0, dur=200, rng=rng)
                mod_w = DataLoader.create_windows(seg_mod, W, step=1, copy=False)

                mr, mw = m_lite.extract_raw_and_whitened_residuals(mod_w, context_len=C, batch_size=512)
                va_ew, me_ew = stabilize_whitened_subblocks(va_white, mw, va_raw, mr, n_cal_w, q_white, q_raw)
                dec_l = panel.evaluate_stream_decisions(
                    panel.calibrate_and_trip_windows(va_ew, me_ew, n_cal_w), len_va, len(seg_mod), "trailing"
                )
                tl = y_ep[dec_l.valid_mask]

                ms1 = _score_1d_windows(m_1d, mod_w, device, use_bank=True)
                mp1, mm1 = aggregate_vector_stream(ms1[:, None], len(seg_mod), C, S, "trailing")
                pred_1d = (mp1[mm1, 0] > th_1d).astype(int)

                ol_sc = np.max(np.abs(seg_mod[dec_l.valid_mask]), axis=1)
                pred_ol = (ol_sc > th_oneliner).astype(int)

                for dname, prd, scr in (
                    ("Laminar-JEPA-Lite", dec_l.test_pred_balanced, dec_l.test_switch_counts),
                    ("Multi-Horizon-1D", pred_1d, mp1[mm1, 0]),
                    ("OneLiner-MaxAbs", pred_ol, ol_sc),
                ):
                    mt = compute_metrics(tl, prd, scores=scr, valid_mask=np.ones_like(tl, dtype=bool), use_pa=False)
                    er, dl = event_stats(tl, prd)
                    rows.append(
                        {
                            "stream": s_key,
                            "seed": sd,
                            "mode": mode,
                            "detector": dname,
                            "f1": float(mt.get("f1", 0.0)),
                            "er": float(er) if er == er else 0.0,
                            "delay": float(dl) if dl == dl else float("nan"),
                            "clean_fpr": float(prd[tl == 0].mean()) if np.any(tl == 0) else 0.0,
                        }
                    )
            del m_lite, m_1d
            atomic_write_csv(pd.DataFrame(rows), out_csv)
        print(f"[WuKeogh {idx:2d}/{len(schedule)}] {s_key} completed", flush=True)

    df = pd.DataFrame(rows)
    sm = (
        df.groupby(["mode", "detector", "stream"], as_index=False)
        .mean(numeric_only=True)
        .groupby(["mode", "detector"], as_index=False)[["f1", "er", "delay", "clean_fpr"]]
        .mean()
        .round(4)
        .to_dict("records")
    )
    atomic_write_json({"wu_keogh_suite": sm}, out_json)
    print("Wu & Keogh suite summary saved to", out_json, flush=True)


if __name__ == "__main__":
    main()
