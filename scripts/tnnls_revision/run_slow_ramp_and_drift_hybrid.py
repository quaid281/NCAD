#!/usr/bin/env python3
"""Experiment 3: Fault Onset-Rate Sensitivity, Mechanistic Diagnostics & Two-Timescale Drift Hybrid.

Addresses Reviewer Weakness 6 & Experiment 3:
  1. Evaluates 6 fault onset profiles across clean nominal segments of the multivariate development
     streams (x 3 seeds `{42, 123, 456}`):
       - `Abrupt_Step`       : Instantaneous +2.0*sigma step (256-step plateau)
       - `Short_Ramp_64`     : Linear ramp L = 64 (= C/4), final +2.0*sigma
       - `Med_Ramp_256`      : Linear ramp L = 256 (= C), final +2.0*sigma
       - `Long_Ramp_1024`    : Linear ramp L = 1024 (= 4*C), final +2.0*sigma
       - `Exp_Degrad_512`    : Exponential sensor degradation `2.5*(1 - exp(-(t-t0)/512))`
       - `Accum_Bias_Slow`   : Slowly accumulating sensor walk `0.001*sigma/step` over 2,000 steps
  2. Directly measures the 4 mechanistic hypotheses during each fault:
       (H1) Pre-LayerNorm encoder shift `||u_ctx(t) - u_nom||_2` vs. post-LayerNorm `||z_ctx(t) - z_nom||_2`
       (H2) Sliding-window context-target co-movement `||z_tgt(t) - z_ctx(t)||_2`
       (H3) Whitened multi-horizon residual RMS `||E_white(t)||_F / sqrt(M*D)`
       (H4) Fraction of active drift vs. sub-horizon position dials
  3. Compares 4 causal detectors on both the 6 onset profiles AND all 14 real GHL industrial ramp streams:
       (a) `Laminar-JEPA-Lite`
       (b) `Multi-Horizon JEPA (1D)` (SPOT)
       (c) `Frozen-Ref-Drift` (`FrozenLatentDriftMonitor` anchored to `X_fit`)
       (d) `Hybrid-TwoScale` (`Laminar-JEPA-Lite OR Frozen-Ref-Drift`)
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
from scripts.tnnls_revision.common_eval import (
    FrozenLatentDriftMonitor,
    atomic_write_csv,
    atomic_write_json,
)
from src.data.data_loader import DataLoader
from src.engine.builder import build_encoder
from src.models.jepa.alien_jepas import SpeculativeJEPAModel
from src.models.laminar_lite import LaminarJEPALiteModel, LaminarLiteEncoder
from src.scoring.circuit_breaker_panel import aggregate_vector_stream, stabilize_whitened_subblocks
from src.scoring.laminar_lite_panel import PositionDriftLitePanel


ONSET_PROFILES: tuple[tuple[str, int, float], ...] = (
    ("Abrupt_Step", 1, 2.0),
    ("Short_Ramp_64", 64, 2.0),
    ("Med_Ramp_256", 256, 2.0),
    ("Long_Ramp_1024", 1024, 2.0),
    ("Exp_Degrad_512", 1536, 2.5),
    ("Accum_Bias_Slow", 2000, 2.0),
)


def _inject_profile(
    seg_std: np.ndarray, profile: str, span: int, amp: float, t0: int, ch_sub: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Inject a single controlled fault episode `[t0, t_end)` and return `(seg_mod, y_ep)`."""
    out = seg_std.copy()
    t_len = len(out)
    plateau = 256
    t_end = min(t_len, t0 + span + plateau)
    y_ep = np.zeros(t_len, dtype=int)
    y_ep[t0:t_end] = 1
    dur = max(1, min(span, t_len - t0))

    if profile == "Abrupt_Step":
        out[t0:t_end, ch_sub] += amp
    elif profile.startswith(("Short_Ramp", "Med_Ramp", "Long_Ramp")):
        ramp = np.linspace(0.0, amp, dur, dtype=np.float32)[:, None]
        out[t0 : t0 + dur, ch_sub] += ramp
        if t0 + dur < t_end:
            out[t0 + dur : t_end, ch_sub] += amp
    elif profile == "Exp_Degrad_512":
        ts = np.arange(t_end - t0, dtype=np.float32)[:, None]
        curve = amp * (1.0 - np.exp(-ts / 512.0))
        out[t0:t_end, ch_sub] += curve
    elif profile == "Accum_Bias_Slow":
        ts = np.arange(t_end - t0, dtype=np.float32)[:, None]
        walk = np.minimum(amp, 0.001 * ts)
        out[t0:t_end, ch_sub] += walk
    else:
        raise ValueError(f"Unknown profile: {profile}")
    return out, y_ep


@torch.no_grad()
def _mechanistic_diagnostics(
    model: LaminarJEPALiteModel, clean_w: np.ndarray, fault_w: np.ndarray, ep_w_mask: np.ndarray
) -> dict[str, float]:
    """Measure context shift, context-target co-movement, and whitened residual RMS inside episode."""
    model.eval()
    dev = next(model.parameters()).device
    idx_ep = np.flatnonzero(ep_w_mask)
    if len(idx_ep) == 0:
        return {"z_ctx_shift": 0.0, "ctx_tgt_diff": 0.0, "white_rms": 0.0}
    sub_idx = idx_ep[:: max(1, len(idx_ep) // 128)]
    cw = torch.from_numpy(np.ascontiguousarray(clean_w[sub_idx])).float().to(dev)
    fw = torch.from_numpy(np.ascontiguousarray(fault_w[sub_idx])).float().to(dev)
    with torch.inference_mode():
        zc_nom, zt_nom, _, _, _ = model.forward(cw[:, :C], cw[:, C:])
        zc_flt, zt_flt, _, _, _ = model.forward(fw[:, :C], fw[:, C:])
        assert zt_nom is not None and zt_flt is not None
        z_ctx_shift = float(torch.linalg.norm(zc_flt - zc_nom, dim=-1).mean().item())
        ctx_tgt_diff = float(torch.linalg.norm(zt_flt - zc_flt, dim=-1).mean().item())
    _, fw_white = model.extract_raw_and_whitened_residuals(fault_w[sub_idx], context_len=C, batch_size=256)
    white_rms = float(np.sqrt(np.mean(fw_white**2)))
    return {"z_ctx_shift": z_ctx_shift, "ctx_tgt_diff": ctx_tgt_diff, "white_rms": white_rms}


def main() -> None:
    prof_csv = ROOT / "reports" / "slow_degradation_profiles_dev19.csv"
    ghl_csv = ROOT / "reports" / "slow_degradation_ghl14.csv"
    sum_json = ROOT / "reports" / "slow_degradation_and_hybrid_summary.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panel = PositionDriftLitePanel(context_size=C, suspect_size=S)

    # Part A: 6 Onset Profiles on Multivariate Dev Streams (3 seeds)
    dev_sched = _build_schedule("dev19_4block")
    prof_rows: list[dict[str, Any]] = pd.read_csv(prof_csv).to_dict("records") if prof_csv.exists() else []
    done_prof = {(str(r["stream"]), int(r["seed"])) for r in prof_rows}

    for idx, (ds, ch, seeds) in enumerate(dev_sched, start=1):
        s_key = f"{ds}:{ch}"
        need = [sd for sd in seeds if (s_key, sd) not in done_prof]
        if not need:
            continue
        train, test, y = load_stream(ds, ch)
        tr_w, q_w, va_w, _, len_va = _split_nominal_dev_4block(train, test)
        k_ch = int(train.shape[1])
        n_cal_w = len(va_w) // 2

        s0, n_run = longest_nominal_run(y)
        n_fit = len(train) - int(len(train) * 0.35) - max(256, int(len(train) * 0.15))
        _, seg_all = _standardize_arrays(
            train[:n_fit], test[s0 : s0 + min(n_run, 5000)] if n_run >= 2800 else train[-min(len(train), 3200) :]
        )
        if len(seg_all) < W + 2400:
            continue
        t0 = W + 64
        clean_w = DataLoader.create_windows(seg_all, W, step=1, copy=False)

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

            # Frozen-Reference Latent Drift Monitor fitted on X_fit, calibrated on X_cal
            drift_mon = FrozenLatentDriftMonitor(ewma_alpha=0.02, safety=1.15, q=0.995)
            f_fit = drift_mon.extract_pre_ln_context_summaries(m_lite, tr_w, context_len=C, batch_size=512)
            f_val = drift_mon.extract_pre_ln_context_summaries(m_lite, va_w, context_len=C, batch_size=512)
            drift_mon.fit_reference(f_fit)

            # 1D MH-JEPA comparator
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
            act = np.flatnonzero(np.std(seg_all[:W], axis=0) > 1e-4)
            if len(act) == 0:
                act = np.arange(k_ch)
            ch_sub = rng.choice(act, size=max(1, int(np.ceil(0.30 * len(act)))), replace=False)

            for pname, pspan, pamp in ONSET_PROFILES:
                seg_mod, y_ep = _inject_profile(seg_all, pname, pspan, pamp, t0, ch_sub)
                mod_w = DataLoader.create_windows(seg_mod, W, step=1, copy=False)
                mr, mw = m_lite.extract_raw_and_whitened_residuals(mod_w, context_len=C, batch_size=512)
                va_ew, me_ew = stabilize_whitened_subblocks(va_white, mw, va_raw, mr, n_cal_w, q_white, q_raw)
                bits = panel.calibrate_and_trip_windows(va_ew, me_ew, n_cal_w)
                dec_l = panel.evaluate_stream_decisions(bits, len_va, len(seg_mod), "trailing")
                tl = y_ep[dec_l.valid_mask]

                # Mechanistic diagnostics
                ep_w_mask = y_ep[W - 1 :] == 1
                mech = _mechanistic_diagnostics(m_lite, clean_w, mod_w, ep_w_mask)
                b_te = np.asarray(bits["bits_pd_te"], dtype=np.float32)
                drift_trip_share = float(b_te[ep_w_mask, 96:].sum() / max(b_te[ep_w_mask].sum(), 1.0))

                # Drift & Hybrid
                f_mod = drift_mon.extract_pre_ln_context_summaries(m_lite, mod_w, context_len=C, batch_size=512)
                dec_d = drift_mon.calibrate_and_score(f_val, f_mod, n_cal_w=n_cal_w, suspect_size=S)
                dp_p, dp_m = aggregate_vector_stream(dec_d.test_pred[:, None].astype(np.float32), len(seg_mod), C, S, "trailing")
                pred_drift = (dp_p[dp_m, 0] > 0.5).astype(int)
                pred_lite = dec_l.test_pred_balanced
                pred_hyb = np.maximum(pred_lite, pred_drift)

                # 1D MH-JEPA
                ms1 = _score_1d_windows(m_1d, mod_w, device, use_bank=True)
                mp1, mm1 = aggregate_vector_stream(ms1[:, None], len(seg_mod), C, S, "trailing")
                pred_1d = (mp1[mm1, 0] > th_1d).astype(int)

                for dname, prd, hfpr in (
                    ("Laminar-JEPA-Lite", pred_lite, float(dec_l.hold_pred_balanced.mean())),
                    ("Multi-Horizon-1D", pred_1d, float("nan")),
                    ("Frozen-Ref-Drift", pred_drift, float(dec_d.hold_pred.mean())),
                    ("Hybrid-TwoScale", pred_hyb, float(np.maximum(dec_l.hold_pred_balanced[: len(dec_d.hold_pred)], dec_d.hold_pred[: len(dec_l.hold_pred_balanced)]).mean())),
                ):
                    er, dl = event_stats(tl, prd)
                    prof_rows.append(
                        {
                            "stream": s_key,
                            "seed": sd,
                            "profile": pname,
                            "detector": dname,
                            "detected": float(er) if er == er else 0.0,
                            "delay": float(dl) if dl == dl else float("nan"),
                            "hold_fpr": hfpr,
                            "clean_tail_fpr": float(prd[tl == 0].mean()) if np.any(tl == 0) else 0.0,
                            "drift_trip_share": drift_trip_share,
                            **mech,
                        }
                    )
            del m_lite, m_1d
            atomic_write_csv(pd.DataFrame(prof_rows), prof_csv)
        print(f"[Exp3-Profiles {idx:2d}/{len(dev_sched)}] {s_key} completed", flush=True)

    # Part B: All 14 Real GHL Slow-Ramp Streams (seeds 42, 123, 456)
    ghl_sched = [(ds, ch, (42, 123, 456)) for ds, ch, _ in _build_schedule("dev33") if ds.lower() == "ghl"]
    ghl_rows: list[dict[str, Any]] = pd.read_csv(ghl_csv).to_dict("records") if ghl_csv.exists() else []
    done_ghl = {(str(r["stream"]), int(r["seed"])) for r in ghl_rows}

    for idx, (ds, ch, seeds) in enumerate(ghl_sched, start=1):
        s_key = f"{ds}:{ch}"
        need = [sd for sd in seeds if (s_key, sd) not in done_ghl]
        if not need:
            continue
        train, test, y = load_stream(ds, ch)
        tr_w, q_w, va_w, te_w, len_va = _split_nominal_dev_4block(train, test)
        k_ch = int(train.shape[1])
        n_cal_w = len(va_w) // 2

        for sd in need:
            torch.manual_seed(sd)
            np.random.seed(sd)
            cfg_l = CSMConfig(model_type="laminar_jepa_lite", context_size=C, suspect_size=S, latent_dim=32)
            enc = LaminarLiteEncoder(k_ch, latent_dim=32, filters=48, tcn_layers=4, proj_dim=64)
            m_lite = LaminarJEPALiteModel(enc, latent_dim=32, predictor_hidden_dim=64, num_horizons=3).to(device)
            _train_fixed_epochs(m_lite, tr_w, cfg_l, epochs=12, device=device)
            m_lite.fit_resolvent_whitening(tr_w, context_len=C, batch_size=512)
            va_raw, va_white = m_lite.extract_raw_and_whitened_residuals(va_w, context_len=C, batch_size=512)
            te_raw, te_white = m_lite.extract_raw_and_whitened_residuals(te_w, context_len=C, batch_size=512)
            q_raw, q_white = m_lite.extract_raw_and_whitened_residuals(q_w, context_len=C, batch_size=512)
            va_ew, te_ew = stabilize_whitened_subblocks(va_white, te_white, va_raw, te_raw, n_cal_w, q_white, q_raw)
            dec_l = panel.evaluate_stream_decisions(panel.calibrate_and_trip_windows(va_ew, te_ew, n_cal_w), len_va, len(test), "trailing")
            tl = y[dec_l.valid_mask]

            drift_mon = FrozenLatentDriftMonitor(ewma_alpha=0.02, safety=1.15, q=0.995)
            f_fit = drift_mon.extract_pre_ln_context_summaries(m_lite, tr_w, context_len=C, batch_size=512)
            f_val = drift_mon.extract_pre_ln_context_summaries(m_lite, va_w, context_len=C, batch_size=512)
            f_te = drift_mon.extract_pre_ln_context_summaries(m_lite, te_w, context_len=C, batch_size=512)
            drift_mon.fit_reference(f_fit)
            dec_d = drift_mon.calibrate_and_score(f_val, f_te, n_cal_w=n_cal_w, suspect_size=S)
            dp_p, dp_m = aggregate_vector_stream(dec_d.test_pred[:, None].astype(np.float32), len(test), C, S, "trailing")
            pred_drift = (dp_p[dp_m, 0] > 0.5).astype(int)
            pred_lite = dec_l.test_pred_balanced
            pred_hyb = np.maximum(pred_lite, pred_drift)
            sc_p, sc_m = aggregate_vector_stream(dec_d.test_scores[:, None], len(test), C, S, "trailing")
            sc_drift = sc_p[sc_m, 0]

            for dname, prd, scr, hfpr in (
                ("Laminar-JEPA-Lite", pred_lite, dec_l.test_switch_counts, float(dec_l.hold_pred_balanced.mean())),
                ("Frozen-Ref-Drift", pred_drift, sc_drift, float(dec_d.hold_pred.mean())),
                ("Hybrid-TwoScale", pred_hyb, np.maximum(dec_l.test_switch_counts / max(dec_l.k_gate_balanced, 1.0), sc_drift / max(dec_d.threshold, 1e-6)), float(np.maximum(dec_l.hold_pred_balanced[: len(dec_d.hold_pred)], dec_d.hold_pred[: len(dec_l.hold_pred_balanced)]).mean())),
            ):
                mt = compute_metrics(tl, prd, scores=scr, valid_mask=np.ones_like(tl, dtype=bool), use_pa=False)
                er, dl = event_stats(tl, prd)
                ghl_rows.append(
                    {
                        "stream": s_key,
                        "seed": sd,
                        "detector": dname,
                        "f1": float(mt.get("f1", 0.0)),
                        "event_recall": float(er) if er == er else float("nan"),
                        "delay": float(dl) if dl == dl else float("nan"),
                        "hold_fpr": hfpr,
                        "test_fpr": float(prd[tl == 0].mean()) if np.any(tl == 0) else 0.0,
                    }
                )
            del m_lite
            atomic_write_csv(pd.DataFrame(ghl_rows), ghl_csv)
        print(f"[Exp3-GHL {idx:2d}/{len(ghl_sched)}] {s_key} completed", flush=True)

    df_p = pd.DataFrame(prof_rows)
    df_g = pd.DataFrame(ghl_rows)
    p_sum = (
        df_p.groupby(["profile", "detector", "stream"], as_index=False)
        .mean(numeric_only=True)
        .groupby(["profile", "detector"], as_index=False)[
            ["detected", "delay", "hold_fpr", "clean_tail_fpr", "z_ctx_shift", "ctx_tgt_diff", "white_rms", "drift_trip_share"]
        ]
        .mean()
        .round(4)
        .to_dict("records")
    )
    g_sum = (
        df_g.groupby(["detector", "stream"], as_index=False)
        .mean(numeric_only=True)
        .groupby("detector", as_index=False)[["f1", "event_recall", "delay", "hold_fpr", "test_fpr"]]
        .mean()
        .round(4)
        .to_dict("records")
    )
    atomic_write_json({"onset_profiles": p_sum, "ghl14_real_ramps": g_sum}, sum_json)
    print("Exp 3 summary saved to", sum_json, flush=True)


if __name__ == "__main__":
    main()
