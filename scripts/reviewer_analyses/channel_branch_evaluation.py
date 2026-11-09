#!/usr/bin/env python3
"""Evaluate a complementary channel-level anomaly branch alongside the 224-bit JEPA panel
across all 17 multivariate streams (K > 1) of the 19-stream cohort and 6 single-channel fault types.

Addresses Reviewer Concern 4:
Evaluates:
  1. Individual channel variance collapse (flatline to mean),
  2. Channel-level prediction residual anomaly (AR(1) innovation violation at fixed variance),
  3. Sensor freezing (sample-and-hold stuck value),
  4. Sensor saturation (rail clipping at nominal 99.5th percentile),
  5. Gradual sensor degradation (linear ramp drift up to 2.5 sigma),
  6. Isolated sensor bias (persistent 1.5 sigma step offset).

Compares four configurations (calibrated strictly on nominal validation windows):
  - Standalone 224-Bit JEPA Panel,
  - Hybrid Variance-Collapse Monitor (224-Bit + rolling variance drop),
  - Channel AR(1) Residual + Bias Monitor Only,
  - Complementary Channel-Level Anomaly Branch + 224-Bit Panel (combines individual channel
    variance, channel-level AR(1) prediction residuals, and rail/bias persistence).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.circuit_breaker_19stream.common import C, S, VAL_FRAC, W, cache_path, load_streams_19
from scripts.experiment_wsp_soliton import extract_raw4_white4_from_latents
from scripts.reviewer_analyses.run_224bit_horizon_sweep import fast_extract_latents
from scripts.run_multiseed_heldout_benchmark import batches, compute_metrics, event_stats, load_stream
from src.config import CSMConfig
from src.data.data_loader import DataLoader
from src.engine.builder import build_ts_jepa_model
from src.scoring.circuit_breaker_panel import MultiPerspectiveCircuitBreakerPanel

FAULT_TYPES = (
    "1_variance_collapse",
    "2_channel_pred_shock",
    "3_sensor_freezing",
    "4_sensor_saturation",
    "5_gradual_degradation",
    "6_isolated_sensor_bias",
)


def inject_single_channel_fault(
    test_norm: np.ndarray,
    y_orig: np.ndarray,
    ch_idx: int,
    fault_type: str,
    train_norm: np.ndarray,
    rng: np.random.Generator,
    n_events: int = 3,
    dur: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject `n_events` single-channel faults of `fault_type` into nominal regions of `test_norm`."""
    x_mod = test_norm.copy()
    n = len(x_mod)
    y_synth = np.zeros(n, dtype=np.int64)
    ch_mean = float(np.mean(train_norm[:, ch_idx]))
    ch_std = max(float(np.std(train_norm[:, ch_idx])), 0.1)
    ch_p995 = float(np.quantile(train_norm[:, ch_idx], 0.995))

    # Pick 3 evenly spaced nominal windows after C+S
    usable_start = W + 50
    usable_end = n - dur - 50
    if usable_end <= usable_start + n_events * dur:
        starts = [max(W, n // 3)]
    else:
        stride = (usable_end - usable_start) // n_events
        starts = [usable_start + i * stride for i in range(n_events)]

    for s in starts:
        e = min(n, s + dur)
        length = e - s
        y_synth[s:e] = 1
        if fault_type == "1_variance_collapse":
            x_mod[s:e, ch_idx] = ch_mean
        elif fault_type == "2_channel_pred_shock":
            # High-frequency AR(1)-violating innovation oscillation with matched variance
            alt = np.where(np.arange(length) % 2 == 0, 1.35, -1.35)
            x_mod[s:e, ch_idx] = ch_mean + alt * ch_std
        elif fault_type == "3_sensor_freezing":
            stuck_val = float(x_mod[max(0, s - 1), ch_idx])
            x_mod[s:e, ch_idx] = stuck_val
        elif fault_type == "4_sensor_saturation":
            x_mod[s:e, ch_idx] = max(ch_p995, ch_mean + 2.2 * ch_std)
        elif fault_type == "5_gradual_degradation":
            ramp = np.linspace(0.0, 2.5 * ch_std, length, dtype=np.float32)
            x_mod[s:e, ch_idx] = x_mod[s:e, ch_idx] + ramp
        elif fault_type == "6_isolated_sensor_bias":
            x_mod[s:e, ch_idx] = x_mod[s:e, ch_idx] + 1.5 * ch_std
        else:
            raise ValueError(f"Unknown fault_type: {fault_type}")
    return x_mod, y_synth


class ComplementaryChannelAnomalyBranch:
    """Lightweight per-channel observation-space anomaly branch operating alongside JEPA.

    Monitors three complementary per-channel statistics over causal rolling windows (L = 64):
      1. Individual channel variance collapse / freeze score: max(0, -log10(sigma_roll / sigma_ref)).
      2. Channel-level AR(1) prediction residual RMS: rolling RMS of innovation e_{t,c} = x_{t,c} - phi_c x_{t-1,c}.
      3. Channel-level rolling baseline bias / saturation excursion: |mean_roll(x_{t,c}) - mu_c| / sigma_c.
    """

    def __init__(self, win: int = 64, min_persist: int = 40) -> None:
        self.win = win
        self.min_persist = min_persist
        self.active_mask: np.ndarray | None = None
        self.phi1: np.ndarray | None = None
        self.mu_c: np.ndarray | None = None
        self.sd_c: np.ndarray | None = None
        self.var_scale: np.ndarray | None = None
        self.thr_var: np.ndarray | None = None
        self.thr_res: np.ndarray | None = None
        self.thr_bias: np.ndarray | None = None

    def _compute_three_channel_streams(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (var_drop_score, ar1_res_rms, rolling_bias_z) each of shape [T, K_active]."""
        xa = x[:, self.active_mask]
        T, Ka = xa.shape
        if T <= self.win:
            z = np.zeros((T, Ka), dtype=np.float64)
            return z, z, z

        # 1. Rolling std -> log variance drop
        w_view = sliding_window_view(xa, self.win, axis=0)  # [T - win + 1, Ka, win]
        r_std = w_view.std(axis=-1)
        pad_std = np.vstack([np.repeat(r_std[:1], self.win - 1, axis=0), r_std])
        var_drop = np.maximum(0.0, -np.log10((pad_std + 1e-6) / self.var_scale))

        # 2. Causal AR(1) one-step prediction residuals
        xc = xa - self.mu_c[None, :]
        pred = np.zeros_like(xc)
        pred[1:] = xc[:-1] * self.phi1[None, :]
        innov = (xc - pred) / self.sd_c[None, :]
        # High-pass innovation difference |x_t - x_{t-1}| vs AR(1) prediction error
        w_innov = sliding_window_view(innov ** 2, self.win, axis=0).mean(axis=-1)
        ar1_rms = np.sqrt(np.vstack([np.repeat(w_innov[:1], self.win - 1, axis=0), w_innov]))

        # 3. Rolling mean excursion / saturation / bias score
        r_mean = np.abs(w_view.mean(axis=-1) - self.mu_c[None, :]) / self.sd_c[None, :]
        bias_z = np.vstack([np.repeat(r_mean[:1], self.win - 1, axis=0), r_mean])
        return var_drop, ar1_rms, bias_z

    def fit(self, tr_norm: np.ndarray, va_cal_norm: np.ndarray) -> "ComplementaryChannelAnomalyBranch":
        # Exclude binary/discrete status channels (must have >= 12 unique values and dynamic std)
        K = tr_norm.shape[1]
        active = []
        for k in range(K):
            col = va_cal_norm[:, k]
            if len(np.unique(np.round(col, 4))) >= 12 and np.std(col) > 0.15:
                active.append(k)
        if not active:
            active = [int(np.argmax(np.std(tr_norm, axis=0)))]
        self.active_mask = np.zeros(K, dtype=bool)
        self.active_mask[active] = True

        xa_tr = tr_norm[:, self.active_mask]
        xa_cal = va_cal_norm[:, self.active_mask]
        self.mu_c = np.median(xa_cal, axis=0)
        self.sd_c = np.maximum(
            1.4826 * np.median(np.abs(xa_cal - self.mu_c[None, :]), axis=0),
            0.20,
        )
        xc = xa_tr - self.mu_c[None, :]
        num = np.sum(xc[1:] * xc[:-1], axis=0)
        den = np.maximum(np.sum(xc[:-1] ** 2, axis=0), 1e-6)
        self.phi1 = np.clip(num / den, -0.98, 0.98)

        w_cal = sliding_window_view(xa_cal, self.win, axis=0).std(axis=-1)
        self.var_scale = np.maximum(np.median(w_cal, axis=0), 0.05)

        v_cal, r_cal, b_cal = self._compute_three_channel_streams(va_cal_norm)
        half = max(32, len(v_cal) // 2)
        # Per-channel calibration drift guard: ratio of second-half to first-half 99th percentile
        g_v = np.maximum(1.0, (np.quantile(v_cal[half:], 0.99, axis=0) + 0.1) / (np.quantile(v_cal[:half], 0.99, axis=0) + 0.1))
        g_r = np.maximum(1.0, (np.quantile(r_cal[half:], 0.99, axis=0) + 0.1) / (np.quantile(r_cal[:half], 0.99, axis=0) + 0.1))
        g_b = np.maximum(1.0, (np.quantile(b_cal[half:], 0.99, axis=0) + 0.1) / (np.quantile(b_cal[:half], 0.99, axis=0) + 0.1))
        k_bonf = 1.0 + 0.10 * np.log(max(len(active), 2))
        self.thr_var = np.maximum(np.quantile(v_cal, 0.9995, axis=0) * g_v * 1.35 * k_bonf, 1.85)
        self.thr_res = np.maximum(np.quantile(r_cal, 0.9995, axis=0) * g_r * 1.45 * k_bonf, 2.50)
        self.thr_bias = np.maximum(np.quantile(b_cal, 0.9995, axis=0) * g_b * 1.45 * k_bonf, 1.65)
        return self

    def _persistence_gate(self, raw_bit: np.ndarray) -> np.ndarray:
        """Require at least `self.min_persist` active steps inside a causal window of length `self.win`."""
        if len(raw_bit) < self.win:
            return raw_bit.astype(np.int64)
        roll = np.convolve(raw_bit.astype(np.int64), np.ones(self.win, dtype=np.int64), mode="full")[: len(raw_bit)]
        return (roll >= self.min_persist).astype(np.int64)

    def predict_branches(self, x_test: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return (binary_pred, continuous_score) for variance-only, residual/bias-only, and full channel branch."""
        v_te, r_te, b_te = self._compute_three_channel_streams(x_test)

        # Normalized ratios relative to per-channel thresholds
        z_var = np.max(v_te / self.thr_var[None, :], axis=1)
        z_res = np.max(r_te / self.thr_res[None, :], axis=1)
        z_bias = np.max(b_te / self.thr_bias[None, :], axis=1)

        pred_var = self._persistence_gate(z_var > 1.0)
        pred_res_bias = self._persistence_gate(np.maximum(z_res, z_bias) > 1.0)
        pred_full_ch = np.maximum(pred_var, pred_res_bias)
        score_full_ch = np.maximum(z_var, np.maximum(z_res, z_bias))

        return {
            "var_only": (pred_var, z_var),
            "res_bias_only": (pred_res_bias, np.maximum(z_res, z_bias)),
            "full_channel": (pred_full_ch, score_full_ch),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports" / "channel_branch_19stream_evaluation.csv"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panel = MultiPerspectiveCircuitBreakerPanel()
    streams = load_streams_19()
    rows: list[dict[str, object]] = []
    t0 = time.time()

    for ds, ch in streams:
        train, test, y_orig = load_stream(ds, ch)
        K = train.shape[1]
        if K <= 1:
            print(f"[SKIP K=1] {ds}:{ch} (univariate stream; no multi-sensor LayerNorm dilution)")
            continue

        if len(test) > 15000:
            anom_idx = np.where(y_orig == 1)[0]
            s_te = max(0, int(anom_idx[0]) - 2000) if len(anom_idx) > 0 else 0
            e_te = min(len(test), s_te + 15000)
            s_te = max(0, e_te - 15000)
            test = test[s_te:e_te]
            y_orig = y_orig[s_te:e_te]

        n_val = int(len(train) * VAL_FRAC)
        tr, va = train[:-n_val], train[-n_val:]
        mu, sd = tr.mean(0), tr.std(0)
        sc = np.where(sd < 1e-8, 1.0, np.maximum(sd, 0.01))
        tr_norm = (tr - mu) / sc
        va_norm = (va - mu) / sc
        te_norm = (test - mu) / sc

        tr_w = DataLoader.create_windows(tr_norm, W, step=16, copy=False)
        va_w = DataLoader.create_windows(va_norm, W, step=1, copy=False)
        n_cal_w = len(va_w) // 2

        torch.manual_seed(42)
        np.random.seed(42)
        cfg = CSMConfig(model_type="soliton_reynolds_jepa", context_size=C, suspect_size=S, latent_dim=32)
        model = build_ts_jepa_model(cfg, input_dim=K, device=device)
        pt_path = cache_path(ds, ch, 42).with_suffix(".pt")
        if pt_path.exists():
            ckpt = torch.load(pt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
            z_ctx_va = ckpt["z_ctx_va"].to(device)
            z_tgt_va = ckpt["z_tgt_va"].to(device)
            z_ctx_te_orig = ckpt["z_ctx_te"].to(device)
            z_tgt_te_orig = ckpt["z_tgt_te"].to(device)
        else:
            opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for _ in range(8):
                model.train()
                for b in batches(tr_w, 128):
                    b = b.to(device)
                    loss, _ = model.compute_objective(b[:, :C], b[:, C:], cfg)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    model.update_target_encoder(None)
            model.fit_alien_bank(tr_w, context_len=C, batch_size=512)
            z_ctx_va, z_tgt_va = fast_extract_latents(model, va_w, C, batch_size=512)
            te_w_orig = DataLoader.create_windows(te_norm, W, step=1, copy=False)
            z_ctx_te_orig, z_tgt_te_orig = fast_extract_latents(model, te_w_orig, C, batch_size=512)
        _, va_white4 = extract_raw4_white4_from_latents(model, z_ctx_va, z_tgt_va)
        va_ew = np.clip(np.nan_to_num(va_white4.astype(np.float64), nan=0.0), -100.0, 100.0)

        # Fit complementary channel-level anomaly branch on nominal validation prefix
        ch_branch = ComplementaryChannelAnomalyBranch(win=64, min_persist=40).fit(
            tr_norm, va_norm[: C + n_cal_w]
        )

        # Pick active dynamic sensor with highest training std
        active_indices = np.where(ch_branch.active_mask)[0]
        target_ch = int(active_indices[np.argmax(sd[active_indices])])
        rng = np.random.default_rng(42)

        for ftype in (*FAULT_TYPES, "0_unmodified_test"):
            if ftype == "0_unmodified_test":
                x_te_cur, y_cur = te_norm, y_orig
                z_ctx_te, z_tgt_te = z_ctx_te_orig, z_tgt_te_orig
            else:
                x_te_cur, y_cur = inject_single_channel_fault(
                    te_norm, y_orig, target_ch, ftype, tr_norm, rng
                )
                te_w = DataLoader.create_windows(x_te_cur, W, step=1, copy=False)
                # Only re-encode windows that overlap the injected fault timestamps y_cur == 1
                fault_idx = np.where(y_cur == 1)[0]
                aff_mask = np.zeros(len(te_w), dtype=bool)
                if len(fault_idx) > 0:
                    # Any window starting in [t - W + 1, t] overlaps timestamp t
                    for t_f in fault_idx:
                        lo = max(0, int(t_f) - W + 1)
                        hi = min(len(te_w), int(t_f) + 1)
                        aff_mask[lo:hi] = True
                z_ctx_te = z_ctx_te_orig.clone()
                z_tgt_te = z_tgt_te_orig.clone()
                if np.any(aff_mask):
                    aff_indices = np.where(aff_mask)[0]
                    zc_sub, zt_sub = fast_extract_latents(model, te_w[aff_indices], C, batch_size=512)
                    z_ctx_te[aff_indices] = zc_sub
                    z_tgt_te[aff_indices] = zt_sub
            _, te_white4 = extract_raw4_white4_from_latents(model, z_ctx_te, z_tgt_te)
            te_ew = np.clip(np.nan_to_num(te_white4.astype(np.float64), nan=0.0), -100.0, 100.0)

            bits = panel.calibrate_and_trip_windows(va_ew, te_ew, n_cal_w)
            dec = panel.evaluate_stream_decisions(bits, len(va_norm), len(x_te_cur), "trailing")
            valid_mask = dec.valid_mask
            tl = y_cur[valid_mask]
            pred_jepa = dec.test_pred_balanced
            score_jepa = dec.test_switch_counts

            ch_out = ch_branch.predict_branches(x_te_cur)
            p_var, s_var = ch_out["var_only"][0][valid_mask], ch_out["var_only"][1][valid_mask]
            p_res, s_res = ch_out["res_bias_only"][0][valid_mask], ch_out["res_bias_only"][1][valid_mask]
            p_fch, s_fch = ch_out["full_channel"][0][valid_mask], ch_out["full_channel"][1][valid_mask]

            variants = [
                ("1_Standalone_JEPA_224Bit", pred_jepa, score_jepa),
                ("2_Hybrid_VarCollapse_Monitor", np.maximum(pred_jepa, p_var), np.maximum(score_jepa, s_var * 10.0)),
                ("3_Channel_PredRes_Bias_Only", p_res, s_res),
                ("4_JEPA_224Bit_+_Full_Channel_Branch", np.maximum(pred_jepa, p_fch), np.maximum(score_jepa, s_fch * 10.0)),
            ]

            for vname, prd, scr in variants:
                mt = compute_metrics(
                    tl, prd, scores=scr, valid_mask=np.ones_like(tl, dtype=bool), use_pa=False
                )
                er, delay = event_stats(tl, prd)
                tfpr = float(prd[tl == 0].mean()) if np.any(tl == 0) else 0.0
                rows.append(
                    {
                        "stream": f"{ds}:{ch}",
                        "K": K,
                        "target_channel": target_ch,
                        "fault_type": ftype,
                        "variant": vname,
                        "f1": float(mt["f1"]),
                        "precision": float(mt["precision"]),
                        "recall": float(mt["recall"]),
                        "event_recall": float(er),
                        "fpr": tfpr,
                        "pr_auc": float(mt["pr_auc"]),
                        "delay": float(delay),
                    }
                )
        print(f"[{time.time() - t0:5.1f}s] {ds}:{ch} (K={K}) 6-fault channel evaluation done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print("\n=== SINGLE-CHANNEL FAULT EVALUATION ACROSS ALL 17 MULTIVARIATE STREAMS (K > 1) ===")
    summary = (
        df.groupby(["fault_type", "variant"])[
            ["event_recall", "f1", "precision", "recall", "fpr", "delay"]
        ]
        .mean()
        .round(4)
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
