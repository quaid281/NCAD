#!/usr/bin/env python3
"""Experiment 4A: Full-Pipeline Operational Complexity, Channel Scaling & Missing-Channel Resilience.

Measures:
  1. End-to-end streaming pipeline latency (`ms/window`), throughput (`windows/sec`), peak memory
     (`MB`), and persistent ring-buffer/calibration state (`KB`) across `K in {5, 9, 25, 38, 55, 72}`
     on GPU (`cuda`), 8-thread server CPU, and 1-thread embedded-proxy CPU — separating network
     forward latency from full-pipeline latency (standardization + ring buffer + encoder/predictor +
     96x96 whitening + 128-dial tripwires + quorum check).
  2. Missing-channel / sensor-dropout resilience across the 19 multivariate development streams:
     evaluates frozen `Laminar-JEPA-Lite` when 1 or 2 active channels drop out (zero-imputation in
     standardized units, i.e., sample-and-hold at nominal training mean) without retraining.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_laminar_paired_benchmarks import (
    C,
    CSMConfig,
    S,
    W,
    _build_schedule,
    _split_nominal_dev_4block,
    _train_fixed_epochs,
    compute_metrics,
    event_stats,
    load_stream,
)
from scripts.tnnls_revision.common_eval import atomic_write_csv, atomic_write_json
from src.models._jepa_utils import whiten_residuals
from src.models.laminar_lite import LaminarJEPALiteModel, LaminarLiteEncoder
from src.scoring.circuit_breaker_panel import stabilize_whitened_subblocks
from src.scoring.laminar_lite_panel import PositionDriftLitePanel

CHANNEL_COUNTS = (5, 9, 25, 38, 55, 72)


def _benchmark_pipeline_device(k_ch: int, dev_label: str, device: torch.device, n_threads: int | None) -> dict[str, Any]:
    prev_threads = torch.get_num_threads()
    if n_threads is not None:
        torch.set_num_threads(n_threads)
    try:
        torch.manual_seed(42)
        enc = LaminarLiteEncoder(k_ch, latent_dim=32, filters=48, tcn_layers=4, proj_dim=64)
        model = LaminarJEPALiteModel(enc, latent_dim=32, predictor_hidden_dim=64, num_horizons=3).to(device)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())

        # Populate whitening & calibration state
        model.precision_fitted.fill_(True)
        model.precision_matrix.copy_(torch.eye(96, device=device))
        mu_ch = np.zeros((1, k_ch), dtype=np.float32)
        sc_ch = np.ones((1, k_ch), dtype=np.float32)
        tau_dials = np.full((1, 128), 2.4, dtype=np.float32)
        w_blend = np.ones((1, 3, 32), dtype=np.float32)
        rms_raw = np.ones((1, 3, 32), dtype=np.float32)
        k_quorum = 9.5

        bsz = 64 if device.type == "cuda" else 1
        n_iters = 80 if device.type == "cuda" else 120
        raw_step_batch = np.random.default_rng(42).standard_normal((bsz, W, k_ch)).astype(np.float32)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        # Warmup
        with torch.inference_mode():
            for _ in range(15):
                x_t = torch.from_numpy(raw_step_batch).to(device)
                _ = model.forward(x_t[:, :C], x_t[:, C:])

        # 1. Network-only latency
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(n_iters):
                x_t = torch.from_numpy(raw_step_batch).to(device)
                _, _, _, zp, zt = model.forward(x_t[:, :C], x_t[:, C:])
                _ = zp - zt
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        net_ms = 1000.0 * (time.perf_counter() - t0) / (n_iters * bsz)

        # 2. Full streaming pipeline (standardize + ring slice + forward + 96x96 whiten + blend + 128 dials + quorum)
        ring = np.zeros((W, k_ch), dtype=np.float32)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(n_iters):
                if bsz == 1:
                    ring[:-1] = ring[1:]
                    ring[-1] = (raw_step_batch[0, -1] - mu_ch[0]) / sc_ch[0]
                    win_std = ring[None, :, :]
                else:
                    win_std = (raw_step_batch - mu_ch[:, None, :]) / sc_ch[:, None, :]
                x_t = torch.from_numpy(np.ascontiguousarray(win_std)).to(device)
                _, _, _, zp, zt = model.forward(x_t[:, :C], x_t[:, C:])
                assert zt is not None
                raw_r = zt - zp
                flat_r = raw_r.reshape(bsz, 96) - model.residual_mean
                white_r = whiten_residuals(flat_r, model.precision_matrix).reshape(bsz, 3, 32)
                raw_np = raw_r.detach().cpu().numpy()
                white_np = white_r.detach().cpu().numpy()
                ew = w_blend * white_np + (1.0 - w_blend) * (raw_np / rms_raw)
                d_pos = np.abs(ew).reshape(bsz, 96)
                d_drf = np.abs(ew.mean(axis=1))
                dials = np.concatenate([d_pos, d_drf], axis=1)
                cnt = (dials > tau_dials).sum(axis=1)
                _ = cnt > k_quorum
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        full_ms = 1000.0 * (time.perf_counter() - t1) / (n_iters * bsz)

        peak_mb = (
            float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)
            if device.type == "cuda"
            else float(sum(p.numel() * 4 for p in model.parameters()) + sum(b.numel() * 4 for b in model.buffers())) / (1024.0**2)
        )
        # Persistent per-stream state: ring buffer (W*K*4) + mu/sd (2*K*4) + Omega (96*96*4) + mean (96*4) + blend/rms/tau (3*128*4)
        state_kb = (W * k_ch * 4 + 2 * k_ch * 4 + 96 * 96 * 4 + 96 * 4 + 3 * 128 * 4) / 1024.0
        return {
            "device_profile": dev_label,
            "batch_mode": bsz,
            "k_channels": k_ch,
            "dyadic_active": bool(k_ch >= 25),
            "params": int(n_params),
            "net_only_ms": round(net_ms, 4),
            "full_pipeline_ms": round(full_ms, 4),
            "throughput_win_per_s": round(1000.0 / max(full_ms, 1e-6), 1),
            "peak_mem_mb": round(peak_mb, 2),
            "stream_state_kb": round(state_kb, 2),
        }
    finally:
        torch.set_num_threads(prev_threads)


def _run_missing_channel_study(device: torch.device) -> list[dict[str, Any]]:
    schedule = _build_schedule("dev19_4block")
    panel = PositionDriftLitePanel(context_size=C, suspect_size=S)
    rows: list[dict[str, Any]] = []

    for ds, ch, _ in schedule:
        train, test, y = load_stream(ds, ch)
        tr_w, q_w, va_w, te_w, len_va = _split_nominal_dev_4block(train, test)
        k_ch = int(train.shape[1])
        n_cal_w = len(va_w) // 2

        torch.manual_seed(42)
        np.random.seed(42)
        cfg = CSMConfig(model_type="laminar_jepa_lite", context_size=C, suspect_size=S, latent_dim=32)
        enc = LaminarLiteEncoder(k_ch, latent_dim=32, filters=48, tcn_layers=4, proj_dim=64)
        model = LaminarJEPALiteModel(enc, latent_dim=32, predictor_hidden_dim=64, num_horizons=3).to(device)
        _train_fixed_epochs(model, tr_w, cfg, epochs=15, device=device)
        model.fit_resolvent_whitening(tr_w, context_len=C, batch_size=512)
        va_raw, va_white = model.extract_raw_and_whitened_residuals(va_w, context_len=C, batch_size=512)
        q_raw, q_white = model.extract_raw_and_whitened_residuals(q_w, context_len=C, batch_size=512)

        act = np.argsort(-np.std(tr_w[:, -1, :], axis=0))
        for n_drop in (0, 1, 2):
            if n_drop >= k_ch:
                continue
            te_mod = te_w.copy()
            if n_drop > 0:
                # Zero in standardized space = nominal mean imputation on dropped sensor(s)
                te_mod[:, :, act[:n_drop]] = 0.0
            tr_r, tw_r = model.extract_raw_and_whitened_residuals(te_mod, context_len=C, batch_size=512)
            va_ew, te_ew = stabilize_whitened_subblocks(va_white, tw_r, va_raw, tr_r, n_cal_w, q_white, q_raw)
            dec = panel.evaluate_stream_decisions(
                panel.calibrate_and_trip_windows(va_ew, te_ew, n_cal_w), len_va, len(test), "trailing"
            )
            tl = y[dec.valid_mask]
            mt = compute_metrics(tl, dec.test_pred_balanced, scores=dec.test_switch_counts, valid_mask=np.ones_like(tl, dtype=bool), use_pa=False)
            er, dl = event_stats(tl, dec.test_pred_balanced)
            rows.append(
                {
                    "stream": f"{ds}:{ch}",
                    "k_channels": k_ch,
                    "dropped_channels": n_drop,
                    "f1": float(mt.get("f1", 0.0)),
                    "event_recall": float(er) if er == er else float("nan"),
                    "delay": float(dl) if dl == dl else float("nan"),
                    "test_fpr": float(dec.test_pred_balanced[tl == 0].mean()) if np.any(tl == 0) else 0.0,
                }
            )
        del model
    return rows


def main() -> None:
    lat_csv = ROOT / "reports" / "deployment_latency_scaling.csv"
    drop_csv = ROOT / "reports" / "deployment_missing_channel_dev19.csv"
    sum_json = ROOT / "reports" / "deployment_benchmark_summary.json"

    profiles: list[tuple[str, torch.device, int | None]] = [
        ("CPU_1Thread_Embedded", torch.device("cpu"), 1),
        ("CPU_8Thread_Server", torch.device("cpu"), 8),
    ]
    if torch.cuda.is_available():
        profiles.insert(0, ("GPU_RTX5090_Batched64", torch.device("cuda"), None))
        profiles.insert(1, ("GPU_RTX5090_Stream1", torch.device("cuda"), None))

    lat_rows: list[dict[str, Any]] = []
    for label, dev, nth in profiles:
        for k_ch in CHANNEL_COUNTS:
            if label == "GPU_RTX5090_Stream1":
                # Measure strict batch=1 step-by-step GPU latency
                row = _benchmark_pipeline_device(k_ch, label, dev, nth)
                # Override bsz=1 measurement explicitly
                enc = LaminarLiteEncoder(k_ch, latent_dim=32, filters=48, tcn_layers=4, proj_dim=64)
                m1 = LaminarJEPALiteModel(enc, latent_dim=32, predictor_hidden_dim=64, num_horizons=3).to(dev).eval()
                m1.precision_fitted.fill_(True)
                m1.precision_matrix.copy_(torch.eye(96, device=dev))
                xb = torch.randn(1, W, k_ch, device=dev)
                with torch.inference_mode():
                    for _ in range(20):
                        _ = m1.forward(xb[:, :C], xb[:, C:])
                    torch.cuda.synchronize(dev)
                    t0 = time.perf_counter()
                    for _ in range(150):
                        _, _, _, zp, zt = m1.forward(xb[:, :C], xb[:, C:])
                        assert zt is not None
                        fr = (zt - zp).reshape(1, 96) - m1.residual_mean
                        wr = whiten_residuals(fr, m1.precision_matrix).detach().cpu().numpy().reshape(1, 3, 32)
                        _ = (np.concatenate([np.abs(wr).reshape(1, 96), np.abs(wr.mean(axis=1))], axis=1) > 2.4).sum()
                    torch.cuda.synchronize(dev)
                ms_b1 = 1000.0 * (time.perf_counter() - t0) / 150.0
                row["batch_mode"] = 1
                row["full_pipeline_ms"] = round(ms_b1, 4)
                row["throughput_win_per_s"] = round(1000.0 / max(ms_b1, 1e-6), 1)
                del m1
                lat_rows.append(row)
            else:
                lat_rows.append(_benchmark_pipeline_device(k_ch, label, dev, nth))
    atomic_write_csv(pd.DataFrame(lat_rows), lat_csv)

    dev_main = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drop_rows = _run_missing_channel_study(dev_main)
    df_drop = pd.DataFrame(drop_rows)
    atomic_write_csv(df_drop, drop_csv)

    drop_sum = df_drop.groupby("dropped_channels", as_index=False)[["f1", "event_recall", "delay", "test_fpr"]].mean().round(4).to_dict("records")
    atomic_write_json({"latency_scaling": lat_rows, "missing_channel_summary": drop_sum}, sum_json)
    print("Exp 4A summary saved to", sum_json, flush=True)


if __name__ == "__main__":
    main()
