"""Real-data GPU training and 5-stage telemetry extraction for Laminar-JEPA-Lite.

Trains ``LaminarJEPALiteModel`` on any stream in ``mTSBench_data/`` using CUDA
(RTX 5090 when available) and extracts real intermediate representations for
interactive visualization:
  1. Normalized multi-channel telemetry + ground-truth anomaly spans.
  2. Latent trajectories before and after barycenter centering + LayerNorm
     hypersphere projection, plus 3-horizon predictions vs. EMA targets.
  3. Raw correlated prediction errors vs. damped ZCA whitened residuals.
  4. Reynolds mean-drift decomposition + 128-dial circuit-breaker states.
  5. Live coordinate quorum vs. 1D scalar SPOT comparator + causal metrics.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.models.laminar_lite import LaminarJEPALiteModel, LaminarLiteEncoder
from src.scoring.circuit_breaker_panel import stabilize_whitened_subblocks
from src.scoring.evt_calibrator import EVTCalibrator
from src.scoring.laminar_lite_panel import PositionDriftLitePanel, build_position_drift_dials

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "mTSBench_data"

CONTEXT_LEN = 256
SUSPECT_LEN = 64
WIN_LEN = CONTEXT_LEN + SUSPECT_LEN

# Curated showcase presets (verified dynamically against mTSBench_data/)
SHOWCASE_PRESETS: list[tuple[str, str, str]] = [
    ("GECCO", "GECCO_water_quality", "GECCO Water Quality (Abrupt Spike)"),
    ("SMAP", "SMAP_A-7", "NASA SMAP A-7 (Contextual Fault)"),
    ("MSL", "MSL_M-1", "NASA MSL M-1 (55-Ch Spacecraft)"),
    ("Daphnet", "Daphnet_S01R01E0", "Daphnet Gait Freeze (Wearable IMU)"),
    ("SMD", "SMD_machine-1-1", "Server Machine Dataset 1-1 (38-Ch)"),
    ("Exathlon", "Exathlon_app1_2_100000_65", "Exathlon Spark Cluster (Drift + Fault)"),
]

_CACHE: OrderedDict[tuple[str, str, int, int], dict[str, Any]] = OrderedDict()
_MAX_CACHE = 8


def get_device_info() -> dict[str, Any]:
    """Return active PyTorch compute device metadata."""
    has_cuda = torch.cuda.is_available()
    return {
        "device": "cuda" if has_cuda else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if has_cuda else "CPU",
        "cuda_available": has_cuda,
    }


def list_available_streams(data_root: Path = DATA_ROOT) -> dict[str, Any]:
    """Scan ``mTSBench_data/`` for paired ``*_train.csv`` / ``*_test.csv`` streams."""
    datasets: dict[str, list[str]] = {}
    if data_root.exists():
        for ds_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            streams: list[str] = []
            for train_csv in sorted(ds_dir.glob("*_train.csv")):
                stem = train_csv.name[: -len("_train.csv")]
                if (ds_dir / f"{stem}_test.csv").exists():
                    streams.append(stem)
            if streams:
                datasets[ds_dir.name] = streams

    presets = [
        {"dataset": ds, "stream": st, "label": lbl}
        for ds, st, lbl in SHOWCASE_PRESETS
        if ds in datasets and st in datasets[ds]
    ]
    return {"presets": presets, "datasets": datasets, "device_info": get_device_info()}


def _read_feature_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """Read an mTSBench CSV cleanly without ``df.iterrows()``."""
    df = pd.read_csv(csv_path)
    labels: np.ndarray | None = None
    if "is_anomaly" in df.columns:
        labels = df["is_anomaly"].to_numpy(dtype=np.int32, copy=True)
    drop_cols = [c for c in ("timestamp", "is_anomaly") if c in df.columns]
    feat_df = df.drop(columns=drop_cols)
    col_names = [str(c) for c in feat_df.columns]
    arr = feat_df.to_numpy(dtype=np.float32, copy=True)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, labels, col_names


def _slice_windows(series: np.ndarray, stride: int) -> np.ndarray:
    """Create contiguous ``(N_win, WIN_LEN, K)`` sliding windows."""
    n_pts = len(series)
    if n_pts < WIN_LEN:
        raise ValueError(f"Series length {n_pts} is shorter than window length {WIN_LEN}.")
    starts = np.arange(0, n_pts - WIN_LEN + 1, stride, dtype=np.int64)
    idx = starts[:, None] + np.arange(WIN_LEN, dtype=np.int64)[None, :]
    return np.ascontiguousarray(series[idx], dtype=np.float32)


def _find_fault_spans(labels: np.ndarray) -> list[list[int]]:
    """Return contiguous ``[start, end]`` inclusive indices of anomalous runs."""
    spans: list[list[int]] = []
    in_seg = False
    start = 0
    for i, v in enumerate(labels.tolist()):
        if v > 0 and not in_seg:
            in_seg = True
            start = i
        elif v == 0 and in_seg:
            in_seg = False
            spans.append([start, i - 1])
    if in_seg:
        spans.append([start, len(labels) - 1])
    return spans


def _select_test_slice(
    test_x: np.ndarray, test_y: np.ndarray, max_test_steps: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Keep a rich contiguous slice centered on real faults when a test stream is huge."""
    n = len(test_x)
    if n <= max_test_steps:
        return test_x, test_y, 0
    spans = _find_fault_spans(test_y)
    if not spans:
        return test_x[:max_test_steps], test_y[:max_test_steps], 0
    # Center around the first fault with generous healthy pre-context before onset
    first_start = spans[0][0]
    offset = max(0, first_start - max(CONTEXT_LEN + 240, max_test_steps // 4))
    if offset + max_test_steps > n:
        offset = max(0, n - max_test_steps)
    end = offset + max_test_steps
    return test_x[offset:end], test_y[offset:end], int(offset)


def _encode_pre_and_post_barycenter(
    encoder: LaminarLiteEncoder, x_batch: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return both pre-barycenter ``fused`` and final hypersphere ``z`` embeddings."""
    x = x_batch.transpose(1, 2)
    if encoder.channel_stem is not None and encoder.stem_gate is not None and x.shape[-1] >= encoder.stem_kernel:
        stem_out = F.gelu(encoder.channel_stem(F.pad(x, (encoder.stem_kernel - 1, 0))))
        stem_rms = torch.sqrt(torch.mean(stem_out.pow(2), dim=-1, keepdim=True) + 1.0)
        x = x + torch.tanh(encoder.stem_gate) * (stem_out / stem_rms)

    x = encoder.input_projection(x)
    if encoder.dyadic_block is not None:
        x = encoder.dyadic_block(x)
    t0 = x

    mid_idx = max(0, (len(encoder.blocks) - 1) // 3)
    t1 = x
    for idx, block in enumerate(encoder.blocks):
        x = block(x)
        if idx == mid_idx:
            t1 = x
    t2 = x

    d = (encoder.latent_dim,)
    temporal_cat = torch.cat([t2[:, :, -1], torch.max(t2, dim=2).values, torch.mean(t2, dim=2)], dim=1)
    deep_fused = torch.cat(
        [encoder.temporal_dense(temporal_cat), encoder.stat_dense(encoder._statistical_descriptors(t2))], dim=1
    )
    z_deep = F.layer_norm(encoder.bottleneck(deep_fused), d)
    tap_vec = torch.cat([encoder._tap_stats(t0), encoder._tap_stats(t1), encoder._tap_stats(t2)], dim=1)
    z_tap = F.layer_norm(encoder.tap_net(tap_vec), d)
    g = 0.5 * torch.sigmoid(encoder.tap_gate)
    fused = (1.0 - g) * z_deep + g * z_tap

    if bool(encoder.has_centroid.item()):
        centered = fused - encoder.running_centroid.unsqueeze(0)
    else:
        centered = fused
    z_post = torch.nan_to_num(F.layer_norm(centered, d))
    return torch.nan_to_num(fused), z_post


def _compute_causal_metrics(preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute stepwise Point-F1, Event Recall, Test FPR, and mean onset latency."""
    y = (labels > 0).astype(np.int32)
    p = (preds > 0).astype(np.int32)
    tp = int(np.sum((p == 1) & (y == 1)))
    fp = int(np.sum((p == 1) & (y == 0)))
    fn = int(np.sum((p == 0) & (y == 1)))
    tn = int(np.sum((p == 0) & (y == 0)))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    fpr = fp / max(fp + tn, 1)

    spans = _find_fault_spans(y)
    caught = 0
    delays: list[float] = []
    for s0, s1 in spans:
        hits = np.where(p[s0 : s1 + 1] == 1)[0]
        if len(hits) > 0:
            caught += 1
            delays.append(float(hits[0]))
    er = caught / max(len(spans), 1) if spans else 0.0
    delay = float(np.mean(delays)) if delays else -1.0
    return {
        "point_f1": round(f1, 4),
        "event_recall": round(er, 4),
        "test_fpr": round(fpr, 4),
        "mean_delay": round(delay, 1),
        "events_caught": caught,
        "events_total": len(spans),
    }


def _fit_2d_pca(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mean, basis_Dx2)`` for clean 2D projection via SVD."""
    mu = mat.mean(axis=0, keepdims=True)
    xc = mat - mu
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    basis = vt[:2].T.astype(np.float32)  # (D, 2)
    return mu.squeeze(0).astype(np.float32), basis


def run_laminar_lite_on_gpu(
    dataset: str,
    stream: str,
    epochs: int = 10,
    seed: int = 42,
    max_test_steps: int = 2400,
    data_root: Path = DATA_ROOT,
) -> dict[str, Any]:
    """Train ``LaminarJEPALiteModel`` on GPU and extract 5-stage live visualization data."""
    cache_key = (dataset, stream, int(epochs), int(seed))
    if cache_key in _CACHE:
        _CACHE.move_to_end(cache_key)
        cached = dict(_CACHE[cache_key])
        cached["from_cache"] = True
        return cached

    t0_wall = time.perf_counter()
    ds_dir = data_root / dataset
    train_csv = ds_dir / f"{stream}_train.csv"
    test_csv = ds_dir / f"{stream}_test.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Stream '{dataset}/{stream}' not found under {data_root}.")

    train_raw, _, col_names = _read_feature_csv(train_csv)
    test_raw_full, test_y_full, _ = _read_feature_csv(test_csv)
    if test_y_full is None:
        test_y_full = np.zeros(len(test_raw_full), dtype=np.int32)

    # Cap massive nominal prefixes for crisp interactive turnaround while preserving chronology
    if len(train_raw) > 18000:
        train_raw = train_raw[-18000:]

    # Chronological disjoint split: X_fit (52%), X_q (13%), X_cal + X_hold (35%) with 64-step gaps
    n_nom = len(train_raw)
    cut_fit = int(n_nom * 0.52)
    cut_q = int(n_nom * 0.65)
    x_fit_raw = train_raw[:cut_fit]
    x_q_raw = train_raw[min(cut_fit + SUSPECT_LEN, n_nom - WIN_LEN) : max(cut_q, cut_fit + SUSPECT_LEN + WIN_LEN)]
    x_va_raw = train_raw[min(cut_q + SUSPECT_LEN, n_nom - WIN_LEN) :]
    if len(x_q_raw) < WIN_LEN + 8:
        x_q_raw = x_fit_raw[-min(len(x_fit_raw), WIN_LEN + 128) :]
    if len(x_va_raw) < WIN_LEN + 32:
        x_va_raw = train_raw[-min(len(train_raw), WIN_LEN + 256) :]

    # Fit z-score strictly on X_fit
    mu_ch = x_fit_raw.mean(axis=0)
    sd_ch = np.maximum(x_fit_raw.std(axis=0), 1e-2)
    x_fit = ((x_fit_raw - mu_ch) / sd_ch).astype(np.float32)
    x_q = ((x_q_raw - mu_ch) / sd_ch).astype(np.float32)
    x_va = ((x_va_raw - mu_ch) / sd_ch).astype(np.float32)

    test_slice_raw, test_y, test_offset = _select_test_slice(test_raw_full, test_y_full, max_test_steps)
    x_te = ((test_slice_raw - mu_ch) / sd_ch).astype(np.float32)

    n_channels = int(x_fit.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Build real Laminar-JEPA-Lite
    encoder = LaminarLiteEncoder(
        input_dim=n_channels,
        latent_dim=32,
        filters=48,
        tcn_layers=4,
        dyadic_min_channels=25,
    )
    model = LaminarJEPALiteModel(
        context_encoder=encoder,
        latent_dim=32,
        predictor_hidden_dim=64,
        ema_decay=0.996,
        num_horizons=3,
    ).to(device)

    # 2. Train on X_fit stride-16 windows on RTX 5090
    fit_windows = _slice_windows(x_fit, stride=16)
    if len(fit_windows) > 1024:
        fit_windows = fit_windows[-1024:]
    fit_tensor = torch.from_numpy(fit_windows).to(device)
    batch_size = min(64, max(16, len(fit_windows) // 8))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    model.train()
    train_losses: list[float] = []
    for _ in range(epochs):
        perm = torch.randperm(len(fit_tensor), device=device)
        ep_loss = 0.0
        n_batches = 0
        for i in range(0, len(fit_tensor), batch_size):
            batch = fit_tensor[perm[i : i + batch_size]]
            if len(batch) < 4:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.compute_objective(batch[:, :CONTEXT_LEN], batch[:, CONTEXT_LEN:])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_target_encoder()
            ep_loss += float(loss.detach().item())
            n_batches += 1
        scheduler.step()
        train_losses.append(round(ep_loss / max(n_batches, 1), 4))

    # 3. Fit damped ZCA whitening on X_fit
    model.eval()
    model.fit_resolvent_whitening(fit_windows, context_len=CONTEXT_LEN, batch_size=256)

    # 4. Extract residuals & intermediate latents on X_q, X_va (cal+hold), and X_te
    q_windows = _slice_windows(x_q, stride=2)
    va_windows = _slice_windows(x_va, stride=1)
    if len(va_windows) > 2200:
        va_windows = va_windows[-2200:]
    te_windows = _slice_windows(x_te, stride=1)

    q_raw, q_white = model.extract_raw_and_whitened_residuals(q_windows, CONTEXT_LEN, batch_size=256)
    va_raw, va_white = model.extract_raw_and_whitened_residuals(va_windows, CONTEXT_LEN, batch_size=256)
    te_raw, te_white = model.extract_raw_and_whitened_residuals(te_windows, CONTEXT_LEN, batch_size=256)

    n_cal_w = max(32, len(va_windows) // 2)
    va_stab, te_stab = stabilize_whitened_subblocks(
        va_white, te_white, va_raw, te_raw, n_cal_w=n_cal_w, adapt_white4=q_white, adapt_raw4=q_raw
    )

    # 5. Calibrate real 128-dial PositionDriftLitePanel
    panel = PositionDriftLitePanel(context_size=CONTEXT_LEN, suspect_size=SUSPECT_LEN)
    va_pd = build_position_drift_dials(va_stab)
    te_pd = build_position_drift_dials(te_stab)
    tau_dials = np.percentile(va_pd[:n_cal_w], 98.5, axis=0, keepdims=True) + 1e-4  # (1, 128)
    te_dial_ratios = (te_pd / np.maximum(tau_dials, 1e-4)).astype(np.float32)  # > 1.0 => tripped

    panel_bits = panel.calibrate_and_trip_windows(va_stab, te_stab, n_cal_w=n_cal_w)
    decision = panel.evaluate_stream_decisions(
        panel_bits, len_va=len(va_windows) + WIN_LEN - 1, len_test=len(te_windows) + WIN_LEN - 1, mapping="trailing"
    )
    k_lite = float(decision.k_gate_balanced)
    lite_counts = decision.test_switch_counts.astype(np.float32)
    lite_preds = decision.test_pred_balanced.astype(np.int32)
    hold_fpr_lite = float(np.mean(decision.hold_pred_balanced)) if len(decision.hold_pred_balanced) else 0.0

    # 6. Real 1D Multi-Horizon Scalar JEPA + SPOT comparator
    va_1d = np.linalg.norm(va_white.reshape(len(va_white), -1), axis=-1)
    te_1d = np.linalg.norm(te_white.reshape(len(te_white), -1), axis=-1)
    evt = EVTCalibrator(risk_level=1e-3, init_percentile=98.0, adaptive_kurtosis=False).fit(va_1d[:n_cal_w])
    tau_1d = float(evt.threshold_)
    preds_1d = (te_1d > tau_1d).astype(np.int32)
    hold_fpr_1d = float(np.mean(va_1d[n_cal_w + SUSPECT_LEN :] > tau_1d)) if len(va_1d) > n_cal_w + SUSPECT_LEN else 0.0

    # Causal labels aligned with window right edge tau = t + WIN_LEN - 1
    aligned_labels = test_y[WIN_LEN - 1 : WIN_LEN - 1 + len(te_windows)]
    metrics_lite = _compute_causal_metrics(lite_preds, aligned_labels)
    metrics_1d = _compute_causal_metrics(preds_1d, aligned_labels)
    metrics_lite["hold_fpr"] = round(hold_fpr_lite, 4)
    metrics_1d["hold_fpr"] = round(hold_fpr_1d, 4)

    # 7. Extract real pre-barycenter vs. post-barycenter latents + predictions + EMA targets
    with torch.inference_mode():
        # Healthy calibration cloud sample (up to 120 points)
        cal_idx = np.linspace(0, n_cal_w - 1, min(120, n_cal_w), dtype=int)
        cal_batch = torch.from_numpy(va_windows[cal_idx, :CONTEXT_LEN]).to(device)
        cal_pre_t, cal_post_t = _encode_pre_and_post_barycenter(model.context_encoder, cal_batch)
        cal_pre = cal_pre_t.detach().cpu().numpy()
        cal_post = cal_post_t.detach().cpu().numpy()
        barycenter_vec = model.context_encoder.running_centroid.detach().cpu().numpy()

        # Test trajectory tensors (batched on GPU)
        te_pre_list, te_post_list, te_pred_list, te_tgt_list = [], [], [], []
        for i in range(0, len(te_windows), 256):
            b = torch.from_numpy(te_windows[i : i + 256]).to(device)
            z_pre, z_post = _encode_pre_and_post_barycenter(model.context_encoder, b[:, :CONTEXT_LEN])
            z_sub_pred = model.predict_horizons(z_post)
            z_sub_tgt = model.encode_target_prefixes(b[:, CONTEXT_LEN:])
            te_pre_list.append(z_pre.detach().cpu().numpy())
            te_post_list.append(z_post.detach().cpu().numpy())
            te_pred_list.append(z_sub_pred.detach().cpu().numpy())
            te_tgt_list.append(z_sub_tgt.detach().cpu().numpy())

        te_pre = np.concatenate(te_pre_list, axis=0)      # (N_te, 32)
        te_post = np.concatenate(te_post_list, axis=0)    # (N_te, 32)
        te_pred = np.concatenate(te_pred_list, axis=0)    # (N_te, 3, 32)
        te_tgt = np.concatenate(te_tgt_list, axis=0)      # (N_te, 3, 32)

    # Project 32D latent space cleanly onto top-2 principal axes of healthy post-barycenter latents
    _, lat_basis = _fit_2d_pca(cal_post)
    lat_scale = float(np.percentile(np.linalg.norm(cal_post @ lat_basis, axis=1), 95)) + 1e-4

    def proj_lat(arr: np.ndarray) -> np.ndarray:
        return np.round((arr @ lat_basis) / lat_scale, 3)

    bary_2d = np.round((barycenter_vec @ lat_basis) / lat_scale, 3).tolist()
    cal_pre_2d = proj_lat(cal_pre).tolist()
    cal_post_2d = proj_lat(cal_post).tolist()

    # Subsample test frames to <= 600 smooth playback frames (preserving all fault regions)
    n_te = len(te_windows)
    step_sub = max(1, n_te // 600)
    frame_idx = np.arange(0, n_te, step_sub, dtype=int)

    # Project Stage 3 raw vs. ZCA whitened residuals (64-step horizon) onto top-2 raw residual axes
    cal_raw64 = va_raw[:n_cal_w, 2, :]
    cal_wh64 = va_white[:n_cal_w, 2, :]
    raw_mu, raw_basis = _fit_2d_pca(cal_raw64)
    wh_scale = float(np.percentile(np.linalg.norm(cal_wh64 @ raw_basis, axis=1), 90)) + 1e-4
    # Normalize both into comparable visual units so the ellipse->circle morph is immediately clear
    raw_proj_cal = (cal_raw64[cal_idx] - raw_mu) @ raw_basis
    raw_norm = float(np.percentile(np.linalg.norm(raw_proj_cal, axis=1), 90)) + 1e-4
    wh_cloud_raw = np.round(raw_proj_cal / raw_norm, 3).tolist()
    wh_cloud_zca = np.round((cal_wh64[cal_idx] @ raw_basis) / wh_scale, 3).tolist()
    wh_live_raw = np.round(((te_raw[frame_idx, 2, :] - raw_mu) @ raw_basis) / raw_norm, 3).tolist()
    wh_live_zca = np.round((te_white[frame_idx, 2, :] @ raw_basis) / wh_scale, 3).tolist()

    # Pick top 4 most-active channels for Stage 1 waveform display
    ch_var = np.std(x_te, axis=0)
    top_ch = np.argsort(-ch_var)[: min(4, n_channels)].tolist()
    top_ch_names = [col_names[i] if i < len(col_names) else f"Ch {i+1}" for i in top_ch]
    # Telemetry aligned to frame_idx right edges
    te_right_edges = frame_idx + WIN_LEN - 1
    waveforms = [np.round(x_te[te_right_edges, ch], 3).tolist() for ch in top_ch]

    elapsed = round(time.perf_counter() - t0_wall, 2)
    payload: dict[str, Any] = {
        "meta": {
            "dataset": dataset,
            "stream": stream,
            "n_channels": n_channels,
            "n_train": int(n_nom),
            "n_cal_windows": int(n_cal_w),
            "n_test_windows": int(n_te),
            "n_frames": int(len(frame_idx)),
            "frame_stride": int(step_sub),
            "test_offset": int(test_offset),
            "epochs": int(epochs),
            "elapsed_sec": elapsed,
            "train_losses": train_losses,
            "device": get_device_info(),
            "channel_names": top_ch_names,
            "from_cache": False,
        },
        "metrics": {"lite": metrics_lite, "scalar_1d": metrics_1d},
        "thresholds": {"k_lite": round(k_lite, 2), "tau_1d": round(tau_1d, 3)},
        "timeline": {
            "labels": aligned_labels[frame_idx].astype(int).tolist(),
            "fault_spans": _find_fault_spans(aligned_labels[frame_idx]),
            "lite_counts": np.round(lite_counts[frame_idx], 2).tolist(),
            "lite_alarms": lite_preds[frame_idx].astype(int).tolist(),
            "scalar_scores": np.round(te_1d[frame_idx], 3).tolist(),
            "scalar_alarms": preds_1d[frame_idx].astype(int).tolist(),
            "waveforms": waveforms,
        },
        "latent": {
            "barycenter_2d": bary_2d,
            "cal_pre_2d": cal_pre_2d,
            "cal_post_2d": cal_post_2d,
            "te_pre_2d": proj_lat(te_pre[frame_idx]).tolist(),
            "te_post_2d": proj_lat(te_post[frame_idx]).tolist(),
            "te_pred_2d": [proj_lat(te_pred[frame_idx, h, :]).tolist() for h in range(3)],
            "te_tgt_2d": [proj_lat(te_tgt[frame_idx, h, :]).tolist() for h in range(3)],
        },
        "whitening": {
            "cal_raw_2d": wh_cloud_raw,
            "cal_zca_2d": wh_cloud_zca,
            "live_raw_2d": wh_live_raw,
            "live_zca_2d": wh_live_zca,
        },
        "dials": {
            # (N_frames, 128) rounded normalized ratios (> 1.0 means tripped)
            "ratios": np.round(np.clip(te_dial_ratios[frame_idx], 0.0, 3.0), 2).tolist(),
            # Signed stabilized residuals (N_frames, 3, 32) for exact Reynolds split visualization
            "signed_res": np.round(np.clip(te_stab[frame_idx], -4.0, 4.0), 2).tolist(),
        },
    }

    _CACHE[cache_key] = payload
    while len(_CACHE) > _MAX_CACHE:
        _CACHE.popitem(last=False)
    return payload
