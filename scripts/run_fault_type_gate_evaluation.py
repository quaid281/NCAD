#!/usr/bin/env python3
"""Per-Fault-Type Evaluation of the Causal Channel Saliency Gate.

Addresses Reviewer Major Concern 8: the variance-based Causal Coordinate
Saliency Gate can suppress channels with intrinsically low nominal variance and
miss quiet faults (flatlines, stuck sensors, dropouts, low-amplitude drift).

Protocol:
- For each multivariate telemetry stream, train a separate TS-JEPA per gating
  mode so that gating is applied symmetrically at training and inference time,
  matching the deployed pipeline described in the manuscript.
- Gating modes:
    variance      : CausalChannelSaliencyGate(mode='causal_context') - paper gate
    robust_mad    : same parametrization but with MAD-based robust scale shares
    global_var    : static gate from full-training-set variance shares (non-causal control)
    uniform       : no gating (gate-free control)
- Inject controlled fault events into a copy of the test partition:
  flatline, stuck_sensor, dropout, low_amplitude_step (all on the lowest
  context-variance channel) and spike_control (on the highest-variance
  channel, a transient the gate is designed to amplify).
- Report per-fault-type point recall, event recall, nominal FPR, and the
  realized gate weight on the faulted channel before/during each event.

Outputs: reports/fault_type_gate_evaluation.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CSMConfig
from src.data.data_loader import DataLoader
from src.engine.trainer import build_ts_jepa_model
from src.models.geometric_layers import CausalChannelSaliencyGate
from src.scoring.event_fusion import (
    aggregate_window_scores,
    calibrate_evt_threshold,
    _extract_segments,
)

C_LEN, S_LEN = 256, 64
W_LEN = C_LEN + S_LEN
EPOCHS = 10
BATCH = 32
N_EVENTS = 6
EVENT_LEN = 128


class MADChannelGate(torch.nn.Module):
    """Robust-scale variant of the causal channel saliency gate.

    Replaces temporal variance with median absolute deviation (MAD) shares
    computed strictly over the historical context window.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def compute_gate(self, x_ctx: torch.Tensor) -> torch.Tensor:
        B, T, K = x_ctx.shape
        if K <= 1:
            return torch.ones((B, 1, K), device=x_ctx.device, dtype=x_ctx.dtype)
        med = x_ctx.median(dim=1, keepdim=True).values
        mad = (x_ctx - med).abs().median(dim=1).values  # (B, K)
        s = mad / (mad.sum(dim=-1, keepdim=True) + self.eps)
        s_bar = 1.0 / float(K)
        sigma_s = torch.std(s, dim=-1, keepdim=True, unbiased=False) + self.eps
        return torch.sigmoid((s - s_bar) / sigma_s).unsqueeze(1)


def make_gate_fn(mode: str, train_vals: np.ndarray, device: torch.device):
    """Return fn(ctx_b, tgt_b) -> (gated_ctx, gated_tgt, g) for the chosen mode."""
    var_gate = CausalChannelSaliencyGate()
    mad_gate = MADChannelGate()
    K = train_vals.shape[1]
    # Static global gate from pooled training variance shares
    gv = torch.tensor(train_vals.var(axis=0), dtype=torch.float32, device=device)
    s_glob = gv / (gv.sum() + 1e-6)
    s_bar = 1.0 / K
    sigma_g = s_glob.std(unbiased=False) + 1e-6
    g_static = torch.sigmoid((s_glob - s_bar) / sigma_g).view(1, 1, K)

    def apply(ctx_b: torch.Tensor, tgt_b: torch.Tensor):
        if mode == "uniform":
            g = torch.ones((ctx_b.shape[0], 1, K), device=ctx_b.device, dtype=ctx_b.dtype)
        elif mode == "variance":
            g = var_gate.compute_gate(ctx_b)
        elif mode == "robust_mad":
            g = mad_gate.compute_gate(ctx_b)
        elif mode == "global_var":
            g = g_static.expand(ctx_b.shape[0], 1, K)
        else:
            raise ValueError(mode)
        return ctx_b * g, (tgt_b * g if tgt_b is not None else None), g

    return apply


def inject_faults(
    test_scaled: np.ndarray,
    base_labels: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]], int]]:
    """Inject per-fault-type events into copies of the scaled test array.

    Returns dict fault_type -> (arr, labels, event_segments, faulted_channel).
    Quiet faults go on the lowest-variance channel; the spike control goes on
    the highest-variance channel. Events are placed in nominally-labeled,
    mutually non-overlapping regions.
    """
    N, K = test_scaled.shape
    nominal_idx = np.where(base_labels == 0)[0]
    if len(nominal_idx) < W_LEN + 2 * (EVENT_LEN + 64):
        return {}

    # Candidate event starts evenly spaced over nominal region; fall back to
    # fewer events on short or heavily-anomalous test partitions.
    segs: List[Tuple[int, int]] = []
    for n_target in (N_EVENTS, 4, 2):
        starts = np.linspace(W_LEN, N - EVENT_LEN - 32, n_target * 3).astype(int)
        segs = []
        for s in starts:
            e = s + EVENT_LEN
            if base_labels[s:e].any():
                continue
            if segs and s - segs[-1][1] < 64:
                continue
            segs.append((s, e))
            if len(segs) >= n_target:
                break
        if len(segs) >= 2:
            break
    if not segs:
        return {}

    ch_var = test_scaled.var(axis=0)
    # Quiet-fault target: lowest-variance channel that is still ALIVE.
    # argmin alone can select a permanently-dead sensor (variance ~ 0 in both
    # train and test), on which a flatline injection is a no-op.
    live = np.where(ch_var > max(1e-3, 0.01 * ch_var.max()))[0]
    if len(live) == 0:
        return {}
    quiet_ch = int(live[np.argmin(ch_var[live])])
    mid_ch = int(live[np.argsort(ch_var[live])[len(live) // 2]])
    loud_ch = int(np.argmax(ch_var))

    fault_channels = {
        "flatline": quiet_ch,
        "flatline_midvar": mid_ch,
        "stuck_sensor": quiet_ch,
        "dropout": quiet_ch,
        "low_amplitude_step": quiet_ch,
        "spike_control": loud_ch,
    }

    out = {}
    for fault, ch in fault_channels.items():
        arr = test_scaled.copy()
        for s, e in segs:
            if fault.startswith("flatline"):
                arr[s:e, ch] = np.median(test_scaled[max(0, s - 64):s, ch])
            elif fault == "stuck_sensor":
                arr[s:e, ch] = test_scaled[s - 1, ch]
            elif fault == "dropout":
                arr[s:e, ch] = 0.0
            elif fault == "low_amplitude_step":
                arr[s:e, ch] += 1.0
            elif fault == "spike_control":
                arr[s:e, ch] = test_scaled[s:e, ch]
                for i in range(0, EVENT_LEN, 16):
                    arr[s + i : s + i + 4, ch] += 5.0
        labels = np.zeros(N, dtype=int)
        # Real anomalies stay anomalous so nominal-FPR accounting stays honest
        labels[base_labels.astype(bool)] = 1
        for s, e in segs:
            labels[s:e] = 1
        out[fault] = (arr, labels, segs, ch)
    return out


def event_recall(labels: np.ndarray, preds: np.ndarray, segs: List[Tuple[int, int]], grace: int = 32) -> float:
    if not segs:
        return float("nan")
    hit = 0
    for s, e in segs:
        window_preds = preds[s : min(e + grace, len(preds))]
        if window_preds.any():
            hit += 1
    return hit / len(segs)


def evaluate_stream(dataset: str, channel: str, device: torch.device) -> List[dict]:
    ds_dir = ROOT / "mTSBench_data" / dataset
    if channel == "default":
        train_p = ds_dir / f"{dataset}_train.csv"
        if not train_p.exists():
            train_p = ds_dir / f"{dataset}.csv"
        test_p = ds_dir / f"{dataset}_test.csv"
    else:
        train_p = ds_dir / f"{dataset}_{channel}_train.csv"
        if not train_p.exists():
            train_p = ds_dir / f"{channel}_train.csv"
        test_p = ds_dir / f"{dataset}_{channel}_test.csv"
        if not test_p.exists():
            test_p = ds_dir / f"{channel}_test.csv"

    train_df = pd.read_csv(train_p)
    test_df = pd.read_csv(test_p)
    numeric_cols = [c for c in train_df.columns if c not in ["timestamp", "is_anomaly", "label"]]
    lbl_col = [c for c in test_df.columns if c in ["is_anomaly", "label", "labels"]]
    base_labels = test_df[lbl_col[0]].to_numpy().astype(int) if lbl_col else np.zeros(len(test_df), dtype=int)

    train_vals = train_df[numeric_cols].to_numpy(dtype=np.float32)
    test_vals = test_df[numeric_cols].to_numpy(dtype=np.float32)
    mean = train_vals.mean(axis=0, keepdims=True)
    std = np.maximum(train_vals.std(axis=0, keepdims=True), 1e-2)
    train_scaled = (train_vals - mean) / std
    test_scaled = (test_vals - mean) / std
    K = train_scaled.shape[1]
    if K < 2:
        print(f"  Skipping {dataset}:{channel} (univariate)")
        return []

    tr_windows = DataLoader.create_windows(train_scaled, W_LEN, step=10, copy=False)
    n_tr = int(len(tr_windows) * 0.8)
    fit_windows, val_windows = tr_windows[:n_tr], tr_windows[n_tr:]

    rng = np.random.default_rng(42)
    fault_pack = inject_faults(test_scaled, base_labels, rng)
    if not fault_pack:
        print(f"  Skipping {dataset}:{channel} (no room for fault injection)")
        return []

    records: List[dict] = []
    for mode in ["variance", "robust_mad", "global_var", "uniform"]:
        print(f"  Training TS-JEPA with gate_mode={mode}...", flush=True)
        torch.manual_seed(42)
        np.random.seed(42)
        gate_fn = make_gate_fn(mode, train_scaled, device)
        cfg = CSMConfig(model_type="ts_jepa", context_size=C_LEN, suspect_size=S_LEN,
                        latent_dim=32, filters=48, tcn_layers=3)
        model = build_ts_jepa_model(cfg, input_dim=K, device=device)
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        for ep in range(EPOCHS):
            model.train()
            perm = np.random.permutation(len(fit_windows))
            for b in range(0, len(perm), BATCH):
                batch = torch.from_numpy(np.ascontiguousarray(fit_windows[perm[b:b + BATCH]])).float().to(device)
                ctx, tgt = batch[:, :C_LEN], batch[:, C_LEN:]
                ctx_g, tgt_g, _ = gate_fn(ctx, tgt)
                loss, _ = model.compute_objective(ctx_g, tgt_g, cfg)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                model.update_target_encoder()

        # Calibrate SPOT threshold on gated nominal validation windows
        model.eval()
        def score_windows(w_arr: np.ndarray, return_gates: bool = False):
            scores, gates = [], []
            with torch.no_grad():
                for i in range(0, len(w_arr), 256):
                    chunk = torch.from_numpy(np.ascontiguousarray(w_arr[i:i + 256])).float().to(device)
                    ctx, tgt = chunk[:, :C_LEN], chunk[:, C_LEN:]
                    ctx_g, tgt_g, g = gate_fn(ctx, tgt)
                    disc = model.compute_predictive_discrepancy(ctx_g, tgt_g)
                    scores.append(disc.cpu().numpy())
                    if return_gates:
                        gates.append(g.squeeze(1).cpu().numpy())
            out = np.concatenate(scores)
            if return_gates:
                return out, np.concatenate(gates)
            return out

        val_scores = score_windows(val_windows)
        evt = calibrate_evt_threshold(val_scores, risk_level=1e-3, init_percentile=98.0, adaptive_kurtosis=True)
        thresh = evt.threshold

        # Evaluate each fault type
        for fault, (arr, labels, segs, ch) in fault_pack.items():
            win = DataLoader.create_windows(arr, W_LEN, step=1, copy=False)
            w_scores, w_gates = score_windows(win, return_gates=True)
            pt_scores, valid_mask = aggregate_window_scores(
                w_scores, n_points=len(arr), context_size=C_LEN, suspect_size=S_LEN,
                step=1, reducer="mean", mapping_method="middle",
            )
            preds = (pt_scores > thresh).astype(int) * valid_mask

            nominal_mask = (labels == 0) & (valid_mask > 0)
            fp = int(((preds > 0) & nominal_mask).sum())
            fpr = fp / max(int(nominal_mask.sum()), 1)

            fault_mask = np.zeros(len(labels), dtype=bool)
            for s, e in segs:
                fault_mask[s:e] = True
            fault_pts = int((fault_mask & (valid_mask > 0)).sum())
            tp = int(((preds > 0) & fault_mask).sum())
            pt_rec = tp / max(fault_pts, 1)
            ev_rec = event_recall(labels, preds, segs)

            # Realized gate weight on the faulted channel inside vs outside events
            # Window i covers time [i, i + W_LEN): map event membership by window overlap
            win_starts = np.arange(len(w_scores))
            win_ends = win_starts + W_LEN
            in_evt = np.zeros(len(w_scores), dtype=bool)
            for s, e in segs:
                in_evt |= (win_starts < e) & (win_ends > s)
            g_in = float(w_gates[in_evt, ch].mean()) if in_evt.any() else float("nan")
            g_out = float(w_gates[~in_evt, ch].mean()) if (~in_evt).any() else float("nan")

            records.append({
                "dataset": dataset, "channel": channel, "gate_mode": mode,
                "fault_type": fault, "faulted_channel": ch, "n_events": len(segs),
                "point_recall": pt_rec, "event_recall": ev_rec,
                "nominal_fpr": fpr, "threshold": thresh,
                "gate_w_faulted_in_event": g_in, "gate_w_faulted_out_event": g_out,
            })
            print(f"    {mode:>10s} | {fault:<18s} pt_rec={pt_rec:.3f} ev_rec={ev_rec:.3f} "
                  f"fpr={fpr:.4f} g_in={g_in:.3f} g_out={g_out:.3f}", flush=True)

    return records


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fault-type gate evaluation on {device}")

    streams = [
        ("SMD", "machine-1-2"),
        ("MSL", "M-1"),
        ("SMAP", "P-3"),
        ("swan", "sf"),
        ("GECCO", "water_quality"),
        ("metro", "traffic-volume"),
    ]

    out_csv = ROOT / "reports" / "fault_type_gate_evaluation.csv"
    all_records: List[dict] = []
    for ds, ch in streams:
        print(f"\n=== {ds}:{ch} ===")
        try:
            all_records.extend(evaluate_stream(ds, ch, device))
        except Exception as e:
            print(f"FAILED {ds}:{ch}: {e}")
        pd.DataFrame(all_records).to_csv(out_csv, index=False)

    df = pd.DataFrame(all_records)
    df.to_csv(out_csv, index=False)
    print("\n=== Summary: per-fault recall by gate mode ===")
    print(df.groupby(["gate_mode", "fault_type"])[["point_recall", "event_recall", "nominal_fpr"]].mean().to_string())
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
