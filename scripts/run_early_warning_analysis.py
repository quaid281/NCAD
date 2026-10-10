#!/usr/bin/env python3
"""Early-Warning Lead-Time Analysis (Buffered Predictive Monitoring).

Addresses Reviewer Major Concern 9: the manuscript claims predictive early
warning, but the review demanded three separately reported outcomes. This
script quantifies the EARLY-WARNING outcome that is distinct from both
strictly-causal online detection and standard buffered detection:

  - Fraction of ground-truth events warned about BEFORE physical onset
  - Lead-time distribution (alarm time minus event onset; positive = early)
  - False warnings per 1,000 nominal operating steps

Because the buffered score at time t aggregates evidence over the suspect
window [t, t+S), an alarm can appear up to ~S steps before the labeled onset.
We attribute an alarm to an event if it falls within [onset - SEARCH_BACK,
event_end + GRACE], where SEARCH_BACK = C + S captures the full buffered
context horizon.

Models: TS-JEPA, TimesNet (same training protocol as the causal/buffered
experiment, 15 epochs). Calibration: SPOT on held-out 20% nominal validation.

Output: reports/early_warning_leadtime.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CSMConfig
from src.data.data_loader import DataLoader
from src.engine.trainer import build_ts_jepa_model, split_train_validation as split_train_val
from src.models.baselines import TimesNet
from src.scoring.event_fusion import (
    aggregate_window_scores,
    calibrate_evt_threshold,
    event_level_filter,
    moving_average,
    positive_robust_z,
    robust_stats,
    _extract_segments,
)

C_LEN, S_LEN = 256, 64
W_LEN = C_LEN + S_LEN
EPOCHS = 15
SEARCH_BACK = C_LEN + S_LEN  # alarms this far before onset can be early warnings
GRACE = 64


def leadtime_stats(labels: np.ndarray, preds: np.ndarray) -> dict:
    """Per-event lead time and warning statistics for buffered scoring."""
    gt_segs = _extract_segments(labels)
    leads, delays, warned, late, missed = [], [], 0, 0, 0
    attributed_alarms = np.zeros(len(preds), dtype=bool)

    for gs, ge in gt_segs:
        lo = max(0, gs - SEARCH_BACK)
        hi = min(len(preds), ge + GRACE)
        alarm_idx = np.where(preds[lo:hi] > 0)[0]
        if len(alarm_idx) == 0:
            missed += 1
            continue
        first = lo + int(alarm_idx[0])
        attributed_alarms[lo:hi] |= preds[lo:hi] > 0
        if first < gs:
            warned += 1
            leads.append(float(gs - first))
        else:
            late += 1
            delays.append(float(first - gs))

    # False warnings: alarms not attributable to any event window
    nominal_pts = int(np.sum(labels == 0))
    false_alarms = int(((preds > 0) & ~attributed_alarms & (labels == 0)).sum())

    return {
        "total_events": len(gt_segs),
        "warned_pre_onset": warned,
        "detected_late": late,
        "missed": missed,
        "frac_warned_pre_onset": warned / max(len(gt_segs), 1),
        "frac_detected_any": (warned + late) / max(len(gt_segs), 1),
        "lead_time_mean": float(np.mean(leads)) if leads else float("nan"),
        "lead_time_median": float(np.median(leads)) if leads else float("nan"),
        "lead_time_max": float(np.max(leads)) if leads else float("nan"),
        "post_onset_delay_mean": float(np.mean(delays)) if delays else float("nan"),
        "false_warnings_per_1k": false_alarms / max(nominal_pts, 1) * 1000.0,
    }


def load_stream(dataset: str, channel: str):
    ds_dir = ROOT / "mTSBench_data" / dataset
    if channel == "default":
        train_path = ds_dir / f"{dataset}_train.csv"
        if not train_path.exists():
            train_path = ds_dir / f"{dataset}.csv"
        test_path = ds_dir / f"{dataset}_test.csv"
    else:
        train_path = ds_dir / f"{dataset}_{channel}_train.csv"
        if not train_path.exists():
            train_path = ds_dir / f"{channel}_train.csv"
        test_path = ds_dir / f"{dataset}_{channel}_test.csv"
        if not test_path.exists():
            test_path = ds_dir / f"{channel}_test.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    numeric_cols = [c for c in train_df.columns if c not in ["timestamp", "is_anomaly", "label"]]
    lbl_col = [c for c in test_df.columns if c in ["is_anomaly", "label", "labels"]]
    labels = test_df[lbl_col[0]].to_numpy().astype(int) if lbl_col else np.zeros(len(test_df), dtype=int)
    train_vals = train_df[numeric_cols].to_numpy(dtype=np.float32)
    test_vals = test_df[numeric_cols].to_numpy(dtype=np.float32)
    mean = train_vals.mean(axis=0)
    std = np.where(train_vals.std(axis=0) < 1e-8, 1.0, np.maximum(train_vals.std(axis=0), 0.01))
    return (train_vals - mean) / std, (test_vals - mean) / std, labels, len(train_df), len(test_df)


def evaluate(dataset: str, channel: str, device: torch.device) -> list[dict]:
    train_scaled, test_scaled, test_labels, n_tr_pts, n_te_pts = load_stream(dataset, channel)
    train_windows = DataLoader.create_windows(train_scaled, W_LEN, step=10, copy=False)
    test_windows = DataLoader.create_windows(test_scaled, W_LEN, step=1, copy=False)
    fit_windows, val_windows = split_train_val(train_windows, val_split=0.2, seed=42,
                                             window_size=W_LEN, step=10)
    input_dim = train_scaled.shape[1]
    records = []

    for m_name in ["TS-JEPA", "TimesNet"]:
        print(f"  Training {m_name}...", flush=True)
        torch.manual_seed(42)
        np.random.seed(42)
        if m_name == "TS-JEPA":
            cfg = CSMConfig(model_type="ts_jepa", context_size=C_LEN, suspect_size=S_LEN, latent_dim=32)
            model = build_ts_jepa_model(cfg, input_dim=input_dim, device=device)
            opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for ep in range(EPOCHS):
                model.train()
                perm = np.random.permutation(len(fit_windows))
                for b in range(0, len(perm), 16):
                    batch = torch.from_numpy(np.ascontiguousarray(fit_windows[perm[b:b + 16]])).float().to(device)
                    loss, _ = model.compute_objective(batch[:, :C_LEN], batch[:, C_LEN:], cfg)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    model.update_target_encoder()

            def get_scores(w_arr):
                model.eval()
                res = []
                with torch.no_grad():
                    for i in range(0, len(w_arr), 128):
                        chunk = torch.from_numpy(np.ascontiguousarray(w_arr[i:i + 128])).float().to(device)
                        res.append(model.compute_predictive_discrepancy(chunk[:, :C_LEN], chunk[:, C_LEN:]).cpu().numpy())
                return np.concatenate(res)
        else:
            model = TimesNet(c_in=input_dim, d_model=48, d_ff=96, e_layers=2).to(device)
            opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for ep in range(EPOCHS):
                model.train()
                perm = np.random.permutation(len(fit_windows))
                for b in range(0, len(perm), 16):
                    batch = torch.from_numpy(np.ascontiguousarray(fit_windows[perm[b:b + 16]])).float().to(device)
                    loss = nn.functional.mse_loss(model(batch), batch)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

            def get_scores(w_arr):
                model.eval()
                res = []
                with torch.no_grad():
                    for i in range(0, len(w_arr), 128):
                        chunk = torch.from_numpy(np.ascontiguousarray(w_arr[i:i + 128])).float().to(device)
                        sc = model.compute_anomaly_scores(chunk)
                        res.append(sc[:, C_LEN:].mean(dim=-1).cpu().numpy())
                return np.concatenate(res)

        # Buffered (smear) scoring: score at t aggregates suspect window [t, t+S)
        train_dense = DataLoader.create_windows(train_scaled, W_LEN, step=1, copy=False)
        train_w_sc = get_scores(train_dense)
        test_w_sc = get_scores(test_windows)
        stats = robust_stats(train_w_sc)
        train_z = positive_robust_z(train_w_sc, stats)
        test_z = positive_robust_z(test_w_sc, stats)

        pt_te, val_te = aggregate_window_scores(test_z, n_points=n_te_pts, context_size=C_LEN,
                                              suspect_size=S_LEN, step=1, reducer="mean",
                                              mapping_method="smear")
        pt_tr, val_tr = aggregate_window_scores(train_z, n_points=n_tr_pts, context_size=C_LEN,
                                              suspect_size=S_LEN, step=1, reducer="mean",
                                              mapping_method="smear")
        pt_te = moving_average(pt_te, 5)
        pt_tr = moving_average(pt_tr, 5)
        evt = calibrate_evt_threshold(pt_tr[val_tr.astype(bool)], risk_level=1e-3,
                                      init_percentile=98.0, adaptive_kurtosis=True)
        preds = event_level_filter(pt_te, evt.threshold, val_te, min_run=2, extreme_factor=1.75) * val_te

        stats_lt = leadtime_stats(test_labels, preds)
        records.append({"dataset": dataset, "channel": channel, "model": m_name,
                        "evaluation": "Buffered Early-Warning", **stats_lt})
        print(f"    {m_name}: warned {stats_lt['warned_pre_onset']}/{stats_lt['total_events']} pre-onset, "
              f"median lead {stats_lt['lead_time_median']:.0f} steps, "
              f"false-warn/1k={stats_lt['false_warnings_per_1k']:.1f}", flush=True)
    return records


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Early-warning lead-time analysis on {device}")
    streams = [
        ("SMAP", "P-3"),
        ("SMD", "machine-1-2"),
        ("MSL", "M-1"),
        ("Daphnet", "S01R01E1"),
        ("GECCO", "water_quality"),
    ]
    out_csv = ROOT / "reports" / "early_warning_leadtime.csv"
    rows = []
    for ds, ch in streams:
        print(f"\n=== {ds}:{ch} ===")
        try:
            rows.extend(evaluate(ds, ch, device))
        except Exception as e:
            print(f"FAILED {ds}:{ch}: {e}")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("\n=== Summary ===")
    print(df.groupby("model")[["frac_warned_pre_onset", "lead_time_median",
                               "post_onset_delay_mean", "false_warnings_per_1k"]].mean().to_string())


if __name__ == "__main__":
    main()
