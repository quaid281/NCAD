#!/usr/bin/env python3
"""Evaluate Soliton-Reynolds 224-Bit Breaker and the Hybrid Union Monitor on the 6-stream single-channel fault-injection suite.
Cohort 3: 6 streams, seed 42, 10 epochs, batch 32, gradient clipping 1.0, Eq. (3) blend applied.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CSMConfig
from src.data.data_loader import DataLoader
from src.engine.trainer import build_encoder
from src.models.jepa.alien_jepas import SolitonReynoldsJEPAModel
from src.scoring.circuit_breaker_panel import (
    MultiPerspectiveCircuitBreakerPanel,
    extract_multihorizon_4subblocks,
    stabilize_whitened_subblocks,
)
from scripts.run_fault_type_gate_evaluation import inject_faults, event_recall, C_LEN, S_LEN, W_LEN
from scripts.run_fault_type_reference_detector import rolling_std, channel_lowvar_scores


def eval_stream(dataset: str, channel: str, device: torch.device) -> list[dict]:
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
    n_val = int(len(train_vals) * 0.35)
    tr_raw, va_raw = train_vals[:-n_val], train_vals[-n_val:]
    mean = tr_raw.mean(axis=0, keepdims=True)
    std = np.maximum(tr_raw.std(axis=0, keepdims=True), 1e-2)
    tr_s = (tr_raw - mean) / std
    va_s = (va_raw - mean) / std
    te_s = (test_vals - mean) / std
    K = tr_s.shape[1]

    rng = np.random.default_rng(42)
    fault_pack = inject_faults(te_s, base_labels, rng)
    if not fault_pack:
        return []

    tr_w = DataLoader.create_windows(tr_s, W_LEN, step=16, copy=False)
    va_w = DataLoader.create_windows(va_s, W_LEN, step=1, copy=False)

    torch.manual_seed(42)
    np.random.seed(42)
    cfg = CSMConfig(model_type="soliton_reynolds_jepa", context_size=C_LEN, suspect_size=S_LEN, latent_dim=32)
    enc = build_encoder(cfg, input_dim=K, device=device)
    model = SolitonReynoldsJEPAModel(context_encoder=enc, latent_dim=32).to(device)
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)

    for _ in range(10):
        model.train()
        perm = np.random.permutation(len(tr_w))
        for i in range(0, len(perm), 32):
            b = torch.from_numpy(np.ascontiguousarray(tr_w[perm[i : i + 32]])).float().to(device)
            if len(b) < 2:
                continue
            loss, _ = model.compute_objective(b[:, :C_LEN], b[:, C_LEN:])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.update_target_encoder()

    model.eval()
    model.fit_alien_bank(tr_w, context_len=C_LEN, batch_size=256)
    va_raw4, va_white4 = extract_multihorizon_4subblocks(model, va_w, C_LEN, device)
    n_cal_w = len(va_white4) // 2

    # Rolling variance-collapse monitor calibrated on va_s[:len(va_s)//2]
    rstd_tr = rolling_std(tr_s)
    rstd_med = np.median(rstd_tr, axis=0)
    rstd_mad = np.median(np.abs(rstd_tr - rstd_med), axis=0)
    cal_lv = channel_lowvar_scores(va_s[: len(va_s) // 2], rstd_med, rstd_mad)
    th_lv = float(np.quantile(cal_lv, 0.999))

    panel = MultiPerspectiveCircuitBreakerPanel(context_size=C_LEN, suspect_size=S_LEN)
    rows = []
    for fault, (arr, labels, segs, ch) in fault_pack.items():
        te_w = DataLoader.create_windows(arr, W_LEN, step=1, copy=False)
        te_raw4, te_white4 = extract_multihorizon_4subblocks(model, te_w, C_LEN, device)
        va_ew, te_ew = stabilize_whitened_subblocks(va_white4, te_white4, va_raw4, te_raw4, n_cal_w)
        bits = panel.calibrate_and_trip_windows(va_ew, te_ew, n_cal_w)
        dec = panel.evaluate_stream_decisions(bits, len(va_s), len(arr), "smear")

        preds_224 = np.zeros(len(arr), dtype=int)
        preds_224[dec.valid_mask] = dec.test_pred_balanced

        sc_lv = channel_lowvar_scores(arr, rstd_med, rstd_mad)
        preds_lv = (sc_lv > th_lv).astype(int)
        preds_hyb = np.maximum(preds_224, preds_lv)

        for mname, prd in [
            ("SolRey_224Bit_Breaker", preds_224),
            ("Var_Collapse_Monitor", preds_lv),
            ("Hybrid_224Bit_Union", preds_hyb),
        ]:
            nominal_mask = (labels == 0) & dec.valid_mask
            fpr = float(((prd > 0) & nominal_mask).sum() / max(nominal_mask.sum(), 1))
            fault_mask = np.zeros(len(labels), dtype=bool)
            for s, e in segs:
                fault_mask[s:e] = True
            pt_rec = float(((prd > 0) & fault_mask).sum() / max(fault_mask.sum(), 1))
            ev_rec = event_recall(labels, prd, segs)
            rows.append(dict(
                stream=f"{dataset}:{channel}", detector=mname, fault_type=fault,
                event_recall=ev_rec, point_recall=pt_rec, nominal_fpr=fpr,
            ))
    return rows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    streams = [
        ("SMD", "machine-1-2"),
        ("MSL", "M-1"),
        ("SMAP", "P-3"),
        ("swan", "sf"),
        ("GECCO", "water_quality"),
        ("metro", "traffic-volume"),
    ]
    rows = []
    for ds, ch in streams:
        rows.extend(eval_stream(ds, ch, device))
        print(f"Done {ds}:{ch}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "reports" / "quiet_fault_224bit_evaluation.csv", index=False)
    print("\n=== QUIET-FAULT EVALUATION WITH SOLITON-REYNOLDS 224-BIT BREAKER ===")
    print(df.groupby(["detector", "fault_type"])[["event_recall", "point_recall", "nominal_fpr"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
