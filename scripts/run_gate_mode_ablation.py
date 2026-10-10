#!/usr/bin/env python3
"""Channel-Gating Regime Ablation on Real Anomaly Benchmarks.

Re-derives the manuscript's Table ablations(b) numbers under an explicit,
reproducible protocol identical to the dyadic front-end ablation
(run_dyadic_frontend_ablation.py): TSJEPAModel on a 3-layer HybridTCNEncoder
(dyadic shells on), 80/20 chronological validation split, SPOT calibration
(q=1e-3, u=p98) on window-level validation scores, middle-mapped point scores.

The channel variance gate is NOT part of the deployed pipeline; it is
evaluated here as an auxiliary input-level variant applied symmetrically at
training and scoring time.

Gating regimes (CausalChannelSaliencyGate modes):
  causal_context : g from x_ctx applied to ctx and tgt (zero-leakage)
  target_only    : g from x_tgt (leaky control)
  independent    : separate gates for ctx and tgt
  uniform        : no gating (deployed pipeline equivalent)

Output: reports/gate_mode_ablation.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CSMConfig
from src.data.data_loader import DataLoader
from src.models.encoders.tcn_encoder import HybridTCNEncoder
from src.models.geometric_layers import CausalChannelSaliencyGate
from src.models.jepa.ts_jepa import TSJEPAModel
from src.scoring.event_fusion import (
    aggregate_window_scores,
    calibrate_evt_threshold,
    compute_metrics,
)
from scripts.run_controlled_backbone_experiment import load_data

C_LEN, S_LEN = 256, 64
W_LEN = C_LEN + S_LEN
EPOCHS = 10
MODES = ["causal_context", "target_only", "independent", "uniform"]


def build_model(input_dim: int, device: torch.device) -> TSJEPAModel:
    encoder = HybridTCNEncoder(
        input_dim=input_dim, latent_dim=32, filters=48, tcn_layers=3,
        use_dyadic_shells=True,
    ).to(device)
    return TSJEPAModel(context_encoder=encoder, latent_dim=32,
                       predictor_hidden_dim=64, predictor_layers=2,
                       ema_decay=0.996, dropout=0.1).to(device)


def run(dataset: str, channel: str, device: torch.device) -> List[dict]:
    train_norm, test_norm, labels = load_data(dataset, channel)
    input_dim = train_norm.shape[1]
    if input_dim < 2:
        print(f"  Skipping {dataset}:{channel} (univariate)")
        return []

    all_tr = DataLoader.create_windows(train_norm, W_LEN, step=10, copy=False)
    n_tr = int(len(all_tr) * 0.8)
    fit_win, val_win = all_tr[:n_tr], all_tr[n_tr:]
    test_win = DataLoader.create_windows(test_norm, W_LEN, step=1, copy=False)

    gate = CausalChannelSaliencyGate()
    rows = []
    for mode in MODES:
        print(f"  Training TS-JEPA gate_mode={mode}...", flush=True)
        torch.manual_seed(42)
        np.random.seed(42)
        cfg = CSMConfig(model_type="ts_jepa", context_size=C_LEN, suspect_size=S_LEN,
                        latent_dim=32)
        model = build_model(input_dim, device)
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        def apply_gate(ctx_b, tgt_b):
            if mode == "uniform":
                return ctx_b, tgt_b
            c_g, t_g, _ = gate(ctx_b, tgt_b, mode=mode)
            return c_g, t_g

        for ep in range(EPOCHS):
            model.train()
            perm = np.random.permutation(len(fit_win))
            for b in range(0, len(perm), 32):
                batch = torch.from_numpy(np.ascontiguousarray(fit_win[perm[b:b + 32]])).float().to(device)
                ctx, tgt = batch[:, :C_LEN], batch[:, C_LEN:]
                ctx_g, tgt_g = apply_gate(ctx, tgt)
                loss, _ = model.compute_objective(ctx_g, tgt_g, cfg)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                model.update_target_encoder()

        def get_scores(w_arr):
            model.eval()
            out = []
            with torch.no_grad():
                for i in range(0, len(w_arr), 256):
                    chunk = torch.from_numpy(np.ascontiguousarray(w_arr[i:i + 256])).float().to(device)
                    ctx_g, tgt_g = apply_gate(chunk[:, :C_LEN], chunk[:, C_LEN:])
                    out.append(model.compute_predictive_discrepancy(ctx_g, tgt_g).cpu().numpy())
            return np.concatenate(out)

        val_res = get_scores(val_win)
        evt = calibrate_evt_threshold(val_res, risk_level=1e-3, init_percentile=98.0,
                                      adaptive_kurtosis=True)
        test_res = get_scores(test_win)
        pt, mask = aggregate_window_scores(test_res, n_points=len(test_norm),
                                         context_size=C_LEN, suspect_size=S_LEN,
                                         step=1, reducer="mean", mapping_method="middle")
        preds = (pt > evt.threshold).astype(int)
        m = compute_metrics(labels, preds, pt)
        fp, tn = m.get("fp", 0), m.get("tn", 0)

        rows.append({
            "dataset": dataset, "channel": channel, "gate_mode": mode,
            "point_f1_buffered": m.get("f1", 0.0),
            "point_precision": m.get("precision", 0.0),
            "point_recall": m.get("recall", 0.0),
            "heldout_fpr": fp / max(fp + tn, 1),
            "event_recall": m.get("event_recall", 0.0),
        })
        print(f"    {mode}: F1={rows[-1]['point_f1_buffered']:.4f} "
              f"FPR={rows[-1]['heldout_fpr']:.4f} ev_rec={rows[-1]['event_recall']:.3f}",
              flush=True)
    return rows


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Gate-mode ablation on {device}")
    streams = [
        ("MSL", "M-1"),
        ("SMAP", "P-3"),
        ("GHL", "17_Lev_fault_Temp_corr_seed_57_vars_23"),
        ("Daphnet", "S01R01E1"),
    ]
    out_csv = ROOT / "reports" / "gate_mode_ablation.csv"
    rows: List[dict] = []
    for ds, ch in streams:
        print(f"\n=== {ds}:{ch} ===")
        try:
            rows.extend(run(ds, ch, device))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"FAILED {ds}:{ch}: {e}")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("\n=== Summary (macro mean) ===")
    print(df.groupby("gate_mode")[["point_f1_buffered", "heldout_fpr", "event_recall"]].mean().to_string())


if __name__ == "__main__":
    main()
