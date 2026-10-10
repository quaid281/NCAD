#!/usr/bin/env python3
"""Dyadic Frequency Front-End Ablation (Reviewer Major Concern 7).

The reviewer asked for "an ablation without the frequency-processing
front-end". This script trains TS-JEPA with the identical HybridTCNEncoder
backbone under two configurations:

  dyadic_on  : use_dyadic_shells=True  (LittlewoodPaleyDyadicBlock active)
  dyadic_off : use_dyadic_shells=False (plain causal TCN front-end)

Same protocol as the matched-backbone experiment: 80/20 chronological
validation split, SPOT calibration (q=1e-3, u=p98), Point-F1/PR-AUC/FPR at
clean SNR. 10 epochs, seed 42.

Output: reports/dyadic_frontend_ablation.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

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
from src.models.jepa.ts_jepa import TSJEPAModel
from src.scoring.event_fusion import (
    aggregate_window_scores,
    calibrate_evt_threshold,
    compute_metrics,
)
from scripts.run_controlled_backbone_experiment import load_data, compute_tail_stats
from sklearn.metrics import average_precision_score

C_LEN, S_LEN = 256, 64
W_LEN = C_LEN + S_LEN
EPOCHS = 10


def build_variant(input_dim: int, use_dyadic: bool, device: torch.device) -> TSJEPAModel:
    encoder = HybridTCNEncoder(
        input_dim=input_dim, latent_dim=32, filters=48, tcn_layers=3,
        use_dyadic_shells=use_dyadic,
    ).to(device)
    return TSJEPAModel(context_encoder=encoder, latent_dim=32,
                       predictor_hidden_dim=64, predictor_layers=2,
                       ema_decay=0.996, dropout=0.1).to(device)


def run(dataset: str, channel: str, device: torch.device) -> list[dict]:
    train_norm, test_norm, labels = load_data(dataset, channel)
    input_dim = train_norm.shape[1]

    all_tr = DataLoader.create_windows(train_norm, W_LEN, step=10, copy=False)
    n_tr = int(len(all_tr) * 0.8)
    fit_win, val_win = all_tr[:n_tr], all_tr[n_tr:]
    test_win = DataLoader.create_windows(test_norm, W_LEN, step=1, copy=False)

    rows = []
    for variant in ["dyadic_on", "dyadic_off"]:
        print(f"  Training {variant}...", flush=True)
        torch.manual_seed(42)
        np.random.seed(42)
        model = build_variant(input_dim, use_dyadic=(variant == "dyadic_on"), device=device)
        n_params = sum(p.numel() for p in model.parameters())
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        cfg = CSMConfig(model_type="ts_jepa", context_size=C_LEN, suspect_size=S_LEN, latent_dim=32)

        for ep in range(EPOCHS):
            model.train()
            perm = np.random.permutation(len(fit_win))
            for b in range(0, len(perm), 32):
                batch = torch.from_numpy(np.ascontiguousarray(fit_win[perm[b:b + 32]])).float().to(device)
                loss, _ = model.compute_objective(batch[:, :C_LEN], batch[:, C_LEN:], cfg)
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
                    out.append(model.compute_predictive_discrepancy(chunk[:, :C_LEN], chunk[:, C_LEN:]).cpu().numpy())
            return np.concatenate(out)

        val_res = get_scores(val_win)
        evt = calibrate_evt_threshold(val_res, risk_level=1e-3, init_percentile=98.0, adaptive_kurtosis=True)
        test_res = get_scores(test_win)
        pt, mask = aggregate_window_scores(test_res, n_points=len(test_norm), context_size=C_LEN,
                                         suspect_size=S_LEN, step=1, reducer="mean",
                                         mapping_method="middle")
        preds = (pt > evt.threshold).astype(int)
        m = compute_metrics(labels, preds, pt)
        fp, tn = m.get("fp", 0), m.get("tn", 0)
        valid = mask > 0
        pr_auc = float(average_precision_score(labels[valid.astype(bool)], pt[valid.astype(bool)])) \
            if labels[valid.astype(bool)].sum() > 0 else float("nan")

        rows.append({
            "dataset": dataset, "channel": channel, "variant": variant,
            "n_params": n_params,
            "point_f1": m.get("f1", 0.0), "point_precision": m.get("precision", 0.0),
            "point_recall": m.get("recall", 0.0), "pr_auc": pr_auc,
            "empirical_fpr": fp / max(fp + tn, 1), "fp": fp,
            "val_kurtosis": compute_tail_stats(val_res)["kurtosis"],
        })
        print(f"    {variant}: F1={rows[-1]['point_f1']:.4f} FPR={rows[-1]['empirical_fpr']:.4f} "
              f"PR-AUC={pr_auc:.4f}", flush=True)
    return rows


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dyadic front-end ablation on {device}")
    streams = [
        ("SMAP", "P-3"),
        ("MSL", "M-1"),
        ("SMD", "machine-1-2"),
        ("Daphnet", "S01R01E1"),
        ("GECCO", "water_quality"),
    ]
    out_csv = ROOT / "reports" / "dyadic_frontend_ablation.csv"
    rows = []
    for ds, ch in streams:
        print(f"\n=== {ds}:{ch} ===")
        try:
            rows.extend(run(ds, ch, device))
        except Exception as e:
            print(f"FAILED {ds}:{ch}: {e}")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("\n=== Summary ===")
    print(df.groupby("variant")[["point_f1", "point_recall", "pr_auc", "empirical_fpr", "val_kurtosis"]].mean().to_string())


if __name__ == "__main__":
    main()
