#!/usr/bin/env python3
"""Run Dyadic Frequency Front-End Ablation (use_dyadic_shells=True vs False)
on SolitonReynoldsJEPAModel + 224-Bit Circuit-Breaker Panel across 5 streams x 3 seeds (42, 123, 456).

Cohort 4: 5 streams x 3 seeds, 12 epochs, batch 32, gradient clipping 1.0, Eq. (3) blend applied.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.data_loader import DataLoader
from src.models.encoders.tcn_encoder import HybridTCNEncoder
from src.models.jepa.alien_jepas import SolitonReynoldsJEPAModel
from src.scoring.circuit_breaker_panel import (
    MultiPerspectiveCircuitBreakerPanel,
    extract_multihorizon_4subblocks,
    stabilize_whitened_subblocks,
)
from src.scoring.event_fusion import compute_metrics
from scripts.run_multiseed_heldout_benchmark import load_stream, event_stats

C_LEN, S_LEN = 256, 64
W_LEN = C_LEN + S_LEN
EPOCHS = 12


def evaluate_stream_seed(ds: str, ch: str, seed: int, use_dyadic: bool, device: torch.device) -> list[dict]:
    train, test, y = load_stream(ds, ch)
    n_val = int(len(train) * 0.35)
    tr, va = train[:-n_val], train[-n_val:]
    mu, sd_arr = tr.mean(0), tr.std(0)
    sc = np.where(sd_arr < 1e-8, 1.0, np.maximum(sd_arr, 0.01))
    tr_s, va_s, te_s = (tr - mu) / sc, (va - mu) / sc, (test - mu) / sc
    in_dim = tr_s.shape[1]
    tr_w = DataLoader.create_windows(tr_s, W_LEN, step=16, copy=False)
    va_w = DataLoader.create_windows(va_s, W_LEN, step=1, copy=False)
    te_w = DataLoader.create_windows(te_s, W_LEN, step=1, copy=False)

    torch.manual_seed(seed)
    np.random.seed(seed)
    enc = HybridTCNEncoder(
        input_dim=in_dim,
        latent_dim=32,
        filters=48,
        tcn_layers=4,
        kernel_size=3,
        dropout=0.0,
        use_dyadic_shells=use_dyadic,
    ).to(device)
    model = SolitonReynoldsJEPAModel(context_encoder=enc, latent_dim=32).to(device)
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)

    for _ in range(EPOCHS):
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
    te_raw4, te_white4 = extract_multihorizon_4subblocks(model, te_w, C_LEN, device)
    n_cal_w = len(va_white4) // 2
    va_ew, te_ew = stabilize_whitened_subblocks(va_white4, te_white4, va_raw4, te_raw4, n_cal_w)

    panel = MultiPerspectiveCircuitBreakerPanel(context_size=C_LEN, suspect_size=S_LEN)
    bits = panel.calibrate_and_trip_windows(va_ew, te_ew, n_cal_w)

    out_rows = []
    for mapping in ("trailing", "smear"):
        dec = panel.evaluate_stream_decisions(bits, len(va_s), len(te_s), mapping)
        tl = y[dec.valid_mask]
        pr_val = float(average_precision_score(tl, dec.test_switch_counts))
        mt = compute_metrics(
            tl, dec.test_pred_balanced, scores=dec.test_pred_balanced.astype(float),
            valid_mask=np.ones_like(tl, bool), use_pa=False,
        )
        er, dl = event_stats(tl, dec.test_pred_balanced)
        out_rows.append(dict(
            stream=f"{ds}:{ch}",
            seed=seed,
            mapping=mapping,
            frontend="Dyadic" if use_dyadic else "StdConv",
            f1=float(mt["f1"]),
            pr_auc=pr_val,
            er=float(er) if er == er else np.nan,
            delay=float(dl) if dl == dl else np.nan,
            hold_fpr=float(dec.hold_pred_balanced.mean()),
        ))
    return out_rows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    streams = [
        ("SMAP", "P-3"),
        ("MSL", "M-1"),
        ("SMD", "machine-1-2"),
        ("Daphnet", "S01R01E1"),
        ("GECCO", "water_quality"),
    ]
    rows = []
    for ds, ch in streams:
        for seed in (42, 123, 456):
            for use_dyadic in (False, True):
                rows.extend(evaluate_stream_seed(ds, ch, seed, use_dyadic, device))
        print(f"Finished {ds}:{ch}", flush=True)

    df = pd.DataFrame(rows)
    out_csv = ROOT / "reports" / "dyadic_frontend_224bit_ablation.csv"
    df.to_csv(out_csv, index=False)
    ps = df.groupby(["stream", "mapping", "frontend"]).mean(numeric_only=True).reset_index()
    for mapping in ("trailing", "smear"):
        print(f"\n=== DYADIC FRONT-END ABLATION UNDER 224-BIT BREAKER ({mapping.upper()}) ===")
        piv = ps[ps["mapping"] == mapping].pivot(index="stream", columns="frontend", values=["f1", "pr_auc", "er", "hold_fpr"])
        print(piv.round(4).to_string())
        print("\nMacro Mean:")
        print(ps[ps["mapping"] == mapping].groupby("frontend")[["f1", "pr_auc", "er", "delay", "hold_fpr"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
