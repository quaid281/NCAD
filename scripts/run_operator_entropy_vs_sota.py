#!/usr/bin/env python3
"""Targeted Benchmark: Operator-Entropy JEPA vs. Leading SOTA Baselines.

This runner provides a focused, head-to-head comparison of our primary architecture
(Operator-Entropy JEPA, alongside standard TS-JEPA) against leading SOTA baselines
(TimesNet, TranAD) under strictly held-out threshold calibration:

  * Train on the first (1 - val_frac) of nominal training sequence.
  * Held-out nominal validation block split chronologically:
      - First half: calibrates EVT/SPOT (u=0.98, q=1e-3) and empirical p99.5 thresholds.
      - Isolation gap of S=64 steps to eliminate overlap contamination.
      - Second half: measures held-out false positive rate (FPR).
  * Measures both buffered ('smear', lookahead window) and strictly causal ('trailing') mappings.
  * Outputs detailed metrics (Point-F1, Event-F1, PR-AUC, AUROC, Held-out FPR with moving-block bootstrap CI).

Output: reports/operator_entropy_vs_sota.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CSMConfig
from src.data.data_loader import DataLoader
from src.engine.trainer import build_ts_jepa_model
from src.models.baselines import TimesNet, TranAD
from src.scoring.event_fusion import (
    _extract_segments,
    aggregate_window_scores,
    compute_metrics,
)
from src.scoring.evt_calibrator import EVTCalibrator, compute_excess_kurtosis

C, S = 256, 64
W = C + S

# Primary paper models vs. leading SOTA baselines
MODELS = ["OpEntropy", "TS-JEPA", "TimesNet", "TranAD"]

DEFAULT_STREAMS = [
    ("SMAP", "P-3"),
    ("MSL", "M-1"),
    ("SMD", "machine-1-2"),
    ("SMAP", "A-1"),
    ("MSL", "C-1"),
    ("SMD", "machine-1-3"),
    ("Daphnet", "S01R01E1"),
    ("GECCO", "water_quality"),
]


def load_stream(ds: str, ch: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load train, test, and ground-truth label traces for a telemetry stream."""
    d = ROOT / "mTSBench_data" / ds
    if ch == "default":
        tr = d / f"{ds}_train.csv"
        tr = tr if tr.exists() else d / f"{ds}.csv"
        te = d / f"{ds}_test.csv"
    else:
        tr = d / f"{ds}_{ch}_train.csv"
        tr = tr if tr.exists() else d / f"{ch}_train.csv"
        te = d / f"{ds}_{ch}_test.csv"
        te = te if te.exists() else d / f"{ch}_test.csv"
    a, b = pd.read_csv(tr), pd.read_csv(te)
    cols = [c for c in a.columns if c not in ["timestamp", "is_anomaly", "label"]]
    lc = [c for c in b.columns if c in ["is_anomaly", "label", "labels"]]
    y = b[lc[0]].to_numpy().astype(int) if lc else np.zeros(len(b), dtype=int)
    return (
        np.nan_to_num(a[cols].to_numpy(), nan=0.0).astype(np.float32),
        np.nan_to_num(b[cols].to_numpy(), nan=0.0).astype(np.float32),
        y,
    )



def train_model(name: str, tr_w: np.ndarray, k: int, device: torch.device, epochs: int):
    """Train the designated model and return a callable window scorer."""
    n_win = len(tr_w)
    bs = min(64, n_win)
    idx = np.arange(n_win)

    if name in ("TS-JEPA", "OpEntropy"):
        model_type = "operator_entropy_jepa" if name == "OpEntropy" else "ts_jepa"
        cfg = CSMConfig(
            model_type=model_type,
            context_size=C,
            suspect_size=S,
            latent_dim=32,
            filters=48,
            tcn_layers=3,
            dropout=0.05,
            alpha_entropy=0.5,
            use_koopman_matrix=True,
        )
        m = build_ts_jepa_model(cfg, input_dim=k, device=device)
        opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)

        for _ in range(epochs):
            np.random.shuffle(idx)
            for s in range(0, n_win, bs):
                b = torch.from_numpy(tr_w[idx[s : s + bs]]).to(device)
                ctx, tgt = b[:, :C], b[:, C:]
                opt.zero_grad()
                loss, _ = m.compute_objective(ctx, tgt, cfg)
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 2.0)
                opt.step()
                if hasattr(m, "update_target_encoder"):
                    m.update_target_encoder()

        m.eval()

        def score(wins: np.ndarray) -> np.ndarray:
            out = []
            with torch.no_grad():
                for s in range(0, len(wins), 128):
                    ch = torch.from_numpy(wins[s : s + 128]).to(device)
                    out.append(
                        m.compute_predictive_discrepancy(ch[:, :C], ch[:, C:])
                        .cpu()
                        .numpy()
                    )
            return np.concatenate(out)

        return score

    elif name in ("TimesNet", "TranAD"):
        if name == "TimesNet":
            m = TimesNet(c_in=k, d_model=48, d_ff=96, e_layers=2).to(device)
        else:
            m = TranAD(c_in=k, d_model=48, n_heads=4, e_layers=2, d_layers=2, d_ff=96).to(device)
        opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)

        for ep in range(epochs):
            m.train()
            np.random.shuffle(idx)
            for s in range(0, n_win, bs):
                b = torch.from_numpy(tr_w[idx[s : s + bs]]).to(device)
                if name == "TimesNet":
                    loss = nn.functional.mse_loss(m(b), b)
                else:
                    r1, r2 = m(b)
                    l1, l2 = m.adversarial_loss(r1, r2, b, epoch=ep + 1)
                    loss = l1 + l2
                opt.zero_grad()
                loss.backward()
                opt.step()
        m.eval()

        def score(wins: np.ndarray) -> np.ndarray:
            out = []
            with torch.no_grad():
                for s in range(0, len(wins), 128):
                    ch = torch.from_numpy(wins[s : s + 128]).to(device)
                    if name == "TimesNet":
                        rec = m(ch)
                    else:
                        r1, _ = m(ch)
                        rec = r1
                    e = (rec[:, C:, :] - ch[:, C:, :]).pow(2).mean(dim=(1, 2))
                    out.append(e.cpu().numpy())
            return np.concatenate(out)

        return score

    raise ValueError(f"Unknown model name: {name}")


def block_bootstrap_ci(ind: np.ndarray, block: int = W, n_boot: int = 400, rng: np.random.Generator | None = None) -> Tuple[float, float]:
    """Moving-block bootstrap 95% CI for the mean of an indicator stream."""
    rng = rng or np.random.default_rng(0)
    n = len(ind)
    if n < 2:
        return float("nan"), float("nan")
    block = max(16, min(block, n // 8))
    n_blocks = int(math.ceil(n / block))
    starts = np.arange(0, n - block + 1)
    means = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.choice(starts, n_blocks)
        means[i] = np.concatenate([ind[j : j + block] for j in s])[:n].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def event_stats(y: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    """Compute event detection recall and first-detection delay."""
    segs = _extract_segments(y)
    if not segs:
        return float("nan"), float("nan")
    delays = []
    for gs, ge in segs:
        hit = np.where(pred[gs:ge] > 0)[0]
        if len(hit):
            delays.append(hit[0])
    return len(delays) / len(segs), (float(np.mean(delays)) if delays else float("nan"))


def compute_thresholds(cal: np.ndarray) -> Tuple[Dict[str, float], float]:
    """Calibrate SPOT (u=0.98, q=1e-3) and empirical p99.5 thresholds."""
    base = EVTCalibrator(risk_level=1e-3, init_percentile=98.0, adaptive_kurtosis=False)
    base.fit(cal)
    t0 = float(base.threshold_)
    kurt = float(compute_excess_kurtosis(cal))
    return {
        "SPOT": t0,
        "p99.5": float(np.percentile(cal, 99.5)),
    }, kurt


def run_benchmark(
    streams: List[Tuple[str, str]],
    seeds: List[int],
    device: torch.device,
    epochs: int,
    val_frac: float,
    out_csv: Path,
):
    """Run head-to-head benchmark and checkpoint rows to out_csv."""
    out_rows = []
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    for ds, ch in streams:
        train, test, y = load_stream(ds, ch)
        n_val = int(len(train) * val_frac)
        tr, va = train[:-n_val], train[-n_val:]
        mu, sd = tr.mean(0), tr.std(0)
        sc = np.where(sd < 1e-8, 1.0, np.maximum(sd, 0.01))
        tr_s, va_s, te_s = (tr - mu) / sc, (va - mu) / sc, (test - mu) / sc

        tr_w = DataLoader.create_windows(tr_s, W, step=16, copy=False)
        va_w = DataLoader.create_windows(va_s, W, step=1, copy=False)
        te_w = DataLoader.create_windows(te_s, W, step=1, copy=False)
        k = train.shape[1]

        print(f"\n== Evaluating Stream: {ds}:{ch} (dim={k}, train={len(tr)}, val={len(va)}, test={len(test)}, anom_rate={y.mean():.4f})", flush=True)

        if len(va_w) < 50:
            print("   [Skip] Held-out validation block too short.", flush=True)
            continue

        for seed in seeds:
            for name in MODELS:
                torch.manual_seed(seed)
                np.random.seed(seed)

                print(f"   --> Model: {name} (seed={seed})...", end=" ", flush=True)
                score_fn = train_model(name, tr_w, k, device, epochs)
                vws, tws = score_fn(va_w), score_fn(te_w)

                for mapping in ("smear", "trailing"):
                    vp, vm = aggregate_window_scores(vws, len(va), C, S, step=1, reducer="mean", mapping_method=mapping)
                    tp, tm = aggregate_window_scores(tws, len(test), C, S, step=1, reducer="mean", mapping_method=mapping)

                    v = vp[vm]
                    half = len(v) // 2
                    cal = v[:half]
                    eval_nom = v[half + S :]

                    if len(cal) < 30 or len(eval_nom) < 30:
                        continue

                    th_dict, kurt = compute_thresholds(cal)
                    te_sc = tp[tm]
                    te_y = y[tm]

                    pr_auc = float(average_precision_score(te_y, te_sc)) if len(np.unique(te_y)) > 1 else float("nan")
                    auroc = float(roc_auc_score(te_y, te_sc)) if len(np.unique(te_y)) > 1 else float("nan")

                    for calib_name, t_val in th_dict.items():
                        ho_pred = (eval_nom > t_val).astype(int)
                        ho_fpr = float(ho_pred.mean())
                        ci_lo, ci_hi = block_bootstrap_ci(ho_pred)

                        te_pred = (te_sc > t_val).astype(int)
                        point_m = compute_metrics(te_y, te_pred)
                        ev_rec, ev_delay = event_stats(te_y, te_pred)

                        row = {
                            "dataset": ds,
                            "channel": ch,
                            "seed": seed,
                            "model": name,
                            "mapping": mapping,
                            "calibrator": calib_name,
                            "threshold": t_val,
                            "excess_kurtosis": kurt,
                            "heldout_fpr": ho_fpr,
                            "heldout_fpr_ci_low": ci_lo,
                            "heldout_fpr_ci_high": ci_hi,
                            "pr_auc": pr_auc,
                            "auroc": auroc,
                            "point_f1": point_m.get("f1", 0.0),
                            "point_precision": point_m.get("precision", 0.0),
                            "point_recall": point_m.get("recall", 0.0),
                            "event_recall": ev_rec,
                            "event_delay": ev_delay,
                        }
                        out_rows.append(row)

                print("Done.", flush=True)

        # Checkpoint after each completed stream
        pd.DataFrame(out_rows).to_csv(out_csv, index=False)
        print(f"   [Checkpointed {len(out_rows)} results to {out_csv}]", flush=True)

    print(f"\nAll benchmark evaluations completed. Output saved to: {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Operator-Entropy JEPA vs. SOTA Benchmarking Runner.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per model")
    parser.add_argument("--val-frac", type=float, default=0.35, help="Fraction of training sequence for held-out validation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456], help="Random seeds")
    parser.add_argument("--streams", type=str, default=None, help="Optional path to streams text file")
    parser.add_argument("--out", type=str, default="reports/operator_entropy_vs_sota.csv", help="Output CSV path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")

    args = parser.parse_args()

    streams = DEFAULT_STREAMS
    if args.streams is not None:
        p = Path(args.streams)
        if p.exists():
            streams = []
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split(":", 1) if ":" in line else line.split(None, 1)
                    if len(parts) == 2:
                        streams.append((parts[0].strip(), parts[1].strip()))

    dev = torch.device(args.device)
    out_path = ROOT / args.out
    print(f"Starting Operator-Entropy vs. SOTA benchmark:")
    print(f"  Models: {MODELS}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Streams: {len(streams)}")
    print(f"  Device: {dev}")
    print(f"  Output CSV: {out_path}")

    run_benchmark(streams, args.seeds, dev, args.epochs, args.val_frac, out_path)


if __name__ == "__main__":
    main()
