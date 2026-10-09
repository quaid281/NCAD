#!/usr/bin/env python3
"""Protocol ablation: which difference explains the 45-stream vs held-out gap?

Each (stream, seed, model) is trained once. The same score streams are then
evaluated under a factorial grid of protocol choices taken from the legacy
45-stream pipeline (scripts/run_benchmark.py, run_modern_baselines.py):

  calib_source : "train"   threshold fit on residuals of windows the model trained on
                 "heldout" threshold fit on a chronologically held-out nominal block
  postproc     : "raw"     threshold applied to raw point scores
                 "ma_evt"  12-step moving average + event_level_filter(min_run=2, 1.75)
  thresholder  : "SPOT"    EVT, no kurtosis multiplier
                 "SPOT+K"  EVT with the adaptive kurtosis multiplier (legacy default)

Epochs are a command-line factor (run the script once per setting).
Output: reports/protocol_ablation.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_multiseed_heldout_benchmark as base  # noqa: E402
from src.data.data_loader import DataLoader  # noqa: E402
from src.scoring.event_fusion import (  # noqa: E402
    aggregate_window_scores,
    compute_metrics,
    event_level_filter,
    moving_average,
)
from src.scoring.evt_calibrator import EVTCalibrator  # noqa: E402

C, S, W = base.C, base.S, base.W


def fit_threshold(cal, kurt):
    cal_ = EVTCalibrator(risk_level=1e-3, init_percentile=98.0, adaptive_kurtosis=kurt)
    cal_.fit(cal)
    if kurt:
        return float(cal_.compute_threshold(cal, risk_level=1e-3).threshold)
    return float(cal_.threshold_)


def run(ds, ch, seeds, device, epochs, val_frac, models, rows):
    train, test, y = base.load_stream(ds, ch)
    n_val = int(len(train) * val_frac)
    tr, va = train[:-n_val], train[-n_val:]
    mu, sd = tr.mean(0), tr.std(0)
    sc = np.where(sd < 1e-8, 1.0, np.maximum(sd, 0.01))
    tr_s, va_s, te_s = (tr - mu) / sc, (va - mu) / sc, (test - mu) / sc
    tr_w = DataLoader.create_windows(tr_s, W, step=16, copy=False)
    va_w = DataLoader.create_windows(va_s, W, step=1, copy=False)
    te_w = DataLoader.create_windows(te_s, W, step=1, copy=False)
    k = train.shape[1]
    print(f"== {ds}:{ch}", flush=True)
    if len(va_w) < 50:
        return
    for seed in seeds:
        for name in models:
            torch.manual_seed(seed)
            np.random.seed(seed)
            score = base.train_model(name, tr_w, k, device, epochs)
            sw = {"train": score(tr_w), "heldout": score(va_w), "test": score(te_w)}
            tp, tm = aggregate_window_scores(sw["test"], len(test), C, S, step=1, reducer="mean")
            tl = y[tm]
            for post in ("raw", "ma_evt"):
                ts_full = moving_average(tp, 12) if post == "ma_evt" else tp
                for src in ("train", "heldout"):
                    if src == "train":
                        cp, cm = aggregate_window_scores(sw["train"], (len(tr_w) - 1) * 16 + W, C, S, step=16, reducer="mean")
                        cs = moving_average(cp, 12) if post == "ma_evt" else cp
                        cal = cs[cm]
                        hold = None
                    else:
                        cp, cm = aggregate_window_scores(sw["heldout"], len(va), C, S, step=1, reducer="mean")
                        cs = moving_average(cp, 12) if post == "ma_evt" else cp
                        v = cs[cm]
                        half = len(v) // 2
                        cal, hold = v[:half], v[half + S:]
                        if len(cal) < 100 or len(hold) < 100:
                            continue
                    for tname, kurt in (("SPOT", False), ("SPOT+K", True)):
                        th = fit_threshold(cal, kurt)
                        if post == "ma_evt":
                            pred = event_level_filter(ts_full, th, tm, min_run=2, extreme_factor=1.75)[tm]
                        else:
                            pred = (ts_full[tm] > th)
                        pred = pred.astype(int)
                        m = compute_metrics(tl, pred, scores=ts_full[tm], valid_mask=np.ones_like(tl, dtype=bool), use_pa=False)
                        er, _ = base.event_stats(tl, pred)
                        rows.append(dict(
                            dataset=ds, channel=ch, seed=seed, model=name, epochs=epochs,
                            calib_source=src, postproc=post, thresholder=tname, threshold=th,
                            cal_fpr=float((cal > th).mean()),
                            hold_fpr=float((hold > th).mean()) if hold is not None else float("nan"),
                            test_fpr=float(m["fp"] / max((tl == 0).sum(), 1)),
                            f1=m.get("f1", 0.0), precision=m.get("precision", 0.0),
                            recall=m.get("recall", 0.0), event_recall=er,
                            pr_auc=float(average_precision_score(tl, ts_full[tm])) if tl.sum() > 0 else float("nan"),
                        ))
            print(f"   seed={seed} {name} done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--val-frac", type=float, default=0.35)
    ap.add_argument("--streams", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=["TS-JEPA", "OpEntropy", "PotentialFlow", "TimesNet", "TranAD"])
    ap.add_argument("--out", default=str(ROOT / "reports" / "protocol_ablation.csv"))
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    streams = [tuple(s.split(":")) for s in a.streams] if a.streams else [s for s in base.STREAMS if s[0] != "GECCO"]
    rows = []
    for ds, ch in streams:
        try:
            run(ds, ch, a.seeds, device, a.epochs, a.val_frac, a.models, rows)
        except Exception as e:
            print(f"   FAILED {ds}:{ch}: {type(e).__name__}: {e}", flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
    print("saved", a.out)


if __name__ == "__main__":
    main()
