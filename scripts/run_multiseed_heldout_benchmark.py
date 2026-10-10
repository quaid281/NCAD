#!/usr/bin/env python3
"""Multi-seed benchmark with strictly held-out threshold calibration.

Addresses the fresh TNNLS review (Major 2, 3, 4, 5):
  * Train on the first (1 - val_frac) of the nominal training sequence.
  * Split the remaining held-out nominal block chronologically into a
    calibration half (fits every threshold) and an evaluation half
    (measures empirical FPR on nominal data the threshold never saw),
    separated by a gap of S=64 points.
  * Same protocol for every model; no thresholds are ever fit on residuals of
    windows the model was trained on.
  * Controls: Operator-Entropy with lambda_ent=0 (same K-matrix architecture),
    Reynolds-Stress with the stress loss off, and Reynolds-Stress scored with
    the mean-only term (isolates the stress term in the score).
  * Calibrators compared on identical score streams: SPOT (u=0.98, q=1e-3),
    SPOT with the tanh kurtosis multiplier (beta=1/6), 99.5th percentile, max.
  * Held-out FPR reported with a moving-block bootstrap 95% interval.
  * Both buffered ("smear", 64-step lookahead) and strictly causal
    ("trailing") score-to-timestamp mappings.

Output: reports/multiseed_heldout_benchmark.csv
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score

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
STREAMS = [
    ("SMAP", "P-3"), ("MSL", "M-1"), ("SMD", "machine-1-2"), ("SMAP", "A-1"),
    ("MSL", "C-1"), ("SMD", "machine-1-3"), ("Daphnet", "S01R01E1"),
    ("GECCO", "water_quality"),
]
MODELS = [
    "TS-JEPA", "OpEntropy", "OpEntropy-lambda0", "Reynolds", "Reynolds-meanscore",
    "Reynolds-alpha0", "PotentialFlow", "TimesNet", "TranAD",
]


def load_stream(ds, ch):
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
    return a[cols].to_numpy(np.float32), b[cols].to_numpy(np.float32), y


def batches(arr, bs):
    perm = np.random.permutation(len(arr))
    for i in range(0, len(perm), bs):
        yield torch.from_numpy(np.ascontiguousarray(arr[perm[i:i + bs]])).float()


def train_model(name, tr_win, k, device, epochs):
    """Return a window-score function for the trained model."""
    if name in ("TimesNet", "TranAD"):
        if name == "TimesNet":
            m = TimesNet(c_in=k, d_model=48, d_ff=96, e_layers=2).to(device)
        else:
            m = TranAD(c_in=k, d_model=48, n_heads=4, e_layers=2, d_layers=2, d_ff=96).to(device)
        opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        for ep in range(epochs):
            m.train()
            for b in batches(tr_win, 16):
                b = b.to(device)
                if name == "TimesNet":
                    loss = nn.functional.mse_loss(m(b), b)
                else:
                    r1, r2 = m(b)
                    l1, l2 = m.adversarial_loss(r1, r2, b, epoch=ep + 1)
                    loss = l1 + l2
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        def score(w):
            m.eval()
            out = []
            with torch.no_grad():
                for i in range(0, len(w), 128):
                    ch = torch.from_numpy(np.ascontiguousarray(w[i:i + 128])).float().to(device)
                    out.append(m.compute_anomaly_scores(ch)[:, C:].mean(dim=-1).cpu().numpy())
            return np.concatenate(out)
        return score

    mt = {"TS-JEPA": "ts_jepa", "OpEntropy": "operator_entropy_jepa",
          "OpEntropy-lambda0": "operator_entropy_jepa", "Reynolds": "reynolds_stress_jepa",
          "Reynolds-meanscore": "reynolds_stress_jepa", "Reynolds-alpha0": "reynolds_stress_jepa",
          "PotentialFlow": "potential_flow_jepa", "PotentialFlow-noregime": "potential_flow_jepa"}[name]
    extra = {"use_regimes": False} if name == "PotentialFlow-noregime" else {}
    cfg = CSMConfig(model_type=mt, context_size=C, suspect_size=S, latent_dim=32, **extra)
    m = build_ts_jepa_model(cfg, input_dim=k, device=device)
    if name == "Reynolds-alpha0":
        m.alpha_stress = 0.0
    opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    kw = {"alpha_entropy": 0.0} if name == "OpEntropy-lambda0" else {}
    for ep in range(epochs):
        m.train()
        for b in batches(tr_win, 16):
            b = b.to(device)
            loss, _ = m.compute_objective(b[:, :C], b[:, C:], cfg, **kw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            m.update_target_encoder()

    mean_only = name in ("Reynolds-meanscore", "Reynolds-alpha0")

    def score(w):
        m.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(w), 128):
                ch = torch.from_numpy(np.ascontiguousarray(w[i:i + 128])).float().to(device)
                if mean_only:
                    zc = m._extract_context(ch[:, :C])
                    d = m.target_encoder(ch[:, C:]) - m.predictor(zc)
                    out.append(m.saliency_gate(d).cpu().numpy())
                else:
                    out.append(m.compute_predictive_discrepancy(ch[:, :C], ch[:, C:]).cpu().numpy())
        return np.concatenate(out)
    return score


def block_bootstrap_ci(ind, block=W, n_boot=400, rng=None):
    """Moving-block bootstrap 95% CI for the mean of a 0/1 exceedance indicator."""
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
        means[i] = np.concatenate([ind[j:j + block] for j in s])[:n].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def event_stats(y, pred):
    segs = _extract_segments(y)
    if not segs:
        return float("nan"), float("nan")
    delays = []
    for gs, ge in segs:
        hit = np.where(pred[gs:ge] > 0)[0]
        if len(hit):
            delays.append(hit[0])
    return len(delays) / len(segs), (float(np.mean(delays)) if delays else float("nan"))


def thresholds(cal):
    base = EVTCalibrator(risk_level=1e-3, init_percentile=98.0, adaptive_kurtosis=False)
    base.fit(cal)
    t0 = float(base.threshold_)
    kurt = float(compute_excess_kurtosis(cal))
    return {
        "SPOT": t0,
        "SPOT+kurt(1/6)": t0 * (1.0 + math.tanh(kurt) / 6.0),
        "p99.5": float(np.percentile(cal, 99.5)),
        "max-val": float(np.max(cal)),
    }, kurt


def run(ds, ch, seeds, device, epochs, val_frac, out_rows):
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
    print(f"== {ds}:{ch} train={len(tr)} val={len(va)} test={len(test)} anomaly_rate={y.mean():.4f}", flush=True)
    if len(va_w) < 50:
        print("   skip: held-out block too short", flush=True)
        return
    for seed in seeds:
        for name in MODELS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            score = train_model(name, tr_w, k, device, epochs)
            vws, tws = score(va_w), score(te_w)
            for mapping in ("smear", "trailing"):
                vp, vm = aggregate_window_scores(vws, len(va), C, S, step=1, reducer="mean", mapping_method=mapping)
                tp, tm = aggregate_window_scores(tws, len(test), C, S, step=1, reducer="mean", mapping_method=mapping)
                v = vp[vm]
                half = len(v) // 2
                # Gap of S points between halves. Short streams (SMAP P-3 has ~2k nominal
                # steps) cannot afford a full-window gap, so residual overlap-correlation is
                # handled by the block bootstrap instead.
                cal, hold = v[:half], v[half + S:]
                if len(cal) < 100 or len(hold) < 100:
                    continue
                ths, kurt = thresholds(cal)
                ts, tl = tp[tm], y[tm]
                pr = float(average_precision_score(tl, ts)) if tl.sum() > 0 else float("nan")
                for cname, th in ths.items():
                    ind = (hold > th).astype(float)
                    lo, hi = block_bootstrap_ci(ind)
                    pred = (ts > th).astype(int)
                    mt = compute_metrics(tl, pred, scores=ts, valid_mask=np.ones_like(tl, dtype=bool), use_pa=False)
                    er, dl = event_stats(tl, pred)
                    out_rows.append(dict(
                        dataset=ds, channel=ch, seed=seed, model=name, mapping=mapping, calibrator=cname,
                        threshold=th, cal_kurtosis=kurt, n_cal=len(cal), n_hold=len(hold),
                        hold_fpr=float(ind.mean()), hold_fpr_lo=lo, hold_fpr_hi=hi,
                        test_fpr=float(mt["fp"] / max((tl == 0).sum(), 1)),
                        f1=mt.get("f1", 0.0), precision=mt.get("precision", 0.0),
                        recall=mt.get("recall", 0.0), pr_auc=pr, event_recall=er, mean_delay=dl,
                    ))
            print(f"   seed={seed} {name} done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--val-frac", type=float, default=0.3)
    ap.add_argument("--streams", nargs="*", default=None, help="DS:channel entries")
    ap.add_argument("--models", nargs="*", default=None, help="subset of MODELS (or PotentialFlow-noregime)")
    ap.add_argument("--out", default=str(ROOT / "reports" / "multiseed_heldout_benchmark.csv"))
    a = ap.parse_args()
    global MODELS
    if a.models:
        MODELS = a.models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.streams and len(a.streams) == 1 and Path(a.streams[0]).exists():
        raw_streams = [line.strip() for line in Path(a.streams[0]).read_text().splitlines() if line.strip()]
        streams = [tuple(s.split(":")) for s in raw_streams]
    elif a.streams:
        streams = [tuple(s.split(":")) for s in a.streams]
    else:
        streams = STREAMS
    rows = []
    for ds, ch in streams:
        try:
            run(ds, ch, a.seeds, device, a.epochs, a.val_frac, rows)
        except Exception as e:  # keep going; log the failure in the output
            print(f"   FAILED {ds}:{ch}: {type(e).__name__}: {e}", flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
    print("saved", a.out)


if __name__ == "__main__":
    main()
