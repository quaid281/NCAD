#!/usr/bin/env python3
"""Summarise reports/multiseed_heldout_benchmark.csv into paired, seed-aware tables.

Unit of analysis: stream. Seeds are averaged inside a stream, then streams are
compared pairwise (Wilcoxon signed-rank + bootstrap CI of the mean difference).
Seed spread is reported as the SD of the stream-macro mean across seeds.
Output: reports/multiseed_summary.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--inputs", nargs="*", default=None)
ap.add_argument("--out", default=str(ROOT / "reports" / "multiseed_summary.md"))
args = ap.parse_args()
if args.inputs:
    df = pd.concat([pd.read_csv(f) for f in args.inputs], ignore_index=True)
else:
    df = pd.read_csv(ROOT / "reports" / "multiseed_heldout_benchmark.csv")
    extra = ROOT / "reports" / "multiseed_heldout_controls_extra.csv"
    if extra.exists():
        df = pd.concat([df, pd.read_csv(extra)], ignore_index=True)
rng = np.random.default_rng(0)
out = []


def p(s=""):
    out.append(s)
    print(s)


def stream_means(d, col):
    return d.groupby(["dataset", "channel", "model"])[col].mean().unstack("model")


def macro_with_seed_sd(d, col, model):
    d = d[d.model == model]
    n_seeds = d.groupby(["dataset", "channel"]).seed.nunique()
    full = set(n_seeds[n_seeds == n_seeds.max()].index)
    d = d[[(a, b) in full for a, b in zip(d.dataset, d.channel)]]
    per_seed = d.groupby(["seed", "dataset", "channel"])[col].mean().groupby("seed").mean()
    return per_seed.mean(), per_seed.std(ddof=1) if len(per_seed) > 1 else float("nan")


def paired(d, col, a, b):
    m = stream_means(d, col)
    if a not in m or b not in m:
        return None
    x = (m[a] - m[b]).dropna()
    if len(x) < 3:
        return None
    boots = [rng.choice(x.values, len(x)).mean() for _ in range(2000)]
    try:
        pv = wilcoxon(x.values).pvalue if np.any(x.values != 0) else 1.0
    except ValueError:
        pv = float("nan")
    return x.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5), pv, int((x > 0).sum()), len(x)


streams = df[["dataset", "channel"]].drop_duplicates().shape[0]
seeds = sorted(df.seed.unique())
p(f"Streams: {streams}; seeds: {seeds}; rows: {len(df)}")
p()
for mapping in ("smear", "trailing"):
    d0 = df[(df.mapping == mapping) & (df.calibrator == "SPOT")]
    label = "Buffered (64-step lookahead)" if mapping == "smear" else "Strictly causal (0 lookahead)"
    p(f"### {label}, SPOT(u=0.98, q=1e-3) fit on held-out calibration half")
    p("| Model | Point-F1 (macro, mean ± seed SD) | PR-AUC | Held-out nominal FPR | Test FPR | Event recall |")
    p("|:--|:-:|:-:|:-:|:-:|:-:|")
    for mdl in [m for m in df.model.unique()]:
        r = []
        for col in ("f1", "pr_auc", "hold_fpr", "test_fpr", "event_recall"):
            mu, sd = macro_with_seed_sd(d0, col, mdl)
            r.append(f"{mu:.4f} ± {sd:.4f}" if col in ("f1",) else f"{mu:.4f}")
        p(f"| {mdl} | " + " | ".join(r) + " |")
    p()

d0 = df[(df.mapping == "smear") & (df.calibrator == "SPOT")]
p("### Paired differences (buffered, SPOT), stream-level, seeds averaged")
p("| Comparison | Metric | Mean diff | 95% bootstrap CI | Wilcoxon p | Streams A>B |")
p("|:--|:--|:-:|:-:|:-:|:-:|")
pairs = [
    ("TS-JEPA", "TimesNet"), ("TS-JEPA", "TranAD"),
    ("OpEntropy", "TS-JEPA"), ("OpEntropy", "OpEntropy-lambda0"),
    ("Reynolds", "TS-JEPA"), ("Reynolds", "Reynolds-meanscore"), ("Reynolds", "Reynolds-alpha0"),
    ("PotentialFlow", "TS-JEPA"), ("PotentialFlow", "PotentialFlow-noregime"),
    ("OpEntropy", "TimesNet"), ("OpEntropy", "TranAD"),
]
for a, b in pairs:
    for col in ("f1", "pr_auc", "hold_fpr"):
        r = paired(d0, col, a, b)
        if r:
            p(f"| {a} vs {b} | {col} | {r[0]:+.4f} | [{r[1]:+.4f}, {r[2]:+.4f}] | {r[3]:.3f} | {r[4]}/{r[5]} |")
p()

p("### Calibrator comparison on identical score streams (buffered)")
p("| Model | Calibrator | Held-out nominal FPR (mean) | Share of rows whose 95% CI excludes q=1e-3 | Point-F1 |")
p("|:--|:--|:-:|:-:|:-:|")
dc = df[df.mapping == "smear"]
for mdl in ("TS-JEPA", "OpEntropy", "TimesNet", "TranAD"):
    for cal in ("SPOT", "SPOT+kurt(1/6)", "p99.5", "max-val"):
        s = dc[(dc.model == mdl) & (dc.calibrator == cal)]
        if len(s) == 0:
            continue
        excl = ((s.hold_fpr_lo > 1e-3) | (s.hold_fpr_hi < 1e-3)).mean()
        p(f"| {mdl} | {cal} | {s.hold_fpr.mean():.4f} | {excl:.2f} | {s.f1.mean():.4f} |")
p()

p("### Calibration gap: SPOT design risk 1e-3 vs measured held-out FPR, by model (buffered)")
g = d0.groupby("model")[["hold_fpr", "hold_fpr_lo", "hold_fpr_hi"]].mean()
p(g.round(4).to_markdown())
Path(args.out).write_text("\n".join(out), encoding="utf-8")
