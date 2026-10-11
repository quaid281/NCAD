#!/usr/bin/env python3
"""Pooled event-recall vs realized-FPR operating curves (Round-4, Q1).

Reads reports/operating_characteristics.csv, aggregates the stream-mean
event recall as a function of *realized* test FPR (not the nominal design
budget) per model in the strictly causal regime, and renders a two-panel
figure: (a) trailing causal curves for the principal models, (b) the same
for the buffered smear regime. Dataset-clustered bootstrap ribbons on the
two principal models (TS-JEPA, TimesNet).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LONG = ROOT / "reports" / "operating_characteristics.csv"
FIGDIR = ROOT / "paper" / "NCAD_CS" / "figures"
N_BOOT = 500
RNG = np.random.default_rng(7)

PRINCIPAL = ["TS-JEPA", "OpEntropy", "Reynolds", "PotentialFlow",
             "TimesNet", "TranAD", "TCN-ObsPred", "TCN-ObsPred-LN",
             "PCA-LinPred-LN", "IsoForest", "ZScoreMax"]
CI_MODELS = ["TS-JEPA", "TimesNet"]


def curve_for(sm, model, mapping):
    g = sm[(sm.model == model) & (sm.mapping == mapping)]
    gg = g.groupby("q").agg(event_recall=("event_recall", "mean"),
                            realized_fpr=("realized_fpr", "mean"))
    return gg.sort_values("realized_fpr")


def boot_band(long, model, mapping, fpr_grid):
    """Dataset-clustered bootstrap band for event recall at realized FPR."""
    d = long[(long.model == model) & (long.mapping == mapping)]
    sm = d.groupby(["dataset", "channel", "q"], as_index=False)[
        ["event_recall", "realized_fpr"]].mean()
    datasets = sm.dataset.unique()
    curves = np.empty((N_BOOT, len(fpr_grid)))
    for i in range(N_BOOT):
        pick = RNG.choice(datasets, len(datasets), replace=True)
        samp = pd.concat([sm[sm.dataset == ds] for ds in pick])
        gg = samp.groupby("q").agg(rf=("realized_fpr", "mean"),
                                   er=("event_recall", "mean")).sort_index()
        # interpolate er vs realized fpr onto grid only where the resampled
        # curve actually spans that FPR
        in_range = (fpr_grid >= gg.rf.min()) & (fpr_grid <= gg.rf.max())
        c = np.full(len(fpr_grid), np.nan)
        c[in_range] = np.interp(fpr_grid[in_range], gg.rf, gg.er)
        curves[i] = c
    cover = np.isfinite(curves).mean(0)
    band = np.nanpercentile(curves, [2.5, 97.5], axis=0)
    band[:, cover < 0.8] = np.nan  # hide ribbon where <80% of boots have data
    return band


def main():
    long = pd.read_csv(LONG)
    sm = (long.groupby(["dataset", "channel", "model", "mapping", "q"],
                       as_index=False)
          [["event_recall", "realized_fpr"]].mean())

    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.6), sharex=True)
    colors = plt.cm.tab20(np.linspace(0, 1, len(PRINCIPAL)))
    style = {m: c for m, c in zip(PRINCIPAL, colors)}
    fpr_grid = np.linspace(0.02, 0.85, 60)

    for ax, mapping, title in zip(axes, ["trailing", "smear"],
                                 ["(a) Strictly causal (trailing)",
                                  "(b) Buffered attribution (smear)"]):
        for model in PRINCIPAL:
            c = curve_for(sm, model, mapping)
            if len(c) == 0:
                continue
            ls = "-" if "JEPA" in model or model in ("OpEntropy", "Reynolds",
                                                     "PotentialFlow") else "--"
            ax.plot(c.realized_fpr, c.event_recall, ls, color=style[model],
                    label=model, lw=1.2)
        for model in CI_MODELS:
            lo, hi = boot_band(long, model, mapping, fpr_grid)
            ax.fill_between(fpr_grid, lo, hi, color=style[model], alpha=0.15)
        ax.set_xscale("log")
        ax.set_xlim(0.02, 0.9)
        ax.set_title(title, fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylabel("Event recall")
    axes[1].set_xlabel("Realized test FPR")
    axes[1].legend(fontsize=5.4, loc="lower right", ncol=2)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"operating_curves.{ext}", dpi=200)
    print("wrote figures/operating_curves.{pdf,png}")


if __name__ == "__main__":
    main()
