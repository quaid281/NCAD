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
    custom_colors = [
        "#1f77b4",  # TS-JEPA (distinct primary blue)
        "#ff7f0e",  # OpEntropy (orange)
        "#2ca02c",  # Reynolds (green)
        "#d62728",  # PotentialFlow (red)
        "#9467bd",  # TimesNet (purple)
        "#8c564b",  # TranAD (brown)
        "#e377c2",  # TCN-ObsPred (pink)
        "#7f7f7f",  # TCN-ObsPred-LN (gray)
        "#bcbd22",  # PCA-LinPred-LN (olive)
        "#17becf",  # IsoForest (cyan)
        "#2b2b2b",  # ZScoreMax (dark charcoal, avoiding collision with TS-JEPA blue)
    ]
    style = {m: c for m, c in zip(PRINCIPAL, custom_colors)}
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
        ax.set_ylim(0.28, 1.02)
        ax.set_title(title, fontsize=8.5, pad=3)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_ylabel("Event recall", fontsize=8.5)
        ax.tick_params(labelsize=7.5)

    # Set explicit, intuitive x-ticks and formatting on log scale
    xticks = [0.02, 0.05, 0.1, 0.2, 0.5]
    axes[1].set_xticks(xticks)
    axes[1].get_xaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}")
    )
    axes[1].set_xlabel("Realized test FPR", fontsize=8.5)

    # Build legend handles including a 12th entry for the 95% bootstrap CI band
    handles, labels = axes[0].get_legend_handles_labels()
    ci_patch = matplotlib.patches.Patch(
        facecolor="#888888", edgecolor="none", alpha=0.35, label="95% Boot. CI"
    )
    handles.append(ci_patch)
    labels.append("95% Boot. CI")

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=3, fontsize=6.2, columnspacing=0.7, handlelength=1.6,
               handletextpad=0.35, labelspacing=0.22, frameon=True,
               edgecolor='#cccccc', framealpha=0.95)

    fig.tight_layout(rect=[0, 0, 1, 0.865])
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"operating_curves.{ext}", dpi=200)
    print("wrote figures/operating_curves.{pdf,png}")


if __name__ == "__main__":
    main()

