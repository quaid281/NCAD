#!/usr/bin/env python3
"""Generate publication-grade true-vector figure for Target-Horizon sweep and Deployable Quantiles.

Compact sleek horizontal layout tailored for IEEE two-column full-width banner:
figsize=(7.14, 1.55) inches.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "paper" / "NCAD_CS" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
PDF_OUT = FIGDIR / "operating_and_horizon_tradeoffs.pdf"
SVG_OUT = FIGDIR / "operating_and_horizon_tradeoffs.svg"

# Palette definitions
C_OPENT_CAUS = "#378d94"   # Deep Teal
C_TSJEPA_CAUS = "#6a408d"  # Churkin Deep Purple
C_TIMESNET_CAUS = "#e13661"# Vibrant Coral / Crimson
C_TRANAD_CAUS = "#4e4e4e"  # Slate Grey
C_OPENT_BUF = "#77b5b6"    # Muted Teal

def set_ieee_style():
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.minor.width": 0.35,
        "ytick.minor.width": 0.35,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.labelsize": 8.8,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7.8,
        "figure.dpi": 300,
    })

def make_figure():
    set_ieee_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.14, 1.55), gridspec_kw={"wspace": 0.30})

    # =========================================================================
    # Panel (a): Target Horizon Length S vs Causal FPR
    # =========================================================================
    x_indices = np.arange(5)
    x_labels = ["$S{=}1$\n(lag 0)", "$S{=}4$\n(lag 3)", "$S{=}16$\n(lag 15)", "$S{=}32$\n(lag 31)", "$S{=}64$\n(lag 63)"]

    fpr_opent_caus = np.array([0.005, 0.007, 0.003, 0.011, 0.020]) * 100
    fpr_tsjepa_caus = np.array([0.004, 0.006, 0.003, 0.003, 0.004]) * 100
    fpr_timesnet_caus = np.array([0.025, 0.031, 0.061, 0.098, 0.213]) * 100
    fpr_opent_buf = np.array([0.005, 0.020, 0.043, 0.057, 0.099]) * 100

    ax1.plot(x_indices, fpr_timesnet_caus, color=C_TIMESNET_CAUS, linestyle="--", linewidth=1.6, marker="s", markersize=4.8, label="TimesNet (caus)", zorder=3)
    ax1.plot(x_indices, fpr_opent_buf, color=C_OPENT_BUF, linestyle=":", linewidth=1.5, marker="^", markersize=4.5, label="OpEnt (buf)", zorder=3)
    ax1.plot(x_indices, fpr_opent_caus, color=C_OPENT_CAUS, linestyle="-", linewidth=1.8, marker="o", markersize=5.2, label="OpEnt (caus)", zorder=4)
    ax1.plot(x_indices, fpr_tsjepa_caus, color=C_TSJEPA_CAUS, linestyle="-", linewidth=1.6, marker="D", markersize=4.5, label="TS-JEPA (caus)", zorder=4)

    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_labels, fontsize=7.5)
    ax1.set_xlabel("Target Horizon $S$ (Score-Emission Lag $S{-}1$)", labelpad=3)
    ax1.set_ylabel("Held-Out FPR (%)", labelpad=3)
    ax1.set_ylim(-0.5, 23.5)
    ax1.set_yticks(np.arange(0, 25, 5))
    ax1.set_yticklabels([f"{y}%" for y in np.arange(0, 25, 5)])

    ax1.grid(True, which="major", linestyle="-", linewidth=0.5, color="#e0e0e0", alpha=0.7)
    ax1.set_axisbelow(True)

    ax1.annotate("9x False-Alarm Explosion\n($2.5\\% \\to 21.3\\%$)", xy=(4, 21.3), xytext=(1.8, 16.5),
                 arrowprops=dict(arrowstyle="->", color=C_TIMESNET_CAUS, lw=0.9),
                 fontsize=7.2, color="#911838", fontweight="semibold")
    ax1.annotate("Stable Sub-2% FPR", xy=(4, 2.0), xytext=(2.2, 5.0),
                 arrowprops=dict(arrowstyle="->", color=C_OPENT_CAUS, lw=0.9),
                 fontsize=7.2, color="#1f5d62", fontweight="semibold")

    ax1.set_title(r"\textbf{(a)} Latency vs. False-Alarm Explosion across $S$", pad=13)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, columnspacing=0.8, handletextpad=0.3, fontsize=7.6)

    # =========================================================================
    # Panel (b): Event Recall vs False Alarm Rate across Deployable Quantiles q*
    # =========================================================================
    models_data = {
        "OpEntropy": {
            "fa": [93.7, 202.8], "er": [0.556, 0.752], "color": C_OPENT_CAUS, "marker": "o", "label": "OpEntropy"
        },
        "TS-JEPA": {
            "fa": [88.1, 177.3], "er": [0.493, 0.730], "color": C_TSJEPA_CAUS, "marker": "D", "label": "TS-JEPA"
        },
        "TimesNet": {
            "fa": [222.4, 262.3], "er": [0.590, 0.618], "color": C_TIMESNET_CAUS, "marker": "s", "label": "TimesNet"
        },
        "TranAD": {
            "fa": [209.6, 269.5], "er": [0.533, 0.602], "color": C_TRANAD_CAUS, "marker": "v", "label": "TranAD"
        },
    }

    ax2.axvspan(70, 210, ymin=0.55, ymax=1.0, color="#77b5b6", alpha=0.10, zorder=1)
    ax2.text(140, 0.765, "JEPA Causal Frontier\n(High Recall, Low Alarms)", fontsize=7.0, color="#21575b",
             ha="center", va="center", style="italic")

    for name, m in models_data.items():
        fa, er = m["fa"], m["er"]
        ax2.annotate("", xy=(fa[1], er[1]), xytext=(fa[0], er[0]),
                     arrowprops=dict(arrowstyle="->", color=m["color"], lw=1.4, mutation_scale=10),
                     zorder=3)
        ax2.scatter(fa[0], er[0], s=30, facecolors="none", edgecolors=m["color"], linewidths=1.4, zorder=4)
        ax2.scatter(fa[1], er[1], s=40, facecolors=m["color"], edgecolors="#222222", linewidths=0.7, marker=m["marker"], label=m["label"], zorder=4)
        ax2.text(fa[0]-4, er[0]-0.015, "$q^*{=}10^{-3}$", fontsize=7.0, ha="right", color=m["color"])
        ax2.text(fa[1]+4, er[1]+0.008, "$q^*{=}0.05$", fontsize=7.0, ha="left", color=m["color"])

    ax2.set_xlabel("False Alarms per 1,000 Nominal Steps", labelpad=3)
    ax2.set_ylabel("Causal Event Recall (ER)", labelpad=3)
    ax2.set_xlim(60, 310)
    ax2.set_ylim(0.44, 0.81)
    ax2.set_yticks(np.arange(0.45, 0.85, 0.05))

    ax2.grid(True, which="major", linestyle="-", linewidth=0.5, color="#e0e0e0", alpha=0.7)
    ax2.set_axisbelow(True)

    ax2.set_title(r"\textbf{(b)} Operating Trade-offs at Deployable Nominal $q^*$", pad=13)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, columnspacing=0.8, handletextpad=0.3, fontsize=7.6)

    fig.savefig(PDF_OUT, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(SVG_OUT, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Generated compact {PDF_OUT} and {SVG_OUT}")

if __name__ == "__main__":
    make_figure()
