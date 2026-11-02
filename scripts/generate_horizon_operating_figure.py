#!/usr/bin/env python3
"""Generate publication-grade true-vector figure for Target-Horizon sweep and Deployable Quantiles.

Clean, spacious two-panel horizontal layout tailored for IEEE two-column full-width banner:
figsize=(7.14, 2.25) inches with zero title/legend overlap and zero bottom-label clipping.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = ROOT / ".agents" / "skills" / "beautiful-research-figures" / "scripts"
if str(SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS))

from ieee_vector_style import save_true_vector_figure  # noqa: E402

FIGDIR = ROOT / "paper" / "NCAD_CS" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
OUT_STEM = FIGDIR / "operating_and_horizon_tradeoffs"

# Palette definitions (Churkin / IEEE grayscale-safe high-contrast)
C_OPENT_CAUS = "#378d94"    # Deep Teal
C_REY0_CAUS = "#1d4e89"     # Deep Navy
C_REY_CAUS = "#2b7a78"      # Emerald Teal
C_TSJEPA_CAUS = "#6a408d"   # Churkin Deep Purple
C_TIMESNET_CAUS = "#e13661" # Vibrant Coral / Crimson
C_TRANAD_CAUS = "#4e4e4e"   # Slate Grey
C_OPENT_BUF = "#77b5b6"     # Muted Teal


def set_ieee_style():
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.minor.width": 0.35,
        "ytick.minor.width": 0.35,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "axes.labelsize": 8.2,
        "axes.titlesize": 8.5,
        "legend.fontsize": 6.8,
        "figure.dpi": 300,
    })


def make_figure():
    set_ieee_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.14, 2.28), gridspec_kw={"wspace": 0.25})
    fig.subplots_adjust(top=0.73, bottom=0.24, left=0.07, right=0.98)

    # =========================================================================
    # Panel (a): Target Horizon Length S vs Causal FPR
    # =========================================================================
    x_indices = np.arange(5)
    x_labels = [
        "$S{=}1$\n(lag 0)", "$S{=}4$\n(lag 3)", "$S{=}16$\n(lag 15)",
        "$S{=}32$\n(lag 31)", "$S{=}64$\n(lag 63)"
    ]

    fpr_opent_caus = np.array([0.005, 0.007, 0.003, 0.011, 0.020]) * 100
    fpr_tsjepa_caus = np.array([0.004, 0.006, 0.003, 0.003, 0.004]) * 100
    fpr_timesnet_caus = np.array([0.025, 0.031, 0.061, 0.098, 0.213]) * 100
    fpr_opent_buf = np.array([0.005, 0.020, 0.043, 0.057, 0.099]) * 100

    ax1.plot(x_indices, fpr_timesnet_caus, color=C_TIMESNET_CAUS, linestyle="--", linewidth=1.6, marker="s", markersize=4.6, label="TimesNet (caus)", zorder=3)
    ax1.plot(x_indices, fpr_opent_buf, color=C_OPENT_BUF, linestyle=":", linewidth=1.5, marker="^", markersize=4.3, label="OpEnt (buf)", zorder=3)
    ax1.plot(x_indices, fpr_opent_caus, color=C_OPENT_CAUS, linestyle="-", linewidth=1.8, marker="o", markersize=5.0, label="OpEnt (caus)", zorder=4)
    ax1.plot(x_indices, fpr_tsjepa_caus, color=C_TSJEPA_CAUS, linestyle="-", linewidth=1.6, marker="D", markersize=4.3, label="TS-JEPA (caus)", zorder=4)

    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_labels, fontsize=7.3)
    ax1.set_xlabel("Target Horizon $S$ (Score-Emission Lag $S{-}1$)", labelpad=4)
    ax1.set_ylabel("Held-Out FPR (%)", labelpad=3)
    ax1.set_ylim(-0.8, 24.0)
    ax1.set_yticks(np.arange(0, 25, 5))
    ax1.set_yticklabels([f"{y}%" for y in np.arange(0, 25, 5)])

    ax1.grid(True, which="major", linestyle="-", linewidth=0.5, color="#e0e0e0", alpha=0.7)
    ax1.set_axisbelow(True)

    ax1.annotate("9x False-Alarm Surge\n($2.5\\% \\to 21.3\\%$)", xy=(4, 21.3), xytext=(1.55, 16.0),
                 arrowprops=dict(arrowstyle="->", color=C_TIMESNET_CAUS, lw=0.9),
                 fontsize=7.0, color="#911838", fontweight="semibold")
    ax1.annotate("Stable Sub-2% Causal FPR", xy=(3.0, 1.1), xytext=(0.15, 10.2),
                 arrowprops=dict(arrowstyle="->", color=C_OPENT_CAUS, lw=0.9),
                 fontsize=7.0, color="#1f5d62", fontweight="semibold")

    ax1.set_title("(a) Latency vs. False-Alarm Explosion across $S$", fontweight="bold", pad=26)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, columnspacing=1.2, handletextpad=0.35, labelspacing=0.25, fontsize=6.8)

    # =========================================================================
    # Panel (b): Event Recall vs False Alarm Rate across Deployable Quantiles q*
    # =========================================================================
    models_data = {
        "Reynolds-alpha0": {
            "fa": [45.0, 112.5], "er": [0.482, 0.756], "color": C_REY0_CAUS, "marker": "p", "label": "Reynolds ($\\alpha{=}0$)"
        },
        "Reynolds": {
            "fa": [64.0, 150.9], "er": [0.497, 0.667], "color": C_REY_CAUS, "marker": "*", "label": "Reynolds"
        },
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

    ax2.axvspan(38, 210, ymin=0.42, ymax=1.0, color="#77b5b6", alpha=0.10, zorder=1)
    ax2.text(43, 0.802, "JEPA Causal Frontier (High Recall, Low Alarms)", fontsize=6.6, color="#21575b",
             ha="left", va="center", style="italic", weight="bold")

    for name, m in models_data.items():
        fa, er = m["fa"], m["er"]
        ax2.annotate("", xy=(fa[1], er[1]), xytext=(fa[0], er[0]),
                     arrowprops=dict(arrowstyle="->", color=m["color"], lw=1.35, mutation_scale=9),
                     zorder=3)
        ax2.scatter(fa[0], er[0], s=26, facecolors="none", edgecolors=m["color"], linewidths=1.3, zorder=4)
        ax2.scatter(fa[1], er[1], s=36, facecolors=m["color"], edgecolors="#222222", linewidths=0.65, marker=m["marker"], label=m["label"], zorder=4)

    ax2.text(44.0, 0.448, "$q^*{=}10^{-3}$", fontsize=6.5, ha="left", color=C_REY0_CAUS, weight="bold")
    ax2.text(118.0, 0.748, "$q^*{=}0.05$", fontsize=6.5, ha="left", color=C_REY0_CAUS, weight="bold")
    ax2.text(218.0, 0.565, "$q^*{=}10^{-3}$", fontsize=6.4, ha="right", color=C_TIMESNET_CAUS)
    ax2.text(265.0, 0.625, "$q^*{=}0.05$", fontsize=6.4, ha="left", color=C_TIMESNET_CAUS)

    ax2.set_xlabel("False Alarms per 1,000 Nominal Steps", labelpad=4)
    ax2.set_ylabel("Causal Event Recall (ER)", labelpad=3)
    ax2.set_xlim(32, 305)
    ax2.set_ylim(0.42, 0.84)
    ax2.set_yticks(np.arange(0.45, 0.85, 0.10))

    ax2.grid(True, which="major", linestyle="-", linewidth=0.5, color="#e0e0e0", alpha=0.7)
    ax2.set_axisbelow(True)

    ax2.set_title("(b) Operating Trade-offs at Deployable Nominal $q^*$", fontweight="bold", pad=26)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, columnspacing=0.65, handletextpad=0.25, labelspacing=0.25, fontsize=6.7)

    saved = save_true_vector_figure(
        fig, OUT_STEM, formats=("svg", "pdf", "png"),
        png_dpi=300, inkscape_postprocess_pdf=True,
        ensure_pure_vector=True, ensure_outside_legend=True
    )
    plt.close(fig)
    print("Saved figure files:", saved)


if __name__ == "__main__":
    make_figure()
