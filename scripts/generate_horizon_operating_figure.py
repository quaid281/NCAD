#!/usr/bin/env python3
"""Generate publication-grade true-vector figure for Target-Horizon sweep and Deployable Quantiles (Figure 4).

Clean, spacious two-panel horizontal layout tailored for IEEE two-column full-width banner:
figsize=(7.14, 2.28) inches with zero internal text callouts, zero title/legend overlap,
and zero bottom-label clipping.
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
C_SOLREY_CAUS = "#1d4e89"   # Deep Navy (Soliton-Reynolds JEPA causal)
C_SOLREY0_CAUS = "#378d94"  # Deep Teal (Soliton-Reynolds alpha_sol=0 causal)
C_SOLREY_BUF = "#77b5b6"    # Muted Teal (Soliton-Reynolds JEPA buffered)
C_TSJEPA_CAUS = "#6a408d"   # Churkin Deep Purple (TS-JEPA causal)
C_TCNLN_CAUS = "#8c6d31"    # Warm Ochre (TCN Obs. Forecaster LN)
C_TIMESNET_CAUS = "#e13661" # Vibrant Coral / Crimson (TimesNet)
C_TRANAD_CAUS = "#4e4e4e"   # Slate Grey (TranAD)


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
    # Panel (a): Target Horizon Length S vs Held-Out FPR (5 streams, 3 seeds)
    # =========================================================================
    x_indices = np.arange(5)
    x_labels = [
        "$S{=}1$\n(lag 0)", "$S{=}4$\n(lag 3)", "$S{=}16$\n(lag 15)",
        "$S{=}32$\n(lag 31)", "$S{=}64$\n(lag 63)"
    ]

    # Exact 5-stream, 3-seed numbers from reports/horizon_latency_sweep.csv
    # and reports/horizon_latency_soliton_reynolds.csv
    fpr_timesnet_caus = np.array([0.0248, 0.0307, 0.0607, 0.0975, 0.2127]) * 100
    fpr_solrey_buf = np.array([0.0038, 0.0050, 0.0271, 0.0440, 0.0942]) * 100
    fpr_tsjepa_caus = np.array([0.0036, 0.0055, 0.0031, 0.0026, 0.0038]) * 100
    fpr_solrey_caus = np.array([0.0038, 0.0022, 0.0021, 0.0058, 0.0026]) * 100

    ax1.plot(x_indices, fpr_timesnet_caus, color=C_TIMESNET_CAUS, linestyle="--", linewidth=1.6, marker="s", markersize=4.6, label="TimesNet (caus.)", zorder=3)
    ax1.plot(x_indices, fpr_solrey_buf, color=C_SOLREY_BUF, linestyle=":", linewidth=1.5, marker="^", markersize=4.3, label="Sol.-Rey. JEPA (buf.)", zorder=3)
    ax1.plot(x_indices, fpr_tsjepa_caus, color=C_TSJEPA_CAUS, linestyle="-.", linewidth=1.5, marker="D", markersize=4.2, label="TS-JEPA (caus.)", zorder=4)
    ax1.plot(x_indices, fpr_solrey_caus, color=C_SOLREY_CAUS, linestyle="-", linewidth=1.8, marker="o", markersize=4.8, label="Sol.-Rey. JEPA (caus.)", zorder=5)

    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_labels, fontsize=7.3)
    ax1.set_xlabel("Target Horizon $S$ (Score-Emission Lag $S{-}1$)", labelpad=4)
    ax1.set_ylabel("Held-Out FPR (%)", labelpad=3)
    ax1.set_ylim(-0.8, 24.0)
    ax1.set_yticks(np.arange(0, 25, 5))
    ax1.set_yticklabels([f"{y}%" for y in np.arange(0, 25, 5)])

    ax1.grid(True, which="major", linestyle="-", linewidth=0.5, color="#e0e0e0", alpha=0.7)
    ax1.set_axisbelow(True)

    ax1.set_title("(a) Latency vs. False-Alarm Explosion across $S$", fontweight="bold", pad=26)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, columnspacing=1.0, handletextpad=0.35, labelspacing=0.25, fontsize=6.8)

    # =========================================================================
    # Panel (b): Event Recall vs False Alarm Rate across Deployable Quantiles q*
    #            (19 non-GHL streams, 5 seeds: SPOT q=1e-3 open -> q*=0.05 filled)
    # =========================================================================
    models_data = {
        "SolitonReynolds": {
            "fa": [5.6, 73.7], "er": [0.478, 0.752], "color": C_SOLREY_CAUS, "marker": "*", "label": "Sol.-Rey. JEPA"
        },
        "SolitonReynolds-a0": {
            "fa": [6.1, 74.4], "er": [0.439, 0.755], "color": C_SOLREY0_CAUS, "marker": "p", "label": "Sol.-Rey. ($\\alpha_{\\mathrm{sol}}{=}0$)"
        },
        "TS-JEPA": {
            "fa": [24.0, 86.3], "er": [0.487, 0.730], "color": C_TSJEPA_CAUS, "marker": "D", "label": "TS-JEPA"
        },
        "TCN-ObsPred-LN": {
            "fa": [117.0, 178.4], "er": [0.421, 0.665], "color": C_TCNLN_CAUS, "marker": "o", "label": "TCN Obs. (LN)"
        },
        "TimesNet": {
            "fa": [243.0, 262.3], "er": [0.588, 0.618], "color": C_TIMESNET_CAUS, "marker": "s", "label": "TimesNet"
        },
        "TranAD": {
            "fa": [222.0, 269.5], "er": [0.533, 0.602], "color": C_TRANAD_CAUS, "marker": "v", "label": "TranAD"
        },
    }

    for name, m in models_data.items():
        fa, er = m["fa"], m["er"]
        ax2.annotate("", xy=(fa[1], er[1]), xytext=(fa[0], er[0]),
                     arrowprops=dict(arrowstyle="->", color=m["color"], lw=1.35, mutation_scale=9),
                     zorder=3)
        ax2.scatter(fa[0], er[0], s=26, facecolors="none", edgecolors=m["color"], linewidths=1.3, zorder=4)
        sz_end = 48 if m["marker"] == "*" else 34
        ax2.scatter(fa[1], er[1], s=sz_end, facecolors=m["color"], edgecolors="#222222", linewidths=0.65, marker=m["marker"], label=m["label"], zorder=4)

    ax2.set_xlabel("False Alarms per 1,000 Nominal Steps", labelpad=4)
    ax2.set_ylabel("Causal Event Recall (ER)", labelpad=3)
    ax2.set_xlim(-8, 290)
    ax2.set_ylim(0.39, 0.86)
    ax2.set_yticks([0.40, 0.50, 0.60, 0.70, 0.80])

    ax2.grid(True, which="major", linestyle="-", linewidth=0.5, color="#e0e0e0", alpha=0.7)
    ax2.set_axisbelow(True)

    ax2.set_title("(b) Operating Trade-offs at Deployable Nominal $q^*$", fontweight="bold", pad=26)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, columnspacing=0.55, handletextpad=0.25, labelspacing=0.25, fontsize=6.6)

    saved = save_true_vector_figure(
        fig, OUT_STEM, formats=("svg", "pdf", "png"),
        png_dpi=300, inkscape_postprocess_pdf=True,
        ensure_pure_vector=True, ensure_outside_legend=True
    )
    plt.close(fig)
    print("Saved figure files:", saved)


if __name__ == "__main__":
    make_figure()
