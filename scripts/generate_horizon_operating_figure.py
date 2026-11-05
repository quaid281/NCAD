#!/usr/bin/env python3
"""Generate publication-grade true-vector figure for Target-Horizon sweep and Operating Curves (Figure 4).

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
C_RAND = "#7f7f7f"          # Neutral Grey (Rate-matched random alerter)


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
    # Panel (b): Event Recall vs False Alarm Rate across Equalized Budgets
    #            (f* in {1%, 5%}) and Deployable Quantiles (SPOT q=1e-3, q*=0.05)
    #            (19 non-GHL streams, 5 seeds)
    # =========================================================================
    # Vertical reference guides at equalized realized FPR budgets f* = 1% (10/1k) and f* = 5% (50/1k)
    ax2.axvline(10.0, color="#9e9e9e", linestyle=":", linewidth=0.9, alpha=0.8, zorder=1)
    ax2.axvline(50.0, color="#9e9e9e", linestyle=":", linewidth=0.9, alpha=0.8, zorder=1)

    # Analytical rate-matched Bernoulli random alerter baseline across FPR
    rand_x = [5.6, 10.0, 24.0, 50.0, 100.0, 150.0, 222.0, 243.0, 270.0]
    rand_y = [0.272, 0.276, 0.283, 0.365, 0.465, 0.535, 0.600, 0.601, 0.612]
    ax2.plot(rand_x, rand_y, color=C_RAND, linestyle=(0, (3, 2)), linewidth=1.15, label="Random Alerter", zorder=2)

    models_data = {
        "SolitonReynolds": {
            "pts": [(5.6, 0.478), (10.0, 0.562), (50.0, 0.698), (73.7, 0.752)],
            "spot": (5.6, 0.478), "q05": (73.7, 0.752),
            "color": C_SOLREY_CAUS, "ls": "-", "marker": "*", "label": "Sol.-Rey. JEPA"
        },
        "SolitonReynolds-a0": {
            "pts": [(6.1, 0.439), (10.0, 0.578), (50.0, 0.711), (74.4, 0.755)],
            "spot": (6.1, 0.439), "q05": (74.4, 0.755),
            "color": C_SOLREY0_CAUS, "ls": "--", "marker": "p", "label": "Sol.-Rey. ($\\alpha_{\\mathrm{sol}}{=}0$)"
        },
        "TS-JEPA": {
            "pts": [(10.0, 0.395), (24.0, 0.487), (50.0, 0.584), (86.3, 0.730)],
            "spot": (24.0, 0.487), "q05": (86.3, 0.730),
            "color": C_TSJEPA_CAUS, "ls": "-.", "marker": "D", "label": "TS-JEPA"
        },
        "TCN-ObsPred-LN": {
            "pts": [(10.0, 0.329), (50.0, 0.541), (178.4, 0.665)],
            "spot": (117.0, 0.421), "q05": (178.4, 0.665),
            "color": C_TCNLN_CAUS, "ls": ":", "marker": "o", "label": "TCN Obs. (LN)"
        },
        "TimesNet": {
            "pts": [(10.0, 0.217), (50.0, 0.407), (243.0, 0.588), (262.3, 0.618)],
            "spot": (243.0, 0.588), "q05": (262.3, 0.618),
            "color": C_TIMESNET_CAUS, "ls": "--", "marker": "s", "label": "TimesNet"
        },
        "TranAD": {
            "pts": [(10.0, 0.149), (50.0, 0.343), (222.0, 0.533), (269.5, 0.602)],
            "spot": (222.0, 0.533), "q05": (269.5, 0.602),
            "color": C_TRANAD_CAUS, "ls": "-.", "marker": "v", "label": "TranAD"
        },
    }

    for name, m in models_data.items():
        xs = [p[0] for p in m["pts"]]
        ys = [p[1] for p in m["pts"]]
        ax2.plot(xs, ys, color=m["color"], linestyle=m["ls"], linewidth=1.35, zorder=3)
        # Plot matched-FPR vertices (f* = 1% at x=10, f* = 5% at x=50)
        for px, py in m["pts"]:
            if px in (10.0, 50.0):
                ax2.scatter(px, py, s=16, facecolors=m["color"], edgecolors="none", zorder=4)
        # Open circle at SPOT (q=1e-3)
        ax2.scatter(m["spot"][0], m["spot"][1], s=28, facecolors="#ffffff", edgecolors=m["color"], linewidths=1.3, zorder=5)
        # Filled distinct marker at q* = 0.05
        sz_end = 48 if m["marker"] == "*" else 34
        ax2.scatter(m["q05"][0], m["q05"][1], s=sz_end, facecolors=m["color"], edgecolors="#222222", linewidths=0.65, marker=m["marker"], label=m["label"], zorder=6)

    ax2.set_xlabel("False Alarms per 1,000 Nominal Steps (Realized FPR)", labelpad=4)
    ax2.set_ylabel("Causal Event Recall (ER)", labelpad=3)
    ax2.set_xlim(-6, 288)
    ax2.set_ylim(0.11, 0.82)
    ax2.set_xticks([10, 50, 100, 150, 200, 250])
    ax2.set_xticklabels(["10\n(1%)", "50\n(5%)", "100\n(10%)", "150", "200", "250\n(25%)"], fontsize=7.0)
    ax2.set_yticks([0.20, 0.35, 0.50, 0.65, 0.80])

    ax2.grid(True, which="major", linestyle="-", linewidth=0.5, color="#e0e0e0", alpha=0.7)
    ax2.set_axisbelow(True)

    ax2.set_title("(b) Matched-FPR ($1\\%, 5\\%$) & Deployable Operating Curves", fontweight="bold", pad=26)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False, columnspacing=0.45, handletextpad=0.22, labelspacing=0.22, fontsize=6.3)

    saved = save_true_vector_figure(
        fig, OUT_STEM, formats=("svg", "pdf", "png"),
        png_dpi=300, inkscape_postprocess_pdf=True,
        ensure_pure_vector=True, ensure_outside_legend=True
    )
    plt.close(fig)
    print("Saved figure files:", saved)


if __name__ == "__main__":
    make_figure()
