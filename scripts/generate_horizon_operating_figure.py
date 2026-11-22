#!/usr/bin/env python3
"""Generate Figure 3: Multi-Horizon Scaling and Development Operating Frontier.

Two-panel horizontal banner tailored for IEEE two-column layout (7.16 in native width):
- Panel (a): Target Horizon S in {1, 4, 16, 32, 64} (reporting lag S-1 in {0, 3, 15, 31, 63})
             vs. Causal Point-F1 (5 exploratory streams, 3 seeds).
- Panel (b): Dev Causal Event Recall vs. Realized FPR (n=19, Table I(c))
             comparing each model's autonomous operating point (large legend marker)
             against post-hoc f* in {1%, 5%} test-FPR budgets (small dots) and Random alarming.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.scale import FuncScale

ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = ROOT / ".agents" / "skills" / "beautiful-research-figures" / "scripts"
if str(SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS))

from ieee_vector_style import save_true_vector_figure  # noqa: E402

FIGDIR = ROOT / "paper" / "NCAD_CS" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
OUT_STEM = FIGDIR / "operating_and_horizon_tradeoffs"

# Curated Churkin / IEEE grayscale-safe high-contrast palette
C_LAMINAR = "#a31d3d"        # Deep Crimson-Burgundy (Laminar-JEPA 224-Dial)
C_LITE = "#0d7377"           # Deep Teal (Laminar-JEPA-Lite 128-Dial)
C_MH1D_CAUS = "#1d4e89"      # Deep Navy (Multi-Horizon 1D causal)
C_TSJEPA_CAUS = "#6a408d"    # Churkin Deep Purple (TS-JEPA 1D causal)
C_TCNLN_CAUS = "#8c6d31"     # Warm Ochre (TCN Obs. Forecaster LN)
C_TIMESNET_CAUS = "#e13661"  # Vibrant Coral-Red (TimesNet)
C_TRANAD_CAUS = "#4e4e4e"    # Slate Grey (TranAD)
C_RAND = "#7f7f7f"           # Neutral Grey (Rate-matched random alerter)


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
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "axes.labelsize": 8.4,
        "axes.titlesize": 8.6,
        "legend.fontsize": 7.0,
        "figure.dpi": 300,
    })


def make_figure():
    set_ieee_style()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.16, 2.28),
        gridspec_kw={"width_ratios": [0.96, 1.04], "wspace": 0.22}
    )
    fig.subplots_adjust(top=0.715, bottom=0.22, left=0.068, right=0.990)

    x_indices = np.arange(5)
    x_labels = [
        "$S{=}1$\n(0)", "$S{=}4$\n(3)", "$S{=}16$\n(15)",
        "$S{=}32$\n(31)", "$S{=}64$\n(63)"
    ]

    # =========================================================================
    # Panel (a): Horizon Scaling — Causal Point-F1 & Relative Gain over 1D
    # =========================================================================
    f1_laminar_caus = np.array([0.0365, 0.0497, 0.0549, 0.0736, 0.0775])
    f1_lite_caus = np.array([0.0298, 0.0362, 0.0519, 0.0516, 0.0592])
    f1_scalar_caus = np.array([0.0142, 0.0115, 0.0318, 0.0311, 0.0444])

    ax1.fill_between(
        x_indices, f1_scalar_caus, np.maximum(f1_laminar_caus, f1_lite_caus),
        color="#f3e5eb", alpha=0.55, zorder=1
    )
    ax1.plot(
        x_indices, f1_laminar_caus, color=C_LAMINAR, linestyle="-", linewidth=1.7,
        marker="*", markersize=6.8, markeredgecolor="#222222", markeredgewidth=0.45,
        label="Laminar (224-Dial)", zorder=6
    )
    ax1.plot(
        x_indices, f1_lite_caus, color=C_LITE, linestyle="--", linewidth=1.6,
        marker="P", markersize=5.0, label="Lite (128-Dial)", zorder=5
    )
    ax1.plot(
        x_indices, f1_scalar_caus, color=C_MH1D_CAUS, linestyle="-.", linewidth=1.5,
        marker="o", markersize=4.4, label="Multi-Horiz. 1D", zorder=4
    )

    callouts = [
        (0, 0.0365, "$2.6{\\times}$", (0, 5)),
        (1, 0.0497, "$4.3{\\times}$", (0, 5)),
        (2, 0.0549, "$1.7{\\times}$", (0, 5)),
        (3, 0.0736, "$2.4{\\times}$", (-2, 5)),
        (4, 0.0775, "$1.7{\\times}$", (0, 5)),
    ]
    for idx, yval, txt, offset in callouts:
        ax1.annotate(
            txt, xy=(idx, yval), xytext=offset, textcoords="offset points",
            ha="center", va="bottom", fontsize=7.0, fontweight="bold", color="#5a1224",
            zorder=7
        )

    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_labels, fontsize=7.4)
    ax1.set_xlabel("Target Horizon $S\\,$ (Target-Window Offset $S{-}1$)", labelpad=3.0)
    ax1.set_ylabel("Causal Point-F1", labelpad=3.0)
    ax1.set_xlim(-0.42, 4.45)
    ax1.set_ylim(0.002, 0.102)
    ax1.set_yticks([0.00, 0.02, 0.04, 0.06, 0.08, 0.10])
    ax1.grid(True, which="major", linestyle="-", linewidth=0.45, color="#e0e0e0", alpha=0.8)
    ax1.set_axisbelow(True)
    ax1.set_title("(a) Horizon Sweep ($n{=}5$)", fontweight="bold", pad=27)
    ax1.legend(
        loc="lower center", bbox_to_anchor=(0.50, 1.03), ncol=3, frameon=False,
        columnspacing=0.75, handletextpad=0.30, labelspacing=0.20, handlelength=1.45, fontsize=6.9
    )

    # =========================================================================
    # Panel (b): Causal Event Recall vs. Realized FPR (Square-Root Scale)
    # Every large filled legend marker = autonomous operating point (Auto.).
    # Small dots at 1% and 5% = post-hoc test-FPR budgets (f* = 0.01, 0.05).
    # =========================================================================
    ax2.set_xscale(FuncScale(ax2, (lambda x: np.sqrt(np.maximum(0.0, x)), lambda x: np.square(x))))

    ax2.axvline(10.0, color="#9e9e9e", linestyle=":", linewidth=0.95, alpha=0.85, zorder=1)
    ax2.axvline(50.0, color="#9e9e9e", linestyle=":", linewidth=0.95, alpha=0.85, zorder=1)

    # Rate-matched random alerter baseline
    rand_x = [5.6, 10.0, 24.0, 50.0, 100.0, 150.0, 222.0, 243.0, 265.0]
    rand_y = [0.272, 0.276, 0.283, 0.365, 0.465, 0.535, 0.600, 0.601, 0.610]
    ax2.plot(
        rand_x, rand_y, color=C_RAND, linestyle=(0, (3, 2)),
        linewidth=1.25, label="Random", zorder=2
    )

    # All 7 models with unified visual grammar:
    # - Small dots at f*=1% and f*=5% connected by model-styled line
    # - Large filled legend marker strictly at the autonomous operating point (Auto.)
    models = [
        {
            "label": "Laminar", "color": C_LAMINAR, "ls": "-", "marker": "*", "ms": 74,
            "f1": (10.0, 0.612), "f5": (50.0, 0.757), "auto": (13.4, 0.673),
            "lw": 1.65, "zorder": 8,
        },
        {
            "label": "Lite", "color": C_LITE, "ls": "-", "marker": "P", "ms": 48,
            "f1": (10.0, 0.560), "f5": (50.0, 0.772), "auto": (10.2, 0.650),
            "lw": 1.55, "zorder": 7,
        },
        {
            "label": "MH-1D", "color": C_MH1D_CAUS, "ls": "--", "marker": "p", "ms": 40,
            "f1": (10.0, 0.562), "f5": (50.0, 0.698), "auto": (5.6, 0.478),
            "lw": 1.35, "zorder": 6,
        },
        {
            "label": "TS-JEPA", "color": C_TSJEPA_CAUS, "ls": "-.", "marker": "D", "ms": 34,
            "f1": (10.0, 0.395), "f5": (50.0, 0.584), "auto": (24.0, 0.487),
            "lw": 1.35, "zorder": 5,
        },
        {
            "label": "TCN (LN)", "color": C_TCNLN_CAUS, "ls": ":", "marker": "o", "ms": 38,
            "f1": (10.0, 0.329), "f5": (50.0, 0.541), "auto": (117.0, 0.421),
            "lw": 1.35, "zorder": 5,
        },
        {
            "label": "TimesNet", "color": C_TIMESNET_CAUS, "ls": "--", "marker": "s", "ms": 36,
            "f1": (10.0, 0.217), "f5": (50.0, 0.407), "auto": (243.0, 0.588),
            "lw": 1.35, "zorder": 5,
        },
        {
            "label": "TranAD", "color": C_TRANAD_CAUS, "ls": "-.", "marker": "v", "ms": 40,
            "f1": (10.0, 0.149), "f5": (50.0, 0.343), "auto": (222.0, 0.533),
            "lw": 1.35, "zorder": 5,
        },
    ]

    for m in models:
        # Connect f*=1%, f*=5%, and Auto. in ascending FPR order
        pts = sorted([m["f1"], m["f5"], m["auto"]], key=lambda p: p[0])
        ax2.plot(
            [p[0] for p in pts], [p[1] for p in pts],
            color=m["color"], linestyle=m["ls"], linewidth=m["lw"], zorder=m["zorder"] - 2
        )
        # Small dots mark post-hoc f*=1% and f*=5% test-FPR budgets
        for px, py in (m["f1"], m["f5"]):
            ax2.scatter(px, py, s=16, facecolors=m["color"], edgecolors="none", zorder=m["zorder"] - 1)
        # Large filled legend marker marks the autonomous (label-free) operating point
        ax2.scatter(
            m["auto"][0], m["auto"][1], s=m["ms"],
            facecolors=m["color"], edgecolors="#1a1a1a", linewidths=0.65,
            marker=m["marker"], label=m["label"], zorder=m["zorder"]
        )

    # Subtle direct labels for the two post-hoc reference lines below top border
    ax2.text(10.0, 0.855, "$f^*{=}1\\%$", ha="center", va="top", fontsize=6.4, color="#555555",
             bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.85}, zorder=9)
    ax2.text(50.0, 0.855, "$f^*{=}5\\%$", ha="center", va="top", fontsize=6.4, color="#555555",
             bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.85}, zorder=9)

    ax2.set_xlabel("Realized FPR ($\\sqrt{\\cdot}$ scale)", labelpad=3.0)
    ax2.set_ylabel("Causal Event Recall", labelpad=3.0)
    ax2.set_xlim(2.5, 278.0)
    ax2.set_ylim(0.11, 0.88)
    ax2.set_xticks([10, 50, 100, 175, 250])
    ax2.set_xticklabels(["1%", "5%", "10%", "17.5%", "25%"], fontsize=7.4)
    ax2.set_yticks([0.20, 0.35, 0.50, 0.65, 0.80])
    ax2.grid(True, which="major", linestyle="-", linewidth=0.45, color="#e0e0e0", alpha=0.8)
    ax2.set_axisbelow(True)
    ax2.set_title("(b) Dev Operating Frontier ($n{=}19$)", fontweight="bold", pad=27)
    ax2.legend(
        loc="lower center", bbox_to_anchor=(0.50, 1.015), ncol=4, frameon=False,
        columnspacing=0.68, handletextpad=0.26, labelspacing=0.18, handlelength=1.3, fontsize=6.8
    )

    saved = save_true_vector_figure(
        fig, OUT_STEM, formats=("svg", "pdf", "png"),
        png_dpi=300, inkscape_postprocess_pdf=True,
        ensure_pure_vector=True, ensure_outside_legend=True
    )
    plt.close(fig)
    print("Saved Figure 3:", saved)


if __name__ == "__main__":
    make_figure()
