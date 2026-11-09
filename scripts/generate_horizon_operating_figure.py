#!/usr/bin/env python3
"""Generate Figure 4: Multi-Horizon Scaling (S in {1,4,16,32,64}) and Operating-Point Curves.

Three-panel horizontal banner tailored for IEEE two-column full-width layout:
- Panel (a): Target Horizon S in {1, 4, 16, 32, 64} vs. Causal Point-F1 (5 streams, 3 seeds),
             comparing Laminar-JEPA (224B), Reynolds 224-Bit, and 1D Scalar SPOT.
- Panel (b): Target Horizon S in {1, 4, 16, 32, 64} vs. Held-Out Nominal FPR (%).
- Panel (c): Causal Event Recall vs. Realized False Alarms per 1,000 Nominal Steps on a
             square-root horizontal scale (FuncScale) across diagnostic budgets f* in {1%, 5%},
             autonomous Laminar-JEPA / 224-Bit / Strict operating points, and q* = 0.05.
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
C_LAMINAR = "#a31d3d"       # Deep Crimson-Burgundy (Laminar-JEPA 224-Bit)
C_BREAKER = "#7b3294"       # Royal Plum (Reynolds 224-Bit unquenched)
C_SOLREY_CAUS = "#1d4e89"   # Deep Navy (Reynolds 1D causal)
C_SOLREY_BUF = "#5ab4ac"    # Muted Teal (Reynolds 1D buffered)
C_TSJEPA_CAUS = "#6a408d"   # Churkin Deep Purple (TS-JEPA 1D causal)
C_TCNLN_CAUS = "#8c6d31"    # Warm Ochre (TCN Obs. Forecaster LN)
C_TIMESNET_CAUS = "#e13661" # Vibrant Coral-Red (TimesNet)
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
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.1,
        "axes.labelsize": 7.8,
        "axes.titlesize": 8.0,
        "legend.fontsize": 6.0,
        "figure.dpi": 300,
    })


def make_figure():
    set_ieee_style()
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(7.16, 2.28),
        gridspec_kw={"width_ratios": [1.0, 1.04, 1.26], "wspace": 0.26}
    )
    fig.subplots_adjust(top=0.70, bottom=0.22, left=0.060, right=0.985)

    x_indices = np.arange(5)
    x_labels = [
        "$S{=}1$\n(lag 0)", "$S{=}4$\n(lag 3)", "$S{=}16$\n(lag 15)",
        "$S{=}32$\n(lag 31)", "$S{=}64$\n(lag 63)"
    ]

    # =========================================================================
    # Panel (a): Horizon Scaling — Causal Point-F1 & Event Recall Gain
    # =========================================================================
    f1_laminar_caus = np.array([0.0369, 0.0496, 0.0554, 0.0740, 0.0781])
    f1_breaker_caus = np.array([0.0268, 0.0408, 0.0512, 0.0535, 0.0871])
    f1_scalar_caus = np.array([0.0144, 0.0115, 0.0325, 0.0309, 0.0442])

    ax1.fill_between(
        x_indices, f1_scalar_caus, np.maximum(f1_laminar_caus, f1_breaker_caus),
        color="#f3e5eb", alpha=0.55, zorder=1
    )
    ax1.plot(
        x_indices, f1_laminar_caus, color=C_LAMINAR, linestyle="-", linewidth=1.65,
        marker="*", markersize=6.0, markeredgecolor="#222222", markeredgewidth=0.45,
        label="Laminar-JEPA (224B)", zorder=6
    )
    ax1.plot(
        x_indices, f1_breaker_caus, color=C_BREAKER, linestyle="--", linewidth=1.5,
        marker="P", markersize=4.5, label="Reynolds 224B", zorder=5
    )
    ax1.plot(
        x_indices, f1_scalar_caus, color=C_SOLREY_CAUS, linestyle="-.", linewidth=1.4,
        marker="o", markersize=4.0, label="Reynolds 1D (SPOT)", zorder=4
    )

    callouts = [
        (0, 0.0369, "$2.6{\\times}$\n(82% ER)", (2, 5)),
        (1, 0.0496, "$4.3{\\times}$\n(84% ER)", (0, 5)),
        (3, 0.0740, "$2.4{\\times}$", (-2, 5)),
        (4, 0.0871, "$2.0{\\times}$", (-5, 4)),
    ]
    for idx, yval, txt, offset in callouts:
        ax1.annotate(
            txt, xy=(idx, yval), xytext=offset, textcoords="offset points",
            ha="center", va="bottom", fontsize=5.9, fontweight="bold", color="#5a1224",
            linespacing=0.88, zorder=7
        )

    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_labels, fontsize=6.7)
    ax1.set_xlabel("Target Horizon $S$ (Reporting Lag $S{-}1$)", labelpad=3.5)
    ax1.set_ylabel("Causal Point-F1", labelpad=2.5)
    ax1.set_xlim(-0.38, 4.38)
    ax1.set_ylim(0.002, 0.106)
    ax1.set_yticks([0.00, 0.02, 0.04, 0.06, 0.08, 0.10])
    ax1.grid(True, which="major", linestyle="-", linewidth=0.45, color="#e0e0e0", alpha=0.75)
    ax1.set_axisbelow(True)
    ax1.set_title("(a) Horizon Scaling: Causal Point-F1", fontweight="bold", pad=27)
    ax1.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False,
        columnspacing=0.55, handletextpad=0.25, labelspacing=0.18, fontsize=5.9
    )

    # =========================================================================
    # Panel (b): Horizon Scaling — Held-Out Nominal FPR (%)
    # =========================================================================
    fpr_timesnet_caus = np.array([0.0248, 0.0307, 0.0607, 0.0975, 0.2127]) * 100
    fpr_solrey_buf = np.array([0.0038, 0.0050, 0.0271, 0.0440, 0.0942]) * 100
    fpr_laminar_caus = np.array([0.0359, 0.0289, 0.0220, 0.0258, 0.0351]) * 100
    fpr_breaker_caus = np.array([0.0206, 0.0240, 0.0150, 0.0163, 0.0291]) * 100
    fpr_solrey_caus = np.array([0.0038, 0.0022, 0.0021, 0.0058, 0.0026]) * 100

    ax2.plot(
        x_indices, fpr_timesnet_caus, color=C_TIMESNET_CAUS, linestyle="--",
        linewidth=1.55, marker="s", markersize=4.2, label="TimesNet", zorder=3
    )
    ax2.plot(
        x_indices, fpr_solrey_buf, color=C_SOLREY_BUF, linestyle=":",
        linewidth=1.45, marker="^", markersize=4.0, label="Reynolds 1D (buf.)", zorder=3
    )
    ax2.plot(
        x_indices, fpr_laminar_caus, color=C_LAMINAR, linestyle="-",
        linewidth=1.6, marker="*", markersize=5.8, markeredgecolor="#222222",
        markeredgewidth=0.4, label="Laminar-JEPA", zorder=6
    )
    ax2.plot(
        x_indices, fpr_breaker_caus, color=C_BREAKER, linestyle="--",
        linewidth=1.45, marker="P", markersize=4.2, label="Reynolds 224B", zorder=5
    )
    ax2.plot(
        x_indices, fpr_solrey_caus, color=C_SOLREY_CAUS, linestyle="-.",
        linewidth=1.4, marker="o", markersize=3.8, label="Reynolds 1D", zorder=4
    )

    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(x_labels, fontsize=6.7)
    ax2.set_xlabel("Target Horizon $S$ (Reporting Lag $S{-}1$)", labelpad=3.5)
    ax2.set_ylabel("Held-Out FPR (%)", labelpad=2.0)
    ax2.set_xlim(-0.38, 4.38)
    ax2.set_ylim(-0.8, 23.8)
    ax2.set_yticks(np.arange(0, 25, 5))
    ax2.set_yticklabels([f"{y}%" for y in np.arange(0, 25, 5)])
    ax2.grid(True, which="major", linestyle="-", linewidth=0.45, color="#e0e0e0", alpha=0.75)
    ax2.set_axisbelow(True)
    ax2.set_title("(b) Horizon Scaling: Held-Out FPR", fontweight="bold", pad=27)
    ax2.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False,
        columnspacing=0.42, handletextpad=0.22, labelspacing=0.18, handlelength=1.45, fontsize=5.8
    )

    # =========================================================================
    # Panel (c): Causal Event Recall vs. Realized FPR on a Square-Root Scale
    # =========================================================================
    ax3.set_xscale(FuncScale(ax3, (lambda x: np.sqrt(np.maximum(0.0, x)), lambda x: np.square(x))))

    ax3.axvline(10.0, color="#9e9e9e", linestyle=":", linewidth=0.85, alpha=0.85, zorder=1)
    ax3.axvline(50.0, color="#9e9e9e", linestyle=":", linewidth=0.85, alpha=0.85, zorder=1)

    rand_x = [5.6, 10.0, 24.0, 50.0, 100.0, 150.0, 222.0, 243.0, 270.0]
    rand_y = [0.272, 0.276, 0.283, 0.365, 0.465, 0.535, 0.600, 0.601, 0.612]
    ax3.plot(
        rand_x, rand_y, color=C_RAND, linestyle=(0, (3, 2)),
        linewidth=1.1, label="Random", zorder=2
    )

    cb_pts = [(10.6, 0.534), (16.7, 0.607), (50.0, 0.728), (72.0, 0.774)]
    ax3.plot([p[0] for p in cb_pts], [p[1] for p in cb_pts], color=C_BREAKER, linestyle="-", linewidth=1.6, zorder=5)
    ax3.scatter([10.6], [0.534], s=26, facecolors="#ffffff", edgecolors=C_BREAKER, linewidths=1.35, zorder=6)
    ax3.scatter(
        [18.2], [0.604], s=36, facecolors=C_BREAKER, edgecolors="#222222",
        linewidths=0.65, marker="P", label="Reynolds 224B", zorder=7
    )
    ax3.scatter(
        [15.2], [0.628], s=60, facecolors=C_LAMINAR, edgecolors="#111111",
        linewidths=0.7, marker="*", label="Laminar-JEPA", zorder=8
    )

    models_data = {
        "SolitonReynolds": {
            "pts": [(5.6, 0.478), (10.0, 0.562), (50.0, 0.698), (73.7, 0.752)],
            "op": (5.6, 0.478), "q05": (73.7, 0.752),
            "color": C_SOLREY_CAUS, "ls": "--", "marker": "p", "label": "Reynolds 1D"
        },
        "TS-JEPA": {
            "pts": [(10.0, 0.395), (24.0, 0.487), (50.0, 0.584), (86.3, 0.730)],
            "op": (24.0, 0.487), "q05": (86.3, 0.730),
            "color": C_TSJEPA_CAUS, "ls": "-.", "marker": "D", "label": "TS-JEPA"
        },
        "TCN-ObsPred-LN": {
            "pts": [(10.0, 0.329), (50.0, 0.541), (178.4, 0.665)],
            "op": (117.0, 0.421), "q05": (178.4, 0.665),
            "color": C_TCNLN_CAUS, "ls": ":", "marker": "o", "label": "TCN (LN)"
        },
        "TimesNet": {
            "pts": [(10.0, 0.217), (50.0, 0.407), (243.0, 0.588), (262.3, 0.618)],
            "op": (243.0, 0.588), "q05": (262.3, 0.618),
            "color": C_TIMESNET_CAUS, "ls": "--", "marker": "s", "label": "TimesNet"
        },
        "TranAD": {
            "pts": [(10.0, 0.149), (50.0, 0.343), (222.0, 0.533), (269.5, 0.602)],
            "op": (222.0, 0.533), "q05": (269.5, 0.602),
            "color": C_TRANAD_CAUS, "ls": "-.", "marker": "v", "label": "TranAD"
        },
    }

    for name, m in models_data.items():
        xs = [p[0] for p in m["pts"]]
        ys = [p[1] for p in m["pts"]]
        ax3.plot(xs, ys, color=m["color"], linestyle=m["ls"], linewidth=1.3, zorder=3)
        for px, py in m["pts"]:
            if px in (10.0, 50.0):
                ax3.scatter(px, py, s=14, facecolors=m["color"], edgecolors="none", zorder=4)
        ax3.scatter(m["op"][0], m["op"][1], s=24, facecolors="#ffffff", edgecolors=m["color"], linewidths=1.25, zorder=5)
        ax3.scatter(
            m["q05"][0], m["q05"][1], s=29, facecolors=m["color"], edgecolors="#222222",
            linewidths=0.6, marker=m["marker"], label=m["label"], zorder=6
        )

    ax3.set_xlabel("False Alarms / 1,000 Steps (Realized FPR, $\\sqrt{\\cdot}$ scale)", labelpad=3.5)
    ax3.set_ylabel("Causal Event Recall (ER)", labelpad=2.0)
    ax3.set_xlim(2.5, 288.0)
    ax3.set_ylim(0.11, 0.82)
    ax3.set_xticks([10, 50, 100, 175, 250])
    ax3.set_xticklabels(["10\n(1%)", "50\n(5%)", "100\n(10%)", "175", "250\n(25%)"], fontsize=6.6)
    ax3.set_yticks([0.20, 0.35, 0.50, 0.65, 0.80])
    ax3.grid(True, which="major", linestyle="-", linewidth=0.45, color="#e0e0e0", alpha=0.75)
    ax3.set_axisbelow(True)
    ax3.set_title("(c) Operating Frontier & Diagnostic $f^*$ Curves", fontweight="bold", pad=27)
    ax3.legend(
        loc="lower center", bbox_to_anchor=(0.48, 1.02), ncol=4, frameon=False,
        columnspacing=0.35, handletextpad=0.18, labelspacing=0.18, handlelength=1.3, fontsize=5.8
    )

    saved = save_true_vector_figure(
        fig, OUT_STEM, formats=("svg", "pdf", "png"),
        png_dpi=300, inkscape_postprocess_pdf=True,
        ensure_pure_vector=True, ensure_outside_legend=True
    )
    plt.close(fig)
    print("Saved Figure 4:", saved)


if __name__ == "__main__":
    make_figure()
