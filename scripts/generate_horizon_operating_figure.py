#!/usr/bin/env python3
"""Generate Figure 4: Multi-Horizon Scaling, Operating Frontier, and Archive N_cal vs. FPR.

Three-panel horizontal banner tailored for IEEE two-column layout (7.16 in native width):
- Panel (a): Target Horizon S in {1, 4, 16, 32, 64} (reporting lag S-1 in {0, 3, 15, 31, 63})
             vs. Causal Point-F1 (5 exploratory streams, 3 seeds).
- Panel (b): Dev Causal Event Recall vs. Realized False Alarms per 1,000 Nominal Steps (n=19)
             on a square-root horizontal scale across diagnostic budgets f* in {1%, 5%}.
- Panel (c): Untouched Archive Stream-Level Calibration Sample Size N_cal vs. Causal Held-Out
             FPR_hold (n=118 streams, 3-seed means), highlighting the N_cal = 500 boundary
             and 1% / 5% operational budgets.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
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
ARCHIVE_CSV = ROOT / "reports" / "laminar_tier2_141stream_5block.csv"

# Curated Churkin / IEEE grayscale-safe high-contrast palette
C_LAMINAR = "#a31d3d"        # Deep Crimson-Burgundy (Laminar-JEPA 224-Bit)
C_LITE = "#0d7377"           # Deep Teal (Laminar-JEPA-Lite 128-Bit)
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
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "axes.labelsize": 8.1,
        "axes.titlesize": 8.3,
        "legend.fontsize": 6.6,
        "figure.dpi": 300,
    })


def load_archive_ncal_fpr():
    df = pd.read_csv(ARCHIVE_CSV)
    ho = df[(df["heldout"] == True) & (df["mapping"] == "trailing")]
    lite = (
        ho[(ho["model"] == "Laminar-JEPA-Lite") & (ho["calibrator"] == "PositionDrift128")]
        .groupby(["dataset", "channel"], as_index=False)
        .agg(n_cal=("n_cal", "mean"), fpr=("hold_fpr", "mean"))
    )
    mh1d = (
        ho[(ho["model"] == "Multi-Horizon-JEPA-1D") & (ho["calibrator"] == "SPOT")]
        .groupby(["dataset", "channel"], as_index=False)
        .agg(n_cal=("n_cal", "mean"), fpr=("hold_fpr", "mean"))
    )
    return lite, mh1d


def make_figure():
    set_ieee_style()
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(7.16, 2.22),
        gridspec_kw={"width_ratios": [0.95, 1.07, 1.02], "wspace": 0.255}
    )
    fig.subplots_adjust(top=0.735, bottom=0.225, left=0.058, right=0.994)

    x_indices = np.arange(5)
    x_labels = [
        "$S{=}1$\n(0)", "$S{=}4$\n(3)", "$S{=}16$\n(15)",
        "$S{=}32$\n(31)", "$S{=}64$\n(63)"
    ]

    # =========================================================================
    # Panel (a): Horizon Scaling — Causal Point-F1 & Event Recall Gain
    # =========================================================================
    f1_laminar_caus = np.array([0.0365, 0.0497, 0.0549, 0.0736, 0.0775])
    f1_lite_caus = np.array([0.0298, 0.0362, 0.0519, 0.0516, 0.0592])
    f1_scalar_caus = np.array([0.0142, 0.0115, 0.0318, 0.0311, 0.0444])

    ax1.fill_between(
        x_indices, f1_scalar_caus, np.maximum(f1_laminar_caus, f1_lite_caus),
        color="#f3e5eb", alpha=0.55, zorder=1
    )
    ax1.plot(
        x_indices, f1_laminar_caus, color=C_LAMINAR, linestyle="-", linewidth=1.6,
        marker="*", markersize=6.2, markeredgecolor="#222222", markeredgewidth=0.45,
        label="Laminar (224B)", zorder=6
    )
    ax1.plot(
        x_indices, f1_lite_caus, color=C_LITE, linestyle="--", linewidth=1.5,
        marker="P", markersize=4.6, label="Lite (128B)", zorder=5
    )
    ax1.plot(
        x_indices, f1_scalar_caus, color=C_MH1D_CAUS, linestyle="-.", linewidth=1.4,
        marker="o", markersize=4.0, label="Multi-Horiz. 1D", zorder=4
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
            ha="center", va="bottom", fontsize=6.5, fontweight="bold", color="#5a1224",
            zorder=7
        )

    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_labels, fontsize=7.0)
    ax1.set_xlabel("Horizon $S$ (Causal Lag $S{-}1$)", labelpad=3.0)
    ax1.set_ylabel("Causal Point-F1", labelpad=2.5)
    ax1.set_xlim(-0.44, 4.48)
    ax1.set_ylim(0.002, 0.103)
    ax1.set_yticks([0.00, 0.02, 0.04, 0.06, 0.08, 0.10])
    ax1.grid(True, which="major", linestyle="-", linewidth=0.45, color="#e0e0e0", alpha=0.75)
    ax1.set_axisbelow(True)
    ax1.set_title("(a) Horizon Sweep ($n{=}5$)", fontweight="bold", pad=23)
    ax1.legend(
        loc="lower center", bbox_to_anchor=(0.50, 1.015), ncol=2, frameon=False,
        columnspacing=0.65, handletextpad=0.30, labelspacing=0.18, fontsize=6.5
    )

    # =========================================================================
    # Panel (b): Causal Event Recall vs. Realized FPR on a Square-Root Scale
    # =========================================================================
    ax2.set_xscale(FuncScale(ax2, (lambda x: np.sqrt(np.maximum(0.0, x)), lambda x: np.square(x))))

    ax2.axvline(10.0, color="#9e9e9e", linestyle=":", linewidth=0.85, alpha=0.85, zorder=1)
    ax2.axvline(50.0, color="#9e9e9e", linestyle=":", linewidth=0.85, alpha=0.85, zorder=1)

    rand_x = [5.6, 10.0, 24.0, 50.0, 100.0, 150.0, 222.0, 243.0, 270.0]
    rand_y = [0.272, 0.276, 0.283, 0.365, 0.465, 0.535, 0.600, 0.601, 0.612]
    ax2.plot(
        rand_x, rand_y, color=C_RAND, linestyle=(0, (3, 2)),
        linewidth=1.1, label="Random", zorder=2
    )

    lam_pts = [(10.0, 0.612), (13.4, 0.673), (50.0, 0.757), (72.0, 0.795)]
    ax2.plot([p[0] for p in lam_pts], [p[1] for p in lam_pts], color=C_LAMINAR, linestyle="-", linewidth=1.55, zorder=6)
    ax2.scatter(
        [13.4], [0.673], s=58, facecolors=C_LAMINAR, edgecolors="#111111",
        linewidths=0.65, marker="*", label="Laminar", zorder=8
    )

    lite_pts = [(10.0, 0.560), (10.2, 0.650), (50.0, 0.772), (68.0, 0.811)]
    ax2.plot([p[0] for p in lite_pts], [p[1] for p in lite_pts], color=C_LITE, linestyle="-", linewidth=1.45, zorder=5)
    ax2.scatter(
        [10.2], [0.650], s=36, facecolors=C_LITE, edgecolors="#222222",
        linewidths=0.6, marker="P", label="Lite", zorder=7
    )

    models_data = {
        "MultiHorizon1D": {
            "pts": [(5.6, 0.478), (10.0, 0.562), (50.0, 0.698), (73.7, 0.752)],
            "op": (5.6, 0.478), "q05": (73.7, 0.752),
            "color": C_MH1D_CAUS, "ls": "--", "marker": "p", "label": "MH-1D"
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

    for _, m in models_data.items():
        xs = [p[0] for p in m["pts"]]
        ys = [p[1] for p in m["pts"]]
        ax2.plot(xs, ys, color=m["color"], linestyle=m["ls"], linewidth=1.25, zorder=3)
        for px, py in m["pts"]:
            if px in (10.0, 50.0):
                ax2.scatter(px, py, s=12, facecolors=m["color"], edgecolors="none", zorder=4)
        ax2.scatter(m["op"][0], m["op"][1], s=22, facecolors="#ffffff", edgecolors=m["color"], linewidths=1.1, zorder=5)
        ax2.scatter(
            m["q05"][0], m["q05"][1], s=26, facecolors=m["color"], edgecolors="#222222",
            linewidths=0.55, marker=m["marker"], label=m["label"], zorder=6
        )

    ax2.set_xlabel("Realized Test FPR ($\\sqrt{\\cdot}$ scale)", labelpad=3.0)
    ax2.set_ylabel("Causal Event Recall", labelpad=2.5)
    ax2.set_xlim(2.5, 288.0)
    ax2.set_ylim(0.11, 0.85)
    ax2.set_xticks([10, 50, 100, 175, 250])
    ax2.set_xticklabels(["1%", "5%", "10%", "17.5%", "25%"], fontsize=7.0)
    ax2.set_yticks([0.20, 0.35, 0.50, 0.65, 0.80])
    ax2.grid(True, which="major", linestyle="-", linewidth=0.45, color="#e0e0e0", alpha=0.75)
    ax2.set_axisbelow(True)
    ax2.set_title("(b) Dev Frontier ($n{=}19$)", fontweight="bold", pad=23)
    ax2.legend(
        loc="lower center", bbox_to_anchor=(0.50, 1.015), ncol=4, frameon=False,
        columnspacing=0.48, handletextpad=0.22, labelspacing=0.18, handlelength=1.2, fontsize=6.3
    )

    # =========================================================================
    # Panel (c): Archive Stream-Level N_cal vs. Causal Held-Out FPR (n=118)
    # =========================================================================
    lite_df, mh1d_df = load_archive_ncal_fpr()
    y_top = float(max(lite_df["fpr"].max(), mh1d_df["fpr"].max())) * 1.15
    ax3.set_xscale("log")
    ax3.set_yscale(FuncScale(ax3, (lambda y: np.sqrt(np.maximum(0.0, y)), lambda y: np.square(y))))

    ax3.axvline(500.0, color="#6a408d", linestyle="--", linewidth=1.0, alpha=0.85, zorder=2)
    ax3.axhline(0.01, color="#8a8a8a", linestyle=":", linewidth=0.85, alpha=0.9, zorder=1)
    ax3.axhline(0.05, color="#8a8a8a", linestyle="-.", linewidth=0.85, alpha=0.8, zorder=1)

    ax3.scatter(
        mh1d_df["n_cal"], mh1d_df["fpr"],
        s=15, facecolors="none", edgecolors=C_MH1D_CAUS, linewidths=0.75,
        alpha=0.72, marker="o", label="MH-1D", zorder=3
    )
    ax3.scatter(
        lite_df["n_cal"], lite_df["fpr"],
        s=18, facecolors=C_LITE, edgecolors="#073b3d", linewidths=0.45,
        alpha=0.82, marker="^", label="Lite (128B)", zorder=4
    )

    # Binned running median for Lite across calibration sizes
    bins = [100, 250, 500, 1200, 3000, 30000]
    med_x, med_y = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sub = lite_df[(lite_df["n_cal"] >= lo) & (lite_df["n_cal"] < hi)]
        if len(sub) >= 3:
            med_x.append(float(np.median(sub["n_cal"])))
            med_y.append(float(np.median(sub["fpr"])))
    ax3.plot(
        med_x, med_y, color="#a31d3d", linestyle="-", linewidth=1.5,
        marker="s", markersize=3.6, label="Lite med.", zorder=5
    )

    ax3.annotate(
        "$N_{\\mathrm{cal}}{=}500$", xy=(500.0, 0.68), xytext=(535.0, 0.68),
        fontsize=5.8, color="#6a408d", fontweight="bold", va="center", ha="left", zorder=6
    )

    ax3.set_xlabel("Calibration Windows $N_{\\mathrm{cal}}$ ($\\log_{10}$)", labelpad=3.0)
    ax3.set_ylabel("Causal $\\mathrm{FPR}_{\\mathrm{hold}}$ ($\\sqrt{\\cdot}$)", labelpad=2.0)
    ax3.set_xlim(90.0, 26000.0)
    ax3.set_ylim(0.0, max(0.86, y_top))
    ax3.set_xticks([100, 500, 2500, 15000])
    ax3.set_xticklabels(["100", "500", "2.5k", "15k"], fontsize=7.0)
    ax3.set_yticks([0.00, 0.01, 0.05, 0.20, 0.50, 0.80])
    ax3.set_yticklabels(["0%", "1%", "5%", "20%", "50%", "80%"], fontsize=7.0)
    ax3.grid(True, which="major", linestyle="-", linewidth=0.45, color="#e0e0e0", alpha=0.75)
    ax3.set_axisbelow(True)
    ax3.set_title("(c) Archive $N_{\\mathrm{cal}}$ vs. FPR ($n{=}118$)", fontweight="bold", pad=23)
    ax3.legend(
        loc="lower center", bbox_to_anchor=(0.50, 1.015), ncol=3, frameon=False,
        columnspacing=0.55, handletextpad=0.22, labelspacing=0.18, handlelength=1.1, fontsize=6.4
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
