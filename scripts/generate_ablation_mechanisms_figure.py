#!/usr/bin/env python3
"""Generate publication-grade true-vector figure for Ablations 2 & 3 (formerly Table II(c) & II(d)):
Panel (a): Factorial and Strength Sweep of the Operator Regularizer across lambda_ent in [0, 2.0].
Panel (b): Injected Quiet-Fault Event & Point Recall across Input Gates and Hybrid Union Monitor.

Follows Dr. Andrey Churkin's 8 Principles of Beautiful Figures and IEEE two-column standards:
- 100% true vector (.svg and .pdf) with TrueType outlines (fonttype=42)
- Strict external-only legends (bbox_to_anchor outside [0, 1] x [0, 1])
- Harmonious grayscale-safe palette with darker edge strokes
- Subtle dual-hierarchy grids behind data (zorder discipline)
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
OUT_STEM = FIGDIR / "regularizer_sweep_and_quiet_faults"

# Churkin / IEEE Grayscale-Safe Palette
C_PURPLE_FILL = "#9671bd"
C_PURPLE_EDGE = "#6a408d"
C_TEAL_FILL = "#77b5b6"
C_TEAL_EDGE = "#378d94"
C_NAVY_FILL = "#4f7cac"
C_NAVY_EDGE = "#1d4e89"
C_GREY_FILL = "#b0b0b0"
C_GREY_EDGE = "#4e4e4e"
C_CORAL_FILL = "#f28482"
C_CORAL_EDGE = "#c1246b"


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
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "axes.labelsize": 8.6,
        "axes.titlesize": 8.8,
        "legend.fontsize": 8.0,
        "figure.dpi": 300,
    })


def make_figure():
    set_ieee_style()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.14, 1.96),
        gridspec_kw={"width_ratios": [1.0, 1.08], "wspace": 0.44}
    )
    fig.subplots_adjust(top=0.70, bottom=0.24, left=0.068, right=0.988)

    # =========================================================================
    # Panel (a): Operator Regularizer Strength Sweep (formerly Table II(c))
    # =========================================================================
    # 7 categorical-ordinal stations along the sweep:
    # 0: Neither (0, 0), 1: Barrier only (0, 1), 2: (0.05, 1), 3: (0.10, 1),
    # 4: Deployed (0.50, 1) [plus Entropy-only (0.50, 0)], 5: (1.00, 1), 6: (2.00, 1)
    x_idx = np.arange(7)
    x_labels = [
        "0\n($\\beta{=}0$)", "0\n($\\beta{=}1$)", "0.05",
        "0.10", "0.50*\n(depl.)", "1.00", "2.00"
    ]

    kappa_k = np.array([1.437, 1.431, 1.395, 1.363, 1.255, 1.204, 1.154])
    cv_loss = np.array([0.565, 0.563, 0.562, 0.559, 0.549, 0.546, 0.545])
    caus_er = np.array([0.369, 0.389, 0.373, 0.421, 0.444, 0.365, 0.397])
    buf_er = np.array([0.250, 0.266, 0.250, 0.250, 0.266, 0.250, 0.250])
    caus_fpr = np.array([0.017, 0.016, 0.017, 0.017, 0.017, 0.017, 0.017])
    buf_fpr = np.array([0.071, 0.074, 0.074, 0.073, 0.072, 0.070, 0.072])
    rank_tgt_ratio = np.array([11.57, 11.58, 11.57, 11.58, 11.56, 11.55, 11.54]) / 32.0

    ax1_r = ax1.twinx()
    ax1_r.spines["right"].set_edgecolor(C_TEAL_EDGE)
    ax1_r.spines["right"].set_linewidth(0.75)
    ax1.spines["left"].set_edgecolor(C_PURPLE_EDGE)

    # Vertical highlight at deployed lambda_ent = 0.50 (index 4)
    ax1.axvline(4, color="#9e9e9e", linestyle="--", linewidth=0.85, alpha=0.65, zorder=1)

    # Left axis: Condition number kappa(K)
    l1, = ax1.plot(
        x_idx, kappa_k, color=C_PURPLE_EDGE, linestyle="-", linewidth=1.8,
        marker="o", markersize=4.8, markerfacecolor=C_PURPLE_FILL,
        markeredgecolor=C_PURPLE_EDGE, label="$\\kappa(\\mathbf{K})$ (left)", zorder=4
    )
    # Entropy-only (0.5, 0) control marker for kappa(K)=1.256 & Causal ER=0.389
    l_ent0, = ax1_r.plot(
        [4], [0.389], color=C_NAVY_EDGE, linestyle="none",
        marker="D", markersize=4.6, markerfacecolor="none", markeredgewidth=1.2,
        label="Caus. ER ($\\beta{=}0$)", zorder=5
    )

    # Right axis: Causal ER, Buffered ER, Target Rank ratio, Causal FPR
    l2, = ax1_r.plot(
        x_idx, caus_er, color=C_TEAL_EDGE, linestyle="-", linewidth=1.7,
        marker="s", markersize=4.4, markerfacecolor=C_TEAL_FILL,
        markeredgecolor=C_TEAL_EDGE, label="Caus. ER (right)", zorder=4
    )
    l3, = ax1_r.plot(
        x_idx, rank_tgt_ratio, color=C_GREY_EDGE, linestyle=":", linewidth=1.4,
        marker="^", markersize=4.0, markerfacecolor=C_GREY_FILL,
        markeredgecolor=C_GREY_EDGE, label="Target Rank / $D$", zorder=3
    )
    l4, = ax1_r.plot(
        x_idx, buf_fpr, color=C_NAVY_EDGE, linestyle="--", linewidth=1.3,
        marker="v", markersize=3.8, markerfacecolor=C_NAVY_FILL,
        markeredgecolor=C_NAVY_EDGE, label="Buf. FPR", zorder=3
    )
    l5, = ax1_r.plot(
        x_idx, caus_fpr, color=C_CORAL_EDGE, linestyle="-.", linewidth=1.35,
        marker="p", markersize=4.0, markerfacecolor=C_CORAL_FILL,
        markeredgecolor=C_CORAL_EDGE, label="Caus. FPR", zorder=3
    )

    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(x_labels, fontsize=8.0)
    ax1.set_xlim(-0.35, 6.35)
    ax1.set_ylim(1.10, 1.48)
    ax1.set_yticks([1.15, 1.25, 1.35, 1.45])
    ax1.set_xlabel("Spectral Entropy Weight (with Barrier $\\beta$)", labelpad=2)
    ax1.set_ylabel("Condition Number $\\kappa(\\mathbf{K})$", color=C_PURPLE_EDGE, labelpad=3)
    ax1.tick_params(axis="y", colors=C_PURPLE_EDGE)

    ax1_r.set_ylim(0.0, 0.50)
    ax1_r.set_yticks([0.0, 0.15, 0.30, 0.45])
    ax1_r.set_ylabel("Event Recall / FPR / Rank Ratio", color=C_TEAL_EDGE, labelpad=3)
    ax1_r.tick_params(axis="y", colors=C_TEAL_EDGE)

    ax1.grid(True, which="major", linestyle="-", linewidth=0.5, color="#e0e0e0", alpha=0.65)
    ax1.set_axisbelow(True)

    ax1.set_title("(a) Operator Regularizer Conditioning & Stability Sweep", fontweight="bold", pad=27)
    handles_a = [l1, l2, l_ent0, l3, l4, l5]
    labels_a = [h.get_label() for h in handles_a]
    ax1.legend(
        handles_a, labels_a, loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=3, frameon=False, columnspacing=0.60, handletextpad=0.28,
        labelspacing=0.20, fontsize=7.8
    )

    # =========================================================================
    # Panel (b): Quiet-Fault Event & Point Recall (formerly Table II(d))
    # =========================================================================
    fault_labels = [
        "Flatline\n(const.)", "Stuck\nsensor", "Sensor\ndropout",
        "Flatline\n(mid)", "Low-amp.\nstep", "$+5\\sigma$\nspike"
    ]
    n_faults = len(fault_labels)
    x_f = np.arange(n_faults)

    # Event recall (bars) and Point recall (overlaid markers)
    monitors = [
        {
            "label": "Unif. (1.9%)",
            "ev": [0.19, 0.19, 0.19, 0.19, 0.19, 0.28],
            "pt": [0.02, 0.02, 0.02, 0.02, 0.03, 0.02],
            "fill": "#d4d4d4", "edge": "#4e4e4e"
        },
        {
            "label": "Var. (6.9%)",
            "ev": [0.28, 0.28, 0.28, 0.28, 0.19, 0.31],
            "pt": [0.04, 0.04, 0.04, 0.05, 0.05, 0.04],
            "fill": C_NAVY_FILL, "edge": C_NAVY_EDGE
        },
        {
            "label": "MAD (5.5%)",
            "ev": [0.17, 0.17, 0.17, 0.19, 0.19, 0.19],
            "pt": [0.06, 0.06, 0.06, 0.07, 0.06, 0.06],
            "fill": C_PURPLE_FILL, "edge": C_PURPLE_EDGE
        },
        {
            "label": "Var. Monitor (14.9%)",
            "ev": [0.26, 0.26, 0.26, 0.39, 0.26, 0.26],
            "pt": [0.18, 0.18, 0.18, 0.32, 0.18, 0.18],
            "fill": C_TEAL_FILL, "edge": C_TEAL_EDGE
        },
        {
            "label": "Hybrid (17.0%)",
            "ev": [0.42, 0.42, 0.42, 0.42, 0.42, 0.50],
            "pt": [0.20, 0.20, 0.20, 0.32, 0.21, 0.21],
            "fill": C_CORAL_FILL, "edge": C_CORAL_EDGE
        },
    ]

    bar_w = 0.15
    offsets = (np.arange(len(monitors)) - 2.0) * bar_w

    bar_handles = []
    pt_handle = None
    for idx, m in enumerate(monitors):
        pos = x_f + offsets[idx]
        b = ax2.bar(
            pos, m["ev"], width=bar_w * 0.90, color=m["fill"],
            edgecolor=m["edge"], linewidth=0.65, label=m["label"], zorder=3
        )
        bar_handles.append(b[0])
        pts = ax2.scatter(
            pos, m["pt"], s=14, marker="o", facecolors="#ffffff",
            edgecolors=m["edge"], linewidths=0.85, zorder=4,
            label="Pt. Recall" if idx == 0 else "_nolegend_"
        )
        if idx == 0:
            pt_handle = pts

    ax2.set_xticks(x_f)
    ax2.set_xticklabels(fault_labels, fontsize=8.0)
    ax2.set_xlim(-0.58, n_faults - 0.42)
    ax2.set_ylim(0.0, 0.56)
    ax2.set_yticks([0.0, 0.15, 0.30, 0.45])
    ax2.set_xlabel("Injected Single-Channel Fault Class ($n{=}6$ Streams)", labelpad=2)
    ax2.set_ylabel("Recall (Bars: Event; Circles: Point)", labelpad=3)

    ax2.grid(True, which="major", axis="y", linestyle="-", linewidth=0.5, color="#e0e0e0", alpha=0.65)
    ax2.set_axisbelow(True)

    ax2.set_title("(b) Quiet-Fault Recovery via Hybrid Sensor Monitor", fontweight="bold", pad=27)
    handles_b = bar_handles + [pt_handle]
    labels_b = [m["label"] for m in monitors] + ["Point Recall"]
    ax2.legend(
        handles_b, labels_b, loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=3, frameon=False, columnspacing=0.60, handletextpad=0.28,
        labelspacing=0.20, fontsize=7.8
    )

    saved = save_true_vector_figure(
        fig, OUT_STEM, formats=("svg", "pdf", "png"),
        png_dpi=300, inkscape_postprocess_pdf=True,
        ensure_pure_vector=True, ensure_outside_legend=True
    )
    plt.close(fig)
    print("Saved figure files:", saved)


if __name__ == "__main__":
    make_figure()
