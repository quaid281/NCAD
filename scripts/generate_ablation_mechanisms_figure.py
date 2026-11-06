#!/usr/bin/env python3
"""Generate Figure 3: Multi-Perspective Circuit-Breaker Progression & Quiet-Fault Recovery.

Panel (a): Progression from 1D Scalar Collapse (TS-JEPA, Sol.-Rey. 1D, alpha_sol=0)
           to 128-Bit Sub-Block Panel, 224-Bit Strict, 224-Bit Breaker (with 32 Onset Relays),
           and 224-Bit + Soft Attention across the 19 primary non-GHL streams.
Panel (b): Injected Single-Channel Quiet-Fault Event & Point Recall across Input Gates
           and the Hybrid Union Monitor.
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
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 8.0,
        "axes.labelsize": 8.4,
        "axes.titlesize": 8.6,
        "legend.fontsize": 7.4,
        "figure.dpi": 300,
    })


def make_figure():
    set_ieee_style()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.14, 2.02),
        gridspec_kw={"width_ratios": [1.02, 1.06], "wspace": 0.42}
    )
    fig.subplots_adjust(top=0.70, bottom=0.24, left=0.068, right=0.988)

    # =========================================================================
    # Panel (a): Perspective Expansion Ladder (1 -> 32 -> 128 -> 160 -> 192 -> 224 Dials)
    # =========================================================================
    x_idx = np.arange(6)
    x_labels = [
        "1D Scalar\n(1 Dial)",
        "32-Bit\n($m{=}4$)",
        "128-Bit\nSub-Blk",
        "160-Bit\n(+Drift)",
        "192-Bit\n(+Fluc.)",
        "224-Bit\nBreaker"
    ]

    # Measured 19-stream metrics across the 6-step perspective expansion ladder:
    # Causal Event Recall, Buffered PR-AUC, Buffered Point-F1, Causal Point-F1, Causal Held-Out FPR
    caus_er = np.array([0.398, 0.643, 0.590, 0.585, 0.575, 0.607])
    pr_buf = np.array([0.370, 0.366, 0.402, 0.404, 0.408, 0.411])
    f1_buf = np.array([0.137, 0.119, 0.192, 0.188, 0.203, 0.228])
    f1_caus = np.array([0.052, 0.101, 0.129, 0.132, 0.131, 0.138])
    fpr_caus = np.array([0.007, 0.009, 0.011, 0.012, 0.012, 0.017])
    delay_caus = np.array([246.6, 114.4, 146.2, 69.1, 179.0, 105.9])

    ax1_r = ax1.twinx()
    ax1_r.spines["right"].set_edgecolor(C_CORAL_EDGE)
    ax1_r.spines["right"].set_linewidth(0.75)
    ax1.spines["left"].set_edgecolor(C_NAVY_EDGE)

    # Vertical separator between 1D Scalar Collapse and Multi-Bit Circuit Breakers
    ax1.axvline(0.5, color="#888888", linestyle=":", linewidth=0.95, alpha=0.80, zorder=1)
    # Highlight deployed 224-Bit Breaker
    ax1.axvline(5.0, color="#9e9e9e", linestyle="--", linewidth=0.85, alpha=0.65, zorder=1)

    l1, = ax1.plot(
        x_idx, caus_er, color=C_TEAL_EDGE, linestyle="-", linewidth=1.75,
        marker="s", markersize=4.5, markerfacecolor=C_TEAL_FILL,
        markeredgecolor=C_TEAL_EDGE, label="Caus. ER", zorder=4
    )
    l2, = ax1.plot(
        x_idx, pr_buf, color=C_NAVY_EDGE, linestyle="-", linewidth=1.6,
        marker="D", markersize=4.2, markerfacecolor=C_NAVY_FILL,
        markeredgecolor=C_NAVY_EDGE, label="Buf. PR-AUC", zorder=4
    )
    l3, = ax1.plot(
        x_idx, f1_buf, color=C_PURPLE_EDGE, linestyle="--", linewidth=1.55,
        marker="^", markersize=4.4, markerfacecolor=C_PURPLE_FILL,
        markeredgecolor=C_PURPLE_EDGE, label="Buf. Pt-F1", zorder=4
    )
    l4, = ax1.plot(
        x_idx, f1_caus, color=C_GREY_EDGE, linestyle="-.", linewidth=1.5,
        marker="o", markersize=4.2, markerfacecolor=C_GREY_FILL,
        markeredgecolor=C_GREY_EDGE, label="Caus. Pt-F1", zorder=4
    )
    l5, = ax1.plot(
        x_idx, fpr_caus, color="#222222", linestyle=":", linewidth=1.3,
        marker="v", markersize=3.8, markerfacecolor="#ffffff",
        markeredgecolor="#222222", label="Caus. FPR", zorder=3
    )
    l6, = ax1_r.plot(
        x_idx, delay_caus, color=C_CORAL_EDGE, linestyle="--", linewidth=1.5,
        marker="p", markersize=4.4, markerfacecolor=C_CORAL_FILL,
        markeredgecolor=C_CORAL_EDGE, label="Delay $\\Delta t$ (right)", zorder=3
    )

    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(x_labels, fontsize=7.4)
    ax1.set_xlim(-0.38, 5.38)
    ax1.set_ylim(-0.02, 0.68)
    ax1.set_yticks([0.0, 0.20, 0.40, 0.60])
    ax1.set_xlabel("Scoring & Circuit-Breaker Architecture ($n{=}19$ Streams)", labelpad=2)
    ax1.set_ylabel("ER / PR-AUC / Point-F1 / FPR", color=C_NAVY_EDGE, labelpad=3)
    ax1.tick_params(axis="y", colors=C_NAVY_EDGE)

    ax1_r.set_ylim(60.0, 280.0)
    ax1_r.set_yticks([80, 140, 200, 260])
    ax1_r.set_ylabel("Causal Delay $\\Delta t$ (steps)", color=C_CORAL_EDGE, labelpad=3)
    ax1_r.tick_params(axis="y", colors=C_CORAL_EDGE)

    ax1.grid(True, which="major", linestyle="-", linewidth=0.5, color="#e0e0e0", alpha=0.65)
    ax1.set_axisbelow(True)

    ax1.set_title("(a) 1D Collapse vs. 224-Bit Circuit-Breaker Panel", fontweight="bold", pad=27)
    handles_a = [l1, l2, l3, l4, l5, l6]
    labels_a = [h.get_label() for h in handles_a]
    ax1.legend(
        handles_a, labels_a, loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=3, frameon=False, columnspacing=0.55, handletextpad=0.25,
        labelspacing=0.20, fontsize=7.3
    )

    # =========================================================================
    # Panel (b): Quiet-Fault Event & Point Recall (n=6 streams, 3 seeds)
    # =========================================================================
    fault_labels = [
        "Flatline\n(const.)", "Stuck\nsensor", "Sensor\ndropout",
        "Flatline\n(mid)", "Low-amp.\nstep", "$+5\\sigma$\nspike"
    ]
    n_faults = len(fault_labels)
    x_f = np.arange(n_faults)

    monitors = [
        {
            "label": "1D JEPA (1.9%)",
            "ev": [0.194, 0.194, 0.194, 0.194, 0.194, 0.278],
            "pt": [0.020, 0.020, 0.020, 0.020, 0.030, 0.020],
            "fill": "#d4d4d4", "edge": "#4e4e4e"
        },
        {
            "label": "Var. Gate (6.9%)",
            "ev": [0.278, 0.278, 0.278, 0.278, 0.194, 0.306],
            "pt": [0.040, 0.040, 0.040, 0.050, 0.050, 0.040],
            "fill": C_PURPLE_FILL, "edge": C_PURPLE_EDGE
        },
        {
            "label": "224-Bit (11.7%)",
            "ev": [0.250, 0.250, 0.250, 0.250, 0.292, 0.333],
            "pt": [0.155, 0.155, 0.151, 0.143, 0.195, 0.256],
            "fill": C_NAVY_FILL, "edge": C_NAVY_EDGE
        },
        {
            "label": "Var. Mon. (10.5%)",
            "ev": [0.260, 0.260, 0.260, 0.390, 0.260, 0.260],
            "pt": [0.180, 0.180, 0.180, 0.320, 0.180, 0.180],
            "fill": C_TEAL_FILL, "edge": C_TEAL_EDGE
        },
        {
            "label": "Hybrid Union",
            "ev": [0.444, 0.444, 0.444, 0.444, 0.486, 0.528],
            "pt": [0.255, 0.255, 0.251, 0.243, 0.295, 0.356],
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
    ax2.set_xticklabels(fault_labels, fontsize=7.6)
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
        labelspacing=0.20, fontsize=7.3
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
