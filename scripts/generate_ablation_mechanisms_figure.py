#!/usr/bin/env python3
"""Generate publication-grade true-vector figure for Ablations 1 & 2 (Figure 3):
Panel (a): Multi-Horizon Whitened Envelope vs. Soliton Concentration Modulation Sweep
           across alpha_sol in [0, 0.24] (19 non-GHL streams, 5 seeds) relative to TS-JEPA.
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
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.2,
        "axes.labelsize": 8.6,
        "axes.titlesize": 8.8,
        "legend.fontsize": 7.8,
        "figure.dpi": 300,
    })


def make_figure():
    set_ieee_style()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.14, 1.98),
        gridspec_kw={"width_ratios": [1.0, 1.08], "wspace": 0.44}
    )
    fig.subplots_adjust(top=0.70, bottom=0.24, left=0.068, right=0.988)

    # =========================================================================
    # Panel (a): Multi-Horizon Envelope & Soliton Modulation Sweep (n=19, 5 seeds)
    # =========================================================================
    # 7 stations:
    # 0: TS-JEPA (Single-Horizon), 1: alpha_sol = 0.00 (Unmodulated Envelope),
    # 2: 0.03, 3: 0.06, 4: 0.12* (Deployed), 5: 0.18, 6: 0.24
    x_idx = np.arange(7)
    x_labels = [
        "TS-JEPA\n(1-Hor.)", "$0.00$\n(Env.)", "$0.03$",
        "$0.06$", "$0.12^*$\n(depl.)", "$0.18$", "$0.24$"
    ]

    # Exact 19-stream, 5-seed metrics from reports/score_dumps/
    ks_drift = np.array([0.214, 0.183, 0.182, 0.181, 0.179, 0.178, 0.177])
    caus_er = np.array([0.487, 0.439, 0.447, 0.452, 0.478, 0.449, 0.428])
    pr_buf = np.array([0.331, 0.330, 0.331, 0.331, 0.331, 0.332, 0.332])
    f1_buf = np.array([0.206, 0.149, 0.149, 0.148, 0.148, 0.149, 0.147])
    fpr_buf = np.array([0.093, 0.083, 0.082, 0.079, 0.078, 0.073, 0.073])
    fpr_caus = np.array([0.024, 0.006, 0.006, 0.006, 0.006, 0.005, 0.005])

    ax1_r = ax1.twinx()
    ax1_r.spines["right"].set_edgecolor(C_TEAL_EDGE)
    ax1_r.spines["right"].set_linewidth(0.75)
    ax1.spines["left"].set_edgecolor(C_PURPLE_EDGE)

    # Vertical separator between Single-Horizon TS-JEPA and Multi-Horizon Soliton-Reynolds
    ax1.axvline(0.5, color="#888888", linestyle=":", linewidth=0.9, alpha=0.75, zorder=1)
    # Vertical highlight at deployed alpha_sol = 0.12 (index 4)
    ax1.axvline(4, color="#9e9e9e", linestyle="--", linewidth=0.85, alpha=0.65, zorder=1)

    # Left axis: KS Drift D_KS
    l1, = ax1.plot(
        x_idx, ks_drift, color=C_PURPLE_EDGE, linestyle="-", linewidth=1.8,
        marker="o", markersize=4.8, markerfacecolor=C_PURPLE_FILL,
        markeredgecolor=C_PURPLE_EDGE, label="$D_{\\mathrm{KS}}$ Drift (left)", zorder=4
    )

    # Right axis: Causal ER, Buffered PR-AUC, Buffered F1, Buffered FPR, Causal FPR
    l2, = ax1_r.plot(
        x_idx, caus_er, color=C_TEAL_EDGE, linestyle="-", linewidth=1.7,
        marker="s", markersize=4.4, markerfacecolor=C_TEAL_FILL,
        markeredgecolor=C_TEAL_EDGE, label="Caus. ER (right)", zorder=4
    )
    l3, = ax1_r.plot(
        x_idx, pr_buf, color=C_NAVY_EDGE, linestyle="-", linewidth=1.5,
        marker="D", markersize=4.1, markerfacecolor=C_NAVY_FILL,
        markeredgecolor=C_NAVY_EDGE, label="Buf. PR-AUC", zorder=4
    )
    l4, = ax1_r.plot(
        x_idx, f1_buf, color=C_GREY_EDGE, linestyle=":", linewidth=1.4,
        marker="^", markersize=4.0, markerfacecolor=C_GREY_FILL,
        markeredgecolor=C_GREY_EDGE, label="Buf. Point-F1", zorder=3
    )
    l5, = ax1_r.plot(
        x_idx, fpr_buf, color=C_NAVY_EDGE, linestyle="--", linewidth=1.25,
        marker="v", markersize=3.8, markerfacecolor="#ffffff",
        markeredgecolor=C_NAVY_EDGE, label="Buf. FPR", zorder=3
    )
    l6, = ax1_r.plot(
        x_idx, fpr_caus, color=C_CORAL_EDGE, linestyle="-.", linewidth=1.35,
        marker="p", markersize=4.0, markerfacecolor=C_CORAL_FILL,
        markeredgecolor=C_CORAL_EDGE, label="Caus. FPR", zorder=3
    )

    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(x_labels, fontsize=7.8)
    ax1.set_xlim(-0.38, 6.38)
    ax1.set_ylim(0.152, 0.228)
    ax1.set_yticks([0.16, 0.18, 0.20, 0.22])
    ax1.set_xlabel("Soliton Modulation Weight $\\alpha_{\\mathrm{sol}}$ ($n{=}19$ Streams)", labelpad=2)
    ax1.set_ylabel("Calibration Drift $D_{\\mathrm{KS}}$", color=C_PURPLE_EDGE, labelpad=3)
    ax1.tick_params(axis="y", colors=C_PURPLE_EDGE)

    ax1_r.set_ylim(-0.02, 0.56)
    ax1_r.set_yticks([0.0, 0.18, 0.36, 0.54])
    ax1_r.set_ylabel("PR-AUC / ER / F1 / FPR", color=C_TEAL_EDGE, labelpad=3)
    ax1_r.tick_params(axis="y", colors=C_TEAL_EDGE)

    ax1.grid(True, which="major", linestyle="-", linewidth=0.5, color="#e0e0e0", alpha=0.65)
    ax1.set_axisbelow(True)

    ax1.set_title("(a) Multi-Horizon Envelope & Soliton Gain Sweep", fontweight="bold", pad=27)
    handles_a = [l1, l2, l3, l4, l5, l6]
    labels_a = [h.get_label() for h in handles_a]
    ax1.legend(
        handles_a, labels_a, loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=3, frameon=False, columnspacing=0.55, handletextpad=0.25,
        labelspacing=0.20, fontsize=7.6
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
        labelspacing=0.20, fontsize=7.6
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
