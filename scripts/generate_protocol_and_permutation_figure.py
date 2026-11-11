#!/usr/bin/env python3
"""Generate Figure 2: Chronological Split, Window Isolation, and Online Causality Protocol.

Single-column (3.5 in) true-vector schematic figure illustrating:
  1. Raw Nominal Telemetry Partition (Training 65% vs. Held-Out Nominal 35%)
  2. Held-Out Causal Indexing (Warm-up W-1=319, Calibration Block [0:H) fitting
     224 dial tripwires tau_j, hijack guards gamma_j & quorum K, Isolation Gap S=64,
     and Held-Out Evaluation [H+S:M_val))
  3. Online Window Causality: Structural Score-Emission Latency (S-1=63) vs.
     Empirical Fault-Detection Delay (Delta t = t_alm - t*).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SKILL_SCRIPTS = ROOT / ".agents" / "skills" / "beautiful-research-figures" / "scripts"
if str(SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS))

from ieee_vector_style import save_true_vector_figure  # noqa: E402

FIG_DIR = ROOT / "paper" / "NCAD_CS" / "figures"


def draw_protocol_schematic(ax: plt.Axes) -> None:
    """Draw exact chronological index schematic & latency vs. detection delay."""
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(-6.0, 96.0)
    ax.axis("off")

    # --- Row 1: 5-Block Disjoint Nominal Telemetry Partition ---
    ax.text(0.0, 93.0, "1. Five-Block Nominal Partition ($T_{\\mathrm{nom}}$; Archive $\\mathbf{X}_q \\perp \\mathbf{X}_{\\mathrm{cal}}$):", fontsize=6.4, fontweight="bold", color="#222222")
    y1, h1 = 76.5, 11.5
    ax.add_patch(Rectangle((0.0, y1), 26.5, h1, fc="#d0e1f2", ec="#1d4e89", lw=0.8))
    ax.text(13.25, y1 + h1 / 2, "Fit $\\mathbf{X}_{\\mathrm{fit}}$\n($E_\\theta, P, \\mathbf{\\Omega}$)",
            ha="center", va="center", fontsize=5.5, color="#0b3c68", fontweight="bold")
    ax.add_patch(Rectangle((27.2, y1), 12.2, h1, fc="#dbe9f6", ec="#1d4e89", lw=0.75))
    ax.text(33.3, y1 + h1 / 2, "Sel. $\\mathbf{X}_{\\mathrm{sel}}$\n(ckpt)",
            ha="center", va="center", fontsize=5.4, color="#0b3c68")
    ax.add_patch(Rectangle((40.1, y1), 13.9, h1, fc="#fff2cc", ec="#996515", lw=0.8))
    ax.text(47.05, y1 + h1 / 2, "Adapt $\\mathbf{X}_q$\n($w_{m,c}$, Q1\u20134)",
            ha="center", va="center", fontsize=5.4, color="#664200", fontweight="bold")
    ax.plot([54.5, 54.5], [y1 - 1.4, y1 + h1 + 1.4], color="#a31d3d", lw=1.0, ls="--")
    ax.text(54.5, y1 - 1.9, "Freeze ($N_{\\mathrm{tr}}$)", ha="center", va="top", fontsize=5.2, color="#a31d3d", fontweight="bold")
    ax.add_patch(Rectangle((55.0, y1), 45.0, h1, fc="#f2e6d8", ec="#8a4a22", lw=0.8))
    ax.text(77.5, y1 + h1 / 2, "Untouched Nominal $\\mathbf{X}_{\\mathrm{val}}$ ($35\\%$)\n$\\mathbf{X}_{\\mathrm{cal}} \\perp \\mathbf{X}_{\\mathrm{hold}}$ ($S{=}64$ target gaps)",
            ha="center", va="center", fontsize=5.5, color="#5c2e12")

    # --- Row 2: Causal Window Indexing inside Untouched Nominal Block ---
    ax.text(0.0, 65.5, "2. Untouched Calibration & Evaluation ($W{=}C{+}S{=}320$, $\\tau{=}t{+}S{-}1$):",
            fontsize=6.4, fontweight="bold", color="#222222")
    y2, h2 = 49.0, 12.0
    ax.add_patch(Rectangle((0.0, y2), 17.5, h2, fc="#f4f4f4", ec="#666666", lw=0.7, ls="--"))
    ax.text(8.75, y2 + h2 / 2, "Warm-up\n$W{-}1{=}319$", ha="center", va="center", fontsize=5.4, color="#333333")
    ax.add_patch(Rectangle((18.0, y2), 31.5, h2, fc="#ccece6", ec="#00706b", lw=0.8))
    ax.text(33.75, y2 + h2 / 2, "Untouched $\\mathbf{X}_{\\mathrm{cal}}$ $[0 : H)$\n(fit $\\tau_j, \\gamma_j$ & quorum $K$)", ha="center", va="center", fontsize=5.4, color="#004d49", fontweight="bold")
    ax.add_patch(Rectangle((50.0, y2), 16.5, h2, fc="#fbe3e8", ec="#a31d3d", lw=0.8))
    ax.text(58.25, y2 + h2 / 2, "Gap $S{=}64$\n(or $W{=}320$)", ha="center", va="center", fontsize=5.5, color="#80112a", fontweight="bold")
    ax.add_patch(Rectangle((67.0, y2), 33.0, h2, fc="#e2d9f3", ec="#5c2d91", lw=0.8))
    ax.text(83.5, y2 + h2 / 2, "Held-Out Eval $[H{+}S : M_{\\mathrm{val}})$\n(measure out-of-sample FPR)", ha="center", va="center", fontsize=5.5, color="#3b1b61", fontweight="bold")

    ax.text(
        50.0, 44.0,
        "At gap $S{=}64$: suspect targets $\\mathbf{x}_{\\mathrm{tgt}}$ are $100\\%$ disjoint ($W{=}320$ check: $\\Delta\\mathrm{FPR} \\leq 0.002$).",
        ha="center", va="center", fontsize=5.5, color="#333333"
    )

    # --- Row 3: Score-Emission Latency vs. Fault-Detection Delay ---
    ax.text(0.0, 35.0, "3. Online Window Causality ($\\tau{=}t{+}S{-}1$): Emission Lag vs. Detection Delay:",
            fontsize=6.5, fontweight="bold", color="#222222")
    y3, h3 = 14.0, 10.5
    ax.add_patch(Rectangle((0.0, y3), 40.0, h3, fc="#d0e1f2", ec="#1d4e89", lw=0.75))
    ax.text(20.0, y3 + h3 / 2, "Past Context $\\mathbf{x}_{\\mathrm{ctx}}$ ($C{=}256$)", ha="center", va="center", fontsize=5.7, color="#0b3c68")
    ax.add_patch(Rectangle((40.0, y3), 28.0, h3, fc="#f9d5b5", ec="#b25900", lw=0.75))
    ax.text(54.0, y3 + h3 / 2, "Target $\\mathbf{x}_{\\mathrm{tgt}}$ ($S{=}64$)", ha="center", va="center", fontsize=5.7, color="#663300")

    ax.plot([40.0, 40.0], [y3 - 4.0, y3 + h3 + 2.2], color="#333333", lw=0.75, ls=":")
    ax.text(39.2, y3 + h3 + 2.6, "$t$", ha="right", va="bottom", fontsize=5.6, color="#222222")

    ax.plot([51.0, 51.0], [y3 - 10.5, y3 + h3 + 2.2], color="#a31d3d", lw=0.95, ls="--")
    ax.text(51.0, y3 + h3 + 2.6, "Onset $t^*$", ha="center", va="bottom", fontsize=5.5, color="#a31d3d", fontweight="bold")

    ax.plot([68.0, 68.0], [y3 - 4.0, y3 + h3 + 2.2], color="#00706b", lw=0.95, ls="-")
    ax.text(68.0, y3 + h3 + 2.6, "Emit $\\tau{=}t{+}63$", ha="center", va="bottom", fontsize=5.5, color="#00706b", fontweight="bold")

    ax.plot([90.0, 90.0], [y3 - 10.5, y3 + h3 + 2.2], color="#5c2d91", lw=0.95, ls="-.")
    ax.scatter([90.0], [y3 + h3 / 2], marker="^", s=28, color="#5c2d91", zorder=5)
    ax.text(90.0, y3 + h3 + 2.6, "1st Trip $t_{\\mathrm{alm}}$", ha="center", va="bottom", fontsize=5.5, color="#5c2d91", fontweight="bold")

    ax.annotate("", xy=(68.0, y3 - 2.6), xytext=(40.0, y3 - 2.6),
                arrowprops=dict(arrowstyle="<->", color="#00706b", lw=0.9))
    ax.text(54.0, y3 - 6.0, "Emission lag $S{-}1{=}63$", ha="center", va="center", fontsize=5.5, color="#00706b", fontweight="bold")

    ax.annotate("", xy=(90.0, y3 - 10.2), xytext=(51.0, y3 - 10.2),
                arrowprops=dict(arrowstyle="<->", color="#5c2d91", lw=0.9))
    ax.text(70.5, y3 - 13.8, "Detection delay $\\Delta t = t_{\\mathrm{alm}} - t^*$", ha="center", va="center", fontsize=5.5, color="#5c2d91", fontweight="bold")


def main() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 6.8,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 1.85))
    fig.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98)

    draw_protocol_schematic(ax)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_stem = FIG_DIR / "protocol_and_permutation_decomposition"
    saved = save_true_vector_figure(
        fig, out_stem, formats=("svg", "pdf", "png"),
        png_dpi=300, inkscape_postprocess_pdf=True,
        ensure_pure_vector=True, ensure_outside_legend=True
    )
    plt.close(fig)
    print("Saved Figure 2:", saved)


if __name__ == "__main__":
    main()
