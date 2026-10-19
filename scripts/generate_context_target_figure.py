#!/usr/bin/env python3
"""Generate Section IV-A schematic figure illustrating x_ctx and x_tgt on a multivariate stream.

Shows:
  - A 3-channel multivariate telemetry waveform X in R^{T x K} partitioned at split index t into:
      * Historical Context Window x_ctx = X[t - C : t, :] (C = 256 steps)
      * Suspect Target Horizon x_tgt = X[t : t + S, :] (S = 64 steps, ending at decision step tau = t + S - 1)
  - How online encoder E_theta(x_ctx) -> z_ctx and predictor K*z_ctx -> z_hat_tgt compare against
    momentum target encoder E_phi(x_tgt) -> z_tgt in latent space R^D (D=32) to form residual e(t),
    filtering out high-frequency channel jitter while capturing structural target anomalies.

Outputs:
  paper/NCAD_CS/figures/context_target_window_schematic.pdf
  paper/NCAD_CS/figures/context_target_window_schematic.svg
  paper/NCAD_CS/figures/context_target_window_schematic.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = ROOT / ".agents" / "skills" / "beautiful-research-figures" / "scripts"
if str(SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS))

from ieee_vector_style import save_true_vector_figure  # noqa: E402

FIG_DIR = ROOT / "paper" / "NCAD_CS" / "figures"


def main() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 7.0,
        "axes.labelsize": 7.3,
        "axes.titlesize": 7.6,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig = plt.figure(figsize=(3.5, 2.35))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.05], hspace=0.34, left=0.10, right=0.95, top=0.91, bottom=0.04)

    # =========================================================================
    # TOP PANEL: Multivariate Telemetry Stream X with Context x_ctx & Target x_tgt
    # =========================================================================
    ax_ts = fig.add_subplot(gs[0])
    rng = np.random.default_rng(42)

    t_grid = np.arange(-256, 64, dtype=float)
    # 3 illustrative channels stacked vertically for clarity
    ch1 = 2.05 + 0.28 * np.sin(2 * np.pi * t_grid / 64.0) + 0.08 * rng.standard_normal(len(t_grid))
    ch2 = 1.00 + 0.25 * np.cos(2 * np.pi * t_grid / 48.0 + 0.6) + 0.08 * rng.standard_normal(len(t_grid))
    ch3 = -0.05 + 0.22 * np.sin(2 * np.pi * t_grid / 80.0 - 0.4) + 0.07 * rng.standard_normal(len(t_grid))

    # Inject a structural fault in the target horizon [0, 64) on channels 1 & 2
    tgt_mask = t_grid >= 0
    t_tgt = t_grid[tgt_mask]
    fault_envelope = 0.68 * (1.0 - np.exp(-t_tgt / 16.0))
    ch1[tgt_mask] += fault_envelope
    ch2[tgt_mask] -= 0.52 * fault_envelope

    # Subtle dual-hierarchy grid (Principle 4)
    ax_ts.grid(True, which="major", linestyle="-", linewidth=0.45, alpha=0.22)
    ax_ts.set_axisbelow(True)

    # Shaded window regions
    ax_ts.axvspan(-256, 0, color="#1d4e89", alpha=0.08, zorder=1)
    ax_ts.axvspan(0, 63, color="#5c2d91", alpha=0.11, zorder=1)

    # Plot clean underlying nominal trajectory in target region (dashed reference)
    ch1_nom_tgt = 2.05 + 0.28 * np.sin(2 * np.pi * t_tgt / 64.0)
    ax_ts.plot(t_tgt, ch1_nom_tgt, color="#1d4e89", ls="--", lw=0.85, alpha=0.8, zorder=2)

    # Plot sensor waveforms
    ax_ts.plot(t_grid[:257], ch1[:257], color="#1d4e89", lw=0.90, zorder=3)
    ax_ts.plot(t_grid[:257], ch2[:257], color="#378d94", lw=0.85, zorder=3)
    ax_ts.plot(t_grid[:257], ch3[:257], color="#6e6e6e", lw=0.80, zorder=3)

    ax_ts.plot(t_grid[256:], ch1[256:], color="#a31d3d", lw=1.10, zorder=4)
    ax_ts.plot(t_grid[256:], ch2[256:], color="#5c2d91", lw=0.95, zorder=4)
    ax_ts.plot(t_grid[256:], ch3[256:], color="#6e6e6e", lw=0.80, zorder=3)

    # Boundary lines at t-C, t, and tau = t+S-1
    ax_ts.axvline(-256, color="#1d4e89", ls=":", lw=0.8, zorder=4)
    ax_ts.axvline(0, color="#222222", ls="--", lw=1.0, zorder=5)
    ax_ts.axvline(63, color="#5c2d91", ls="-", lw=1.1, zorder=5)

    # Region header banners placed cleanly in the top band [2.95, 3.55] above waveforms
    ax_ts.text(
        -128, 3.48,
        r"Context Window $\mathbf{x}_{\mathrm{ctx}} = \mathbf{X}[t{-}C : t, :]$ ($C{=}256$)",
        ha="center", va="top", fontsize=5.8, color="#0b4f8a", weight="bold",
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#99c2ff", alpha=0.94),
        zorder=6
    )
    ax_ts.text(
        31.5, 3.48,
        r"Target $\mathbf{x}_{\mathrm{tgt}}$ ($S{=}64$)",
        ha="center", va="top", fontsize=5.7, color="#4a1f75", weight="bold",
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#c8b2e8", alpha=0.94),
        zorder=6
    )

    # Channel labels placed cleanly in troughs with white backing
    ax_ts.text(
        -246, 2.48, r"Ch $1$", fontsize=5.2, color="#0b4f8a", weight="bold",
        bbox=dict(boxstyle="round,pad=0.06", fc="white", ec="none", alpha=0.75), zorder=5
    )
    ax_ts.text(
        -246, 1.36, r"Ch $2$", fontsize=5.2, color="#286b70", weight="bold",
        bbox=dict(boxstyle="round,pad=0.06", fc="white", ec="none", alpha=0.75), zorder=5
    )
    ax_ts.text(
        -246, 0.28, r"Ch $K$", fontsize=5.2, color="#444444", weight="bold",
        bbox=dict(boxstyle="round,pad=0.06", fc="white", ec="none", alpha=0.75), zorder=5
    )

    # Fault callout in target window pointing cleanly to the target excursion
    ax_ts.annotate(
        "Unobserved fault\nexcursion in $\\mathbf{x}_{\\mathrm{tgt}}$",
        xy=(28, ch1[256 + 28]), xytext=(-85, 2.52),
        fontsize=5.2, color="#a31d3d", weight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.13", fc="#fff8f8", ec="#e6a8b5", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#a31d3d", lw=0.85, shrinkA=1, shrinkB=2),
        zorder=6
    )

    ax_ts.set_xlim(-262, 70)
    ax_ts.set_ylim(-0.55, 3.72)
    ax_ts.set_xticks([-256, -128, 0, 63])
    ax_ts.set_xticklabels([r"$t{-}256$", r"$t{-}128$", r"Split $t$", r"$\tau{=}t{+}63$"])
    ax_ts.set_yticks([])
    ax_ts.set_ylabel(r"Stream $\mathbf{X}{\in}\mathbb{R}^{T\times K}$", labelpad=3)
    ax_ts.set_title(
        r"Sliding Window Partitioning ($W = C + S = 320$ steps, Stride $1$)",
        fontweight="bold", pad=3
    )

    # =========================================================================
    # BOTTOM PANEL: Mapping into Latent Space R^D (Prediction vs. Reconstruction)
    # =========================================================================
    ax_lat = fig.add_subplot(gs[1])
    ax_lat.set_xlim(0, 100)
    ax_lat.set_ylim(0, 36)
    ax_lat.axis("off")

    def draw_node(x, y, w, h, text, fc, ec, tc="#111111", fs=5.4, bold=False):
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
            facecolor=fc, edgecolor=ec, linewidth=0.85, zorder=2
        )
        ax_lat.add_patch(box)
        ax_lat.text(
            x + w / 2.0, y + h / 2.0, text,
            ha="center", va="center", fontsize=fs, color=tc,
            weight="bold" if bold else "normal", zorder=3
        )

    # Context branch (left -> middle)
    draw_node(
        1.0, 20.0, 27.5, 13.0,
        "Online Context Encoder\n$E_\\theta(\\mathbf{x}_{\\mathrm{ctx}}) \\to z_{\\mathrm{ctx}} \\in \\mathbb{R}^{32}$",
        fc="#eef5fc", ec="#1d4e89", tc="#0b4f8a", fs=5.3, bold=True
    )
    draw_node(
        34.0, 20.0, 29.0, 13.0,
        "Latent Predictor\n$\\hat{z}_{\\mathrm{tgt}} = \\mathbf{K} z_{\\mathrm{ctx}}$ (or MLP $g_\\psi$)",
        fc="#f4effa", ec="#5c2d91", tc="#4a1f75", fs=5.3, bold=True
    )

    # Target branch (right)
    draw_node(
        67.5, 20.0, 31.5, 13.0,
        "EMA Target Encoder\n$E_\\phi(\\mathbf{x}_{\\mathrm{tgt}}) \\to z_{\\mathrm{tgt}} \\in \\mathbb{R}^{32}$\n(filters channel noise)",
        fc="#f4effa", ec="#5c2d91", tc="#4a1f75", fs=5.2, bold=True
    )

    # Bottom residual & SPOT scoring box (wide enough so zero text touches borders)
    draw_node(
        3.5, 1.5, 93.0, 12.0,
        "Latent Predictive Residual: $\\mathbf{e}(t) = z_{\\mathrm{tgt}}(t) - \\hat{z}_{\\mathrm{tgt}}(t) \\in \\mathbb{R}^{32}$\n"
        "Coordinate Saliency Gate $\\mathcal{D}_{\\mathrm{sal}}(\\mathbf{e}) \\;\\to\\;$ SPOT Threshold $\\tau_{\\mathrm{SPOT}}$ at Step $\\tau$",
        fc="#fff8f6", ec="#a31d3d", tc="#7a122b", fs=5.3, bold=True
    )

    # Connecting arrows from top waveform regions to encoders
    ax_lat.annotate(
        "", xy=(14.7, 33.3), xytext=(14.7, 36.0),
        arrowprops=dict(arrowstyle="->", color="#1d4e89", lw=1.0), zorder=4
    )
    ax_lat.annotate(
        "", xy=(83.2, 33.3), xytext=(83.2, 36.0),
        arrowprops=dict(arrowstyle="->", color="#5c2d91", lw=1.0), zorder=4
    )

    # Arrow: Context Encoder -> Latent Predictor
    ax_lat.annotate(
        "", xy=(33.7, 26.5), xytext=(28.8, 26.5),
        arrowprops=dict(arrowstyle="->", color="#1d4e89", lw=1.0), zorder=4
    )

    # Arrows: Latent Predictor & Target Encoder -> Residual Box (clean arrows + labels beside shafts)
    ax_lat.annotate(
        "", xy=(48.5, 13.8), xytext=(48.5, 19.7),
        arrowprops=dict(arrowstyle="->", color="#5c2d91", lw=1.0), zorder=4
    )
    ax_lat.text(46.0, 16.7, r"$\hat{z}_{\mathrm{tgt}}$", fontsize=5.4, color="#4a1f75", ha="right", va="center")

    ax_lat.annotate(
        "", xy=(72.0, 13.8), xytext=(81.0, 19.7),
        arrowprops=dict(arrowstyle="->", color="#5c2d91", lw=1.0), zorder=4
    )
    ax_lat.text(79.5, 16.5, r"$z_{\mathrm{tgt}}$", fontsize=5.4, color="#4a1f75", ha="left", va="center")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_stem = FIG_DIR / "context_target_window_schematic"
    saved = save_true_vector_figure(
        fig, out_stem, formats=("svg", "pdf", "png"),
        png_dpi=300, inkscape_postprocess_pdf=True,
        ensure_pure_vector=True, ensure_outside_legend=False
    )
    print("Saved context/target schematic figure:")
    for fmt, p in saved.items():
        print(f"  [{fmt.upper()}] {p}")


if __name__ == "__main__":
    main()
