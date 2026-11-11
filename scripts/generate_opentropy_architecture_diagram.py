#!/usr/bin/env python3
"""Generate Figure 1: Four-Stage Laminar-JEPA & Laminar-JEPA-Lite Circuit-Breaker Architecture.

100% true-vector diagram (.svg, .pdf, .png) illustrating:
  (A) Multiscale Windowing, 3-Depth TCN Hierarchy Taps & Barycenter Centering
  (B) Multi-Horizon Latent Rollout (M=4 Sequential / M=3 Parallel Lite) & Two-Term Whitening Defect Loss
  (C) Horizon Mean-Plus-Fluctuation Dial Expansion (224-Bit Full / 128-Bit Lite P+D) & Stabilized Whitening
  (D) Coordinate Boolean Circuit-Breaker Panel & Fast Right-Edge Onset Relays
"""

from __future__ import annotations

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path(__file__).resolve().parent.parent
SKILL_SCRIPTS = ROOT / ".agents" / "skills" / "beautiful-research-figures" / "scripts"
if str(SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS))

from ieee_vector_style import save_true_vector_figure  # noqa: E402

OUT_FIG_DIR = ROOT / "paper" / "NCAD_CS" / "figures"
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)


def make_figure() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig, ax = plt.subplots(figsize=(13.6, 4.9), dpi=300)
    fig.subplots_adjust(left=0.006, right=0.994, top=0.99, bottom=0.01)
    ax.set_xlim(0.0, 16.0)
    ax.set_ylim(0.0, 5.8)
    ax.axis("off")

    c_dark = "#1e293b"
    c_muted = "#475569"
    c_bdr = "#cbd5e1"

    c_ctx_bg, c_ctx_bdr, c_ctx_txt = "#eff6ff", "#1d4e89", "#0b3c68"
    c_tgt_bg, c_tgt_bdr, c_tgt_txt = "#f4f0fa", "#6a408d", "#4a256b"
    c_hor_bg, c_hor_bdr, c_hor_txt = "#e8f6f6", "#378d94", "#1b5e63"
    c_brk_bg, c_brk_bdr, c_brk_txt = "#fdf2f4", "#a31d3d", "#7a122b"

    def draw_card(x, y, w, h, title="", subtitle="", fc="#ffffff", ec=c_bdr, lw=1.0, rx=0.10, title_color=c_dark):
        box = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={rx}",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2
        )
        ax.add_patch(box)
        if title and subtitle:
            ax.text(x + w / 2, y + h * 0.67, title, ha="center", va="center", fontsize=8.0, fontweight="bold", color=title_color, zorder=3)
            ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=7.2, color=c_muted, zorder=3)
        elif title:
            ax.text(x + w / 2, y + h * 0.50, title, ha="center", va="center", fontsize=8.0, fontweight="bold", color=title_color, zorder=3)
        return box

    def draw_arrow(x1, y1, x2, y2, color="#334155", lw=1.2, ls="-"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, linestyle=ls, mutation_scale=9.0, shrinkA=0, shrinkB=0),
            zorder=5
        )

    outer = patches.FancyBboxPatch(
        (0.08, 0.08), 15.84, 5.64, boxstyle="round,pad=0.0,rounding_size=0.14",
        facecolor="#fafafa", edgecolor="#cbd5e1", linewidth=1.1, zorder=1
    )
    ax.add_patch(outer)

    # =========================================================================
    # STAGE 1: Multiscale Windowing & Dual Hierarchy Encoders
    # =========================================================================
    draw_card(0.22, 0.22, 3.65, 5.36, fc="#ffffff", ec=c_bdr, lw=1.0)
    ax.text(2.04, 5.32, "(1) Multiscale Encoding & Barycenter", ha="center", va="center", fontsize=8.8, fontweight="bold", color=c_ctx_txt)

    draw_card(0.38, 4.02, 1.48, 0.96, "Context Window", "$\\mathbf{x}_{\\mathrm{ctx}} \\in \\mathbb{R}^{256 \\times K}$", fc=c_ctx_bg, ec=c_ctx_bdr)
    draw_card(2.14, 4.02, 1.56, 0.96, "Online Encoder $E_\\theta$", "rFFT ($K{\\geq}25$) + TCN", fc=c_ctx_bg, ec=c_ctx_bdr)
    draw_arrow(1.86, 4.50, 2.14, 4.50, color=c_ctx_bdr)

    draw_card(0.38, 2.28, 3.32, 1.38, fc="#f8fafc", ec="#cbd5e1", lw=0.9)
    ax.text(2.04, 3.38, "3-Depth TCN Taps & Barycenter Shift", ha="center", va="center", fontsize=7.8, fontweight="bold", color=c_dark)
    ax.text(
        2.04, 2.74,
        "Shallow, mid & deep TCN taps ($L_{\\mathrm{tap}}{=}3$)\n"
        "Subtract context barycenter $\\bar{\\boldsymbol{\\mu}}_{\\mathrm{ctx}}$\n"
        "LayerNorm summary in $\\mathbb{R}^{D}$ ($D{=}32$)",
        ha="center", va="center", fontsize=7.0, color=c_muted
    )

    draw_card(0.38, 0.42, 1.48, 0.96, "Nested Prefixes", "$\\mathbf{x}_{\\mathrm{tgt}}^{(m)} \\in \\mathbb{R}^{S_m \\times K}$", fc=c_tgt_bg, ec=c_tgt_bdr)
    draw_card(2.14, 0.42, 1.56, 0.96, "EMA Target $E_\\phi$", "$\\phi \\leftarrow m\\phi{+}(1{-}m)\\theta$", fc=c_tgt_bg, ec=c_tgt_bdr)
    draw_arrow(1.86, 0.90, 2.14, 0.90, color=c_tgt_bdr)
    draw_arrow(2.92, 4.02, 2.92, 3.68, color=c_ctx_bdr, ls="--")
    draw_arrow(2.92, 2.26, 2.92, 1.40, color=c_tgt_bdr, ls="--")

    # =========================================================================
    # STAGE 2: Nested Multi-Horizon Rollout & Two-Term Loss
    # =========================================================================
    draw_card(4.02, 0.22, 3.75, 5.36, fc="#ffffff", ec=c_bdr, lw=1.0)
    ax.text(5.90, 5.32, "(2) Multi-Horizon Latent Rollout", ha="center", va="center", fontsize=8.8, fontweight="bold", color=c_hor_txt)

    draw_card(4.20, 4.02, 1.45, 0.96, "Context Latent", "$z_{\\mathrm{ctx}} \\in \\mathbb{R}^{32}$", fc=c_ctx_bg, ec=c_ctx_bdr)
    draw_arrow(3.70, 4.50, 4.20, 4.50, color=c_ctx_bdr)

    draw_card(5.92, 4.02, 1.68, 0.96, "Predictor ($M{=}4 / 3$)", "$S_m \\in \\{16..64\\}$", fc=c_hor_bg, ec=c_hor_bdr)
    draw_arrow(5.65, 4.50, 5.92, 4.50, color=c_hor_bdr)

    draw_card(4.20, 2.28, 3.40, 1.38, fc=c_hor_bg, ec=c_hor_bdr, lw=0.9)
    ax.text(5.90, 3.38, "Two-Term Objective ($\\mathcal{L}_{\\mathrm{Lam}}$)", ha="center", va="center", fontsize=7.8, fontweight="bold", color=c_hor_txt)
    ax.text(
        5.90, 2.74,
        "$\\sum_{m=1}^{M} \\frac{w_m}{D}\\left(\\|\\hat{z}_{\\mathrm{tgt}}^{(m)} - \\mathrm{sg}(z_{\\mathrm{tgt}}^{(m)})\\|_2^2 + \\|\\boldsymbol{\\Delta}_\\gamma(\\hat{z}_{\\mathrm{tgt}}^{(m)})\\|_F^2\\right)$\n"
        "Frobenius whitening defect ($\\mathrm{RankMe}{=}19.8$)",
        ha="center", va="center", fontsize=7.0, color=c_dark
    )

    draw_card(4.20, 0.42, 1.45, 0.96, "Target Latents", "$z_{\\mathrm{tgt}}^{(1..M)} \\in \\mathbb{R}^{M \\times 32}$", fc=c_tgt_bg, ec=c_tgt_bdr)
    draw_arrow(3.70, 0.90, 4.20, 0.90, color=c_tgt_bdr)

    draw_card(5.92, 0.42, 1.68, 0.96, "Residuals $\\mathbf{E}$", "$\\mathbf{e}_m = z_{\\mathrm{tgt}}^{(m)} - \\hat{z}_{\\mathrm{tgt}}^{(m)}$", fc="#ffffff", ec=c_hor_bdr)
    draw_arrow(5.65, 0.90, 5.92, 0.90, color=c_tgt_bdr)
    draw_arrow(6.76, 4.02, 6.76, 3.68, color=c_hor_bdr)
    draw_arrow(6.76, 2.26, 6.76, 1.40, color=c_hor_bdr)

    # =========================================================================
    # STAGE 3: Horizon Dial Expansion
    # =========================================================================
    draw_card(7.92, 0.22, 3.95, 5.36, fc="#ffffff", ec=c_bdr, lw=1.0)
    ax.text(9.90, 5.32, "(3) Multi-Horizon Dial Expansion", ha="center", va="center", fontsize=8.8, fontweight="bold", color=c_tgt_txt)

    draw_card(
        8.10, 4.02, 3.58, 0.96,
        "Stabilized Whitening & Optional Quench ($\\tilde{\\mathbf{E}} \\in \\mathbb{R}^{M \\times 32}$)",
        "$\\tilde{\\mathbf{e}} = \\mathbf{\\Omega}^{1/2}(\\mathrm{vec}(\\mathbf{E}) - \\bar{\\boldsymbol{\\mu}}_{\\mathrm{res}})$ blended via $w_{m,c}$",
        fc=c_tgt_bg, ec=c_tgt_bdr
    )
    draw_arrow(7.60, 0.90, 8.10, 4.46, color=c_hor_bdr)

    draw_card(8.10, 2.88, 1.72, 0.90, "1. Sub-Block ($MD$)", "$|\\tilde{e}_{m,c}(t)|,\\; m \\in \\{1..M\\}$", fc="#f8fafc", ec=c_ctx_bdr)
    draw_card(9.96, 2.88, 1.72, 0.90, "2. Mean Drift (32)", "$|\\bar{e}_c(t)| = |\\frac{1}{M}\\sum_{m=1}^M \\tilde{e}_{m,c}|$", fc="#f8fafc", ec=c_ctx_bdr)
    draw_card(8.10, 1.78, 1.72, 0.90, "3. Fluctuation (32)", "$\\sigma'_c(t) = \\sqrt{\\frac{1}{M}\\sum (\\tilde{e}_{m,c}{-}\\bar{e}_c)^2}$", fc="#f8fafc", ec=c_hor_bdr)
    draw_card(9.96, 1.78, 1.72, 0.90, "4. Endpoint Diff (32)", "$|\\tilde{e}_{M,c}(t) - \\tilde{e}_{1,c}(t)|$", fc="#f8fafc", ec=c_brk_bdr)

    draw_arrow(8.96, 4.02, 8.96, 3.78, color=c_tgt_bdr)
    draw_arrow(10.82, 4.02, 10.82, 3.78, color=c_tgt_bdr)

    draw_card(
        8.10, 0.42, 3.58, 1.14,
        "Horizon Gram Concentration Bounds",
        "$\\mathbf{G} = \\mathbf{F}\\mathbf{F}^\\top \\succeq \\mathbf{0},\\; \\frac{1}{M-1} \\leq R_{\\mathrm{LT}}(\\mathbf{F}) \\leq 1$ (Prop. 1)",
        fc="#f8fafc", ec="#94a3b8"
    )

    # =========================================================================
    # STAGE 4: Coordinate Circuit-Breaker Panel
    # =========================================================================
    draw_card(12.02, 0.22, 3.74, 5.36, fc="#ffffff", ec=c_bdr, lw=1.0)
    ax.text(13.89, 5.32, "(4) Coordinate Circuit-Breaker Panel", ha="center", va="center", fontsize=8.8, fontweight="bold", color=c_brk_txt)

    draw_card(
        12.20, 3.92, 3.38, 1.06,
        "Hijack Guards & Per-Dial Breakers",
        "$b_j(t) = \\mathbb{I}\\left(z_j(t) > q_{0.985}(z_{j,\\mathrm{cal}})\\,\\gamma_j^{\\mathrm{cal}}\\,\\gamma_j^{\\mathrm{sc}}\\right)$",
        fc=c_brk_bg, ec=c_brk_bdr
    )
    draw_arrow(11.68, 3.33, 12.20, 4.45, color=c_brk_bdr)
    draw_arrow(11.68, 2.23, 12.20, 4.25, color=c_brk_bdr)

    draw_card(
        12.20, 2.56, 3.38, 1.08,
        "32 Right-Edge Onset Relays (Full 224B)",
        "$r_c(t) = b_{4,c}^{\\mathrm{edge}}(t) \\wedge \\left(b_{\\mathrm{accel},c}^{\\mathrm{edge}} \\vee b_{\\mathrm{fluc},c}^{\\mathrm{edge}} \\vee b_{3,c}^{\\mathrm{edge}}\\right)$",
        fc="#fff8f1", ec="#c2410c"
    )
    draw_arrow(13.89, 3.92, 13.89, 3.64, color=c_brk_bdr)

    draw_card(12.20, 0.42, 3.38, 1.86, fc=c_brk_bg, ec=c_brk_bdr, lw=1.1)
    ax.text(13.89, 2.02, "Boolean Quorum Alarm Decision", ha="center", va="center", fontsize=7.8, fontweight="bold", color=c_brk_txt)
    ax.text(
        13.89, 1.16,
        "• Laminar-JEPA (224B): $N_{224} > K_{224} \\vee N_{128} > K_{128} \\vee N_{\\mathrm{edge}} > K_{\\mathrm{edge}}$\n"
        "• Laminar-JEPA-Lite (128B P+D): $N_{128}(t) > K_{\\mathrm{lite}}$ (1 quorum)\n"
        "• Lite cuts params $35\\%{-}52\\%$ & runs $2.4\\times$ faster\n"
        "• Causal ($\\tau{=}t{+}S{-}1$) & prefix-routed buffered",
        ha="center", va="center", fontsize=6.7, color=c_dark
    )
    draw_arrow(13.89, 2.56, 13.89, 2.28, color=c_brk_bdr)

    out_stem = OUT_FIG_DIR / "opentropy_jepa_architecture"
    saved = save_true_vector_figure(
        fig, out_stem, formats=("svg", "pdf", "png"),
        png_dpi=300, inkscape_postprocess_pdf=True,
        ensure_pure_vector=True, ensure_outside_legend=True
    )
    plt.close(fig)
    print("Saved Figure 1:", saved)


if __name__ == "__main__":
    make_figure()
