"""Generate the true-vector IEEE TNNLS Figure 1 architecture diagram (Laminar-JEPA-Lite).

Draws a clean, single-model four-stage pipeline for Laminar-JEPA-Lite (M=3 horizons,
128 dials, single healthy quorum) with vertically rotated Context / Target (Suspect)
inputs on the far left feeding horizontally into the Online and EMA Target encoders.
Exports 100% pure-vector SVG and TrueType PDF artifacts.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "NCAD_CS" / "figures"
SVG_PATH = FIG_DIR / "opentropy_jepa_architecture.svg"
PDF_PATH = FIG_DIR / "opentropy_jepa_architecture.pdf"

# Harmonious grayscale-safe palette (Slate Blue / Plum Purple / Deep Teal / Coral)
C_NAVY = "#1E2D3D"
C_PANEL_BG = "#FAFBFD"
C_PANEL_EDGE = "#CFD9E3"
C_TEXT = "#1B2836"
C_SUBTEXT = "#4A5E72"

C_CTX_FILL = "#6B8BA4"
C_CTX_EDGE = "#2B465E"
C_CTX_TRACK = "#EDF3F8"
C_CTX_ENC = "#C9DAE8"

C_TGT_FILL = "#74539B"
C_TGT_EDGE = "#432766"
C_TGT_TRACK = "#F4EFFA"
C_TGT_ENC = "#583D7C"
C_PREF = ["#B89ED6", "#9371BC", "#6A4796"]

C_TEAL = "#2F7D80"
C_TEAL_LIGHT = "#EBF6F6"
C_TEAL_EDGE = "#5B9E9F"
C_TEAL_CELLS = ["#D4ECEC", "#9BD0D0", "#5FA8AA", "#2F7D80"]

C_ALARM = "#C0394B"
C_ALARM_BG = "#FDF2F3"
C_OBJ_BG = "#EEF3F7"


def _rbox(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    fc: str,
    ec: str,
    lw: float = 1.4,
    r: float = 0.08,
    ls: str = "-",
    zorder: int = 3,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _arrow(
    ax: plt.Axes,
    p0: tuple[float, float],
    p1: tuple[float, float],
    color: str = C_NAVY,
    lw: float = 1.7,
    ls: str = "-",
    zorder: int = 6,
) -> None:
    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>,head_length=6.0,head_width=3.6",
        color=color,
        linewidth=lw,
        linestyle=ls,
        shrinkA=0.0,
        shrinkB=0.0,
        zorder=zorder,
    )
    ax.add_patch(arr)


def _polyline_arrow(
    ax: plt.Axes,
    pts: list[tuple[float, float]],
    color: str = C_NAVY,
    lw: float = 1.7,
    ls: str = "-",
    zorder: int = 6,
) -> None:
    if len(pts) > 2:
        xs = [p[0] for p in pts[:-1]]
        ys = [p[1] for p in pts[:-1]]
        ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls, solid_capstyle="round", zorder=zorder)
    _arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls, zorder=zorder)


def _draw_stage_frames(ax: plt.Axes) -> None:
    stages = [
        (0.08, 4.66, "STAGE 1: INPUT & JEPA ENCODERS"),
        (4.90, 3.12, "STAGE 2: PREDICT & WHITEN"),
        (8.18, 2.84, "STAGE 3: 128-DIAL EXPANSION"),
        (11.18, 2.74, "STAGE 4: QUORUM & ALARM"),
    ]
    for x, w, title in stages:
        _rbox(ax, x, 0.08, w, 4.18, C_PANEL_BG, C_PANEL_EDGE, lw=1.1, r=0.09, zorder=1)
        _rbox(ax, x, 4.34, w, 0.48, C_NAVY, C_NAVY, lw=1.0, r=0.08, zorder=2)
        ax.text(
            x + w / 2.0,
            4.58,
            title,
            ha="center",
            va="center",
            fontsize=12.8,
            fontweight="bold",
            color="white",
            zorder=4,
        )


def _draw_stage1(ax: plt.Axes) -> tuple[tuple[float, float], tuple[float, float]]:
    # Left vertical time rail cleanly inside Stage 1
    _arrow(ax, (0.31, 4.00), (0.31, 0.48), color=C_SUBTEXT, lw=1.3, zorder=5)
    ax.text(0.20, 2.24, "Time", ha="center", va="center", rotation=90, fontsize=13.5, color=C_SUBTEXT, fontweight="bold", zorder=5)

    # Context window (top vertical block, Slate Blue)
    _rbox(ax, 0.42, 2.12, 0.90, 1.90, C_CTX_FILL, C_CTX_EDGE, lw=1.7, r=0.06, zorder=4)
    ax.text(0.87, 3.54, "Context", ha="center", va="center", fontsize=14.0, fontweight="bold", color="white", zorder=5)
    ax.text(0.87, 3.22, "Window", ha="center", va="center", fontsize=14.0, fontweight="bold", color="white", zorder=5)
    ax.text(0.87, 2.78, r"$\mathbf{x}_{\mathrm{ctx}}$", ha="center", va="center", fontsize=15.5, color="white", zorder=5)
    ax.text(0.87, 2.42, r"($C\!=\!256$)", ha="center", va="center", fontsize=13.0, color="#ECF3FA", zorder=5)

    # Split boundary at index t
    ax.plot([0.36, 1.72], [2.04, 2.04], color=C_NAVY, lw=1.4, ls="--", zorder=5)
    ax.text(
        0.34,
        2.04,
        r"$t$",
        ha="right",
        va="center",
        fontsize=14.5,
        fontweight="bold",
        color=C_NAVY,
        bbox=dict(boxstyle="round,pad=0.06", fc=C_PANEL_BG, ec="none"),
        zorder=6,
    )

    # Target (Suspect) window (bottom vertical block, Plum Purple)
    _rbox(ax, 0.42, 0.50, 0.90, 1.46, C_TGT_FILL, C_TGT_EDGE, lw=1.7, r=0.06, zorder=4)
    ax.text(0.87, 1.64, "Target", ha="center", va="center", fontsize=14.0, fontweight="bold", color="white", zorder=5)
    ax.text(0.87, 1.34, "(Suspect)", ha="center", va="center", fontsize=13.0, color="#F0E8FA", zorder=5)
    ax.text(0.87, 0.98, r"$\mathbf{x}_{\mathrm{tgt}}$", ha="center", va="center", fontsize=15.5, color="white", zorder=5)
    ax.text(0.87, 0.69, r"($S\!=\!64$)", ha="center", va="center", fontsize=13.0, color="#F0E8FA", zorder=5)

    # Nested M=3 vertical prefix bars starting at y=1.96 downward (flush with Target bottom y=0.50)
    bar_specs = [
        (1.38, 0.365, C_PREF[0]),
        (1.49, 0.730, C_PREF[1]),
        (1.60, 1.460, C_PREF[2]),
    ]
    for bx, bh, col in bar_specs:
        rect = Rectangle((bx, 1.96 - bh), 0.085, bh, facecolor=col, edgecolor=C_TGT_EDGE, linewidth=0.95, zorder=4)
        ax.add_patch(rect)

    # Bottom input summary label cleanly inside Stage 1
    ax.text(
        1.02,
        0.24,
        r"$W\!=\!320 \;\vert\; M\!=\!3\!:\!\{16,32,64\}$",
        ha="center",
        va="center",
        fontsize=12.0,
        color=C_TGT_EDGE,
        fontweight="bold",
        zorder=5,
    )

    # Top track container (Online)
    _rbox(ax, 1.90, 2.38, 2.74, 1.72, C_CTX_TRACK, "#B6CADB", lw=1.2, r=0.08, zorder=2)
    ax.text(3.04, 3.92, "TOP TRACK (Online)", ha="center", va="center", fontsize=13.0, fontweight="bold", color=C_CTX_EDGE, zorder=4)

    # Straight horizontal arrow: Context -> Online Encoder
    _arrow(ax, (1.32, 3.10), (2.02, 3.10), color=C_CTX_EDGE, lw=2.0, zorder=6)

    # Online Encoder trapezoid
    trap = Polygon(
        [(2.02, 2.56), (3.08, 2.68), (3.08, 3.52), (2.02, 3.64)],
        closed=True,
        facecolor=C_CTX_ENC,
        edgecolor=C_CTX_EDGE,
        linewidth=1.5,
        zorder=4,
    )
    ax.add_patch(trap)
    ax.text(2.54, 3.28, r"Online $E_\theta$", ha="center", va="center", fontsize=13.8, fontweight="bold", color=C_TEXT, zorder=5)
    ax.text(2.54, 2.92, "rFFT + TCN", ha="center", va="center", fontsize=12.5, color=C_CTX_EDGE, zorder=5)

    # Barycenter & LayerNorm box
    _arrow(ax, (3.08, 3.10), (3.24, 3.10), color=C_CTX_EDGE, lw=1.6, zorder=6)
    _rbox(ax, 3.24, 2.70, 0.78, 0.80, C_TEAL, "#1F585A", lw=1.4, r=0.06, zorder=4)
    ax.text(3.63, 3.24, r"$-\bar{\mu}_{\mathrm{ctx}}$", ha="center", va="center", fontsize=14.0, fontweight="bold", color="white", zorder=5)
    ax.text(3.63, 2.90, "LN", ha="center", va="center", fontsize=13.2, fontweight="bold", color="#D9F2F2", zorder=5)

    # z_ctx node
    _arrow(ax, (4.02, 3.10), (4.22, 3.10), color=C_TEAL, lw=1.6, zorder=6)
    z_ctx_pos = (4.37, 3.10)
    ax.add_patch(Circle(z_ctx_pos, 0.135, fc=C_TEAL, ec="#1B4D4F", lw=1.5, zorder=5))
    ax.text(4.37, 3.42, r"$z_{\mathrm{ctx}}$", ha="center", va="center", fontsize=14.8, fontweight="bold", color=C_TEXT, zorder=5)
    ax.text(4.37, 2.76, "32D", ha="center", va="center", fontsize=12.8, color=C_SUBTEXT, zorder=5)

    # EMA update dashed vertical arrow
    _arrow(ax, (2.56, 2.56), (2.56, 1.56), color=C_SUBTEXT, lw=1.5, ls="--", zorder=5)
    ax.text(
        2.66,
        2.12,
        "EMA update",
        ha="left",
        va="center",
        fontsize=12.8,
        fontweight="bold",
        color=C_SUBTEXT,
        bbox=dict(boxstyle="round,pad=0.10", fc=C_PANEL_BG, ec="none"),
        zorder=6,
    )

    # Bottom track container (EMA Target)
    _rbox(ax, 1.90, 0.42, 2.74, 1.52, C_TGT_TRACK, "#CBB9E2", lw=1.2, r=0.08, zorder=2)
    ax.text(3.04, 1.76, "BOTTOM TRACK (EMA)", ha="center", va="center", fontsize=13.0, fontweight="bold", color=C_TGT_EDGE, zorder=4)

    # Straight horizontal arrow: Target Prefixes -> EMA Encoder with stop-grad //
    _arrow(ax, (1.70, 1.06), (2.34, 1.06), color=C_TGT_EDGE, lw=2.0, zorder=6)
    ax.plot([1.95, 2.02], [0.95, 1.17], color=C_TGT_EDGE, lw=1.9, zorder=7)
    ax.plot([2.04, 2.11], [0.95, 1.17], color=C_TGT_EDGE, lw=1.9, zorder=7)
    ax.text(2.03, 0.73, "stop-\ngrad", ha="center", va="center", fontsize=11.8, linespacing=0.95, color=C_TGT_EDGE, zorder=6)

    # EMA Encoder trapezoid (widened for generous E_phi padding)
    trap_ema = Polygon(
        [(2.34, 0.58), (3.80, 0.68), (3.80, 1.44), (2.34, 1.54)],
        closed=True,
        facecolor=C_TGT_ENC,
        edgecolor=C_TGT_EDGE,
        linewidth=1.5,
        zorder=4,
    )
    ax.add_patch(trap_ema)
    ax.text(3.05, 1.21, r"EMA Encoder $E_\phi$", ha="center", va="center", fontsize=13.4, fontweight="bold", color="white", zorder=5)
    ax.text(3.05, 0.88, "(Shared Arch.)", ha="center", va="center", fontsize=12.5, color="#E8DCF8", zorder=5)

    # 3 target embedding nodes z_tgt^(1..3)
    _arrow(ax, (3.80, 1.06), (4.20, 1.06), color=C_TGT_EDGE, lw=1.7, zorder=6)
    for dy, col in zip([0.21, 0.0, -0.21], C_PREF):
        ax.add_patch(Circle((4.37, 1.06 + dy), 0.08, fc=col, ec=C_TGT_EDGE, lw=1.2, zorder=5))
    ax.text(4.37, 1.52, r"$z_{\mathrm{tgt}}^{(1..3)}$", ha="center", va="center", fontsize=14.2, fontweight="bold", color=C_TGT_EDGE, zorder=5)
    ax.text(4.37, 0.60, r"$3{\times}32$D", ha="center", va="center", fontsize=12.5, color=C_SUBTEXT, zorder=5)

    return z_ctx_pos, (4.46, 1.06)


def _draw_stage2(
    ax: plt.Axes,
    z_ctx_pos: tuple[float, float],
    z_tgt_pos: tuple[float, float],
) -> tuple[float, float]:
    # Parallel Multi-Horizon Predictor
    _rbox(ax, 5.06, 3.16, 2.80, 0.90, "#2B3E50", C_NAVY, lw=1.6, r=0.08, zorder=4)
    ax.text(6.46, 3.74, "Parallel Predictor", ha="center", va="center", fontsize=14.5, fontweight="bold", color="white", zorder=5)
    ax.text(6.46, 3.38, r"1-LN MLP ($z_{\mathrm{ctx}}\!+\!\mathbf{p}_m,\,M\!=\!3$)", ha="center", va="center", fontsize=13.0, color="#DCE8F2", zorder=5)

    # Route z_ctx cleanly through gutter (x=4.82) into predictor
    _polyline_arrow(ax, [(z_ctx_pos[0] + 0.135, z_ctx_pos[1]), (4.82, 3.10), (4.82, 3.60), (5.06, 3.60)], color=C_CTX_EDGE, lw=1.8)

    # Subtraction node (-)
    sub_pos = (5.44, 2.18)
    ax.add_patch(Circle(sub_pos, 0.20, fc="white", ec=C_NAVY, lw=1.8, zorder=5))
    ax.text(sub_pos[0], sub_pos[1], r"$-$", ha="center", va="center", fontsize=18.0, fontweight="bold", color=C_NAVY, zorder=6)

    # Down arrow from predictor to (-)
    _arrow(ax, (5.44, 3.16), (5.44, 2.38), color=C_NAVY, lw=1.8, zorder=6)
    ax.text(5.86, 2.78, r"$\hat{z}_{\mathrm{tgt}}^{(1..3)}$", ha="center", va="center", fontsize=14.2, fontweight="bold", color=C_TEXT, zorder=6)

    # Route z_tgt^(1..3) into (-)
    _polyline_arrow(ax, [z_tgt_pos, (5.44, z_tgt_pos[1]), (5.44, 1.98)], color=C_TGT_EDGE, lw=1.8)

    # Whitening arrow (-) -> E_tilde
    _arrow(ax, (5.64, 2.18), (6.40, 2.18), color=C_TEAL, lw=2.0, zorder=6)
    ax.text(6.02, 2.44, r"$\mathbf{\Omega}^{1/2}$", ha="center", va="center", fontsize=14.8, fontweight="bold", color=C_TEAL, zorder=6)
    ax.text(6.02, 1.92, "Eq. (4)", ha="center", va="center", fontsize=12.8, color=C_SUBTEXT, zorder=6)

    # Whitened residual tensor grid (3 rows x 8 cols representing 3 x 32)
    gx, gy, gw, gh = 6.44, 1.66, 1.42, 1.05
    _rbox(ax, gx - 0.04, gy - 0.04, gw + 0.08, gh + 0.08, "white", C_TEAL, lw=1.6, r=0.05, zorder=3)
    pattern = [
        [0, 1, 2, 1, 0, 2, 1, 0],
        [1, 2, 3, 2, 1, 1, 2, 1],
        [0, 1, 2, 3, 2, 1, 3, 2],
    ]
    cw, ch = gw / 8.0, gh / 3.0
    for r_idx in range(3):
        for c_idx in range(8):
            rect = Rectangle(
                (gx + c_idx * cw, gy + (2 - r_idx) * ch),
                cw,
                ch,
                facecolor=C_TEAL_CELLS[pattern[r_idx][c_idx]],
                edgecolor="white",
                linewidth=0.9,
                zorder=4,
            )
            ax.add_patch(rect)
    ax.text(gx + gw / 2.0, 2.90, r"Whitened $\tilde{\mathbf{E}}$", ha="center", va="center", fontsize=14.4, fontweight="bold", color=C_TEAL, zorder=5)
    ax.text(gx + gw / 2.0, 1.40, r"$3{\times}32$ residual tensor", ha="center", va="center", fontsize=13.0, color=C_SUBTEXT, zorder=5)

    # Training Objective box at bottom of Stage 2
    _rbox(ax, 5.14, 0.24, 2.70, 0.78, C_OBJ_BG, "#5D7690", lw=1.4, r=0.07, zorder=4)
    ax.text(6.49, 0.78, "Training Objective (Eq. 1)", ha="center", va="center", fontsize=13.5, fontweight="bold", color=C_NAVY, zorder=5)
    ax.text(6.49, 0.45, r"MSE + Anti-Collapse ($\mathcal{R}_\gamma$)", ha="center", va="center", fontsize=13.0, color=C_TEXT, zorder=5)
    # Clean dotted connector from subtraction node to objective box
    ax.plot([5.58, 5.78, 5.78], [2.04, 1.80, 1.02], color="#7A8FA3", lw=1.3, ls=":", zorder=3)

    return (gx + gw + 0.04, 2.18)


def _draw_stage3(ax: plt.Axes, e_out: tuple[float, float]) -> tuple[float, float]:
    # Fork from E_tilde cleanly in gutter (x=8.10) into two branches
    fork_x = 8.10
    ax.plot([e_out[0], fork_x], [e_out[1], e_out[1]], color=C_NAVY, lw=1.8, zorder=6)
    _polyline_arrow(ax, [(fork_x, e_out[1]), (fork_x, 3.16), (8.36, 3.16)], color=C_TEAL, lw=1.8)
    _polyline_arrow(ax, [(fork_x, e_out[1]), (fork_x, 1.24), (8.36, 1.24)], color=C_TGT_EDGE, lw=1.8)

    # Top card: 96 Position Dials
    _rbox(ax, 8.38, 2.34, 2.18, 1.72, C_TEAL_LIGHT, C_TEAL_EDGE, lw=1.5, r=0.08, zorder=3)
    for idx, (dx, dy, col) in enumerate([(0.0, 0.16, "#B9DFDF"), (0.10, 0.08, "#6BB3B5"), (0.20, 0.0, C_TEAL)]):
        _rbox(ax, 8.85 + dx, 3.40 + dy, 1.04, 0.30, col, "#1F585A", lw=1.0, r=0.03, zorder=4 + idx)
    ax.text(9.47, 3.16, "Position Dials (96)", ha="center", va="center", fontsize=14.0, fontweight="bold", color="#1C4E50", zorder=6)
    ax.text(9.47, 2.82, r"$|\tilde{e}_{m,c}(t)|$", ha="center", va="center", fontsize=14.8, fontweight="bold", color=C_TEXT, zorder=6)
    ax.text(9.47, 2.52, r"Per-horizon ($3{\times}32$)", ha="center", va="center", fontsize=12.8, color=C_SUBTEXT, zorder=6)

    # Bottom card: 32 Reynolds Mean-Drift Dials
    _rbox(ax, 8.38, 0.34, 2.18, 1.72, C_TGT_TRACK, "#9F84C4", lw=1.5, r=0.08, zorder=3)
    for yy in [1.76, 1.62, 1.48]:
        _arrow(ax, (8.70, yy), (9.32, yy), color="#8C74AE", lw=1.3, zorder=5)
    _arrow(ax, (9.44, 1.62), (10.22, 1.62), color=C_TGT_EDGE, lw=2.8, zorder=6)
    ax.text(9.47, 1.18, "Mean-Drift Dials (32)", ha="center", va="center", fontsize=13.8, fontweight="bold", color=C_TGT_EDGE, zorder=6)
    ax.text(9.47, 0.84, r"$|\bar{e}_c(t)| = |\frac{1}{3}\sum_m \tilde{e}_{m,c}|$", ha="center", va="center", fontsize=13.8, fontweight="bold", color=C_TEXT, zorder=6)
    ax.text(9.47, 0.52, "Cross-horizon flow", ha="center", va="center", fontsize=12.8, color=C_SUBTEXT, zorder=6)

    # Right merge bracket combining 96 + 32 = 128 dials
    bx = 10.76
    ax.plot([10.56, bx, bx, 10.56], [3.16, 3.16, 1.24, 1.24], color=C_NAVY, lw=1.7, zorder=5)
    ax.text(
        bx,
        2.18,
        "128\nDials",
        ha="center",
        va="center",
        fontsize=13.2,
        fontweight="bold",
        color=C_NAVY,
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec=C_NAVY, lw=1.3),
        zorder=7,
    )
    return (bx + 0.16, 2.18)


def _draw_stage4(ax: plt.Axes, d_out: tuple[float, float]) -> None:
    # Arrow from 128 Dials cleanly through gutter (x=11.10) into Tripwires card
    _polyline_arrow(ax, [d_out, (11.10, 2.18), (11.10, 3.34), (11.30, 3.34)], color=C_NAVY, lw=1.8)

    # Card 1: Per-Dial Quantile Tripwires
    _rbox(ax, 11.32, 2.58, 2.46, 1.52, "white", "#4E6882", lw=1.5, r=0.08, zorder=3)
    ax.text(12.55, 3.88, "Per-Dial Tripwires (Eq. 6)", ha="center", va="center", fontsize=13.8, fontweight="bold", color=C_NAVY, zorder=5)

    # 8 schematic circuit-breaker switches (3 tripped in coral, 5 quiet in slate)
    tripped = {1, 3, 5}
    for i in range(8):
        sx = 11.50 + i * 0.275
        sy = 3.24
        is_on = i in tripped
        fc = C_ALARM if is_on else "#DDE4EC"
        ec = "#8A1F2E" if is_on else "#7B8E9F"
        _rbox(ax, sx, sy, 0.18, 0.34, fc, ec, lw=1.2, r=0.03, zorder=4)
        ky = sy + 0.22 if is_on else sy + 0.05
        _rbox(ax, sx + 0.025, ky, 0.13, 0.08, "white", ec, lw=0.9, r=0.015, zorder=5)

    ax.text(12.55, 2.98, r"$b_j(t)=\mathbf{1}(d_j>\tau_j)\in\{0,1\}$", ha="center", va="center", fontsize=13.4, color=C_TEXT, zorder=5)
    ax.text(12.55, 2.72, r"$\tau_j=q_{0.985}+10^{-4}$ (per dial)", ha="center", va="center", fontsize=12.6, color=C_SUBTEXT, zorder=5)

    # Arrow down -> Single Healthy Quorum
    _arrow(ax, (12.55, 2.58), (12.55, 2.26), color=C_TEAL, lw=1.8, zorder=6)

    # Card 2: Single Healthy Quorum
    _rbox(ax, 11.32, 1.36, 2.46, 0.90, C_TEAL_LIGHT, C_TEAL, lw=1.5, r=0.08, zorder=3)
    ax.text(12.55, 2.04, "Healthy Quorum (Eq. 7)", ha="center", va="center", fontsize=14.0, fontweight="bold", color="#1C4E50", zorder=5)
    ax.text(12.55, 1.74, r"$N_{128}(\tau)=\sum b_j > K_{\mathrm{Lite}}$", ha="center", va="center", fontsize=13.6, fontweight="bold", color=C_TEXT, zorder=5)
    ax.text(12.55, 1.49, r"$K_{\mathrm{Lite}}=\max(1.12\,q_{0.995},\,3.5)$", ha="center", va="center", fontsize=12.6, color=C_SUBTEXT, zorder=5)

    # Arrow down -> Real-Time Causal Alarm
    _arrow(ax, (12.55, 1.36), (12.55, 1.06), color=C_ALARM, lw=1.9, zorder=6)

    # Card 3: Causal Alarm output with mini spike trace
    _rbox(ax, 11.32, 0.22, 2.46, 0.84, C_ALARM_BG, C_ALARM, lw=1.7, r=0.08, zorder=3)
    wx = [11.44, 11.54, 11.64, 11.73, 11.81, 11.88, 11.96, 12.04]
    wy = [0.46, 0.52, 0.44, 0.50, 0.48, 0.88, 0.56, 0.48]
    ax.plot([11.42, 12.06], [0.68, 0.68], color=C_ALARM, lw=1.1, ls="--", zorder=4)
    ax.plot(wx, wy, color=C_NAVY, lw=1.5, zorder=5)
    ax.plot([11.88], [0.88], marker="o", ms=5.0, color=C_ALARM, zorder=6)

    ax.text(12.92, 0.78, r"Causal Alarm $\hat{y}(\tau)$", ha="center", va="center", fontsize=14.0, fontweight="bold", color="#8F1D2C", zorder=5)
    ax.text(12.92, 0.46, r"at $\tau = t + S - 1$", ha="center", va="center", fontsize=13.4, color="#5C2029", zorder=5)


def build_architecture_figure() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig = plt.figure(figsize=(14.0, 4.9), facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, 14.0)
    ax.set_ylim(0.0, 4.9)
    ax.axis("off")

    _draw_stage_frames(ax)
    z_ctx_pos, z_tgt_pos = _draw_stage1(ax)
    e_out = _draw_stage2(ax, z_ctx_pos, z_tgt_pos)
    d_out = _draw_stage3(ax, e_out)
    _draw_stage4(ax, d_out)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SVG_PATH, format="svg", facecolor="white")
    fig.savefig(PDF_PATH, format="pdf", facecolor="white")
    plt.close(fig)
    print(f"[OK] Saved true-vector SVG: {SVG_PATH}")
    print(f"[OK] Saved true-vector PDF: {PDF_PATH}")


if __name__ == "__main__":
    build_architecture_figure()
