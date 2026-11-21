#!/usr/bin/env python3
"""Generate publication-grade 100% true-vector 2-panel geometric figure:
(a) Raw Correlated Errors (Naive 1D Box & Global Scalar Norm failures) vs.
(b) ZCA-Whitened Space + Per-Dial Tripwires & Quorum Zones (exact symmetric ZCA).

Follows Dr. Andrey Churkin's 8 Principles of Beautiful Figures:
- 100% true vector (.svg + .pdf, pdf.fonttype=42, svg.fonttype='none', zero raster)
- 1:1 IEEE two-column canvas scaling (7.2 in -> ~7.0 in print width)
- Strict external-only frameless legend above both panels
- Harmonious grayscale-safe Teal / Purple / Ochre / Crimson / Slate palette
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "NCAD_CS" / "figures"
OUT_PDF = FIG_DIR / "zca_quorum_geometry.pdf"
OUT_SVG = FIG_DIR / "zca_quorum_geometry.svg"

# Harmonious grayscale-safe palette (Teal / Purple / Ochre / Crimson / Slate)
COL_HEALTHY = "#216e75"
COL_HEALTHY_FILL = "#cfe7e9"
COL_BOX = "#475569"
COL_SCALAR = "#6a408d"
COL_HIJACKER = "#b8741e"
COL_HIJACKER_FILL = "#fef3df"
COL_FAULT = "#c23548"
COL_QUORUM_FILL = "#fbe6e9"
COL_TEXT = "#1e293b"


def build_figure(preview_png: Path | None = None) -> None:
    # 1:1 IEEE two-column scale (7.2 in canvas printed at ~6.8-7.1 in)
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 10.0,
        "axes.labelsize": 10.8,
        "axes.titlesize": 11.2,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.6,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    # Exact 2D correlated Gaussian healthy distribution:
    # Axis 1 has 1.9x higher std (noisy/hijacker-prone), correlation rho = 0.88
    rng = np.random.default_rng(42)
    s1, s2, rho = 1.90, 1.00, 0.88
    Sigma = np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    W_zca = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    W_inv = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    # Draw healthy samples and apply exact symmetric ZCA whitening
    E_raw = rng.multivariate_normal([0.0, 0.0], Sigma, size=420)
    E_white = E_raw @ W_zca.T

    # Thresholds (98.5% two-sided normal quantile z_q = 2.432)
    z_q = 2.432
    tau1_raw = z_q * s1   # 4.62
    tau2_raw = z_q * s2   # 2.43
    r_glob = float(np.quantile(np.linalg.norm(E_raw, axis=1), 0.985))
    tau_w = z_q           # 2.43 in whitened space

    # Exact test points mapped faithfully between Panel (a) and Panel (b) via W_zca
    p_coord_raw = np.array([-2.35, 1.48])
    p_coord_w = W_zca @ p_coord_raw

    p_hj_w = np.array([3.30, 0.20])
    p_hj_raw = W_inv @ p_hj_w

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.2, 3.55), facecolor="white",
        gridspec_kw={"wspace": 0.42, "left": 0.08, "right": 0.98, "top": 0.77, "bottom": 0.145},
    )

    halo = dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.88)

    # =========================================================================
    # PANEL (a): Raw Correlated Errors (e1, e2)
    # =========================================================================
    ax1.set_title("(a) Raw Correlated Errors", fontweight="bold", color=COL_TEXT, pad=7)

    chi2_r = np.sqrt(8.40)  # 98.5% chi2_2 contour
    angle_deg = float(np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1])))
    ax1.add_patch(Ellipse(
        (0, 0),
        width=2 * chi2_r * np.sqrt(eigvals[1]),
        height=2 * chi2_r * np.sqrt(eigvals[0]),
        angle=angle_deg,
        facecolor=COL_HEALTHY_FILL, edgecolor=COL_HEALTHY, lw=1.55, alpha=0.55, zorder=2,
    ))
    ax1.scatter(E_raw[:, 0], E_raw[:, 1], s=9, color=COL_HEALTHY, alpha=0.35, linewidths=0, zorder=3)

    # Naive 1D Independent Threshold Box (discards covariance)
    ax1.add_patch(Rectangle(
        (-tau1_raw, -tau2_raw), 2 * tau1_raw, 2 * tau2_raw,
        fill=False, edgecolor=COL_BOX, lw=1.45, ls="--", zorder=4,
    ))

    # Global Scalar L2 Threshold Circle (inflated by noisy axis e1)
    ax1.add_patch(Circle(
        (0, 0), radius=r_glob,
        fill=False, edgecolor=COL_SCALAR, lw=1.5, ls=":", zorder=4,
    ))

    ax1.scatter(
        [p_coord_raw[0]], [p_coord_raw[1]],
        marker="*", s=160, color=COL_FAULT, edgecolor="white", lw=0.8, zorder=7,
    )
    ax1.scatter(
        [p_hj_raw[0]], [p_hj_raw[1]],
        marker="D", s=58, color=COL_HIJACKER, edgecolor="white", lw=0.8, zorder=7,
    )

    # Zero-overlap annotations in Panel (a) — Coordinated Shift placed cleanly outside dotted circle
    ax1.annotate(
        "Coordinated Shift\n(Missed by 1D Box\n& Scalar Norm)",
        xy=(p_coord_raw[0], p_coord_raw[1]),
        xytext=(-6.45, 4.95),
        fontsize=8.2, fontweight="bold", color=COL_FAULT, ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=COL_FAULT, lw=1.15, shrinkA=2, shrinkB=6),
        bbox=halo, zorder=8,
    )

    ax1.annotate(
        "Mode Hijacker\n(False Alarm)",
        xy=(p_hj_raw[0], p_hj_raw[1]),
        xytext=(4.05, 5.20),
        fontsize=8.1, fontweight="bold", color=COL_HIJACKER, ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color=COL_HIJACKER, lw=1.1, shrinkA=2, shrinkB=5),
        bbox=halo, zorder=8,
    )

    ax1.annotate(
        "Naive 1D Box",
        xy=(tau1_raw, -tau2_raw),
        xytext=(1.75, -4.25),
        fontsize=8.3, fontweight="bold", color=COL_BOX, ha="left", va="top",
        arrowprops=dict(arrowstyle="->", color=COL_BOX, lw=1.0, shrinkA=2, shrinkB=3),
        bbox=halo, zorder=8,
    )

    ax1.annotate(
        "Scalar Norm\n(Inflated by $e_1$)",
        xy=(-3.1, -3.75),
        xytext=(-6.40, -5.00),
        fontsize=8.2, fontweight="bold", color=COL_SCALAR, ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=COL_SCALAR, lw=1.0, shrinkA=2, shrinkB=3),
        bbox=halo, zorder=8,
    )

    ax1.set_xlim(-6.75, 6.75)
    ax1.set_ylim(-6.20, 6.50)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("Raw Residual $e_1$ (Noisy Dial)")
    ax1.set_ylabel("Raw Residual $e_2$ (Quiet Dial)")
    ax1.set_xticks([-4, 0, 4])
    ax1.set_yticks([-4, 0, 4])

    # =========================================================================
    # PANEL (b): ZCA-Whitened Space (e_tilde_1, e_tilde_2)
    # =========================================================================
    ax2.set_title("(b) ZCA-Whitened Space + Quorum", fontweight="bold", color=COL_TEXT, pad=7)
    lim_w = 5.45

    for y0, h in [(tau_w, lim_w - tau_w), (-lim_w, lim_w - tau_w)]:
        ax2.add_patch(Rectangle((-tau_w, y0), 2 * tau_w, h, facecolor=COL_HIJACKER_FILL, ec="none", alpha=0.75, zorder=0))
    for x0, w in [(tau_w, lim_w - tau_w), (-lim_w, lim_w - tau_w)]:
        ax2.add_patch(Rectangle((x0, -tau_w), w, 2 * tau_w, facecolor=COL_HIJACKER_FILL, ec="none", alpha=0.75, zorder=0))
    for x0 in (-lim_w, tau_w):
        for y0 in (-lim_w, tau_w):
            ax2.add_patch(Rectangle((x0, y0), lim_w - tau_w, lim_w - tau_w, facecolor=COL_QUORUM_FILL, ec="none", alpha=0.90, zorder=0))

    ax2.add_patch(Circle(
        (0, 0), radius=chi2_r,
        facecolor=COL_HEALTHY_FILL, edgecolor=COL_HEALTHY, lw=1.55, alpha=0.50, zorder=2,
    ))
    ax2.scatter(E_white[:, 0], E_white[:, 1], s=9, color=COL_HEALTHY, alpha=0.35, linewidths=0, zorder=3)

    for t_val in (-tau_w, tau_w):
        ax2.axvline(t_val, color=COL_BOX, lw=1.35, ls="--", zorder=4)
        ax2.axhline(t_val, color=COL_BOX, lw=1.35, ls="--", zorder=4)

    # Explicit dynamic tripwire labels (+/- tau_j) so ticks (-4, 0, 4) are never mistaken for quorum
    ax2.text(tau_w + 0.14, 5.18, r"$+\tau_1$", fontsize=8.0, fontweight="bold", color=COL_BOX, ha="left", va="top", zorder=6)
    ax2.text(-tau_w + 0.14, 5.18, r"$-\tau_1$", fontsize=8.0, fontweight="bold", color=COL_BOX, ha="left", va="top", zorder=6)
    ax2.text(5.22, tau_w + 0.16, r"$+\tau_2$", fontsize=8.0, fontweight="bold", color=COL_BOX, ha="right", va="bottom", zorder=6)
    ax2.text(-5.22, -tau_w - 0.16, r"$-\tau_2$", fontsize=8.0, fontweight="bold", color=COL_BOX, ha="left", va="top", zorder=6)

    ax2.scatter(
        [p_coord_w[0]], [p_coord_w[1]],
        marker="*", s=160, color=COL_FAULT, edgecolor="white", lw=0.8, zorder=7,
    )
    ax2.scatter(
        [p_hj_w[0]], [p_hj_w[1]],
        marker="D", s=58, color=COL_HIJACKER, edgecolor="white", lw=0.8, zorder=7,
    )

    ax2.annotate(
        "Quorum Fires!\n(2/2 Dials Trip)",
        xy=(p_coord_w[0], p_coord_w[1]),
        xytext=(-1.45, 4.18),
        fontsize=8.3, fontweight="bold", color=COL_FAULT, ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=COL_FAULT, lw=1.15, shrinkA=2, shrinkB=6),
        bbox=halo, zorder=8,
    )

    ax2.annotate(
        "1-Dial Hijacker\n(Suppressed)",
        xy=(p_hj_w[0], p_hj_w[1]),
        xytext=(3.78, -1.65),
        fontsize=8.3, fontweight="bold", color=COL_HIJACKER, ha="center", va="top",
        arrowprops=dict(arrowstyle="->", color=COL_HIJACKER, lw=1.1, shrinkA=2, shrinkB=5),
        bbox=halo, zorder=8,
    )

    ax2.text(
        -3.95, -4.18, "Quorum\n(2/2 Dials)", fontsize=8.0, fontweight="bold",
        color=COL_FAULT, ha="center", va="center", zorder=6,
    )
    ax2.text(
        0.0, -4.05, "1-Dial Band\n(No Alarm)", fontsize=8.1, fontweight="bold",
        color=COL_HIJACKER, ha="center", va="center", zorder=6,
    )

    ax2.set_xlim(-lim_w, lim_w)
    ax2.set_ylim(-lim_w, lim_w)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel(r"Whitened Dial $\tilde{e}_1$ ($\sigma$ units)")
    ax2.set_ylabel(r"Whitened Dial $\tilde{e}_2$ ($\sigma$ units)")
    ax2.set_xticks([-4, 0, 4])
    ax2.set_yticks([-4, 0, 4])

    for ax in (ax1, ax2):
        ax.axhline(0, color="#cbd5e1", lw=0.7, zorder=1)
        ax.axvline(0, color="#cbd5e1", lw=0.7, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#64748b")
        ax.spines["bottom"].set_color("#64748b")
        ax.tick_params(colors="#334155")

    # Crisp transformation arrow between Panel (a) and Panel (b)
    arrow = FancyArrowPatch(
        (0.452, 0.455), (0.542, 0.455),
        transform=fig.transFigure,
        arrowstyle="Simple,tail_width=2.5,head_width=7.5,head_length=7",
        color=COL_SCALAR,
        zorder=10,
    )
    fig.patches.append(arrow)
    fig.text(
        0.497, 0.495, "ZCA\nWhitening\n" + r"($\mathbf{\Omega}^{1/2}{=}\mathbf{\Sigma}^{-1/2}$)",
        ha="center", va="bottom", fontsize=8.1, fontweight="bold", color=COL_SCALAR,
    )

    # Strict external frameless legend above both panels
    handles = [
        Patch(facecolor=COL_HEALTHY_FILL, edgecolor=COL_HEALTHY, lw=1.3, label="Healthy 98.5% Region"),
        Line2D([0], [0], color=COL_BOX, lw=1.4, ls="--", label="Per-Coordinate Tripwires"),
        Line2D([0], [0], color=COL_SCALAR, lw=1.5, ls=":", label="Global Scalar Norm"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=COL_FAULT, markersize=11, label="Coordinated Fault"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=COL_HIJACKER, markersize=7, label="1-Dial Hijacker"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.985),
        ncol=3,
        frameon=False,
        columnspacing=1.15,
        handlelength=1.7,
        handletextpad=0.45,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf")
    fig.savefig(OUT_SVG, format="svg")
    if preview_png is not None:
        preview_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_png, format="png", dpi=240)
    plt.close(fig)
    print(f"Saved: {OUT_PDF}")
    print(f"Saved: {OUT_SVG}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 2-panel ZCA whitening & quorum geometry figure.")
    parser.add_argument("--preview-png", type=Path, default=None, help="Optional PNG preview path.")
    args = parser.parse_args()
    build_figure(preview_png=args.preview_png)
