#!/usr/bin/env python3
"""
Generate Publication-Grade IEEE TNNLS Architectural Diagram: Operator-Entropy JEPA
-----------------------------------------------------------------------------------
Generates a crisp, publication-ready vector diagram depicting the complete
Operator-Entropy JEPA architecture:
(A) Multiscale Input & Asymmetric Encoders (Context TCN vs. Target Momentum EMA with Stop-Grad)
(B) Geometric Latent Filtering & Koopman Transition Operator K \in R^{D x D}
(C) Dual Spectral Entropy Anti-Collapse Objective (Von Neumann entropy + Bregman barrier)
(D) Latent Coordinate Saliency & Inductive Tail Calibration (Streaming SPOT / EVT)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path(__file__).resolve().parent.parent
OUT_FIG_DIR = ROOT / "paper" / "NCAD_CS" / "figures"
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Publication Typography
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman']
plt.rcParams['mathtext.fontset'] = 'cm'

fig, ax = plt.subplots(figsize=(15.8, 7.2), dpi=300)
ax.set_xlim(0, 15.8)
ax.set_ylim(0, 7.2)
ax.axis('off')

# IEEE Academic Color Palette (Flat, High-Contrast, Professional)
c_slate_dark  = '#0F172A'  # Slate-900 (Main text & dark borders)
c_slate_muted = '#475569'  # Slate-600 (Subtext & annotations)
c_slate_bdr   = '#CBD5E1'  # Slate-300 (Card borders)

c_ctx_bg      = '#EFF6FF'  # Blue-50 (Context pipeline)
c_ctx_bdr     = '#2563EB'  # Blue-600
c_ctx_txt     = '#1E3A8A'  # Blue-900

c_tgt_bg      = '#FFFBEB'  # Amber-50 (Target EMA pipeline)
c_tgt_bdr     = '#D97706'  # Amber-600
c_tgt_txt     = '#92400E'  # Amber-900

c_koop_bg     = '#F0FDF4'  # Emerald-50 (Operator Dynamics)
c_koop_bdr    = '#059669'  # Emerald-600
c_koop_txt    = '#065F46'  # Emerald-900

c_loss_bg     = '#FAF5FF'  # Purple-50 (Dual Spectral Entropy Loss)
c_loss_bdr    = '#9333EA'  # Purple-600
c_loss_txt    = '#581C87'  # Purple-900

c_tail_bg     = '#FFF7ED'  # Orange-50 (SPOT Tail Calibration)
c_tail_bdr    = '#EA580C'  # Orange-600
c_tail_txt    = '#9A3412'  # Orange-900


def draw_card(x, y, w, h, title="", subtitle="", facecolor='#FFFFFF', edgecolor=c_slate_bdr, lw=1.0, rx=0.12):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={rx}",
                                facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=2)
    ax.add_patch(box)
    if title and subtitle:
        ax.text(x + w / 2, y + h * 0.68, title, ha='center', va='center', fontsize=8.6, fontweight='bold', color=c_slate_dark, zorder=3)
        ax.text(x + w / 2, y + h * 0.32, subtitle, ha='center', va='center', fontsize=7.4, color=c_slate_muted, zorder=3)
    elif title:
        ax.text(x + w / 2, y + h * 0.50, title, ha='center', va='center', fontsize=8.6, fontweight='bold', color=c_slate_dark, zorder=3)
    return box


def draw_arrow(x1, y1, x2, y2, color='#334155', lw=1.2, ls='-', style="-|>", rad=0.0):
    conn = f"arc3,rad={rad}" if rad != 0.0 else "arc3,rad=0"
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, lw=lw, color=color, linestyle=ls,
                                mutation_scale=9, shrinkA=0, shrinkB=0, connectionstyle=conn), zorder=5)


# Canvas Outer Boundary
main_box = patches.FancyBboxPatch((0.2, 0.2), 15.4, 6.8, boxstyle="round,pad=0.0,rounding_size=0.20",
                                 facecolor='#FAFAFA', edgecolor='#E2E8F0', linewidth=1.2, zorder=1)
ax.add_patch(main_box)

# Header Banner
ax.text(0.5, 6.70, "Operator-Entropy JEPA: Latent Dynamical Predictive Architecture",
        fontsize=12.5, fontweight='bold', color=c_slate_dark, ha='left', va='center')
ax.text(0.5, 6.40, r"Spectral Koopman Operator $\mathbf{K} \in \mathbb{R}^{D \times D}$ with Non-Contrastive Dual Von Neumann Entropy & Logarithmic Bregman Barrier",
        fontsize=8.5, color=c_slate_muted, ha='left', va='center')


# =========================================================================
# PANEL A: Asymmetric Dual-Stream Encoders
# =========================================================================
draw_card(0.5, 0.5, 3.8, 5.65, facecolor='#FFFFFF', edgecolor=c_slate_bdr, lw=1.0)
ax.text(2.4, 5.85, "(A) Multiscale Input & Encoders", ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_ctx_txt)

# Context Stream (Top)
draw_card(0.8, 4.4, 1.4, 1.05, "Context Window", r"$\mathbf{x}_{\mathrm{ctx}} \in \mathbb{R}^{C \times K}$", facecolor=c_ctx_bg, edgecolor=c_ctx_bdr)
draw_card(2.6, 4.4, 1.5, 1.05, "Online Encoder $E_\\theta$", "Dyadic LP + TCN", facecolor=c_ctx_bg, edgecolor=c_ctx_bdr)
draw_arrow(2.2, 4.92, 2.6, 4.92, color=c_ctx_bdr, lw=1.4)

# Target Stream (Bottom)
draw_card(0.8, 0.8, 1.4, 1.05, "Suspect Horizon", r"$\mathbf{x}_{\mathrm{tgt}} \in \mathbb{R}^{S \times K}$", facecolor=c_tgt_bg, edgecolor=c_tgt_bdr)
draw_card(2.6, 0.8, 1.5, 1.05, "Target Encoder $E_\\phi$", "EMA Replica (Stop-Grad)", facecolor=c_tgt_bg, edgecolor=c_tgt_bdr)
draw_arrow(2.2, 1.32, 2.6, 1.32, color=c_tgt_bdr, lw=1.4)

# Stop-gradient symbol on target stream
ax.text(2.68, 2.05, r"$\perp$ (Stop-Gradient)", fontsize=7.2, color='#B45309', fontweight='bold')

# EMA Momentum parameter transfer arrow
draw_arrow(3.35, 4.4, 3.35, 1.9, color='#9333EA', lw=1.3, ls='--')
ax.text(3.30, 3.15, r"$\phi \leftarrow m\phi + (1-m)\theta$" "\n" r"($m \in [0.996, 0.9995]$)",
        fontsize=6.8, color='#7E22CE', ha='right', va='center')

# Multi-scale filterbank explanation card
draw_card(0.8, 2.3, 2.0, 1.65, facecolor='#F8FAFC', edgecolor='#E2E8F0', lw=0.8)
ax.text(1.8, 3.70, "Multiscale Dyadic Split", ha='center', va='center', fontsize=7.8, fontweight='bold', color=c_slate_dark)
ax.text(1.8, 3.10, "• $C=256$, $S=64$, $K$ channels\n• rFFT dyadic partition of unity\n• Smooth partition: $\\sum_j \\Psi_j(\\omega) = 1$\n• Causal receptive field $\\geq C$",
        ha='center', va='center', fontsize=6.8, color=c_slate_muted)


# =========================================================================
# PANEL B: Latent Geometric Filtering & Koopman Operator
# =========================================================================
draw_card(4.5, 0.5, 4.4, 5.65, facecolor='#FFFFFF', edgecolor=c_slate_bdr, lw=1.0)
ax.text(6.7, 5.85, "(B) Geometric Filters & Koopman Transition", ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_koop_txt)

# Latent representation cards
draw_card(4.7, 4.4, 1.2, 1.05, r"$\mathbf{z}_{\mathrm{ctx}} \in \mathbb{R}^D$", "$D=32$", facecolor='#FFFFFF', edgecolor=c_ctx_bdr)
draw_arrow(4.1, 4.92, 4.7, 4.92, color=c_ctx_bdr, lw=1.4)

# Geometric filter blocks
draw_card(6.1, 4.4, 1.2, 1.05, "Filters", "Radial Heat\n+ Cohn-Elkies", facecolor='#F8FAFC', edgecolor=c_slate_bdr)
draw_arrow(5.9, 4.92, 6.1, 4.92, color=c_ctx_bdr, lw=1.2)

# Koopman transition operator card
draw_card(7.5, 4.4, 1.2, 1.05, "Koopman", r"$\mathbf{K} \in \mathbb{R}^{D \times D}$", facecolor=c_koop_bg, edgecolor=c_koop_bdr, lw=1.4)
draw_arrow(7.3, 4.92, 7.5, 4.92, color=c_koop_bdr, lw=1.4)

# Latent prediction card
draw_card(7.5, 2.7, 1.2, 1.05, r"$\hat{\mathbf{z}}_{\mathrm{tgt}} \in \mathbb{R}^D$", r"$\mathbf{K} \mathbf{z}_{\mathrm{ctx}}$", facecolor='#FFFFFF', edgecolor=c_koop_bdr)
draw_arrow(8.1, 4.4, 8.1, 3.75, color=c_koop_bdr, lw=1.4)

# Target latent card
draw_card(4.7, 0.8, 1.2, 1.05, r"$\mathbf{z}_{\mathrm{tgt}} \in \mathbb{R}^D$", "$D=32$", facecolor='#FFFFFF', edgecolor=c_tgt_bdr)
draw_arrow(4.1, 1.32, 4.7, 1.32, color=c_tgt_bdr, lw=1.4)

# Target filters card
draw_card(6.1, 0.8, 1.2, 1.05, "Filters", "Radial Heat\n+ Cohn-Elkies", facecolor='#F8FAFC', edgecolor=c_slate_bdr)
draw_arrow(5.9, 1.32, 6.1, 1.32, color=c_tgt_bdr, lw=1.2)

# Target Latent Clean
draw_card(7.5, 0.8, 1.2, 1.05, r"$\mathbf{z}_{\mathrm{tgt}} \in \mathbb{R}^D$", "Target Latent", facecolor='#FFFFFF', edgecolor=c_tgt_bdr)
draw_arrow(7.3, 1.32, 7.5, 1.32, color=c_tgt_bdr, lw=1.2)

# Structural note on Koopman operator
draw_card(4.7, 2.1, 2.5, 1.95, facecolor='#F0FDF4', edgecolor='#A7F3D0', lw=0.8)
ax.text(5.95, 3.80, "Koopman Operator Priors", ha='center', va='center', fontsize=7.6, fontweight='bold', color=c_koop_txt)
ax.text(5.95, 3.00, "• $\\hat{\\mathbf{z}}_{\\mathrm{tgt}} = \\mathbf{K} \\mathbf{z}_{\\mathrm{ctx}}$ (1,024 params)\n• Initialized: $\\mathbf{K} \\sim \\mathbf{I} + 0.01\\mathcal{N}(\\mathbf{0},\\mathbf{I})$\n• Singular values: $\\sigma_1 \\geq \\dots \\geq \\sigma_D$\n• Prevents dynamic collapse\n• Optional: Cayley unitary transform",
        ha='center', va='center', fontsize=6.8, color=c_slate_dark)


# =========================================================================
# PANEL C: Dual Spectral Entropy & Anti-Collapse Objective
# =========================================================================
draw_card(9.1, 0.5, 3.5, 5.65, facecolor='#FFFFFF', edgecolor=c_slate_bdr, lw=1.0)
ax.text(10.85, 5.85, "(C) Dual Spectral Entropy Objective", ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_loss_txt)

# Discrepancy node
circ = patches.Circle((9.7, 2.35), 0.28, facecolor='#FFFFFF', edgecolor=c_slate_dark, lw=1.3, zorder=4)
ax.add_patch(circ)
ax.text(9.7, 2.35, r"$\mathcal{D}$", ha='center', va='center', fontsize=10.0, fontweight='bold', color=c_slate_dark, zorder=5)

draw_arrow(8.7, 3.0, 9.45, 2.45, color=c_koop_bdr, lw=1.3)
draw_arrow(8.7, 1.5, 9.45, 2.25, color=c_tgt_bdr, lw=1.3)

# Composite Loss Breakdown Card
draw_card(9.3, 3.2, 3.1, 2.35, facecolor=c_loss_bg, edgecolor=c_loss_bdr, lw=1.0)
ax.text(10.85, 5.25, "Composite Objective Function", ha='center', va='center', fontsize=8.2, fontweight='bold', color=c_loss_txt)
ax.text(10.85, 4.88, r"$\mathcal{L} = \|\hat{\mathbf{z}}_{\mathrm{tgt}} - \mathbf{z}_{\mathrm{tgt}}\|_2^2 + \alpha_K \mathcal{R}_K + \alpha_Z \mathcal{R}_Z$",
        ha='center', va='center', fontsize=7.6, color=c_slate_dark)
ax.text(10.85, 4.15, "1. Operator Entropy Deficiency:\n" r"   $\Delta\mathcal{H}(\mathbf{K}) = \ln D - \mathcal{H}_{\mathrm{spectral}}(\mathbf{K})$" "\n"
                      "2. Itakura-Saito Energy Barrier:\n" r"   $\Phi(r) = r - 1 - \ln r \geq 0$" "\n"
                      "3. Representation Covariance Entropy:\n" r"   $\mathcal{R}_Z = \Delta\mathcal{H}(\mathbf{\Sigma}_Z)$ on $\frac{1}{2}(\mathbf{\Sigma}_{\mathrm{ctx}}+\mathbf{\Sigma}_{\mathrm{tgt}})$",
        ha='center', va='center', fontsize=6.8, color=c_slate_dark)

# Anti-collapse theorem box
draw_card(9.3, 0.8, 3.1, 1.25, facecolor='#FFFFFF', edgecolor='#DDD6FE', lw=0.8)
ax.text(10.85, 1.25, "Floored barrier (" r"$\epsilon=10^{-6}$" ") guarantees " r"$\mathbf{K}=\mathbf{0}$" "\nis strictly non-minimizing without divergence.\nOperator maintains full rank (" r"$\mathrm{RankMe} \approx 32$" ").",
        ha='center', va='center', fontsize=6.7, color=c_slate_muted)


# =========================================================================
# PANEL D: Saliency Gating & Tail Calibration
# =========================================================================
draw_card(12.8, 0.5, 2.8, 5.65, facecolor='#FFFFFF', edgecolor=c_slate_bdr, lw=1.0)
ax.text(14.2, 5.85, "(D) Gating & Tail Calibration", ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_tail_txt)

# Arrow from discrepancy to saliency
draw_arrow(9.98, 2.35, 13.0, 4.6, color=c_slate_dark, lw=1.4)

# Coordinate Saliency Gate
draw_card(13.0, 4.1, 2.4, 1.15, "Coordinate Saliency", r"$\mathbf{s} = \text{Softmax}(\mathbf{z}/\tau) \odot \mathbf{z}$", facecolor='#FFFFFF', edgecolor='#EA580C')
ax.text(14.2, 4.30, "Weights informative coordinates", ha='center', va='center', fontsize=6.8, color=c_slate_muted)

draw_arrow(14.2, 4.1, 14.2, 3.4, color='#EA580C', lw=1.3)

# Bilateral Anomaly Score
draw_card(13.0, 2.5, 2.4, 0.9, "Anomaly Score $S_t$", r"$S_t = (\mathbf{e}^\top \boldsymbol{\Sigma}_{\text{res}}^{-1} \mathbf{e})^{1/2} + \lambda_{\text{leak}}$", facecolor='#FFFFFF', edgecolor='#EA580C')

draw_arrow(14.2, 2.5, 14.2, 1.8, color='#EA580C', lw=1.3)

# Streaming SPOT EVT Calibration
draw_card(13.0, 0.8, 2.4, 1.0, "Streaming SPOT (EVT)", r"$\tau_q = u + \frac{\hat{\sigma}}{\hat{\xi}}\left[\left(\frac{q N_u}{N}\right)^{-\hat{\xi}} - 1\right]$", facecolor=c_tail_bg, edgecolor=c_tail_bdr, lw=1.2)
ax.text(14.2, 1.00, "Strict Causal Held-Out SPOT", ha='center', va='center', fontsize=6.8, color=c_tail_txt)


plt.tight_layout()
plt.savefig(OUT_FIG_DIR / "opentropy_jepa_architecture.png", dpi=300)
plt.savefig(OUT_FIG_DIR / "opentropy_jepa_architecture.pdf")
plt.close()
print(f"[OK] Generated {OUT_FIG_DIR / 'opentropy_jepa_architecture.png'}")
print(f"[OK] Generated {OUT_FIG_DIR / 'opentropy_jepa_architecture.pdf'}")
