#!/usr/bin/env python3
"""
Generate Publication-Grade IEEE / NeurIPS Architectural Overview Diagram (Figure 1)
-------------------------------------------------------------------------------------
Depicts:
(A) The Dilemma: Observation Drowning vs. Latent Representation Prediction
(B) Unified Self-Supervised JEPA Framework (EMA momentum encoder & stop-gradient)
(C) The Flagship Physics-Informed Inductive Invariant Heads (Koopman, Reynolds, Potential Flow)
(D) Pure Inductive Tail Calibration (Kurtosis-Adaptive EVT & Streaming SPOT)
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

fig, ax = plt.subplots(figsize=(16.0, 7.4), dpi=300)
ax.set_xlim(0, 16.0)
ax.set_ylim(0, 7.4)
ax.axis('off')

# Modern Academic Palette
c_dark      = '#0F172A'  # Slate-900
c_muted     = '#475569'  # Slate-600
c_blue_bg   = '#EFF6FF'  # Blue-50
c_blue_bdr  = '#2563EB'  # Blue-600
c_blue_txt  = '#1E3A8A'  # Blue-900
c_amber_bg  = '#FFFBEB'  # Amber-50
c_amber_bdr = '#D97706'  # Amber-600
c_amber_txt = '#92400E'  # Amber-900
c_green_bg  = '#F0FDF4'  # Emerald-50
c_green_bdr = '#059669'  # Emerald-600
c_green_txt = '#065F46'  # Emerald-900
c_purple_bg = '#FAF5FF'  # Purple-50
c_purple_bdr= '#9333EA'  # Purple-600
c_purple_txt= '#581C87'  # Purple-900
c_red_bg    = '#FEF2F2'  # Rose-50
c_red_bdr   = '#E11D48'  # Rose-600
c_red_txt   = '#9F1239'  # Rose-900


def draw_card(x, y, w, h, title="", subtitle="", facecolor='#FFFFFF', edgecolor='#CBD5E1', lw=1.0, rx=0.15):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={rx}",
                                facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=2)
    ax.add_patch(box)
    if title and subtitle:
        ax.text(x + w / 2, y + h * 0.70, title, ha='center', va='center', fontsize=8.8, fontweight='bold', color=c_dark, zorder=3)
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha='center', va='center', fontsize=7.5, color=c_muted, zorder=3)
    elif title:
        ax.text(x + w / 2, y + h * 0.50, title, ha='center', va='center', fontsize=8.8, fontweight='bold', color=c_dark, zorder=3)
    return box


def draw_arrow(x1, y1, x2, y2, color='#334155', lw=1.2, ls='-', style="-|>", rad=0.0):
    conn = f"arc3,rad={rad}" if rad != 0.0 else "arc3,rad=0"
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, lw=lw, color=color, linestyle=ls,
                                mutation_scale=10, shrinkA=0, shrinkB=0, connectionstyle=conn), zorder=5)


# Canvas Container
main_box = patches.FancyBboxPatch((0.2, 0.2), 15.6, 7.0, boxstyle="round,pad=0.0,rounding_size=0.25",
                                 facecolor='#FAFAFA', edgecolor='#E2E8F0', linewidth=1.2, zorder=1)
ax.add_patch(main_box)

# Title Banner
ax.text(0.5, 6.90, "Physics-Informed Joint-Embedding Predictive Architecture (JEPA) Triad",
        fontsize=13.0, fontweight='bold', color=c_dark, ha='left', va='center')
ax.text(0.5, 6.58, "Representation-space latent predictive dynamics constrained by fundamental physical conservation invariants",
        fontsize=9.0, color=c_muted, ha='left', va='center')


# =========================================================================
# PANEL A: Observation Drowning vs. Latent JEPA
# =========================================================================
draw_card(0.5, 0.5, 3.8, 5.85, facecolor='#FFFFFF', edgecolor='#CBD5E1', lw=1.1)
ax.text(2.4, 6.05, "(A) Observation Drowning vs. JEPA", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_blue_txt)

# Reconstructive Baselines Box (Warning)
draw_card(0.7, 3.4, 3.4, 2.35, facecolor=c_red_bg, edgecolor=c_red_bdr, lw=0.9)
ax.text(2.4, 5.45, "Observation-Space Baselines", ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_red_txt)
ax.text(2.4, 5.15, "TimesNet / TranAD / AnomalyTransf.", ha='center', va='center', fontsize=7.5, color='#881337')
ax.text(2.4, 4.75, r"$\min_\theta \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2$ in raw sensor domain", ha='center', va='center', fontsize=8.0, color=c_dark)
ax.text(2.4, 4.15, "• Forces networks to fit high-frequency\n  stochastic sensor noise & benign jitter.\n• High reconstruction residual $\\neq$ true anomaly.\n• Generates $>1{,}000{,}000$ false alarms in telemetry.",
        ha='center', va='center', fontsize=7.2, color='#881337')

# Proposed Latent JEPA Box (Success)
draw_card(0.7, 0.8, 3.4, 2.35, facecolor=c_green_bg, edgecolor=c_green_bdr, lw=0.9)
ax.text(2.4, 2.85, "Proposed Latent JEPA Triad", ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_green_txt)
ax.text(2.4, 2.55, "Operator Entropy / Reynolds / Potential Flow", ha='center', va='center', fontsize=7.5, color='#064E3B')
ax.text(2.4, 2.15, r"$\min_{\theta, \psi} \mathcal{D}(T_\psi(\mathbf{z}_{\mathrm{ctx}}), \mathbf{z}_{\mathrm{tgt}})$ in $\mathbb{R}^D$", ha='center', va='center', fontsize=8.0, color=c_dark)
ax.text(2.4, 1.55, "• Predicts abstract invariant dynamical state.\n• Discards pixel/point stochastic noise.\n• Immune to representation collapse via\n  non-contrastive physical conservation heads.",
        ha='center', va='center', fontsize=7.2, color='#064E3B')


# =========================================================================
# PANEL B: Unified Predictive Architecture
# =========================================================================
draw_card(4.6, 0.5, 6.2, 5.85, facecolor='#FFFFFF', edgecolor='#CBD5E1', lw=1.1)
ax.text(7.7, 6.05, "(B) Unified Predictive Latent Architecture", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_blue_txt)

# Context Pipeline (Top)
draw_card(4.9, 4.5, 1.6, 1.1, "Context Window", r"$\mathbf{x}_{\mathrm{ctx}} \in \mathbb{R}^{C \times K}$", facecolor=c_blue_bg, edgecolor=c_blue_bdr)
draw_card(6.9, 4.5, 1.8, 1.1, "Context Encoder $E_\\theta$", "Dyadic TCN + Saliency", facecolor=c_blue_bg, edgecolor=c_blue_bdr)
draw_arrow(6.5, 5.05, 6.9, 5.05, color=c_blue_bdr, lw=1.5)

draw_card(9.1, 4.5, 1.4, 1.1, r"$\mathbf{z}_{\mathrm{ctx}} \in \mathbb{R}^D$", "Latent State", facecolor='#FFFFFF', edgecolor=c_blue_bdr)
draw_arrow(8.7, 5.05, 9.1, 5.05, color=c_blue_bdr, lw=1.5)

# Target Pipeline (Bottom)
draw_card(4.9, 0.9, 1.6, 1.1, "Target Window", r"$\mathbf{x}_{\mathrm{tgt}} \in \mathbb{R}^{S \times K}$", facecolor=c_amber_bg, edgecolor=c_amber_bdr)
draw_card(6.9, 0.9, 1.8, 1.1, "Target Encoder $E_\\phi$", "Momentum Replica (EMA)", facecolor=c_amber_bg, edgecolor=c_amber_bdr)
draw_arrow(6.5, 1.45, 6.9, 1.45, color=c_amber_bdr, lw=1.5)

draw_card(9.1, 0.9, 1.4, 1.1, r"$\mathbf{z}_{\mathrm{tgt}} \in \mathbb{R}^D$", r"$\perp$ (Stop-Grad)", facecolor='#FFFFFF', edgecolor=c_amber_bdr)
draw_arrow(8.7, 1.45, 9.1, 1.45, color=c_amber_bdr, lw=1.5)

# EMA connection on the far left without intersecting anything
draw_arrow(7.8, 4.5, 7.8, 2.0, color='#9333EA', lw=1.4, ls='--')
ax.text(7.7, 3.25, r"$\phi \leftarrow m\phi + (1-m)\theta$" "\n(EMA Schedule)", fontsize=7.2, color='#7E22CE', ha='right', va='center')

# Latent Predictor (Middle Right)
draw_card(9.1, 2.7, 1.4, 1.1, "Predictor $T_\\psi$", "Latent ResMLP", facecolor='#F8FAFC', edgecolor='#475569')
draw_arrow(9.8, 4.5, 9.8, 3.8, color=c_blue_bdr, lw=1.5)

# Discrepancy comparator node
circ = patches.Circle((10.45, 2.15), 0.22, facecolor='#FFFFFF', edgecolor='#0F172A', lw=1.2, zorder=4)
ax.add_patch(circ)
ax.text(10.45, 2.15, r"$\mathcal{D}$", ha='center', va='center', fontsize=9.0, fontweight='bold', color='#0F172A', zorder=5)

draw_arrow(9.8, 2.7, 10.25, 2.25, color='#475569', lw=1.2)
ax.text(10.2, 2.65, r"$\hat{\mathbf{z}}_{\mathrm{tgt}}$", fontsize=7.5, color='#334155')

draw_arrow(9.8, 2.0, 10.25, 2.15, color=c_amber_bdr, lw=1.2)
ax.text(9.9, 1.75, r"$\mathbf{z}_{\mathrm{tgt}}$", fontsize=7.5, color=c_amber_txt)


# =========================================================================
# PANEL C: The Three Physics Inductive Heads & Tail Calibration
# =========================================================================
draw_card(11.1, 0.5, 4.4, 5.85, facecolor='#FFFFFF', edgecolor='#CBD5E1', lw=1.1)
ax.text(13.3, 6.05, "(C) Physics Heads & Tail Calibration", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_blue_txt)

# Head 1: Operator Entropy
draw_card(11.3, 4.6, 4.0, 1.15, facecolor=c_blue_bg, edgecolor=c_blue_bdr)
ax.text(13.3, 5.45, "1. Operator-Entropy JEPA", ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_blue_txt)
ax.text(13.3, 5.15, "Koopman Dynamic Spectral Entropy (Unitary Spectrum)", ha='center', va='center', fontsize=7.2, color='#1E40AF')
ax.text(13.3, 4.82, r"$\mathcal{K}_{\mathbf{z}} \mathbf{z}_t \approx \mathbf{z}_{t+1}, \quad \mathcal{H}_K = -\sum_i \lambda_i \ln \lambda_i$",
        ha='center', va='center', fontsize=7.8, color=c_dark)

# Head 2: Reynolds Stress
draw_card(11.3, 3.25, 4.0, 1.15, facecolor=c_green_bg, edgecolor=c_green_bdr)
ax.text(13.3, 4.10, "2. Reynolds-Stress JEPA", ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_green_txt)
ax.text(13.3, 3.80, "Turbulent Momentum Transfer & Shear Divergence", ha='center', va='center', fontsize=7.2, color='#047857')
ax.text(13.3, 3.47, r"$\mathbf{u} = \bar{\mathbf{u}} + \mathbf{u}', \quad \mathbf{R} = \overline{\mathbf{u}' \otimes \mathbf{u}'} \quad (\nabla \cdot \mathbf{R} \neq \mathbf{0})$",
        ha='center', va='center', fontsize=7.8, color=c_dark)

# Head 3: Potential Flow
draw_card(11.3, 1.9, 4.0, 1.15, facecolor=c_purple_bg, edgecolor=c_purple_bdr)
ax.text(13.3, 2.75, "3. Potential-Flow JEPA", ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_purple_txt)
ax.text(13.3, 2.45, "Conservative Irrotational Potential Field", ha='center', va='center', fontsize=7.2, color='#6D28D9')
ax.text(13.3, 2.12, r"$\mathbf{v} = \nabla \Phi, \quad \nabla \times \mathbf{v} = \mathbf{0} \quad (\text{Vorticity-Free Conservation})$",
        ha='center', va='center', fontsize=7.8, color=c_dark)

# Tail Calibration Box (Bottom)
draw_card(11.3, 0.7, 4.0, 1.0, facecolor='#FFFBEB', edgecolor='#D97706')
ax.text(13.3, 1.42, "Pure Inductive EVT & Streaming SPOT", ha='center', va='center', fontsize=8.2, fontweight='bold', color='#B45309')
ax.text(13.3, 1.15, "Kurtosis-adaptive Generalized Pareto distribution tail fitting", ha='center', va='center', fontsize=7.2, color='#78350F')
ax.text(13.3, 0.88, r"$\tau(t) = \text{Med}(t) + (\tau_0 - \text{Med}_0) \cdot \frac{\text{IQR}(t)}{\text{IQR}_0}$",
        ha='center', va='center', fontsize=7.6, color=c_dark)

# Output Arrow from Discrepancy to Physics Invariants
draw_arrow(10.67, 2.15, 11.3, 2.15, color='#0F172A', lw=1.5)

plt.tight_layout()
plt.savefig(OUT_FIG_DIR / "physics_jepa_triad_architecture.png", dpi=300)
plt.savefig(OUT_FIG_DIR / "physics_jepa_triad_architecture.pdf")
plt.close()
print(f"[OK] Generated {OUT_FIG_DIR / 'physics_jepa_triad_architecture.png'}")
