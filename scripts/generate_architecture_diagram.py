from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]

# Set true IEEE publication typography (Times / Computer Modern serif style)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 9.0
plt.rcParams['axes.linewidth'] = 0.8

fig, ax = plt.subplots(figsize=(15.0, 6.2), dpi=300)
ax.set_xlim(0, 15.0)
ax.set_ylim(0, 6.2)
ax.axis('off')

# IEEE Academic Color Palette (Muted slate, classic academic navy/amber/forest, crisp black borders)
c_box_bg_ctx = '#F1F5F9'       # Slate-50
c_box_bg_tgt = '#FFFBEB'       # Amber-50
c_box_bg_pred = '#F0FDF4'      # Emerald-50
c_box_bg_loss = '#FDF4FF'      # Fuchsia-50
c_box_bg_infer = '#F8FAFC'     # Slate-50

c_line_dark = '#0F172A'        # Slate-900
c_blue = '#1E40AF'             # Blue-800
c_amber = '#B45309'            # Amber-700
c_green = '#065F46'            # Emerald-800
c_purple = '#6B21A8'           # Purple-800
c_gray = '#64748B'             # Slate-500

# Helper: Draw IEEE Standard Rectangular Functional Block
def draw_block(x, y, w, h, title, subtitle="", bg_col='#FFFFFF', border_col='#0F172A', lw=1.1, ax=ax):
    rect = patches.Rectangle((x, y), w, h, facecolor=bg_col, edgecolor=border_col, linewidth=lw, zorder=2)
    ax.add_patch(rect)
    if subtitle:
        ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', fontsize=9.0, fontweight='bold', color=c_line_dark, zorder=3)
        ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center', fontsize=7.5, color='#334155', zorder=3)
    else:
        ax.text(x + w/2, y + h/2, title, ha='center', va='center', fontsize=8.8, fontweight='bold', color=c_line_dark, zorder=3)
    return rect

# Helper: Draw IEEE Sum/Diff/Product Node
def draw_node(x, y, r=0.22, symbol=r"\oplus", bg='#FFFFFF', border='#0F172A', ax=ax):
    circ = patches.Circle((x, y), r, facecolor=bg, edgecolor=border, lw=1.1, zorder=4)
    ax.add_patch(circ)
    ax.text(x, y, f"${symbol}$", ha='center', va='center', fontsize=11.0, fontweight='bold', color=border, zorder=5)

# Helper: Draw Arrow with standardized IEEE styling
def draw_arrow(x1, y1, x2, y2, color=c_line_dark, lw=1.2, ls='-', connectionstyle="arc3,rad=0", ax=ax):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, linestyle=ls,
                                mutation_scale=11, connectionstyle=connectionstyle), zorder=5)

# Helper: Draw Orthogonal (Manhattan) Arrow
def draw_manhattan_arrow(points, color=c_line_dark, lw=1.2, ls='-', ax=ax):
    # points is a list of (x, y) coordinates: [(x1, y1), (x2, y2), ..., (xn, yn)]
    for i in range(len(points) - 2):
        ax.plot([points[i][0], points[i+1][0]], [points[i][1], points[i+1][1]], color=color, lw=lw, linestyle=ls, zorder=4)
    draw_arrow(points[-2][0], points[-2][1], points[-1][0], points[-1][1], color=color, lw=lw, ls=ls, ax=ax)

# ==============================================================================
# SECTION A: FLOW-JEPA TRAINING PIPELINE (Top & Middle Region: y in [1.8, 6.0])
# ==============================================================================

# Outer Boundary for Training Stage
train_box = patches.Rectangle((0.2, 1.85), 14.6, 4.2, facecolor='#FAFAFA', edgecolor='#CBD5E1', lw=0.9, linestyle='--', zorder=0)
ax.add_patch(train_box)
ax.text(0.4, 5.82, "(a) Flow-JEPA Self-Supervised Training Architecture", fontsize=10.0, fontweight='bold', color=c_line_dark)

# 1. Inputs
draw_block(0.4, 4.4, 1.8, 1.0, "Context Window", r"$\mathbf{x}_{\mathrm{ctx}} \in \mathbb{R}^{C \times K}$", bg_col=c_box_bg_ctx, border_col=c_blue)
draw_block(0.4, 2.3, 1.8, 1.0, "Target Window", r"$\mathbf{x}_{\mathrm{tgt}} \in \mathbb{R}^{S \times K}$", bg_col=c_box_bg_tgt, border_col=c_amber)

# Arrows from Inputs to Encoders
draw_arrow(2.2, 4.9, 2.7, 4.9, color=c_blue)
draw_arrow(2.2, 2.8, 2.7, 2.8, color=c_amber)

# 2. Dual Encoders
draw_block(2.7, 4.4, 2.3, 1.0, r"Context Encoder $E_\theta$", "Dilated Causal TCN / Patch Transf.", bg_col=c_box_bg_ctx, border_col=c_blue)
draw_block(2.7, 2.3, 2.3, 1.0, r"Target Encoder $E_\phi$", "Momentum Replica (EMA)", bg_col=c_box_bg_tgt, border_col=c_amber)

# EMA Update Arrow
draw_arrow(3.85, 4.4, 3.85, 3.3, color=c_purple, lw=1.2, ls='--', connectionstyle="arc3,rad=-0.35")
ax.text(3.18, 3.85, r"EMA: $\phi \leftarrow m\phi + (1-m)\theta$", fontsize=7.0, color=c_purple, ha='center')

# Arrows from Encoders to Latent Vectors
draw_arrow(5.0, 4.9, 5.5, 4.9, color=c_blue)
draw_arrow(5.0, 2.8, 5.5, 2.8, color=c_amber)

# Latent Representation Nodes
ax.text(5.95, 4.9, r"$\mathbf{z}_{\mathrm{ctx}} \in \mathbb{R}^D$", fontsize=8.5, fontweight='bold', color=c_blue,
        ha='center', va='center', bbox=dict(boxstyle='square,pad=0.25', facecolor='#EFF6FF', edgecolor=c_blue, lw=0.9), zorder=4)

ax.text(5.95, 2.8, r"$\mathbf{z}_{\mathrm{tgt}} \in \mathbb{R}^D$", fontsize=8.5, fontweight='bold', color=c_amber,
        ha='center', va='center', bbox=dict(boxstyle='square,pad=0.25', facecolor='#FFFBEB', edgecolor=c_amber, lw=0.9), zorder=4)
ax.text(5.95, 2.35, r"$\bot$ (stop-grad)", fontsize=7.2, color='#DC2626', ha='center')

# 3. Base Noise & Probability Path Generator
draw_block(5.25, 3.55, 1.4, 0.65, r"$\mathbf{z}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$", "Base Prior", bg_col='#F8FAFC', border_col=c_gray)
draw_block(7.2, 3.2, 2.2, 1.2, "OT Probability Path", r"$\mathbf{z}_t = (1-t)\mathbf{z}_0 + t\mathbf{z}_{\mathrm{tgt}}$" + "\n" + r"$\mathbf{u}_t = \mathbf{z}_{\mathrm{tgt}} - \mathbf{z}_0$", bg_col='#F0FDFA', border_col='#0D9488')

draw_arrow(6.65, 3.88, 7.2, 3.88, color=c_gray)
draw_arrow(6.4, 2.8, 7.2, 3.45, color=c_amber)

# Time input
ax.text(8.3, 4.7, r"$t \sim \mathcal{U}(0, 1)$", fontsize=8.0, color=c_line_dark, ha='center')
draw_arrow(8.3, 4.55, 8.3, 4.4, color=c_line_dark)

# 4. Continuous Flow Predictor Network
draw_block(10.1, 3.75, 2.3, 1.7, r"Flow Predictor $v_\psi$",
           "Time Embedding $\\mathbf{e}(t)$\nAdaLN / Cross-Attention\nDeep Residual MLP", bg_col=c_box_bg_pred, border_col=c_green, lw=1.3)

# Routing inputs into Flow Predictor
# Context conditioning: neat horizontal line above
draw_manhattan_arrow([(6.4, 4.9), (6.7, 5.2), (10.1, 5.2)], color=c_blue)
ax.text(8.0, 5.35, r"$\mathbf{z}_{\mathrm{ctx}}$ (condition)", fontsize=7.5, color=c_blue, ha='center')

# Time t into predictor
draw_manhattan_arrow([(8.75, 4.7), (10.1, 4.7)], color=c_line_dark)

# z_t into predictor
draw_arrow(9.4, 3.95, 10.1, 3.95, color='#0D9488')
ax.text(9.75, 4.12, r"$\mathbf{z}_t$", fontsize=8.0, color='#0D9488', ha='center')


# Predicted velocity vector
draw_arrow(12.4, 4.6, 13.0, 4.6, color=c_green)
ax.text(12.7, 4.85, r"$\hat{v}_\psi \in \mathbb{R}^D$", fontsize=8.5, fontweight='bold', color=c_green, ha='center')

# 5. Losses
draw_node(13.2, 4.6, r=0.20, symbol="-", bg='#FFFFFF', border=c_green)
# Target velocity to subtraction node (orthogonal Manhattan path)
draw_manhattan_arrow([(9.4, 3.45), (13.2, 3.45), (13.2, 4.4)], color='#0D9488')
ax.text(11.3, 3.28, r"Target Velocity $\mathbf{u}_t = \mathbf{z}_{\mathrm{tgt}} - \mathbf{z}_0$", fontsize=7.2, color='#0D9488', ha='center')

draw_arrow(13.4, 4.6, 13.8, 4.6, color=c_green)
ax.text(14.2, 4.6, r"$\mathcal{L}_{\mathrm{CFM}}$", fontsize=9.0, fontweight='bold', color=c_green,
        ha='center', va='center', bbox=dict(boxstyle='square,pad=0.2', facecolor=c_box_bg_pred, edgecolor=c_green, lw=0.9), zorder=4)

# Manifold Regularization Box
draw_block(10.1, 2.1, 2.3, 1.0, "Manifold Regularization",
           r"$\mathcal{L}_{\mathrm{var}}$: Variance Hinge" + "\n" + r"$\mathcal{L}_{\mathrm{cov}}$: Operator Entropy",
           bg_col=c_box_bg_loss, border_col=c_purple)

draw_arrow(6.4, 2.8, 10.1, 2.6, color=c_amber)
draw_arrow(12.4, 2.6, 13.8, 2.6, color=c_purple)
ax.text(14.2, 2.6, r"$\mathcal{L}_{\mathrm{reg}}$", fontsize=9.0, fontweight='bold', color=c_purple,
        ha='center', va='center', bbox=dict(boxstyle='square,pad=0.2', facecolor=c_box_bg_loss, edgecolor=c_purple, lw=0.9), zorder=4)

# Total Loss summation
draw_node(14.2, 3.6, r=0.22, symbol=r"\Sigma", bg='#FFFFFF', border=c_line_dark)
draw_arrow(14.2, 4.35, 14.2, 3.82, color=c_green)
draw_arrow(14.2, 2.85, 14.2, 3.38, color=c_purple)
draw_arrow(14.42, 3.6, 14.65, 3.6, color=c_line_dark)
ax.text(14.7, 3.85, r"$\mathcal{L}_{\mathrm{total}}$", fontsize=8.5, fontweight='bold', color=c_line_dark, ha='left')


# ==============================================================================
# SECTION B: DETERMINISTIC ZERO-VARIANCE INFERENCE (Bottom Region: y in [0.2, 1.65])
# ==============================================================================

infer_box = patches.Rectangle((0.2, 0.15), 14.6, 1.55, facecolor='#FFFFFF', edgecolor='#CBD5E1', lw=0.9, linestyle='--', zorder=0)
ax.add_patch(infer_box)
ax.text(0.4, 1.45, "(b) Deterministic Zero-Variance Inference & Extreme Value Theory (EVT) Tail Calibration", fontsize=9.5, fontweight='bold', color=c_line_dark)

# Step 1: Deterministic Midpoint Ray
draw_block(0.4, 0.3, 3.4, 0.95, "1. OT Midpoint Ray ($t=0.5$)",
           r"$\mathbf{z}_0 = \mathbf{0} \rightarrow \mathbf{z}_{0.5} = 0.5 \mathbf{z}_{\mathrm{tgt}}$" + "\n" +
           r"$\hat{\mathbf{v}}_{\mathrm{mid}} = v_\psi(0.5\mathbf{z}_{\mathrm{tgt}}, 0.5, \mathbf{z}_{\mathrm{ctx}})$",
           bg_col=c_box_bg_infer, border_col=c_amber)

draw_arrow(3.8, 0.77, 4.3, 0.77, color=c_line_dark)

# Step 2: Velocity Residual
draw_block(4.3, 0.3, 2.7, 0.95, "2. Velocity Residual",
           r"$\mathbf{e} = \hat{\mathbf{v}}_{\mathrm{mid}} - \mathbf{z}_{\mathrm{tgt}}$" + "\n" +
           r"$\mathbf{u}_{\mathrm{true}} = \mathbf{z}_{\mathrm{tgt}} - \mathbf{0} = \mathbf{z}_{\mathrm{tgt}}$",
           bg_col=c_box_bg_infer, border_col=c_green)

draw_arrow(7.0, 0.77, 7.5, 0.77, color=c_line_dark)

# Step 3: Precision Whitening
draw_block(7.5, 0.3, 3.4, 0.95, "3. Mahalanobis Whitening",
           r"$\mathbf{\Sigma}^{-1} = (\mathrm{Cov}(\mathbf{e}_{\mathrm{cal}}) + \lambda \mathbf{I})^{-1}$" + "\n" +
           r"$D_M = \sqrt{ \mathbf{e}^T \mathbf{\Sigma}^{-1} \mathbf{e} }$",
           bg_col=c_box_bg_infer, border_col=c_blue)

draw_arrow(10.9, 0.77, 11.4, 0.77, color=c_line_dark)

# Step 4: EVT / SPOT Decision
draw_block(11.4, 0.3, 3.2, 0.95, "4. EVT / SPOT Threshold",
           r"$\mathrm{GPD}(\hat{\xi}, \hat{\sigma}) \rightarrow \tau_{\mathrm{EVT}}$" + "\n" +
           r"Decision: $\hat{y} = \mathbb{I}(D_M \geq \tau_{\mathrm{EVT}})$",
           bg_col='#FEF2F2', border_col='#DC2626')


# Save figures
out_png = ROOT / "paper" / "NCAD_CS" / "figures" / "flow_jepa_architecture.png"
out_pdf = ROOT / "paper" / "NCAD_CS" / "figures" / "flow_jepa_architecture.pdf"

plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
print(f"Generated classic IEEE Transactions block diagram:\n- PNG: {out_png}\n- PDF: {out_pdf}")
