import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "NCAD_CS" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Set IEEE publication typography (Times / Computer Modern serif style)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 9.0
plt.rcParams['axes.linewidth'] = 0.8

# Color Palette (IEEE Standard Muted Academic Palette)
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
c_red = '#DC2626'              # Red-600

def draw_block(ax, x, y, w, h, title, subtitle="", bg_col='#FFFFFF', border_col='#0F172A', lw=1.1):
    rect = patches.Rectangle((x, y), w, h, facecolor=bg_col, edgecolor=border_col, linewidth=lw, zorder=2)
    ax.add_patch(rect)
    if subtitle:
        ax.text(x + w/2, y + h*0.64, title, ha='center', va='center', fontsize=9.0, fontweight='bold', color=c_line_dark, zorder=3)
        ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center', fontsize=7.6, color='#334155', zorder=3)
    else:
        ax.text(x + w/2, y + h/2, title, ha='center', va='center', fontsize=8.8, fontweight='bold', color=c_line_dark, zorder=3)
    return rect

def draw_node(ax, x, y, r=0.22, symbol=r"\oplus", bg='#FFFFFF', border='#0F172A'):
    circ = patches.Circle((x, y), r, facecolor=bg, edgecolor=border, lw=1.1, zorder=4)
    ax.add_patch(circ)
    ax.text(x, y, f"${symbol}$", ha='center', va='center', fontsize=11.0, fontweight='bold', color=border, zorder=5)

def draw_arrow(ax, x1, y1, x2, y2, color=c_line_dark, lw=1.2, ls='-', connectionstyle="arc3,rad=0"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, linestyle=ls,
                                mutation_scale=11, connectionstyle=connectionstyle), zorder=5)

def draw_manhattan_arrow(ax, points, color=c_line_dark, lw=1.2, ls='-'):
    for i in range(len(points) - 2):
        ax.plot([points[i][0], points[i+1][0]], [points[i][1], points[i+1][1]], color=color, lw=lw, linestyle=ls, zorder=4)
    draw_arrow(ax, points[-2][0], points[-2][1], points[-1][0], points[-1][1], color=color, lw=lw, ls=ls)


# ==============================================================================
# DIAGRAM 1: FLOW-JEPA SELF-SUPERVISED TRAINING ARCHITECTURE
# ==============================================================================
def generate_training_diagram():
    fig, ax = plt.subplots(figsize=(14.5, 4.8), dpi=300)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 4.8)
    ax.axis('off')

    # Border framing
    frame = patches.Rectangle((0.2, 0.15), 14.1, 4.5, facecolor='#FAFAFA', edgecolor='#CBD5E1', lw=1.0, zorder=0)
    ax.add_patch(frame)
    ax.text(0.4, 4.42, "Flow-JEPA Self-Supervised Training Architecture", fontsize=10.5, fontweight='bold', color=c_line_dark)

    # 1. Inputs (x in [0.5, 2.3])
    draw_block(ax, 0.5, 3.0, 1.8, 1.1, "Context Window", r"$\mathbf{x}_{\mathrm{ctx}} \in \mathbb{R}^{C \times K}$", bg_col=c_box_bg_ctx, border_col=c_blue)
    draw_block(ax, 0.5, 0.45, 1.8, 1.1, "Target Window", r"$\mathbf{x}_{\mathrm{tgt}} \in \mathbb{R}^{S \times K}$", bg_col=c_box_bg_tgt, border_col=c_amber)

    # Arrows to Encoders
    draw_arrow(ax, 2.3, 3.55, 2.8, 3.55, color=c_blue)
    draw_arrow(ax, 2.3, 1.0, 2.8, 1.0, color=c_amber)

    # 2. Dual Encoders (x in [2.8, 5.2])
    draw_block(ax, 2.8, 3.0, 2.4, 1.1, r"Context Encoder $E_\theta$", "Dilated Causal TCN / Transf.\n(Active Gradients $\\nabla_\\theta$)", bg_col=c_box_bg_ctx, border_col=c_blue)
    draw_block(ax, 2.8, 0.45, 2.4, 1.1, r"Target Encoder $E_\phi$", "Momentum Replica (EMA)\n($\\phi \\leftarrow m\\phi + (1-m)\\theta$)", bg_col=c_box_bg_tgt, border_col=c_amber)

    # EMA Parameter bridge
    draw_arrow(ax, 4.0, 3.0, 4.0, 1.55, color=c_purple, lw=1.3, ls='--', connectionstyle="arc3,rad=-0.32")
    ax.text(3.35, 2.25, r"EMA Update", fontsize=7.2, fontweight='bold', color=c_purple, ha='center')

    # Arrows from Encoders to Latents
    draw_arrow(ax, 5.2, 3.55, 5.7, 3.55, color=c_blue)
    draw_arrow(ax, 5.2, 1.0, 5.7, 1.0, color=c_amber)

    # Latents z_ctx and z_tgt (x around 6.15)
    ax.text(6.15, 3.55, r"$\mathbf{z}_{\mathrm{ctx}} \in \mathbb{R}^D$", fontsize=9.0, fontweight='bold', color=c_blue,
            ha='center', va='center', bbox=dict(boxstyle='square,pad=0.28', facecolor='#EFF6FF', edgecolor=c_blue, lw=1.0), zorder=4)

    ax.text(6.15, 1.0, r"$\mathbf{z}_{\mathrm{tgt}} \in \mathbb{R}^D$", fontsize=9.0, fontweight='bold', color=c_amber,
            ha='center', va='center', bbox=dict(boxstyle='square,pad=0.28', facecolor='#FFFBEB', edgecolor=c_amber, lw=1.0), zorder=4)
    ax.text(6.15, 0.50, r"$\bot$ (stop-grad)", fontsize=7.5, fontweight='bold', color=c_red, ha='center')

    # 3. Base Noise & Optimal Transport Probability Path (x in [5.4, 9.4])
    draw_block(ax, 5.4, 2.0, 1.5, 0.75, r"$\mathbf{z}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$", "Base Prior", bg_col='#F8FAFC', border_col=c_gray)
    draw_block(ax, 7.3, 1.55, 2.2, 1.35, "OT Probability Path",
               r"$\mathbf{z}_t = (1-t)\mathbf{z}_0 + t\mathbf{z}_{\mathrm{tgt}}$" + "\n" +
               r"$\mathbf{u}_t = \mathbf{z}_{\mathrm{tgt}} - \mathbf{z}_0$",
               bg_col='#F0FDFA', border_col='#0D9488', lw=1.2)

    draw_arrow(ax, 6.9, 2.35, 7.3, 2.35, color=c_gray)
    draw_arrow(ax, 6.6, 1.0, 7.3, 1.7, color=c_amber)

    # Time sampling t ~ U(0, 1)
    ax.text(8.4, 3.25, r"$t \sim \mathcal{U}(0, 1)$", fontsize=8.2, fontweight='bold', color=c_line_dark, ha='center')
    draw_arrow(ax, 8.4, 3.08, 8.4, 2.9, color=c_line_dark)

    # 4. Continuous Flow Predictor Network (x in [10.0, 12.3])
    draw_block(ax, 10.0, 2.05, 2.3, 2.05, r"Flow Predictor $v_\psi$",
               "Sinusoidal Time $\\mathbf{e}(t)$\nAdaLN / Cross-Attention\nDeep Residual MLP",
               bg_col=c_box_bg_pred, border_col=c_green, lw=1.3)

    # Routing signals into Flow Predictor
    # Context conditioning z_ctx (clean top route)
    draw_manhattan_arrow(ax, [(6.6, 3.55), (7.0, 3.9), (10.0, 3.9)], color=c_blue, lw=1.3)
    ax.text(8.4, 4.05, r"$\mathbf{z}_{\mathrm{ctx}}$ (Conditioning)", fontsize=7.6, color=c_blue, fontweight='bold', ha='center')

    # Time t into predictor
    draw_manhattan_arrow(ax, [(8.85, 3.25), (10.0, 3.25)], color=c_line_dark, lw=1.2)

    # z_t into predictor
    draw_arrow(ax, 9.5, 2.4, 10.0, 2.4, color='#0D9488', lw=1.3)
    ax.text(9.75, 2.56, r"$\mathbf{z}_t$", fontsize=8.2, fontweight='bold', color='#0D9488', ha='center')

    # Predicted Velocity v_psi
    draw_arrow(ax, 12.3, 3.05, 12.85, 3.05, color=c_green, lw=1.4)
    ax.text(12.58, 3.30, r"$\hat{v}_\psi \in \mathbb{R}^D$", fontsize=8.5, fontweight='bold', color=c_green, ha='center')

    # 5. Losses and Regularization (x in [12.8, 14.1])
    # Velocity subtraction node
    draw_node(ax, 13.05, 3.05, r=0.20, symbol="-", bg='#FFFFFF', border=c_green)
    # Target velocity u_t routed along horizontal path at y=1.75
    draw_manhattan_arrow(ax, [(9.5, 1.75), (13.05, 1.75), (13.05, 2.85)], color='#0D9488', lw=1.3)
    ax.text(11.2, 1.58, r"Target Velocity $\mathbf{u}_t = \mathbf{z}_{\mathrm{tgt}} - \mathbf{z}_0$", fontsize=7.2, color='#0D9488', ha='center')

    # CFM Loss box
    draw_arrow(ax, 13.25, 3.05, 13.65, 3.05, color=c_green, lw=1.3)
    ax.text(13.9, 3.05, r"$\mathcal{L}_{\mathrm{CFM}}$", fontsize=9.2, fontweight='bold', color=c_green,
            ha='center', va='center', bbox=dict(boxstyle='square,pad=0.22', facecolor=c_box_bg_pred, edgecolor=c_green, lw=1.0), zorder=4)

    # Manifold Regularization (from y=0.35 to y=1.40, completely clear of text at 1.58 and arrow at 1.75)
    draw_block(ax, 10.0, 0.35, 2.3, 1.05, "Manifold Regularization",
               r"$\mathcal{L}_{\mathrm{var}}$: Variance Hinge" + "\n" + r"$\mathcal{L}_{\mathrm{cov}}$: Operator Entropy",
               bg_col=c_box_bg_loss, border_col=c_purple, lw=1.2)

    draw_arrow(ax, 6.6, 1.0, 10.0, 0.88, color=c_amber, lw=1.2)
    draw_arrow(ax, 12.3, 0.88, 13.65, 0.88, color=c_purple, lw=1.3)
    ax.text(13.9, 0.88, r"$\mathcal{L}_{\mathrm{reg}}$", fontsize=9.2, fontweight='bold', color=c_purple,
            ha='center', va='center', bbox=dict(boxstyle='square,pad=0.22', facecolor=c_box_bg_loss, edgecolor=c_purple, lw=1.0), zorder=4)

    # Total Loss Summation node
    draw_node(ax, 13.9, 1.95, r=0.22, symbol=r"\Sigma", bg='#FFFFFF', border=c_line_dark)
    draw_arrow(ax, 13.9, 2.78, 13.9, 2.17, color=c_green, lw=1.3)
    draw_arrow(ax, 13.9, 1.15, 13.9, 1.73, color=c_purple, lw=1.3)
    draw_arrow(ax, 14.12, 1.95, 14.30, 1.95, color=c_line_dark, lw=1.3)
    ax.text(14.33, 2.15, r"$\mathcal{L}_{\mathrm{total}}$", fontsize=8.8, fontweight='bold', color=c_line_dark, ha='left')

    out_png = FIG_DIR / "flow_jepa_training.png"
    out_pdf = FIG_DIR / "flow_jepa_training.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"Generated Training Architecture Diagram:\n- {out_png}\n- {out_pdf}")


# ==============================================================================
# DIAGRAM 2: DETERMINISTIC INFERENCE & TAIL CALIBRATION PIPELINE
# ==============================================================================
def generate_inference_diagram():
    fig, ax = plt.subplots(figsize=(14.5, 4.2), dpi=300)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 4.2)
    ax.axis('off')

    # Border framing
    frame = patches.Rectangle((0.2, 0.15), 14.1, 3.85, facecolor='#FAFAFA', edgecolor='#CBD5E1', lw=1.0, zorder=0)
    ax.add_patch(frame)
    ax.text(0.4, 3.82, "Deterministic Zero-Variance Inference & Extreme Value Theory (EVT) Calibration", fontsize=10.5, fontweight='bold', color=c_line_dark)

    # Stage 1: Deterministic OT Midpoint Ray (x in [0.5, 3.6])
    s1_box = patches.Rectangle((0.5, 0.4), 3.1, 3.0, facecolor='#EFF6FF', edgecolor=c_blue, lw=1.3, zorder=2)
    ax.add_patch(s1_box)
    ax.text(2.05, 3.05, "1. Deterministic OT Midpoint ($t=0.5$)", ha='center', va='center', fontsize=9.0, fontweight='bold', color=c_line_dark, zorder=3)
    ax.text(2.05, 2.35, "$\\mathbf{z}_0 = \\mathbf{0} \\rightarrow$ Zero-Variance Ray\n" +
                        "$\\mathbf{z}_{0.5} = 0.5\\,\\mathbf{z}_{\\mathrm{tgt}}$ at $t=0.5$",
            ha='center', va='center', fontsize=7.8, color='#1E293B', zorder=3)
    ax.text(2.05, 1.45, "Velocity Evaluation:\n" +
                        "$\\hat{\\mathbf{v}}_{\\mathrm{mid}} = v_\\psi(0.5\\mathbf{z}_{\\mathrm{tgt}}, 0.5, \\mathbf{z}_{\\mathrm{ctx}})$",
            ha='center', va='center', fontsize=7.8, fontweight='bold', color=c_blue, zorder=3)
    ax.text(2.05, 0.75, "$\\Rightarrow$ Zero Monte Carlo noise", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#047857', zorder=3)

    draw_arrow(ax, 3.6, 1.9, 4.0, 1.9, color=c_line_dark, lw=1.5)

    # Stage 2: Velocity Residual Vector (x in [4.0, 6.9])
    s2_box = patches.Rectangle((4.0, 0.4), 2.9, 3.0, facecolor='#F0FDF4', edgecolor=c_green, lw=1.3, zorder=2)
    ax.add_patch(s2_box)
    ax.text(5.45, 3.05, "2. Dynamic Velocity Residual", ha='center', va='center', fontsize=9.0, fontweight='bold', color=c_line_dark, zorder=3)
    ax.text(5.45, 2.35, "Target Velocity Field:\n" +
                        "$\\mathbf{u}_{\\mathrm{true}} = \\mathbf{z}_{\\mathrm{tgt}} - \\mathbf{0} = \\mathbf{z}_{\\mathrm{tgt}}$",
            ha='center', va='center', fontsize=7.8, color='#1E293B', zorder=3)
    ax.text(5.45, 1.45, "Discrepancy Error Vector:\n" +
                        "$\\mathbf{e} = \\hat{\\mathbf{v}}_{\\mathrm{mid}} - \\mathbf{z}_{\\mathrm{tgt}} \\in \\mathbb{R}^D$",
            ha='center', va='center', fontsize=7.8, fontweight='bold', color=c_green, zorder=3)
    ax.text(5.45, 0.75, "$\\Rightarrow$ Physical dynamic deviation", ha='center', va='center', fontsize=7.5, color='#334155', zorder=3)

    draw_arrow(ax, 6.9, 1.9, 7.3, 1.9, color=c_line_dark, lw=1.5)

    # Stage 3: Covariance Whitening / Mahalanobis Metric (x in [7.3, 10.4])
    s3_box = patches.Rectangle((7.3, 0.4), 3.1, 3.0, facecolor='#FFFBEB', edgecolor=c_amber, lw=1.3, zorder=2)
    ax.add_patch(s3_box)
    ax.text(8.85, 3.05, "3. Covariance Whitening ($D_M$)", ha='center', va='center', fontsize=9.0, fontweight='bold', color=c_line_dark, zorder=3)
    ax.text(8.85, 2.35, "Precision Matrix Calibration:\n" +
                        "$\\mathbf{\\Sigma}^{-1} = (\\mathrm{Cov}(\\mathbf{e}_{\\mathrm{cal}}) + \\lambda \\mathbf{I}_D)^{-1}$",
            ha='center', va='center', fontsize=7.8, color='#1E293B', zorder=3)
    ax.text(8.85, 1.45, "Whitened Mahalanobis Metric:\n" +
                        "$D_M = \\sqrt{ \\mathbf{e}^T \\mathbf{\\Sigma}^{-1} \\mathbf{e} }$",
            ha='center', va='center', fontsize=8.0, fontweight='bold', color=c_amber, zorder=3)
    ax.text(8.85, 0.75, "$\\Rightarrow$ Mode-variance scaled", ha='center', va='center', fontsize=7.5, color='#334155', zorder=3)

    draw_arrow(ax, 10.4, 1.9, 10.8, 1.9, color=c_line_dark, lw=1.5)

    # Stage 4: Extreme Value Theory (SPOT) Calibration (x in [10.8, 14.1])
    s4_box = patches.Rectangle((10.8, 0.4), 3.3, 3.0, facecolor='#FEF2F2', edgecolor=c_red, lw=1.3, zorder=2)
    ax.add_patch(s4_box)
    ax.text(12.45, 3.05, "4. EVT Tail Decision", ha='center', va='center', fontsize=9.0, fontweight='bold', color=c_line_dark, zorder=3)

    # Mini GPD Plot inside Stage 4 box
    px = np.linspace(0, 2.0, 50)
    py = np.exp(-px*1.8)
    gx = 11.2 + (px / 2.0) * 1.3
    gy = 1.65 + py * 0.9
    ax.plot(gx, gy, color=c_red, lw=1.5, zorder=4)
    ax.plot([11.2, 12.5], [1.65, 1.65], color=c_line_dark, lw=0.9, zorder=4)
    ax.plot([11.2, 11.2], [1.65, 2.60], color=c_line_dark, lw=0.9, zorder=4)
    # Threshold dashed line
    tau_x = 11.2 + (1.1 / 2.0) * 1.3
    ax.plot([tau_x, tau_x], [1.65, 2.45], color=c_red, lw=1.2, linestyle='--', zorder=5)
    ax.text(tau_x + 0.05, 2.35, r"$\tau_{\mathrm{EVT}}$", fontsize=7.2, fontweight='bold', color=c_red, zorder=5)
    ax.text(12.6, 1.65, "$D_M$", fontsize=7.0, color=c_line_dark, ha='left', va='center', zorder=5)
    ax.text(12.8, 2.2, "Upper 2%\nGPD Tail", fontsize=6.8, color='#991B1B', ha='center', zorder=5)

    # Analytic Formula & Decision
    ax.text(12.45, 1.15, r"$\tau_{\mathrm{EVT}} = u + \frac{\sigma}{\xi}\left[\left(\frac{N_u}{N}q\right)^{-\xi} - 1\right]$",
            ha='center', va='center', fontsize=7.5, fontweight='bold', color='#991B1B', zorder=3)
    ax.text(12.45, 0.70, r"Decision: $\hat{y} = \mathbb{I}(D_M \geq \tau_{\mathrm{EVT}})$" + "\n" + r"(Zero Label-Leakage Calibrated)",
            ha='center', va='center', fontsize=7.4, color='#1E293B', zorder=3)

    out_png = FIG_DIR / "flow_jepa_inference.png"
    out_pdf = FIG_DIR / "flow_jepa_inference.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"Generated Inference Architecture Diagram:\n- {out_png}\n- {out_pdf}")


if __name__ == "__main__":
    generate_training_diagram()
    generate_inference_diagram()
