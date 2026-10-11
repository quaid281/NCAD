#!/usr/bin/env python3
"""Generate the inference-regime timeline diagram (Reviewer Major Concern 3).

Shows exactly which observations are available when each score is computed and
where the alarm is attributed, for the buffered ("smear") and strictly causal
("trailing") score-to-timestamp mappings.

Output: paper/NCAD_CS/figures/timeline_inference_regimes.{pdf,png}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "NCAD_CS" / "figures"

C, S = 256, 64
CTX_LO, CTX_HI = 0, C
TGT_LO, TGT_HI = C, C + S
TAU = C + S - 1
TSTAR = C + 22

fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.0), sharex=True,
                         gridspec_kw=dict(hspace=0.62))
t_axis_lo, t_axis_hi = -14, C + S + 40


def draw_window(ax):
    ax.add_patch(Rectangle((CTX_LO, 0.55), CTX_HI - CTX_LO, 0.34,
                           fc="#4C72B0", ec="k", lw=0.6, alpha=0.9, zorder=3))
    ax.add_patch(Rectangle((TGT_LO, 0.55), TGT_HI - TGT_LO, 0.34,
                           fc="#DD8452", ec="k", lw=0.6, alpha=0.9, zorder=3))
    ax.text((CTX_LO + CTX_HI) / 2, 0.72, r"context $\mathbf{x}_{\mathrm{ctx}}$ "
            r"($C{=}256$)", ha="center", va="center", color="white", fontsize=8)
    ax.text((TGT_LO + TGT_HI) / 2, 0.72, r"target $\mathbf{x}_{\mathrm{tgt}}$ "
            r"($S{=}64$)", ha="center", va="center", color="white", fontsize=8)
    ax.scatter([TAU], [0.45], marker="v", s=55, c="k", zorder=4)
    ax.axvline(TSTAR, color="#C44E52", ls="--", lw=1.2, zorder=2)
    ax.text(TSTAR - 5, 0.10, r"onset $t^{*}$", color="#C44E52", fontsize=8,
            ha="right", va="center")
    ax.axhline(0.45, color="k", lw=0.8, zorder=1)
    ax.set_ylim(-0.55, 1.5)
    ax.set_yticks([])


def emission_note(ax):
    ax.annotate(r"score $s_i$ computable at $\tau=t{+}S{-}1$"
                "\n(all window observations received)", xy=(TAU, 0.42),
                xytext=(TAU - 165, -0.38), fontsize=8, va="center",
                arrowprops=dict(arrowstyle="->", lw=0.8))


ax = axes[0]
draw_window(ax)
emission_note(ax)
ax.add_patch(Rectangle((TGT_LO, 1.0), TGT_HI - TGT_LO, 0.22, fc="#DD8452",
                       ec="k", lw=0.6, alpha=0.35, zorder=3))
ax.annotate("", xy=(TGT_LO + 4, 1.02), xytext=(TAU, 0.9),
            arrowprops=dict(arrowstyle="->", color="#8a4a22", lw=1.3,
                            connectionstyle="arc3,rad=-0.3"))
ax.text(TGT_LO - 10, 1.11, "score attributed back over\n"
        r"suspect region $[t{+}1,\,t{+}S]$", fontsize=7.5, color="#8a4a22",
        ha="right", va="center")
ax.text(TAU + 8, 0.72, "pre-onset alarms score as FP\n"
        "under pointwise evaluation", fontsize=7.5, color="#8a4a22", va="center")
ax.set_title("(a) Buffered monitoring (``smear''): "
             r"$\mathcal{F}_{t+S-1}$-measurable; alarms reach back $S$ steps",
             fontsize=9, loc="left")

ax = axes[1]
draw_window(ax)
emission_note(ax)
ax.scatter([TAU], [0.98], marker="o", s=55, fc="#55A868", ec="k", zorder=4)
ax.annotate("", xy=(TAU, 0.94), xytext=(TAU, 0.52),
            arrowprops=dict(arrowstyle="->", color="#2d5a3a", lw=1.3))
ax.text(TGT_LO - 10, 0.98, "score emitted at $\\tau$; alarm attributed\n"
        r"to $\tau$ only (online, no reach-back)", fontsize=7.5,
        color="#2d5a3a", ha="right", va="center")
ax.set_title("(b) Strictly causal streaming (``trailing''): "
             r"$\mathcal{F}_{\tau}$-measurable; alarm at window completion",
             fontsize=9, loc="left")

for ax in axes:
    ax.set_xlim(t_axis_lo, t_axis_hi)
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.set_xticks([CTX_LO, TSTAR, TAU])
    ax.set_xticklabels([r"$t{-}C{+}1$", r"$t^{*}$", r"$\tau=t{+}S{-}1$"], fontsize=8)

axes[1].set_xlabel("time (steps)", fontsize=9)
fig.savefig(FIG / "timeline_inference_regimes.pdf", bbox_inches="tight")
fig.savefig(FIG / "timeline_inference_regimes.png", dpi=300, bbox_inches="tight")
print("wrote", FIG / "timeline_inference_regimes.{pdf,png}")
