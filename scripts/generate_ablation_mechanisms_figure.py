"""
Generate Figure 3: Perspective Expansion & Quiet-Fault Recovery
(`paper/NCAD_CS/figures/regularizer_sweep_and_quiet_faults.{pdf,svg,png}`).

Follows `.agents/skills/beautiful-research-figures/SKILL.md`:
  - Churkin Purple-Grey-Teal harmonious grayscale-safe palette
  - Strict external-only frameless horizontal legends (`bbox_to_anchor` outside axes)
  - Subtle dual-hierarchy gridlines strictly behind data (`set_axisbelow(True)`)
  - Clean two-panel layout with zero dual-axis clutter and zero overlapping symbols:
      Panel (a): 6-step fluid-dynamic perspective expansion ladder (1D Scalar -> 224-Bit Breaker)
                 showing strictly causal and buffered Point-F1 with shaded gain envelope ($n=19$).
      Panel (b): Injected single-channel fault Event Recall (%) comparing 1D TS-JEPA,
                 standalone Soliton-Reynolds 224-Bit Breaker, and Hybrid 224-Bit Union ($n=6$).
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / ".agents" / "skills" / "beautiful-research-figures" / "scripts")
)
from ieee_vector_style import (
    apply_churkin_style,
    style_subtle_grid,
    save_true_vector_figure,
)


def main() -> None:
    apply_churkin_style(font_family="Times New Roman", font_size=9)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.16, 2.38), gridspec_kw={"width_ratios": [1.02, 1.05], "wspace": 0.26}
    )

    # =========================================================================
    # Panel (a): 6-Step Fluid-Dynamic Perspective Expansion Ladder (n=19)
    # =========================================================================
    stages = [
        "1D Scalar\n(1 Dial)",
        "32-Bit\n(m=4)",
        "128-Bit\nSub-Blk",
        "160-Bit\n(+Drift)",
        "192-Bit\n(+Fluc.)",
        "224-Bit\nBreaker",
    ]
    x_a = np.arange(len(stages), dtype=float)

    # Verified 19-stream, 3-seed perspective expansion ladder (Table II(a))
    causal_f1 = np.array([0.052, 0.101, 0.129, 0.132, 0.131, 0.138])
    buffered_f1 = np.array([0.137, 0.119, 0.192, 0.188, 0.203, 0.228])

    # Churkin palette: Purple (#6a408d / #9671bd) for Buffered, Teal (#1f6f78 / #77b5b6) for Causal
    c_buf_line, c_buf_fill = "#6a408d", "#9671bd"
    c_cau_line, c_cau_fill = "#1f6f78", "#77b5b6"

    # Baseline reference dashed lines from 1D Scalar Collapse
    ax_a.axhline(
        buffered_f1[0], color=c_buf_line, linestyle=":", linewidth=0.9, alpha=0.55, zorder=1
    )
    ax_a.axhline(
        causal_f1[0], color=c_cau_line, linestyle=":", linewidth=0.9, alpha=0.55, zorder=1
    )

    # Shaded gain envelopes relative to 1D Scalar baseline
    ax_a.fill_between(
        x_a,
        buffered_f1[0],
        buffered_f1,
        where=(buffered_f1 >= buffered_f1[0]),
        color=c_buf_fill,
        alpha=0.16,
        interpolate=True,
        zorder=1,
    )
    ax_a.fill_between(
        x_a,
        causal_f1[0],
        causal_f1,
        color=c_cau_fill,
        alpha=0.20,
        interpolate=True,
        zorder=1,
    )

    # Plot progression curves
    ax_a.plot(
        x_a,
        buffered_f1,
        marker="o",
        markersize=5.6,
        linewidth=2.0,
        color=c_buf_line,
        markerfacecolor=c_buf_fill,
        markeredgecolor=c_buf_line,
        markeredgewidth=1.2,
        label="Buffered Point-F1 (\u00d71.66 gain)",
        zorder=3,
    )
    ax_a.plot(
        x_a,
        causal_f1,
        marker="s",
        markersize=5.4,
        linewidth=2.0,
        color=c_cau_line,
        markerfacecolor=c_cau_fill,
        markeredgecolor=c_cau_line,
        markeredgewidth=1.2,
        label="Strictly Causal Point-F1 (\u00d72.65 gain)",
        zorder=4,
    )

    # Direct value labels on endpoints and key steps (clean, non-overlapping)
    for idx in [0, 2, 3, 5]:
        ax_a.annotate(
            f".{int(round(buffered_f1[idx]*1000)):03d}",
            xy=(x_a[idx], buffered_f1[idx]),
            xytext=(0, 5.5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.3,
            fontweight="bold" if idx in (0, 5) else "normal",
            color=c_buf_line,
        )
        ax_a.annotate(
            f".{int(round(causal_f1[idx]*1000)):03d}",
            xy=(x_a[idx], causal_f1[idx]),
            xytext=(0, -10.5 if idx not in (0, 3) else 5.0),
            textcoords="offset points",
            ha="center",
            va="top" if idx not in (0, 3) else "bottom",
            fontsize=7.3,
            fontweight="bold" if idx in (0, 5) else "normal",
            color=c_cau_line,
        )

    # Callout annotation for delay reduction at 224-Bit Breaker
    ax_a.annotate(
        "Causal delay:\n247 \u2192 106 steps",
        xy=(x_a[-1], causal_f1[-1]),
        xytext=(x_a[3] - 0.15, 0.058),
        fontsize=7.0,
        color="#333333",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#f8f8fa", edgecolor="#bcbcbc", lw=0.6),
        arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8, connectionstyle="arc3,rad=0.15"),
        zorder=5,
    )

    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(stages, fontsize=7.2)
    ax_a.set_xlim(-0.38, len(stages) - 0.55)
    ax_a.set_ylim(0.025, 0.265)
    ax_a.set_yticks([0.05, 0.10, 0.15, 0.20, 0.25])
    ax_a.set_xlabel("Fluid-Dynamic Perspective Expansion (n=19 Streams)", fontsize=8.5, labelpad=3)
    ax_a.set_ylabel("Unadjusted Point-F1", fontsize=8.5)
    ax_a.set_title("(a) 1D Collapse vs. 224-Bit Perspective Ladder", fontsize=9.0, fontweight="bold", pad=22)

    style_subtle_grid(ax_a, lock_square_aspect=False)
    ax_a.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        frameon=False,
        fontsize=7.3,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    # =========================================================================
    # Panel (b): Single-Channel Quiet-Fault Recovery (3 Clean Bars per Fault)
    # =========================================================================
    fault_labels = [
        "Flatline\n(const.)",
        "Stuck\nsensor",
        "Sensor\ndropout",
        "Flatline\n(mid)",
        "Low-amp.\nstep",
        "+5\u03c3\nspike",
    ]
    x_b = np.arange(len(fault_labels), dtype=float)
    width = 0.25

    # Verified 6-stream single-channel fault Event Recall (%)
    er_1d_jepa = np.array([19.4, 19.4, 19.4, 19.4, 19.4, 27.8])
    er_224bit = np.array([25.0, 25.0, 25.0, 25.0, 29.2, 33.3])
    er_hybrid = np.array([44.4, 44.4, 44.4, 44.4, 48.6, 52.8])

    series = [
        (er_1d_jepa, -width, "#b8b8b8", "#4e4e4e", "1D TS-JEPA"),
        (er_224bit, 0.0, "#9671bd", "#5b3280", "224-Bit Breaker"),
        (er_hybrid, width, "#77b5b6", "#1f6f78", "Hybrid 224-Bit Union"),
    ]

    for vals, offset, fill_c, edge_c, label in series:
        bars = ax_b.bar(
            x_b + offset,
            vals,
            width=width * 0.92,
            color=fill_c,
            edgecolor=edge_c,
            linewidth=0.9,
            label=label,
            zorder=3,
        )
        # Annotate the Hybrid Union and representative bars cleanly
        if "Hybrid" in label:
            for bx, v in zip(x_b + offset, vals):
                ax_b.text(
                    bx,
                    v + 1.2,
                    f"{v:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=6.8,
                    fontweight="bold",
                    color=edge_c,
                )

    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(fault_labels, fontsize=7.2)
    ax_b.set_xlim(-0.55, len(fault_labels) - 0.45)
    ax_b.set_ylim(0.0, 62.0)
    ax_b.set_yticks([0, 15, 30, 45, 60])
    ax_b.set_yticklabels(["0%", "15%", "30%", "45%", "60%"])
    ax_b.set_xlabel("Injected Single-Channel Fault Class (n=6 Streams)", fontsize=8.5, labelpad=3)
    ax_b.set_ylabel("Event Recall (%)", fontsize=8.5)
    ax_b.set_title("(b) Single-Channel Quiet-Fault Event Recall", fontsize=9.0, fontweight="bold", pad=22)

    style_subtle_grid(ax_b, lock_square_aspect=False)
    ax_b.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,
        frameon=False,
        fontsize=7.2,
        handletextpad=0.35,
        columnspacing=0.8,
    )

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.19, top=0.80)

    out_dir = Path("paper/NCAD_CS/figures")
    paths = save_true_vector_figure(fig, out_dir / "regularizer_sweep_and_quiet_faults")
    fig.savefig(out_dir / "regularizer_sweep_and_quiet_faults.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {paths}")


if __name__ == "__main__":
    main()
