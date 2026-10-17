#!/usr/bin/env python3
"""
Adapted from Dr. Andrey Churkin's `02_3d_price_surface/Python/scripts/beautiful_3d_price_surface.py`
(https://github.com/AndreyChurkin/BeautifulFigures).

Demonstrates honest, publication-ready 3D surface plotting:
1. Coherent Purple-Orange Sunset colormap (`#2d0b4e` low plateau -> `#ff6f4b` spike peak)
2. `FACET_COLOUR_RULE = "max"` so sharp spikes reach the true colormap peak instead of being
   averaged down by Matplotlib's default `"mean"` corner shading
3. `USE_SHADES = False` (`shade=False`) so facet colors match the colorbar 1:1
4. `linewidth=0` on `plot_surface` and transparent 3D panes
5. Standalone `ScalarMappable` colorbar and no redundant Z-axis label
6. True-vector `.svg` and `.pdf` export
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "scripts"))

from ieee_vector_style import (
    DARK_TEAL_BG_EDGE,
    DARK_TEAL_GRIDLINE,
    DARK_TEAL_PRIMARY_INK,
    apply_churkin_style,
    build_sunset_surface_cmap,
    compute_surface_facecolors,
    save_true_vector_figure,
)


def synthesize_spiky_surface(n_days: int = 90, n_hours: int = 25):
    """Create a realistic Day x Hour-of-Day surface with calm plateau and sharp spikes."""
    rng = np.random.default_rng(7)
    days = np.arange(n_days)
    hours = np.linspace(0, 24, n_hours)
    day_grid, hour_grid = np.meshgrid(days, hours)

    # Morning (8h) and evening (19h) duck-curve profile + calm baseline
    base = 45.0 + 18.0 * np.exp(-0.5 * ((hour_grid - 8.0) / 2.2) ** 2) + 25.0 * np.exp(-0.5 * ((hour_grid - 19.0) / 2.5) ** 2)
    noise = rng.normal(0.0, 4.0, size=day_grid.shape)
    surface = np.clip(base + noise, 10.0, None)

    # Inject sharp localized price/energy spikes
    spike_coords = [(18, 19), (42, 18), (43, 19), (67, 8), (74, 19)]
    for d_idx, h_idx in spike_coords:
        if d_idx < n_days and h_idx < n_hours:
            surface[h_idx, d_idx] = 260.0 + 40.0 * rng.random()

    return day_grid, hour_grid, surface


def build_beautiful_3d_surface(output_stem: Path, facet_colour_rule: str = "max", use_shades: bool = False) -> dict:
    apply_churkin_style(font_family="Courier New", font_size=13, keep_svg_text_editable=True)

    day_grid, hour_grid, z_grid = synthesize_spiky_surface()
    cmap = build_sunset_surface_cmap(low_fraction=0.10)
    facecolors, _, mappable = compute_surface_facecolors(z_grid, cmap=cmap, rule=facet_colour_rule)

    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor(DARK_TEAL_BG_EDGE)

    ax = fig.add_subplot(111, projection="3d", facecolor=DARK_TEAL_BG_EDGE)
    ax.set_box_aspect(None, zoom=1.1)

    ax.plot_surface(
        day_grid,
        hour_grid,
        z_grid,
        facecolors=facecolors,
        linewidth=0,
        shade=use_shades,
        rstride=1,
        cstride=1,
    )

    fig.suptitle(
        "Day-Ahead Signal Surface (Max-Corner Facet Rule)",
        color=DARK_TEAL_PRIMARY_INK,
        fontsize=18,
        fontweight="bold",
        y=0.96,
    )
    ax.set_xlabel("Day Index", color=DARK_TEAL_PRIMARY_INK, fontsize=13, labelpad=12)
    ax.set_ylabel("Hour of Day", color=DARK_TEAL_PRIMARY_INK, fontsize=13, labelpad=6)

    hour_ticks = [0, 4, 8, 12, 16, 20, 24]
    ax.set_yticks(hour_ticks)
    ax.set_yticklabels([str(h) for h in hour_ticks], color=DARK_TEAL_PRIMARY_INK)
    ax.tick_params(colors=DARK_TEAL_PRIMARY_INK, labelsize=11)

    grid_rgba = to_rgba(DARK_TEAL_GRIDLINE, alpha=0.35)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0, 0, 0, 0))
        axis.pane.set_edgecolor(DARK_TEAL_GRIDLINE)
        axis._axinfo["grid"].update(color=grid_rgba, linewidth=0.5)
    ax.grid(True)
    ax.view_init(elev=30, azim=-60)

    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.12)
    if getattr(cbar, "solids", None) is not None:
        cbar.solids.set_rasterized(False)
    cbar.set_label("Amplitude / Price", color=DARK_TEAL_PRIMARY_INK, fontsize=13, labelpad=12)
    cbar.ax.yaxis.set_tick_params(color=DARK_TEAL_PRIMARY_INK, labelsize=11)
    plt.setp(cbar.ax.get_yticklabels(), color=DARK_TEAL_PRIMARY_INK)

    saved = save_true_vector_figure(fig, output_stem, formats=("svg", "pdf", "png"), png_dpi=300)
    plt.close(fig)
    return saved


if __name__ == "__main__":
    out_dir = SCRIPT_DIR.parent / "output_figures"
    saved_files = build_beautiful_3d_surface(out_dir / "beautiful_3d_surface")
    print("Saved 3D vector figures:", saved_files)
