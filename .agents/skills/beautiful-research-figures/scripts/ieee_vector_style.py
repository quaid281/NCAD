#!/usr/bin/env python3
"""
Reusable Matplotlib styling, axis range calculation, 3D surface facet coloring,
and true-vector SVG/PDF export utilities based on Dr. Andrey Churkin's
Principles of Beautiful Figures for Research Papers and IEEE Transactions rules.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm, to_rgb, to_rgba

# ============================================================================
# CURATED HARMONIOUS PALETTES (GRAYSCALE-SAFE, NON-TAB10, NON-RAINBOW)
# ============================================================================

CHURKIN_PURPLE_TEAL_FILLS: List[str] = ["#9671bd", "#7e7e7e", "#77b5b6"]
CHURKIN_PURPLE_TEAL_EDGES: List[str] = ["#6a408d", "#4e4e4e", "#378d94"]
CHURKIN_NEUTRAL_LINE: str = "#8a8a8a"
CHURKIN_SEAFOAM_LINE: str = "#7f9fa1"

SUNSET_LOW_PLATEAU: str = "#2d0b4e"
SUNSET_RAMP: List[str] = ["#a11477", "#c1246b", "#e13661", "#fd4c55", "#ff6f4b"]

DARK_TEAL_BG_CENTER: str = "#12302f"
DARK_TEAL_BG_EDGE: str = "#071414"
DARK_TEAL_PRIMARY_INK: str = "#f2f2f0"
DARK_TEAL_GRIDLINE: str = "#2c4a48"

DEFAULT_INKSCAPE_WIN_PATH = r"C:\Program Files\Inkscape\bin\inkscape.exe"


def compute_ieee_scaled_fontsize(
    canvas_width_in: float = 10.0,
    target_print_width_in: float = 3.15,
    target_print_pt: float = 8.0,
) -> int:
    """
    Compute the Matplotlib font size (in pt) required so that when a figure of
    width `canvas_width_in` is scaled down to `target_print_width_in` (e.g.,
    0.9 * IEEE columnwidth = 3.15 in), the rendered text equals `target_print_pt`
    (minimum 8 pt for IEEE Transactions).

    Note: Using Andrey Churkin's empirical 10x10 in -> 0.9*columnwidth mapping,
    20 pt in Matplotlib renders at ~8 pt in print once outer whitespace is trimmed.
    """
    if canvas_width_in <= 0 or target_print_width_in <= 0:
        raise ValueError("Widths must be positive.")
    raw_scale = canvas_width_in / target_print_width_in
    # Account for typical ~20% outer axes margin on standard Matplotlib canvases
    effective_pt = target_print_pt * (raw_scale * 0.7875)
    return max(int(round(effective_pt)), int(round(target_print_pt)))


def apply_churkin_style(
    font_family: str = "Courier New",
    font_size: int = 20,
    keep_svg_text_editable: bool = True,
) -> dict:
    """
    Configure Matplotlib rcParams for true-vector publication figures:
    - Embeds TrueType fonts (Type 42) in PDF and PS exports
    - Preserves editable <text> nodes in SVG (`svg.fonttype = 'none'`) for Inkscape
    - Sets consistent academic typography across title, axis labels, ticks, and legend
    """
    params = {
        "font.family": font_family,
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "figure.titlesize": font_size,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none" if keep_svg_text_editable else "path",
    }
    plt.rcParams.update(params)
    return params


def style_subtle_grid(
    ax: plt.Axes,
    major_lw: float = 0.75,
    major_alpha: float = 0.25,
    minor_lw: float = 0.25,
    minor_alpha: float = 0.15,
    lock_square_aspect: bool = True,
) -> None:
    """
    Apply Andrey Churkin's dual-hierarchy subtle grid behind all plotted data,
    and optionally lock the axes to a 1:1 square aspect ratio.
    """
    if lock_square_aspect:
        ax.set_aspect("equal", adjustable="datalim")

    ax.grid(True, which="major", linestyle="-", linewidth=major_lw, alpha=major_alpha)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle="-", linewidth=minor_lw, alpha=minor_alpha)
    ax.set_axisbelow(True)


def set_centered_zoom_limits(
    ax: plt.Axes,
    data_x: Iterable[float],
    data_y: Iterable[float],
    zoom_out: float = 0.6,
    equal_span: bool = True,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute plotting limits centered on the midpoint `(min + max) / 2` of the
    primary data points (ignoring long regression/extrapolation lines), adding
    `zoom_out` margin equally in all directions.
    """
    x_arr = np.asarray(list(data_x), dtype=float).ravel()
    y_arr = np.asarray(list(data_y), dtype=float).ravel()
    if x_arr.size == 0 or y_arr.size == 0:
        raise ValueError("data_x and data_y must be non-empty.")

    x_min, x_max = float(np.min(x_arr)), float(np.max(x_arr))
    y_min, y_max = float(np.min(y_arr)), float(np.max(y_arr))

    x_median = (x_min + x_max) / 2.0
    y_median = (y_min + y_max) / 2.0
    x_range = x_max - x_min
    y_range = y_max - y_min

    if equal_span:
        span_x = span_y = max(x_range, y_range) + float(zoom_out)
    else:
        span_x = x_range + float(zoom_out)
        span_y = y_range + float(zoom_out)

    xlim = (x_median - span_x / 2.0, x_median + span_x / 2.0)
    ylim = (y_median - span_y / 2.0, y_median + span_y / 2.0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return xlim, ylim


def is_bbox_outside_axes(bbox_to_anchor: Optional[Sequence[float]]) -> bool:
    """
    Return True only if `bbox_to_anchor` places the legend strictly OUTSIDE the
    interior `(0, 1) x (0, 1)` axes canvas (e.g., `y > 1.0` above the plot,
    `y < 0.0` below the plot, `x < 0.0` left, or `x > 1.0` right).
    """
    if bbox_to_anchor is None or len(bbox_to_anchor) < 2:
        return False
    bx, by = float(bbox_to_anchor[0]), float(bbox_to_anchor[1])
    return not (0.0 <= bx <= 1.0 and 0.0 <= by <= 1.0)


def place_top_horizontal_legend(
    ax: plt.Axes,
    desired_order: Optional[Sequence[int]] = None,
    ncol: int = 4,
    bbox_to_anchor: Tuple[float, float] = (0.5, 1.10),
    loc: str = "upper center",
) -> plt.Legend:
    """
    Place a clean, frameless horizontal legend OUTSIDE the axes (default: above
    the plot at `bbox_to_anchor=(0.5, 1.10)`) so it NEVER sits inside the figure
    canvas or occludes data. Raises ValueError if `bbox_to_anchor` is inside `[0, 1] x [0, 1]`.
    """
    if not is_bbox_outside_axes(bbox_to_anchor):
        raise ValueError(
            f"Legend bbox_to_anchor={bbox_to_anchor} is inside the [0, 1]x[0, 1] axes box. "
            "Rule violation: No legends inside the figure — place outside always (e.g. (0.5, 1.10))."
        )

    handles, labels = ax.get_legend_handles_labels()
    if desired_order is not None:
        handles = [handles[i] for i in desired_order]
        labels = [labels[i] for i in desired_order]

    leg = ax.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        frameon=False,
    )
    leg._outside_anchor = tuple(bbox_to_anchor)  # type: ignore[attr-defined]
    return leg


place_outside_legend = place_top_horizontal_legend


def enforce_outside_legends(
    fig: plt.Figure,
    default_bbox_to_anchor: Tuple[float, float] = (0.5, 1.10),
    default_loc: str = "upper center",
    default_ncol: int = 4,
) -> int:
    """
    Inspect all axes in `fig` and relocate any interior legend to a frameless
    external legend outside the figure (`default_bbox_to_anchor=(0.5, 1.10)`).
    Returns the number of interior legends that were automatically relocated.
    """
    relocated_count = 0
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        outside_anchor = getattr(leg, "_outside_anchor", None)
        if outside_anchor is not None and is_bbox_outside_axes(outside_anchor):
            continue
        # Check if user explicitly anchored outside via bbox_to_anchor in axes coords
        bbox_attr = getattr(leg, "_bbox_to_anchor", None)
        is_external = False
        if bbox_attr is not None:
            try:
                # Transform bbox back to axes coordinates to check if outside [0, 1] x [0, 1]
                inv = ax.transAxes.inverted()
                pts = inv.transform(bbox_attr.extents.reshape(2, 2))
                if not (0.0 <= pts[0, 0] <= 1.0 and 0.0 <= pts[0, 1] <= 1.0):
                    is_external = True
            except Exception:
                is_external = False
        if not is_external:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ncol = min(max(len(handles), 1), default_ncol)
                new_leg = ax.legend(
                    handles,
                    labels,
                    loc=default_loc,
                    bbox_to_anchor=default_bbox_to_anchor,
                    ncol=ncol,
                    frameon=False,
                )
                new_leg._outside_anchor = tuple(default_bbox_to_anchor)  # type: ignore[attr-defined]
            else:
                leg.set_frame_on(False)
                leg.set_bbox_to_anchor(default_bbox_to_anchor, transform=ax.transAxes)
                leg._outside_anchor = tuple(default_bbox_to_anchor)  # type: ignore[attr-defined]
            relocated_count += 1
    return relocated_count


def build_sunset_surface_cmap(low_fraction: float = 0.1) -> LinearSegmentedColormap:
    """
    Build the Purple-Orange Sunset colormap with a calm deep-purple plateau
    at the bottom `low_fraction` of the range and a monotonic ramp to bright orange.
    """
    positions = np.linspace(low_fraction, 1.0, len(SUNSET_RAMP) + 1)[1:]
    stops = [(0.0, SUNSET_LOW_PLATEAU), (low_fraction, SUNSET_LOW_PLATEAU)]
    stops += list(zip(positions, SUNSET_RAMP))
    return LinearSegmentedColormap.from_list("churkin_sunset", stops)


def compute_surface_facecolors(
    z_grid: np.ndarray,
    cmap: matplotlib.colors.Colormap,
    rule: str = "max",
    nonlinear_gamma: Optional[float] = None,
) -> Tuple[np.ndarray, Normalize, cm.ScalarMappable]:
    """
    Compute 3D surface facet colors from corner Z-values:
    - `rule="max"` colors each facet by the maximum of its 4 corners so sharp
      spikes reach the true peak color of the colormap (fixing Matplotlib's
      default `"mean"` averaging dilution).
    - Returns `(facecolors, norm, mappable)` so `fig.colorbar(mappable, ...)`
      stays 100% honest to the plotted surface.
    """
    z = np.asarray(z_grid, dtype=float)
    if z.ndim != 2 or z.shape[0] < 2 or z.shape[1] < 2:
        raise ValueError("z_grid must be a 2D array with shape >= (2, 2).")

    if nonlinear_gamma is not None:
        norm: Normalize = PowerNorm(gamma=nonlinear_gamma, vmin=float(z.min()), vmax=float(z.max()))
    else:
        norm = Normalize(vmin=float(z.min()), vmax=float(z.max()))

    if rule == "max":
        face_value = np.maximum.reduce(
            [z[:-1, :-1], z[1:, :-1], z[:-1, 1:], z[1:, 1:]]
        )
    elif rule == "mean":
        face_value = 0.25 * (z[:-1, :-1] + z[1:, :-1] + z[:-1, 1:] + z[1:, 1:])
    else:
        raise ValueError(f"Unknown rule {rule!r}; expected 'max' or 'mean'.")

    facecolors = cmap(norm(face_value))
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    return facecolors, norm, mappable


def find_inkscape_executable() -> Optional[str]:
    """Locate Inkscape executable on Windows or PATH.
    Prefers inkscape.com on Windows because it is the native console application
    which runs synchronously without detaching from the command line."""
    which_path = shutil.which("inkscape")
    if which_path:
        return which_path
    com_path = r"C:\Program Files\Inkscape\bin\inkscape.com"
    if os.path.exists(com_path):
        return com_path
    if os.path.exists(DEFAULT_INKSCAPE_WIN_PATH):
        return DEFAULT_INKSCAPE_WIN_PATH
    return None


def vectorize_figure_collections(fig: plt.Figure) -> int:
    """
    Ensure all collections (especially Matplotlib 3.8+ `Colorbar.solids`, which
    defaults to `rasterized=True`) are set to `rasterized=False` so `.svg` and
    `.pdf` exports remain 100% pure vector with zero embedded `<image>` streams.
    """
    unrasterized_count = 0
    for ax in fig.axes:
        for col in getattr(ax, "collections", []):
            if col.get_rasterized():
                col.set_rasterized(False)
                unrasterized_count += 1
        for patch in getattr(ax, "patches", []):
            if patch.get_rasterized():
                patch.set_rasterized(False)
                unrasterized_count += 1
    return unrasterized_count


def save_true_vector_figure(
    fig: plt.Figure,
    out_stem: Union[str, Path],
    formats: Sequence[str] = ("svg", "pdf", "png"),
    png_dpi: int = 300,
    inkscape_postprocess_pdf: bool = False,
    ensure_pure_vector: bool = True,
    ensure_outside_legend: bool = True,
) -> dict:
    """
    Save `fig` to `.svg`, `.pdf` (true vector), and optionally `.png` preview.
    Automatically un-rasterizes `Colorbar.solids` and collection artists when
    `ensure_pure_vector=True`, and relocates any interior legend outside the
    plotting area when `ensure_outside_legend=True`.
    """
    stem_path = Path(out_stem)
    stem_path.parent.mkdir(parents=True, exist_ok=True)

    if ensure_pure_vector:
        vectorize_figure_collections(fig)
    if ensure_outside_legend:
        enforce_outside_legends(fig)

    saved_paths = {}
    facecolor = fig.get_facecolor()

    for ext in formats:
        ext_clean = ext.lstrip(".").lower()
        target = stem_path.with_suffix(f".{ext_clean}")
        if ext_clean == "png":
            fig.savefig(target, dpi=png_dpi, facecolor=facecolor)
        else:
            fig.savefig(target, facecolor=facecolor)
        saved_paths[ext_clean] = str(target.resolve())

    if inkscape_postprocess_pdf and "svg" in saved_paths:
        inkscape_bin = find_inkscape_executable()
        if inkscape_bin:
            pdf_target = stem_path.with_suffix(".pdf").resolve()
            subprocess.run(
                [
                    inkscape_bin,
                    saved_paths["svg"],
                    "--export-type=pdf",
                    f"--export-filename={pdf_target}",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            saved_paths["pdf"] = str(pdf_target)

    return saved_paths

