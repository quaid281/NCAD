#!/usr/bin/env python3
"""
Adapted from Dr. Andrey Churkin's `01_main_simple_example/Python/scripts/beautiful_figure_example.py`
(https://github.com/AndreyChurkin/BeautifulFigures).

Demonstrates all 2D principles for a two-column IEEE research manuscript:
1. 10x10 inch locked-aspect square canvas (`ax.set_aspect('equal', adjustable='datalim')`)
2. 20 pt 'Courier New' typography (renders as ~8 pt at 0.9 * columnwidth in IEEE two-column layout)
3. Dual-hierarchy subtle major/minor grids (`alpha=0.25` / `0.15`) placed behind data (`set_axisbelow(True)`)
4. Harmonious Purple-Grey-Teal palette (`#9671bd`, `#7e7e7e`, `#77b5b6`) with darker marker borders
5. Explicit `zorder=3` for scatter markers over `zorder=2` regression curves
6. Top horizontal frameless legend (`bbox_to_anchor=(0.5, 1.10)`, `ncol=4`, `frameon=False`)
7. Center-median `zoom_out = 0.6` plotting range calculation so regression lines do not blow up axis limits
8. True-vector `.svg` and `.pdf` export with editable text (`svg.fonttype='none'`, `pdf.fonttype=42`)
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Import shared helper from ../scripts
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "scripts"))

from ieee_vector_style import (
    CHURKIN_NEUTRAL_LINE,
    CHURKIN_PURPLE_TEAL_EDGES,
    CHURKIN_PURPLE_TEAL_FILLS,
    apply_churkin_style,
    place_top_horizontal_legend,
    save_true_vector_figure,
    set_centered_zoom_limits,
    style_subtle_grid,
)


def _generate_or_load_iris():
    """Load Iris dataset via scikit-learn if available, else deterministic fallback."""
    try:
        from sklearn.datasets import load_iris
        iris = load_iris()
        return iris.data, iris.target, iris.feature_names, list(iris.target_names)
    except ImportError:
        rng = np.random.default_rng(42)
        n = 50
        setosa = np.column_stack([rng.normal(5.0, 0.35, n), rng.normal(3.4, 0.38, n)])
        versicolor = np.column_stack([rng.normal(5.9, 0.51, n), rng.normal(2.77, 0.31, n)])
        virginica = np.column_stack([rng.normal(6.58, 0.63, n), rng.normal(2.97, 0.32, n)])
        X = np.vstack([setosa, versicolor, virginica])
        y = np.array([0] * n + [1] * n + [2] * n)
        return X, y, ["sepal length (cm)", "sepal width (cm)"], ["setosa", "versicolor", "virginica"]


def build_beautiful_2d_figure(output_stem: Path) -> dict:
    X_data, y_target, feature_names, target_names = _generate_or_load_iris()
    x_col, y_col = 0, 1
    polynomial_degree = 2
    datasets_to_plot = ["setosa", "virginica"]

    # 1. Configure fonts & true-vector rcParams
    apply_churkin_style(font_family="Courier New", font_size=20, keep_svg_text_editable=True)

    # 2. Create 10x10 square figure & subtle grids
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel(f"{feature_names[x_col].capitalize()}")
    ax.set_ylabel(f"{feature_names[y_col].capitalize()}")
    style_subtle_grid(ax, major_lw=0.75, major_alpha=0.25, minor_lw=0.25, minor_alpha=0.15, lock_square_aspect=True)

    all_data_x_to_plot = []
    all_data_y_to_plot = []
    regression_flag = 0

    for class_name in datasets_to_plot:
        class_index = target_names.index(class_name)
        class_mask = y_target == class_index
        data_x = X_data[class_mask, x_col]
        data_y = X_data[class_mask, y_col]

        # Scatter plot on top (zorder=3)
        ax.scatter(
            data_x,
            data_y,
            label=class_name.capitalize(),
            s=90,
            color=CHURKIN_PURPLE_TEAL_FILLS[class_index],
            edgecolors=CHURKIN_PURPLE_TEAL_EDGES[class_index],
            linewidths=1.5,
            zorder=3,
        )

        # Linear & polynomial regressions using numpy polyfit (zero extra dependencies)
        x_range = np.linspace(data_x.min() - 2.0, data_x.max() + 2.0, 100)
        lin_coef = np.polyfit(data_x, data_y, deg=1)
        y_pred_linear = np.polyval(lin_coef, x_range)
        ax.plot(
            x_range,
            y_pred_linear,
            label="LR" if regression_flag == 0 else "_nolegend_",
            linewidth=2.6,
            color=CHURKIN_NEUTRAL_LINE,
            zorder=2,
        )

        poly_coef = np.polyfit(data_x, data_y, deg=polynomial_degree)
        y_pred_poly = np.polyval(poly_coef, x_range)
        ax.plot(
            x_range,
            y_pred_poly,
            label="PR" if regression_flag == 0 else "_nolegend_",
            linewidth=2.6,
            color=CHURKIN_NEUTRAL_LINE,
            linestyle="--",
            zorder=2,
        )

        all_data_x_to_plot.extend(data_x.tolist())
        all_data_y_to_plot.extend(data_y.tolist())
        regression_flag += 1

    # 3. Top horizontal frameless legend
    place_top_horizontal_legend(ax, desired_order=[0, 3, 1, 2], ncol=4, bbox_to_anchor=(0.5, 1.10))

    # 4. Center-median zoom_out range
    set_centered_zoom_limits(ax, all_data_x_to_plot, all_data_y_to_plot, zoom_out=0.6, equal_span=True)

    # 5. Export vector SVG & PDF (plus PNG preview)
    saved = save_true_vector_figure(fig, output_stem, formats=("svg", "pdf", "png"), png_dpi=100)
    plt.close(fig)
    return saved


if __name__ == "__main__":
    out_dir = SCRIPT_DIR.parent / "output_figures"
    saved_files = build_beautiful_2d_figure(out_dir / "beautiful_2d_scatter_regression")
    print("Saved 2D vector figures:", saved_files)
