---
name: beautiful-research-figures
description: >-
  Create publication-grade, 100% true-vector scientific figures (.svg and .pdf)
  following Dr. Andrey Churkin's 8 Principles of Beautiful Figures, IEEE
  two-column typography scaling math, strict external-only legends (never inside
  the figure/axes), harmonious grayscale-safe palettes, and Inkscape MCP
  (inkmcp) vector post-processing. Use this skill whenever creating, styling,
  auditing, or post-processing research plots, charts, 2D/3D scientific
  visualizations, or manuscript figures.
---

# Beautiful Research Figures (`beautiful-research-figures`)

This skill implements a complete, end-to-end workflow for creating publication-ready, **100% true-vector** figures tailored for two-column academic manuscripts (such as IEEE Transactions and `docs/our_paper/main.tex`), synthesizing **Dr. Andrey Churkin's 8 Principles of Beautiful Figures** with automated **Inkscape MCP (`inkmcp`)** vector manipulation and verification.

---

## 1. The 4-Stage Iterative Workflow

Never treat `plt.savefig()` as the final step. Every research figure follows four stages:

1. **Stage 1 — Code in Python (`matplotlib`)**: Parameterize canvas size, font scaling, median-centered zoom-out ranges, subtle dual-hierarchy grids, explicit `zorder`, external-only legends, and grayscale-safe color schemes using [`scripts/ieee_vector_style.py`](./scripts/ieee_vector_style.py).
2. **Stage 2 — Refine in Vector Graphics Software (`inkmcp` / Inkscape)**: Load the exported `.svg` in the `inkmcp` MCP server (`set_active_svg`), inspect/adjust vector groups, add callouts or mathematical annotations without data overlap, and export clean PDF/SVG via `inkscape_cli_export`.
3. **Stage 3 — Test in the Target Manuscript (`main.tex`)**: Place the `.pdf` figure into the actual two-column LaTeX template (`0.9\columnwidth` or `\textwidth`) and inspect rendered font sizes, line weights, and layout balance at 100% page zoom.
4. **Stage 4 — Iterate**: Verify vector purity using [`scripts/verify_vector_figure.py`](./scripts/verify_vector_figure.py) (or the `inspect_vector_figure` MCP tool) and refine margins, legend placement, or `zoom_out` until zero visual defects remain.

---

## 2. The 8 Core Principles & Concrete Parameter Rules

### Principle 1: Overall Quality — True Vector Graphics First
- **NEVER** publish raster `.png` or `.jpg` figures for line plots, scatter plots, bar charts, or schematics (Matplotlib's default 640×480 px at 100 DPI produces blurry, pixelated artifacts in print).
- Always export **both** `.pdf` (for LaTeX `\includegraphics`) and `.svg` (for Inkscape editing).
- Enforce editable TrueType text and live `<text>` nodes in `rcParams`:
  ```python
  plt.rcParams.update({
      "pdf.fonttype": 42,       # TrueType outlines in PDF (searchable, scalable, IEEE-compliant)
      "ps.fonttype": 42,        # TrueType in EPS/PS
      "svg.fonttype": "none",   # Keep text as editable <text> elements in SVG for Inkscape
  })
  ```
- Avoid `rasterized=True` on vector artists. Note that Matplotlib 3.8+ defaults `cbar.solids.set_rasterized(True)` on colorbars; always call `cbar.solids.set_rasterized(False)` (handled automatically by `save_true_vector_figure()`).

### Principle 2: Readability & IEEE Two-Column Font Scaling Math
- In an IEEE Transactions two-column layout, a single column is `3.5 in` wide (`0.9\columnwidth` $\approx 3.15\text{ in}$).
- When a `figsize=(10, 10)` figure is placed in a single column, it scales down by a factor of $s \approx 0.315\text{--}0.40$:
  $$\text{Printed Font Size (pt)} = \text{Matplotlib Font Size (pt)} \times \frac{\text{Rendered Width in LaTeX (in)}}{\text{Matplotlib Figure Width (in)}}$$
- **Rule of 20 pt**: On a `10×10 in` canvas placed in one column, set **`font.size = 20`** so it renders at **~8 pt** in print (the minimum legible font size in IEEE Transactions captions/annotations). For a `3.5×3.5 in` native single-column canvas, set `font.size = 8–9`.
- **Typography**: Avoid Matplotlib's default `DejaVu Sans` or `Calibri`. Use deliberate academic fonts:
  - `'Courier New'` (formal technical monospace, Churkin signature style)
  - `'Times New Roman'` or `'STIXGeneral'` (matches IEEE Roman body text)
  - `'Arial'` or `'Helvetica'` (clean IEEE sans-serif subfigures)

### Principle 3: Controlled Proportions & Center-Median "Zoom Out" Range
- Lock canvas dimensions across companion figures (`figsize=(10, 10)` and `ax.set_aspect('equal', adjustable='datalim')` for square phase/scatter plots) so side-by-side figures in a manuscript never stretch arbitrarily.
- **Do NOT let Matplotlib autoscale to long regression or extrapolation lines.** Compute explicit limits centered on the **data midpoint** (`(min + max) / 2`) with a manual `zoom_out` margin:
  ```python
  x_min, x_max = np.min(data_x), np.max(data_x)
  y_min, y_max = np.min(data_y), np.max(data_y)
  x_median, y_median = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
  plotting_range = max(x_max - x_min, y_max - y_min) + zoom_out
  ax.set_xlim(x_median - plotting_range / 2.0, x_median + plotting_range / 2.0)
  ax.set_ylim(y_median - plotting_range / 2.0, y_median + plotting_range / 2.0)
  ```
- **Avoid `plt.tight_layout()` on locked-aspect or 3D figures**: `tight_layout()` overrides square plotting areas and fails on 3D projections.

### Principle 4: Simplify, Declutter & Visual Hierarchy (`zorder`) + STRICT EXTERNAL-ONLY LEGENDS
- **Subtle Dual-Hierarchy Grids**: Help readers trace values without visual clutter by using thin, highly transparent gridlines strictly behind the data:
  ```python
  ax.grid(True, which="major", linestyle="-", linewidth=0.75, alpha=0.25)
  ax.minorticks_on()
  ax.grid(True, which="minor", linestyle="-", linewidth=0.25, alpha=0.15)
  ax.set_axisbelow(True)
  ```
- **Explicit Layering (`zorder`)**: Primary empirical data points must sit *above* fitted/reference curves:
  - Scatter / primary observations: `s=90`, `linewidths=1.5`, darker `edgecolors`, **`zorder=3`**
  - Regression / model curves: `linewidth=2.6`, neutral/muted tone (`#8a8a8a`), **`zorder=2`**
- **MANDATORY RULE — NO LEGENDS INSIDE THE FIGURE (OUTSIDE ALWAYS)**:
  - **NEVER** place a legend inside the plotting canvas/axes box (e.g., `loc="upper right"`, `loc="best"`, `loc="lower left"` without an external `bbox_to_anchor` are **strictly forbidden**). Even when an interior corner looks empty on one dataset, interior legends violate visual balance, create box clutter, and risk colliding with data or grid structure.
  - **ALWAYS** place legends **outside** the plotting area—preferrably as a frameless horizontal bar directly above the axes (`bbox_to_anchor=(0.5, 1.10)`, `loc="upper center"`, `frameon=False`) or below the axes (`bbox_to_anchor=(0.5, -0.15)`, `loc="upper center"`, `frameon=False`), and deduplicate repeated model labels with `"_nolegend_"`:
  ```python
  ax.legend(
      [handles[i] for i in desired_order],
      [labels[i] for i in desired_order],
      loc="upper center",
      bbox_to_anchor=(0.5, 1.10),  # Strictly OUTSIDE the [0, 1] x [0, 1] axes box
      ncol=4,
      frameon=False,
  )
  ```

### Principle 5: Color Harmony & Grayscale Safety
- **Reject Default `tab10` & Rainbow Colormaps**:
  - Default blue/orange looks like unstyled Excel; uncontextualized red/green lines trigger false "good vs. bad/error" semantics.
  - **NEVER** use `jet`, `rainbow`, `gist_rainbow`, or `hsv`. They have non-monotonic luminance, become unreadable in grayscale print, and blind color-vision-deficient readers.
- **Curated Palettes**:
  - **Churkin Purple–Grey–Teal Categorical Palette** (with darker stroke borders):
    - Fill colors: `['#9671bd', '#7e7e7e', '#77b5b6']`
    - Edge/line colors: `['#6a408d', '#4e4e4e', '#378d94']`
    - Neutral regression/reference line: `'#8a8a8a'` (or `'#7f9fa1'`)
  - **Purple–Orange Sunset Sequential Palette** (monotonic luminance, high dynamic range):
    - Plateau low: `'#2d0b4e'` $\rightarrow$ ramp `['#a11477', '#c1246b', '#e13661', '#fd4c55', '#ff6f4b']`

### Principle 6: Message, Story & Honest 3D Surfaces
- Avoid pie charts and gratuitous 3D plots whenever a 2D plot communicates the comparison more clearly.
- When 3D surfaces *are* justified (e.g., `Day × Hour × Price/Signal` regimes with sharp spikes):
  - **Use `FACET_COLOUR_RULE = "max"`**: Matplotlib's default flat shading averages the 4 corner Z-values (`"mean"`), pulling sharp single-cell peaks down toward their neighbors so they *never* reach the top of the colormap. Computing `face_value = np.maximum.reduce([Z[:-1,:-1], Z[1:,:-1], Z[:-1,1:], Z[1:,1:]])` ensures peaks show their true maximum color.
  - **Set `USE_SHADES = False` (`shade=False`)**: Directional lighting distorts facet colors so they no longer match the colorbar.
  - **Set `linewidth=0`** on `plot_surface` and make 3D panes transparent (`axis.pane.set_facecolor((0, 0, 0, 0))`).
  - **Omit redundant Z-axis text labels** when the colorbar already states the variable and units.

### Principles 7 & 8: Consistent Style Across the Paper & Iterative Time Investment
- Centralize all style constants in [`scripts/ieee_vector_style.py`](./scripts/ieee_vector_style.py) so every figure in the manuscript shares identical font families, tick sizes, spine widths, and color semantics.

---

## 3. Executable Helpers & Examples

- **Style & Vector Export Helper**: [`scripts/ieee_vector_style.py`](./scripts/ieee_vector_style.py)
  Provides `apply_churkin_style()`, `set_centered_zoom_limits()`, `style_subtle_grid()`, `place_top_horizontal_legend()`, `compute_surface_facecolors()`, and `save_true_vector_figure()`.
- **Vector Purity & IEEE Font Validator**: [`scripts/verify_vector_figure.py`](./scripts/verify_vector_figure.py)
  Run via CLI to verify that any `.svg` or `.pdf` figure contains true vector paths/text, zero unintended raster `<image>` leaks, and IEEE-compliant scaled font sizes:
  ```powershell
  & "C:\Users\andre\anaconda3\envs\pinknoise\python.exe" .agents/skills/beautiful-research-figures/scripts/verify_vector_figure.py path/to/figure.svg --target-width-in 3.15
  ```
- **Complete 2D & 3D Reference Implementations**:
  - [`examples/beautiful_2d_scatter_regression.py`](./examples/beautiful_2d_scatter_regression.py)
  - [`examples/beautiful_3d_surface.py`](./examples/beautiful_3d_surface.py)
- **Deep-Dive Reference & Inkscape MCP Guide**:
  - [`references/churkin_principles_and_inkscape_workflow.md`](./references/churkin_principles_and_inkscape_workflow.md)

---

## 4. Using the Inkscape MCP (`inkmcp`) Server for Stage 2 Post-Processing

When the `inkmcp` MCP server is configured, use its tools to inspect, annotate, and convert vector figures natively on Windows:

1. **`set_active_svg(svg_path, create_if_missing=True)`**: Point `inkmcp` at the `.svg` figure exported by Matplotlib.
2. **`inkscape_operation(command)`**:
   - Inspect document structure: `command="get-info"`
   - Add vector callouts, arrows, or labels: `command="text id=peak_label x=240 y=120 text='Resonance Peak' font-family='Courier New' font-size=20 fill=#6a408d"`
   - Run custom `inkex` Python transformations on the SVG DOM: `command="execute-code code='el = get_element_by_id(\"legend_1\"); el.set(\"transform\", \"translate(0,-10)\") if el else None'"`
   - Render a quick viewport check: `command="export-document-image format=png return_base64=true"`
3. **`inspect_vector_figure(file_path)`**: Audit vector purity (`<path>`, `<text>`, `<g>` vs. `<image>` count, font families, dimensions) before manuscript inclusion.
4. **`inkscape_cli_export(input_path, output_path, export_type="pdf", text_to_path=False, export_area_drawing=False, actions="")`**: Convert the refined `.svg` to publication-ready `.pdf` / `.eps` / `.emf` using headless Inkscape (`C:\Program Files\Inkscape\bin\inkscape.exe`).
