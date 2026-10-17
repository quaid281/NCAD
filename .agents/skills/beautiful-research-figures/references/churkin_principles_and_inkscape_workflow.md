# Dr. Andrey Churkin's 8 Principles & Inkscape MCP (`inkmcp`) Vector Workflow

This reference manual compiles the complete methodology from Dr. Andrey Churkin's *Principles of Beautiful Figures for Research Papers* (Part 1) and *How to Create Beautiful Research Figures — Complete Tutorial* (Part 2, [`AndreyChurkin/BeautifulFigures`](https://github.com/AndreyChurkin/BeautifulFigures)), alongside the Windows-native Inkscape MCP (`inkmcp`) pipeline.

---

## Part I: The 8 Principles of Beautiful Figures for Research Papers

### 1. The Overall Quality — Vector Graphics Always
- **Problem**: Saving plots with `plt.savefig("fig.png")` uses Matplotlib's default 640×480 pixel raster canvas at 100 DPI. When placed in a manuscript and zoomed by reviewers, text and curves appear blurry and pixelated.
- **Solution**:
  - Export every figure in **vector formats**: `.pdf` (for LaTeX/Overleaf inclusion) and `.svg` (for post-processing in Inkscape).
  - Set `pdf.fonttype = 42`, `ps.fonttype = 42`, and `svg.fonttype = 'none'` so fonts are embedded as TrueType outlines in PDF and preserved as editable `<text>` elements in SVG (rather than converted into hundreds of uneditable glyph paths).
  - Keep `.png` (at `dpi=300`) strictly for quick README previews or slide decks.

### 2. Readability — Typography & The IEEE Two-Column Scaling Equation
- **Problem**: Default fonts (`DejaVu Sans` at 10 pt) look informal and shrink to an unreadable ~3–4 pt when a large figure is placed inside a single column of a two-column IEEE journal.
- **The Scaling Math**:
  - Standard IEEE Transactions column width: $\text{columnwidth} \approx 3.5\text{ in}$ ($88.9\text{ mm}$).
  - Typical single-column figure width: $0.9 \times \text{columnwidth} \approx 3.15\text{ in}$ ($80\text{ mm}$).
  - Standard double-column span width: $\text{textwidth} \approx 7.16\text{ in}$ ($182\text{ mm}$).
  - If your Python script creates `fig, ax = plt.subplots(figsize=(10, 10))` and LaTeX scales it to $0.9\text{ columnwidth}$, the linear scaling ratio is:
    $$s = \frac{3.15\text{ in}}{10.0\text{ in}} = 0.315 \quad (\text{or } \approx 0.40 \text{ after margin cropping})$$
  - Thus, a **`20 pt`** font in Matplotlib renders at **$20 \times 0.40 = 8\text{ pt}$** in the compiled IEEE PDF—matching the minimum 8 pt font size required by IEEE Transactions for figure captions and labels.
- **Font Selection**:
  - Set `font.family` explicitly before creating axes (`'Courier New'`, `'Times New Roman'`, `'STIXGeneral'`, or `'Arial'`).
  - Synchronize `font.size`, `axes.titlesize`, `axes.labelsize`, `xtick.labelsize`, `ytick.labelsize`, `legend.fontsize`, and `figure.titlesize`.

### 3. Controlled Proportions & The Center-Median `zoom_out` Algorithm
- **Problem**: Matplotlib's default `ax.autoscale()` computes axis limits from *every* plotted artist—including long linear/polynomial regression lines (`x_range = np.linspace(min - 2, max + 2)`). This pushes the actual data cluster into a tiny region in the middle of the canvas. Furthermore, arbitrary aspect ratios make side-by-side manuscript figures look mismatched.
- **Solution**:
  1. Fix `figsize=(10, 10)` and lock axis geometry with `ax.set_aspect('equal', adjustable='datalim')`.
  2. Track only the **primary data coordinates** (`all_data_x_to_plot`, `all_data_y_to_plot`) while plotting in loops.
  3. Compute the bounding-box midpoint (`x_median`, `y_median`) and the maximum span across both axes, then add a controlled `zoom_out` margin:
     ```python
     x_min, x_max = min(all_data_x_to_plot), max(all_data_x_to_plot)
     x_median = (x_min + x_max) / 2
     x_range = x_max - x_min

     y_min, y_max = min(all_data_y_to_plot), max(all_data_y_to_plot)
     y_median = (y_min + y_max) / 2
     y_range = y_max - y_min

     plotting_range = max([x_range, y_range]) + zoom_out

     ax.set_xlim(x_median - plotting_range / 2, x_median + plotting_range / 2)
     ax.set_ylim(y_median - plotting_range / 2, y_median + plotting_range / 2)
     ```
  4. **Never call `plt.tight_layout()`** when preserving locked square geometry or 3D box aspect ratios, as `tight_layout()` mutates the subplot box dimensions unpredictably.

### 4. Simplify, Declutter & Visual Hierarchy
- **Grid Lines**:
  - Default grids are either absent (making quantitative lookup hard) or heavy and opaque (competing with the data).
  - Use a two-tier subtle grid placed strictly behind all data artists:
    - Major grid: `linewidth=0.75, alpha=0.25, linestyle='-'`
    - Minor grid (`ax.minorticks_on()`): `linewidth=0.25, alpha=0.15, linestyle='-'`
    - Always call `ax.set_axisbelow(True)`.
- **Layer Ordering (`zorder`) & Marker Framing**:
  - By default, lines plotted after `ax.scatter()` draw *on top of* the scatter points.
  - Set `zorder=3` on primary `ax.scatter()` calls (`s=90, linewidths=1.5` with a darker border `edgecolors`) and `zorder=2` on regression/fit `ax.plot()` calls (`linewidth=2.6`).
- **Legend Optimization (STRICT RULE: NO LEGENDS INSIDE THE FIGURE — OUTSIDE ALWAYS)**:
  - **Never** place a legend inside the axes/plotting area (`loc='best'`, `'upper right'`, `'upper left'`, `'lower right'`, `'lower left'` without an external `bbox_to_anchor` are strictly forbidden). In-canvas legends occlude data or gridlines and break visual symmetry.
  - Always place the legend **outside** the figure canvas—ideally as a frameless horizontal bar above the plot (`bbox_to_anchor=(0.5, 1.10), loc='upper center', ncol=4, frameon=False`) or below the plot (`bbox_to_anchor=(0.5, -0.15), loc='upper center', frameon=False`).
  - Suppress duplicate regression labels across classes using `label="LR" if regression_flag == 0 else "_nolegend_"`.
  - Reorder handles (`desired_order = [0, 3, 1, 2]`) so data series and model types group logically.

### 5. Color Harmony & Perceptual Integrity
- **Why Default Colors Fail**:
  - Matplotlib's default `tab10` pairs saturated blue and orange, followed by traffic-light green and red. Readers subconsciously interpret red as "error/infeasible/bad" and green as "optimal/good" even when they represent neutral categories.
  - Rainbow colormaps (`jet`, `rainbow`, `gist_rainbow`, `hsv`) create artificial sharp bands where luminance reverses direction, completely destroying structure when printed in black-and-white.
- **Recommended Palettes**:
  | Palette | Use Case | Hex Codes |
  | :--- | :--- | :--- |
  | **Purple–Grey–Teal (Fills)** | 2–3 categorical classes | `#9671bd` (Purple), `#7e7e7e` (Grey), `#77b5b6` (Teal) |
  | **Purple–Grey–Teal (Borders)** | Marker edges / emphasis | `#6a408d` (Dark Purple), `#4e4e4e` (Dark Grey), `#378d94` (Dark Teal) |
  | **Neutral Fit Line** | Regression / baseline | `#8a8a8a` (Neutral Grey) or `#7f9fa1` (Seafoam Grey) |
  | **Purple–Orange Sunset** | Continuous surface / heatmap | Base `#2d0b4e` $\rightarrow$ `#a11477` $\rightarrow$ `#c1246b` $\rightarrow$ `#e13661` $\rightarrow$ `#fd4c55` $\rightarrow$ `#ff6f4b` |
  | **Dark Teal Canvas** | Dark-mode 3D surface | Center `#12302f`, Edge `#071414`, Ink `#f2f2f0`, Grid `#2c4a48` |

### 6. Message, Story & Honest 3D Surface Visualization
- **Avoid 3D and Pie Charts by Default**: 3D perspective foreshortening and occlusion make it hard to read exact values. Only use 3D when showing a genuinely 3-dimensional manifold (e.g., `Day × Hour-of-Day × Price` over a full year) where the dramatic contrast between a calm plateau and sharp spikes *is* the story.
- **Fixing Matplotlib's 3D Surface Artifacts (`02_3d_price_surface`)**:
  1. **`FACET_COLOUR_RULE = "max"` vs. `"mean"`**:
     By default, `ax.plot_surface` colors each quadrilateral facet by the **mean** of its 4 corner Z-values. For a narrow 1-hour price spike, averaging with 3 calm neighbor corners dilutes the facet color by up to 75%, so the spike *never* takes the peak color shown on the colorbar! Computing `face_value = np.maximum.reduce([Z[:-1, :-1], Z[1:, :-1], Z[:-1, 1:], Z[1:, 1:]])` and passing `facecolors=cmap(norm(face_value))` guarantees that true data peaks match the colorbar maximum.
  2. **`USE_SHADES = False` (`shade=False`)**:
     Matplotlib's directional 3D lighting darkens facets based on surface normal angles, breaking the 1:1 correspondence between surface color and the colorbar.
  3. **Standalone `ScalarMappable` for the Colorbar**:
     Passing explicit `facecolors` bypasses `plot_surface`'s internal scalar mapping; create `mappable = cm.ScalarMappable(norm=color_norm, cmap=price_cmap)` for `fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.12)`.
  4. **Box Zoom & Clean Panes**:
     Use `ax.set_box_aspect(None, zoom=1.1)` to fill the canvas, set `axis.pane.set_facecolor((0, 0, 0, 0))` for transparent panes, and omit the redundant Z-axis label when the colorbar is labeled.

### 7. Consistent Style Across the Manuscript
- Every figure in a paper should look like it was drawn by the same hand. Keep a single shared style module (`ieee_vector_style.py`) per project.

### 8. Allocate Time for Iteration (The 4 Stages)
- Plan for multiple rounds across Python $\rightarrow$ Inkscape $\rightarrow$ LaTeX two-column compile $\rightarrow$ refinement.

---

## Part II: Inkscape MCP (`inkmcp`) Vector Manipulation Reference

On Windows, the `inkmcp` server (`C:\Users\andre\AppData\Roaming\inkscape\extensions\inkmcp_repo\inkmcp\inkscape_mcp_server.py`) operates directly on `.svg` files using Inkscape's `inkex` library and headless CLI (`C:\Program Files\Inkscape\bin\inkscape.exe`), without requiring Linux `gdbus`.

### Available MCP Tools

1. **`set_active_svg(svg_path: str, create_if_missing: bool = True, width: str = "1000", height: str = "1000")`**
   - Sets the target `.svg` file that subsequent `inkscape_operation` calls read and modify in-place.
   - Always call this first before running `inkscape_operation` on a Matplotlib-generated `.svg` figure.

2. **`inkscape_operation(command: str)`**
   - Executes `ElementCreator` from `inkscape_mcp.py` against the active `.svg` file and saves changes back to disk automatically.
   - **Querying the SVG**:
     - `"get-info"` — Returns `dimensions`, `viewBox`, and `elementCounts` (`path`, `text`, `g`, `use`, etc.).
     - `"get-info-by-id id=axes_1"` — Returns attributes and parsed CSS styles of a specific SVG element.
     - `"export-document-image format=png max_size=800 area=page return_base64=true"` — Renders the current SVG via Inkscape CLI and returns an `ImageContent` preview.
   - **Adding Vector Annotations**:
     - `"text id=annotation_1 x=320 y=180 text='Optimal Regime' font-family='Courier New' font-size=20 fill=#6a408d"`
     - `"rect id=highlight_box x=280 y=140 width=180 height=90 fill=none stroke=#378d94 stroke-width=2 stroke-dasharray='6,4'"`
   - **Programmatic DOM Surgery (`execute-code`)**:
     - Uses `svg` (the root `inkex.SvgDocumentElement`) and `get_element_by_id(id)`:
       ```text
       execute-code code='el = get_element_by_id("text_1"); el.set("style", "font-family:Courier New;font-size:20px;fill:#4e4e4e") if el else None'
       ```

3. **`inspect_vector_figure(file_path: str)`**
   - Audits `.svg` or `.pdf` files for **true vector purity**:
     - Counts vector primitives (`<path>`, `<text>`, `<tspan>`, `<g>`, `<line>`, `<polyline>`, `<polygon>`, `<circle>`, `<rect>`, `<use>`) vs. embedded raster `<image>` elements.
     - Lists all detected font families and font sizes so you can verify IEEE 8 pt scaled readability.
     - Flags a warning if raster `<image>` elements are present in a 2D figure.

4. **`inkscape_cli_export(input_path: str, output_path: str, export_type: str = "pdf", text_to_path: bool = False, export_area_drawing: bool = False, actions: str = "")`**
   - Invokes `C:\Program Files\Inkscape\bin\inkscape.exe` headlessly to export `.svg` $\leftrightarrow$ `.pdf` / `.eps` / `.emf` / `.png`, optionally executing Inkscape action chains (e.g. `actions="select-all;FitCanvasToDrawing;export-do"`).
