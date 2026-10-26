---
name: beautiful-research-tables
description: >-
  Design, refactor, and audit publication-grade LaTeX tables following the
  canonical principles of TeX.SE #112343 ("Beautiful table samples"), TeX.SE
  #681429 ("How to make better looking tables for a publication"), Simon Fear's
  booktabs manual, and IEEE Transactions two-column typesetting standards.
  Enforces strict booktabs rules (no vertical pipes or \hline), bans \resizebox
  font distortion on tabular environments, aligns numbers with siunitx S
  columns, hoists units into hierarchical \cmidrule(lr) headers, groups rows
  with \addlinespace, and verifies tables via an automated Python linter. Use
  this skill whenever creating, styling, refactoring, or auditing LaTeX tables
  in research manuscripts.
---

# Beautiful Research Tables (`beautiful-research-tables`)

This skill implements a complete workflow for creating, refactoring, and auditing publication-grade LaTeX tables for academic manuscripts (such as `IEEEtran` two-column journals and conference proceedings). It synthesizes the canonical community standards from **TeX.SE #112343** (*"Beautiful table samples"*), **TeX.SE #681429** (*"How to make better looking tables for e.g. a publication"*), **Simon Fear's `booktabs` guidelines**, and **Lapo Mori's `siunitx` / `threeparttable` discipline**.

---

## 1. The 4-Stage Iterative Table Workflow

Never treat a raw `\begin{tabular}{|c|c|c|}` or a `\resizebox{\textwidth}{!}{...}` wrapper as acceptable for publication. Every manuscript table follows four stages:

1. **Stage 1 — Audit & Structural Design**: Run [`scripts/audit_latex_tables.py`](./scripts/audit_latex_tables.py) on the target `.tex` file to identify anti-patterns (`\resizebox`, vertical pipes `|`, `\hline`, repeated body units, unaligned numeric columns). Decide whether the table belongs in a single column (`table`, width `3.5 in` / `\columnwidth`) or double column (`table*`, width `7.16 in` / `\textwidth`).
2. **Stage 2 — Typeset with `booktabs` + `siunitx` (or `tabularray`)**: Use `\toprule`, `\midrule`, `\cmidrule(lr){a-b}`, and `\bottomrule` for horizontal structure; `S[table-format=...]` for decimal alignment; hoisted units in headers; and `\addlinespace` to separate logical row cohorts.
3. **Stage 3 — Fit & Compile in the Target Manuscript**: Replace `\resizebox` with native typographic sizing (`\footnotesize` or `\scriptsize`, calibrated `\tabcolsep`, or `tabular*` with `@{\extracolsep{\fill}}`). Compile the manuscript PDF and inspect the `.log` file for `Overfull \hbox` warnings.
4. **Stage 4 — Verify Data Integrity, Visual Balance & Page Budget**: Re-run [`scripts/audit_latex_tables.py`](./scripts/audit_latex_tables.py) to confirm zero P0/P1 violations, verify that all numerical values, `\label{...}`, and `\cite{...}` keys remain 100% intact, and verify the manuscript page count via `pypdf`.

---

## 2. The 8 Golden Principles of Publication Tables

### Principle 1: Zero Vertical Rules & Minimalist `booktabs` Horizon
- **NEVER use vertical lines (`|` or `\vline`)**: Vertical rules create visual "chart junk" (boxed prisons around cells) and disrupt horizontal eye scanning across rows. Use whitespace (`\tabcolsep` or `@{\extracolsep{\fill}}`) to separate columns.
- **NEVER use `\hline`, `\cline`, or double rules (`\hline\hline`)**: Standard `\hline` has zero vertical padding, causing glyphs to collide with rules.
- **Use only the four `booktabs` rules**:
  1. `\toprule` — heavy rule (`0.08em`) at the very top of the table.
  2. `\cmidrule(lr){a-b}` — light sub-rule (`0.03em`), **always trimmed with `(lr)`** (or `(l)` / `(r)`) so adjacent spanner rules never merge into a single deceptive line.
  3. `\midrule` — medium rule (`0.05em`) separating the header block from the data body (or separating the body from a summary/total footer row).
  4. `\bottomrule` — heavy rule (`0.08em`) closing the bottom of the table.

### Principle 2: Strictly Ban `\resizebox` and `\scalebox` on Tables
- **Why `\resizebox` is a credibility killer**: Wrapping a `tabular` in `\resizebox{\columnwidth}{!}{...}` or `\resizebox{0.98\textwidth}{!}{...}` scales the font size, glyph stroke widths, and `booktabs` rule weights by an arbitrary ratio (e.g., `6.3 pt` in one table and `8.7 pt` in the next). Reviewers and typesetters immediately spot the mismatched typography.
- **How to fit wide tables without `\resizebox`**:
  1. **Use explicit LaTeX font commands**: Use `\footnotesize` (8 pt in `IEEEtran`, the standard IEEE table size) or `\scriptsize` (7 pt for dense multi-column benchmarks).
  2. **Tune `\tabcolsep` locally**: Inside the `table` environment, set `\setlength{\tabcolsep}{3.5pt}` (default is `6pt`) or use `tabular*` to stretch exact inter-column glue:
     ```latex
     \setlength{\tabcolsep}{0pt}
     \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}} l S[table-format=2.1] S[table-format=2.1] @{}}
     ```
  3. **Wrap multi-word headers** onto two lines and hoist repeated units out of data cells.
  4. **Transpose the table** if there are more than 10 metrics for only 3–4 models/methods.

### Principle 3: Align Numbers on Decimals (`siunitx`)
- **Never center (`c`) or left-align (`l`) quantitative columns** when numbers vary in magnitude, sign, or decimal places.
- **Alignment rules by column data type**:
  - **Row labels / Categorical names / Text**: Left-aligned (`l`), never centered.
  - **Quantitative numbers (metrics, latencies, p-values, weights)**: Decimal-aligned using `siunitx` `S[table-format=...]` (or right-aligned `r` when all cells share identical fixed decimal precision).
  - **Short fixed codes / Binary indicators (`\checkmark`, `--`, `Yes`/`No`, integer counts of equal width)**: Centered (`c`).
- **Protecting Headers & Bold Best Results in `S` Columns**:
  - Wrap text headers in curly braces `{Header}` so `siunitx` does not parse words as numbers.
  - To bold the best metric in an `S` column without breaking decimal alignment, configure `siunitx` in the preamble (or locally):
    ```latex
    \usepackage{siunitx}
    \usepackage{etoolbox}
    \robustify\bfseries
    \sisetup{
        detect-all,
        mode = text,
        group-separator = {,},
        group-minimum-digits = 4
    }
    ```
    Then write `\bfseries 98.4` directly inside the `S` cell (without outer braces around the number).

### Principle 4: Hoist Units & Strip Repetitive Inline Symbols
- **Never repeat units (`%`, `ms`, `s`, `GB`, `dB`, `\$`) on every row of a column.** Repeating `100.0%`, `90.9%`, `93.9%` down 15 rows wastes horizontal width and forces visual clutter.
- **Hoist the unit into the column header or sub-header**:
  - Bad: `\textbf{Coverage}` with body cells `98.4\%`, `96.7\%`
  - Good: `\textbf{Coverage (\%)}` with body cells `98.4`, `96.7`
- **Consistent Decimal Precision**: Within any single column, format all measurements to the same number of decimal places (e.g., `90.0` instead of `90`, `0.120` instead of `0.12`).

### Principle 5: Hierarchical Spanner Headers (`\multicolumn` + `\cmidrule(lr)`)
- When several columns share a domain, metric family, or experimental cohort, group them under a top-level spanner header rather than repeating prefixes in every column title:
  - Bad: `DSE Pass | LIH Coverage | Total Coverage | LLM Mean Latency (ms) | LLM Med Latency (ms)`
  - Good: Group `DSE`, `LIH`, `Total` under `\multicolumn{3}{c}{\textbf{Coverage (\%)}}` with `\cmidrule(lr){4-6}`, and `Mean`, `Median` under `\multicolumn{2}{c}{\textbf{LIH Latency (ms)}}` with `\cmidrule(lr){7-8}`.
- Remove outer padding at the left and right edges of the table using `@{}` in the column specification (`\begin{tabular}{@{} l ... @{}}`) so the text aligns flush with the ends of `\toprule` and `\bottomrule`.

### Principle 6: Group Rows with Whitespace (`\addlinespace`), Not Extra Rules
- **Do not put a `\midrule` between every row or every minor group.** The classic `booktabs` rule is three horizontal lines total (`\toprule`, one `\midrule` below the header, `\bottomrule`), plus optional `\cmidrule(lr)` in the header and at most one `\midrule` before an overall summary/average row.
- **Use `\addlinespace[3pt]` (or `[0.4em]`)** to visually separate logical row cohorts (e.g., baseline architectures vs. proposed method, deterministic vs. generative arms, or training vs. held-out splits).
- **Avoid heavy zebra striping (`\rowcolors`)**: Colored alternating rows add ink without information. Only use subtle row shading (`\rowcolor{gray!8}`) in dense wide tables ($>10$ columns and $>15$ rows) where horizontal eye tracking across a double-column span would otherwise drift.

### Principle 7: Self-Contained Top Captions & `threeparttable` Notes
- **Captions Always Above Tables**: Unlike figures (where captions sit below), table captions sit **above** `\toprule` because readers scan tables from top to bottom.
- **Stand-Alone Table Notes Below `\bottomrule`**: Every abbreviation, statistical test notation ($p$-value method, bootstrap resample count, confidence interval type), cohort size ($N$), and boldface convention ("Best results in **bold**; second-best \underline{underlined}") must be explained in a note block flush-left beneath `\bottomrule` using `threeparttable` or `\parbox{\columnwidth}{\footnotesize ...}`.

### Principle 8: `IEEEtran` Page-Budget & Zero-Corruption Guardrails
- **Compact `booktabs` Spacing for Tight Page Budgets**: When a manuscript operates under a strict page ceiling (such as 13 pages in IEEE TIFS), preserve `booktabs` proportions while tightening vertical padding locally inside the `table` environment instead of crushing `\arraystretch` below `0.90`:
  ```latex
  \setlength{\aboverulesep}{0.4ex}
  \setlength{\belowrulesep}{0.5ex}
  \renewcommand{\arraystretch}{0.95}
  ```
- **Zero Data or Citation Corruption**: Refactoring a table's layout must **never** alter a single digit, metric value, `\label{...}`, or `\cite{...}` key.

---

## 3. Canonical Before & After Transformation

### Before (Common Anti-Patterns)
```latex
\begin{table*}[!t]
    \centering
    \renewcommand{\arraystretch}{0.85}
    \caption{Cross-Model Benchmark Summary}
    \label{tab:bad_example}
    \resizebox{0.98\textwidth}{!}{
    \begin{tabular}{|l|c|c|c|c|}
        \hline
        \textbf{Scenario} & \textbf{Models} & \textbf{LIH Coverage} & \textbf{Total Coverage} & \textbf{LLM latency, mean / med (ms)} \\
        \hline\hline
        GPU Attacker & 11 & 90.9\% & 98.4\% & 1,739 / 339 ms \\
        Web Exploitation & 5 & 90.7\% & 98.3\% & 1,177 / 291 ms \\
        DB Exploitation & 4 & 86.2\% & 96.7\% & 1,134 / 550 ms \\
        \hline
    \end{tabular}
    }
\end{table*}
```

### After (`booktabs` + `siunitx` + Hierarchical Spanners, Zero `\resizebox`)
```latex
\begin{table}[!t]
    \centering
    \footnotesize
    \setlength{\tabcolsep}{4.5pt}
    \caption{Cross-Model Benchmark Summary by Attack Scenario}
    \label{tab:good_example}
    \begin{tabular*}{\columnwidth}{
        @{\extracolsep{\fill}}
        l
        S[table-format=2.0]
        S[table-format=2.1]
        S[table-format=2.1]
        S[table-format=4.0]
        S[table-format=3.0]
        @{}
    }
        \toprule
        & & \multicolumn{2}{c}{\textbf{Coverage (\%)}} & \multicolumn{2}{c}{\textbf{LIH Latency (ms)}} \\
        \cmidrule(lr){3-4} \cmidrule(lr){5-6}
        \textbf{Scenario} & {\textbf{Models}} & {\textbf{LIH}} & {\textbf{Total}} & {\textbf{Mean}} & {\textbf{Median}} \\
        \midrule
        GPU Attacker     & 11 & \bfseries 90.9 & \bfseries 98.4 & 1739 & 339 \\
        Web Exploitation &  5 & 90.7           & 98.3           & 1177 & \bfseries 291 \\
        DB Exploitation  &  4 & 86.2           & 96.7           & \bfseries 1134 & 550 \\
        \bottomrule
    \end{tabular*}
    \vspace{2pt}
    \parbox{\columnwidth}{\scriptsize\emph{Note:} Coverage values report unweighted scenario means across $N=20$ evaluated model-scenario runs. Best performance per column is shown in \textbf{bold}.}
\end{table}
```

---

## 4. Executable Helpers, Templates & References

- **Automated LaTeX Table Linter**: [`scripts/audit_latex_tables.py`](./scripts/audit_latex_tables.py)
  Scans `.tex` files or directories for `\resizebox` around tables, vertical pipes `|`, `\hline`/`\cline`, untrimmed `\cmidrule`, repeated body units (`\%`, `ms`), and caption placement:
  ```powershell
  python .agents/skills/beautiful-research-tables/scripts/audit_latex_tables.py Paper/
  ```
- **Ready-to-Use LaTeX Templates**: [`examples/beautiful_ieee_tables.tex`](./examples/beautiful_ieee_tables.tex)
  Contains four complete, copy-pasteable templates:
  1. Single-column `IEEEtran` quantitative table (`tabular*` + `booktabs` + `siunitx`).
  2. Double-column `table*` benchmark with hierarchical `\cmidrule(lr)` spanners and 95% confidence intervals.
  3. Modern `tabularray` (`tblr`) scientific table with native `booktabs` and `siunitx` libraries.
  4. Qualitative/architectural feature matrix (`\ding{51}` / `--`) with `\addlinespace` cohort grouping.
- **Deep-Dive Typographic Reference**: [`references/tex_se_112343_681429_guide.md`](./references/tex_se_112343_681429_guide.md)
  Detailed guide to TeX.SE #112343 and #681429 patterns, confidence interval formatting, `siunitx` v3 syntax, and `IEEEtran` float placement optimization.
