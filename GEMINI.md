# Project Instructions & Workspace Rules: NCAD-CS

This workspace contains both the core Python implementation for **NCAD-CS** (TS-JEPA, Flow Matching, VICReg) and its accompanying academic paper in `paper/NCAD_CS/`.

---

## 1. Academic Paper & LaTeX Rules (`paper/`)

When inspecting, reviewing, editing, or generating content for `.tex` and `.bib` files in `paper/`:

### A. Zero Syntax Corruption Policy
- **Never alter or break LaTeX commands**: Preserve all macros, environments, and formatting commands (e.g., `\section`, `\begin{algorithm}`, `\begin{table}`, `\caption`, etc.).
- **Protect Citations & Cross-References**: Keep `\cite{...}`, `\citep{...}`, `\citet{...}`, `\ref{...}`, `\eqref{...}`, and `\label{...}` tags completely intact. Do not invent or alter citation keys.
- **Preserve Mathematical Notation**: Do not reformat or modify inline math (`$...$`) or display math (`\begin{equation}...\end{equation}`, `\begin{align}...\end{align}`) unless specifically requested to correct a mathematical formulation.
- **Maintain Comments & Structure**: Retain structural comments (`% ...`), section markers, and layout spacing.

### B. High-Impact Academic Tone (Top-Tier ML Venues: NeurIPS / ICLR / ICML / KDD)
- **Active & Direct Voice**: Favor active, authoritative constructions over passive or verbose phrasing (e.g., replace *"it can be observed that the model achieves"* with *"the model achieves"*).
- **Conciseness & Precision**: Eliminate empty fillers, redundancies, and weak hedges (e.g., *"in order to"*, *"widely considered to be"*, *"serves as a way to"*).
- **Claims Grounded in Evidence**: Ensure empirical claims strictly reflect the benchmark metrics (AUROC, PR-AUC, F1-Score) and theoretical foundations (VICReg, Flow Matching, EVT/SPOT).

### C. Safe Editing Protocol
- When asked to polish or revise a section, first provide a concise diagnosis of improvements (clarity, narrative flow, conciseness) and present the proposed text changes or diff clearly before applying direct modifications.

---

## 2. Codebase & Implementation Guidelines (`src/`, `tests/`)

- **Documentation & Integrity**: Preserve existing comments, docstrings, and type annotations unless specifically updating them.
- **Consistency**: Adhere to the established TS-JEPA modular architectures and clean separation of concerns across models, trainers, datasets, and utilities.

---

## 3. Mandatory Writing Standard: Eliminate "AI-isms" (`docs/SKILL.md`)

Whenever writing, editing, reviewing, or generating any prose, paper sections, documentation, reports, or explanations in this workspace, **ALWAYS strictly enforce the anti-AI-writing rules defined in [`docs/SKILL.md`](docs/SKILL.md)**:

### A. Tone Calibration & Cadence
- **Direct, Specific, & Human**: Demonstrate confidence through concrete evidence, not rhetorical assertion.
- **Vary Sentence Cadence**: Mix short, punchy statements with longer, explanatory structures. Avoid monotonous, uniform paragraph lengths.
- **Be Concrete**: Ground arguments in exact metrics, formulas, or hardware/benchmark details rather than vague abstractions.

### B. Prohibited AI Patterns & Banned Tells
- **P0 Credibility Killers (Zero Tolerance)**:
  - No chatbot meta-chatter ("I hope this helps", "Great question!").
  - No cutoff disclaimers ("As of my knowledge cutoff...").
  - No vague, unsourced attributions ("Researchers have shown", "Experts believe").
  - No significance inflation of routine engineering or empirical findings.
- **P1 Word & Phrase Blacklist (Never Use in Drafts or Edits)**:
  - Banned words: *delve, leverage, harness, robust, pivotal, multifaceted, testament, beacon, tapestry, foster, elevate, underscore, nuanced, intricate, seamlessly, poised, holistic, paramount*.
  - Banned formulaic openers & transitions: *"In the rapidly evolving world of..."*, *"Not only X, but also Y"*, *"Moreover"*, *"Furthermore"*, *"Additionally"*, *"In conclusion"*, *"It is worth noting that"*.
  - Banned copula avoidance: Avoid *"serves as"*, *"boasts"*, *"features"* when a direct *"is"* or active verb is clearer.
  - No hedge-stacked predictions (*"could potentially"*, *"may eventually"*).
  - No bare noun-phrase bullet lists (always use complete, verb-driven clauses).
  - No synonym cycling within the same paragraph.

### C. Content Boundaries
- Apply these checks strictly to editable prose.
- **Never alter** quoted material, code blocks, tables, or raw experimental data carried in `.tex` or `.md` files.

