# Project Instructions & LaTeX Safety Rules: NCAD-CS

Core project facts, hard constraints ($\le 13$-page IEEE TNNLS ceiling, no supplement, honest null-result reporting, loss/code parsimony, anti-AI writing rules, and readability invariants) are defined in [`AGENTS.md`](AGENTS.md). In addition, enforce the following LaTeX safety rules when editing `.tex` and `.bib` files in `paper/NCAD_CS/`:

## LaTeX Zero Syntax Corruption Policy (`paper/NCAD_CS/`)
- **Preserve LaTeX Commands & Structure**: Keep all macros, environments (`\section`, `\begin{table}`, `\caption`, etc.), and structural comments intact.
- **Protect Citations & Cross-References**: Keep `\cite{...}`, `\ref{...}`, `\eqref{...}`, and `\label{...}` tags intact; never invent citation keys.
- **Preserve Mathematical Notation**: Do not alter inline or display math unless explicitly updating a formulation or wrapping long multi-term equations with `aligned` to eliminate `Overfull \hbox` margin spills.
- **Safe Editing Protocol**: Always run `C:/Users/andre/anaconda3/envs/alpha/python.exe scripts/build_and_verify_manuscript.py` after any manuscript edit to verify standalone reconstruction, zero unresolved references, $\le 13$ pages, zero `Overfull \hbox` spills, zero undefined acronyms, and zero writing violations per `.agents/skills/avoid-ai-writing/SKILL.md`.
