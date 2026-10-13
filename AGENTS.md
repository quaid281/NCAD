# AGENTS.md — NCAD_CS

## Hard Rules

- **Page limit: the manuscript must stay at or below 12 pages.** Target venue
  is IEEE Transactions on Neural Networks and Learning Systems (TNNLS).
  Standard full paper length is 10 pages; over-length charge is $200/page
  beyond 10 pages up to a 15-page cap. The author budget allows at most 2
  extra pages, setting a hard ceiling of 12 pages total. Before committing
  manuscript changes, compile `paper/NCAD_CS/NCAD_CS_standalone.tex` and verify
  page count. If over budget, condense elsewhere — tighten prose, shrink
  tables, or prune secondary discussions. Everything must fit and be addressed
  directly in the main manuscript.
- **No supplementary papers (fully self-contained manuscript):** There must be
  no companion supplement paper (no `NCAD_CS_supplement.tex` or external
  S-sections). Reviewers evaluate the main paper alone; central claims, causal
  vs. buffered evaluation tables, drift analyses, baseline comparisons
  (including NCAD), and core ablations must never depend on an external
  supplement. All claims, evidence, methodology, and proofs must be addressed
  directly within the main manuscript (`paper/NCAD_CS/NCAD_CS_standalone.tex`
  built from `sections/`) while strictly respecting the 12-page ceiling. Never
  defer reviewer-facing findings to a supplement. Non-manuscript artifacts and
  raw diagnostic logs belong in repository `reports/`, not in a submitted
  supplement document.
- **Never let manuscript claims exceed the measured evidence.** Latent
  predictive residuals reduce false alarms; the physics-inspired
  regularizers show no reliable aggregate benefit — report them as honest
  null results, not advantages. The held-out calibration benchmark
  (65/17.5/17.5 + 64-step isolation gap, standard SPOT u=0.98, q=1e-3) is
  the primary evidence; buffered ("smear") and strictly causal ("trailing")
  mappings must always be reported separately and labeled.
- **Do not commit large CSV artifacts** (gitignored) except the reviewer-
  facing diagnostics whitelisted in `.gitignore` (`stream_manifest.csv`,
  `evt_tail_diagnostics.csv`, `lp_filterbank_*.csv`,
  `heldout_reynolds_covmodes.csv`).
- **Always scan the manuscript with `docs/SKILL.md`:** Before committing any
  manuscript or documentation changes, always audit all paper sections
  against `docs/SKILL.md` (`C:/Users/andre/OneDrive/Desktop/NCAD_CS/docs/SKILL.md`)
  via `C:/Users/andre/anaconda3/envs/alpha/python.exe scripts/scan_ai_writing.py`
  (also enforced automatically by `scripts/build_and_verify_manuscript.py`).
  Zero tolerance for P0 credibility killers (including codebase relative paths in
  manuscript prose: `reports/`, `scripts/`, `src/`, `.csv`, `.py`, `.txt`; readers
  lack repository access), P1 word-list violations (`delve`,
  `leverage`, `harness`, `robust`, `pivotal`, `multifaceted`, `testament`,
  `beacon`, `tapestry`, `foster`, `elevate`, `underscore`, `nuanced`,
  `intricate`, `seamlessly`, `poised`, `holistic`, `paramount`, `cutting-edge`,
  `paradigm`), formulaic transitions (`Moreover`, `Furthermore`, `Additionally`,
  `In conclusion`), copula avoidance (`serves as`, `boasts`, `features` as verb),
  and hedge stacking (`could potentially`). Em-dash rate in prose must remain
  at or below 1.0 per 1,000 words.
- **Git:** work on `dev`; the user syncs/pushes `main`. Use the standard
  Devin commit trailer (`Generated with [Devin]` / Co-Authored-By).

## IEEE TNNLS Guidelines & Article Structure

Guidelines references:
- Submission guidelines: https://cis.ieee.org/publications/t-neural-networks-and-learning-systems/tnnls-guidelines-for-paper-submission
- Article structure: https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-the-text-of-your-article/structure-your-article/

Key requirements:
- **Page Limits (Full Papers):** Standard length is 10 pages. Overlength cap
  is 15 pages ($200 per page over 10 pages). Author budget allows at most 2
  extra pages, setting a hard ceiling of 12 pages.
- **Title:** Specific, concise, and descriptive. Avoid assertions of novelty
  such as "new" or "novel".
- **Abstract:** Exactly one paragraph, up to 250 words. Self-contained:
  no math symbols, equations, footnotes, or citations. Include 3–5 standardized
  keywords from the IEEE Thesaurus.
- **First Footnote (Unnumbered):** Structured into three mandatory paragraphs:
  1. Dates (received, revised, accepted); corresponding author name;
     financial support and funding sources; disclosure of any prior
     conference presentation (with DOI) or thesis material.
  2. Author affiliations: department, institution, city, state/province,
     country, and corresponding author email address.
  3. IEEE-provided notice for color figures (omit supplementary materials
     notice; the manuscript is fully self-contained without a supplement).
- **Section Ordering:** Title $\rightarrow$ Abstract $\rightarrow$ Keywords
  $\rightarrow$ First Footnote $\rightarrow$ Introduction $\rightarrow$
  Methodology $\rightarrow$ Equations (typeset via IEEE guidelines)
  $\rightarrow$ Results (tables for numerical values, figures for trends)
  $\rightarrow$ Discussion $\rightarrow$ Conclusion (no inflated claims)
  $\rightarrow$ References (strictly supportive, no citation padding)
  $\rightarrow$ Acknowledgments.

## Environment

- **Python:** `C:/Users/andre/anaconda3/envs/alpha/python.exe` (the system
  `python` = Python 3.13 has no torch). Run all scripts with this
  interpreter.
- Data lives in `mTSBench_data/<dataset>/`; stream manifests in
  `reports/streams_*.txt`.

## Key Commands

```bash
# Rebuild, compile, and verify manuscript (builds standalone,
# scans for unresolved '??' / '[?]' markers, audits SKILL.md anti-AI rules, and enforces <= 12-page budget)
C:/Users/andre/anaconda3/envs/alpha/python.exe scripts/build_and_verify_manuscript.py

# Audit manuscript prose against anti-AI writing patterns (docs/SKILL.md)
C:/Users/andre/anaconda3/envs/alpha/python.exe scripts/scan_ai_writing.py

# Regenerate standalone manuscript from sections/
C:/Users/andre/anaconda3/envs/alpha/python.exe scripts/build_standalone.py

# Primary held-out benchmark (adds rows per stream x seed x model x
# mapping x calibrator; checkpoints CSV after each stream)
C:/Users/andre/anaconda3/envs/alpha/python.exe \
  scripts/run_multiseed_heldout_benchmark.py \
  --seeds 42 123 456 --streams reports/streams_19_nonghl.txt \
  --out reports/<out>.csv

# Aggregate/paired statistics (filter calibrator=='SPOT', mapping smear|trailing)
C:/Users/andre/anaconda3/envs/alpha/python.exe scripts/analyze_multiseed.py \
  --inputs reports/heldout_45stream.csv <more.csv> ...

# Stream manifest (dataset, K, lengths, role, actual seed coverage)
C:/Users/andre/anaconda3/envs/alpha/python.exe scripts/build_stream_manifest.py
```

## Project Facts (verified)

- 45 streams = 33 primary (19 non-GHL at 5 seeds {42,123,456,7,2024};
  14 GHL at 1–3 seeds, >1M obs each) + 12 case studies. 11 datasets.
- Independent statistical unit = telemetry stream; seeds averaged within
  stream before paired Wilcoxon tests; no equivalence/noninferiority
  claims — report effect sizes + 95% bootstrap CIs.
- Deployed pipeline uses *latent* coordinate saliency; the input-channel
  `CausalChannelSaliencyGate` is auxiliary (fault-type study only) and has
  an explicit `K<=1` identity short-circuit.
- The "Littlewood–Paley" front-end is a smooth rFFT-domain dyadic-shell
  split (exact amplitude PoU, err 1.2e-7) + per-shell causal convs —
  *less* frequency-localized than Haar/DB4 DWT bands; described as a
  multiscale front-end, not an exact LP decomposition.
- Reynolds `compute_observed_stress` supports `cov_mode` in
  {instant, temporal, batch}; estimator choice does not change the null
  regularizer result.
- SPOT overshoots its design risk primarily due to inter-block drift
  (tail diagnostics: KS rejects in only ~22–30% of cells while held-out
  FPR spans 0–0.67); empirical p99.5 ≈ SPOT reliability.
