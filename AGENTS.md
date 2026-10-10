# AGENTS.md — NCAD_CS

## Hard Rules

- **Page limit: the manuscript must stay under 14 pages.** Every revision
  round (reviewer responses, new tables/paragraphs/experiments) grows the
  paper; before committing manuscript changes, compile
  `paper/NCAD_CS/NCAD_CS_standalone.tex` and check the page count. If over
  budget, condense elsewhere — tighten prose, shrink tables, move detail to
  `reports/` artifacts — rather than exceeding the limit. Prefer adding
  evidence via supplementary artifacts (`reports/*.csv`, `scripts/`) cited
  in text over adding pages of in-paper detail.
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
- **Git:** work on `dev`; the user syncs/pushes `main`. Use the standard
  Devin commit trailer (`Generated with [Devin]` / Co-Authored-By).

## Environment

- **Python:** `C:/Users/andre/anaconda3/envs/alpha/python.exe` (the system
  `python` = Python 3.13 has no torch). Run all scripts with this
  interpreter.
- Data lives in `mTSBench_data/<dataset>/`; stream manifests in
  `reports/streams_*.txt`.

## Key Commands

```bash
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
