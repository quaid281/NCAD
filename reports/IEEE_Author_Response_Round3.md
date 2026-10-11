# Author Response — Round 3 (Draft)

Manuscript: "On Latent Predictive Residuals and Automated Threshold Calibration for
Multivariate Time-Series Anomaly Detection: An Empirical Investigation"

We thank the reviewer for the detailed and constructive assessment. We agree that the
submitted version established a promising empirical observation more strongly than a
reliable methodology, and we have restructured the revision accordingly: the central
claim is now an *empirical characterization* of latent predictive residuals — a
conservative, low-false-alarm operating regime — supported by matched-backbone controls,
a systematic calibration-failure study, matched-FPR operating-point comparisons, a
strictly causal deployment evaluation, and a complete statistical test map. Where the
evidence is null (physics-inspired regularizers, calibration reliability), the revised
manuscript says so explicitly.

Below we respond point by point. "Sec." / "Tab." refer to the revised manuscript;
"Supp." to the supplementary document; artifact paths refer to the released repository.

---

## Major Concern 1 — Architectural confounding vs. genuine latent-residual conditioning

**Response.** We agree the previous comparison confounded representation learning,
objective, and capacity. The revision adds a *matched-backbone study under the primary
held-out protocol* (65/17.5/17.5 chronological split, 64-step isolation gap, identical
SPOT u=0.98/q=1e-3 calibration, five seeds) with five arms on the shared HybridTCNEncoder
backbone (C=256 context, S=64 horizon, d=32 latent, F=48 filters, three dilated TCN
layers, identical AdamW budget of 20 epochs, batch 16, lr 1e-3):

| Arm | Objective | Isolates |
| --- | --- | --- |
| TCN-ObsRecon | reconstruct full 320-step window in observation space | reconstruction objective |
| TCN-ObsPred | forecast 64-step target from context in observation space | prediction objective |
| JEPA-CtxRecon | predict EMA encoding of the *context* (not future) | latent compression + EMA without prediction |
| JEPA-NoEMA | latent prediction with online (hard-copy) target | EMA target encoder |
| TS-JEPA | latent prediction with EMA target | proposed model |

All arms share input windows, standardization, threshold rules, and evaluation code.
Raw per-timestep score arrays are dumped for every arm, enabling the requested residual
diagnostics.

**New analyses reported (Sec. / Supp. Tab.):**

- Residual-distribution diagnostics per arm: tail quantiles (q90–q99.9), excess
  kurtosis, extremal clustering (runs-declustered exceedance count and extremal index),
  residual autocorrelation (lag-1…lag-64 ACF), and out-of-sample exceedance rate at the
  calibrated threshold — separating *lighter tails / better threshold conditioning* from
  *better anomaly-vs-nominal separation* (PR-AUC). [PENDING: reports/residual_diagnostics.csv]
- The prior 15-stream × 4-SNR controlled experiment is retained as Supp. Tab.
  `controlled_backbone` for noise-robustness context; the headline claim now rests on
  the primary-protocol matched arms above.

**Outcome (full coverage — 19 streams × 5 seeds, primary protocol):**
`reports/residual_diagnostics.csv` (2,220 cells) + Supp. Table `residual_diagnostics`.

- Observation-space arms produce statistically indistinguishable heavy tails regardless
  of direction (TCN-Obs-Recon vs. TCN-Obs-Pred: γ₂ 8.31/8.24, held-out FPR 0.303/0.310,
  extremal index θ ≈ 0.12). Latent arms compress tails ≈8× (γ₂ 1.0–1.6, θ ≈ 0.18) and
  cut FPR ≈3×: TS-JEPA −0.230 vs. TCN-Obs-Pred (Wilcoxon p=0.031) and −0.223 vs.
  TCN-Obs-Recon (p=0.058), with PR-AUC differences inconclusive (p ≥ 0.16).
- **Mechanism attribution (the reviewer's core question):** JEPA-CtxRecon — which never
  predicts — retains most of the conditioning (−0.203 vs. TCN-Obs-Pred, p=0.055; only
  −0.028 below TS-JEPA, p=0.23), and JEPA-NoEMA changes little (−0.033, p=0.17). The
  FPR reduction is therefore attributable primarily to *scoring in the contracted
  latent space*; prediction and the EMA target are secondary. The revised manuscript
  states this explicitly rather than attributing the effect to prediction per se —
  the conditioning is a representation property, separation requires the predictive
  objective.

## Major Concern 2 — Calibration does not achieve advertised risk control

**Response.** We agree and have reframed the result as a *calibration-failure study*
rather than a calibrated-risk pipeline. Concretely:

1. **Calibrator comparison** — Table (revised Table IV) already reports SPOT,
   SPOT+kurt(1/6), empirical p99.5, and validation-max under identical splits. The
   revision adds (a) runs-declustered SPOT (dependence-aware) and (b) an explicitly
   stated finding: no calibrator reliably meets q=1e-3 under inter-block drift;
   empirical p99.5 is approximately as reliable as SPOT.
2. **Multiple evaluation blocks** — `evt_multiwindow_drift.csv` partitions each held-out
   block into sequential sub-blocks and tracks FPR drift; Supp. discussion now reports
   the per-block range and identifies inter-block drift — not GPD misspecification — as
   the dominant failure mode (KS rejects the calibration distribution in only ~22–30%
   of cells while held-out FPR spans 0–0.67).
3. **Exceedance CIs + effective cluster counts** — all held-out FPR estimates now carry
   moving-block-bootstrap CIs (block L=64) and the supplement reports the
   runs-declustered effective exceedance count per stream (median ~21 raw exceedances
   → far fewer independent clusters).
4. **Controlled drift sweep (DONE, full coverage)** —
   `reports/controlled_drift_calibration.csv` (83,700 cells) injects controlled shifts
   into the *real* dumped residual streams (mean shifts +0.5σ/+1.0σ, variance ×1.5/×2.0,
   ACF 0.6/0.9, regime change +1σ, continuous ramp) and re-evaluates five calibrators
   across multiple evaluation blocks. Findings (Supp. Table `drift_calibration`):
   static SPOT overshoots q=10⁻³ by 60–110× under every condition; declustered SPOT is
   no better; rolling calibrators control risk to 1–7% under step/variance/ACF shifts
   but fail catastrophically under continuous ramps (0.33–0.39). **Effective cluster
   counts:** runs declustering leaves only 5–9 effective exceedances per cell of a
   median 21 raw — GPD tail fits rest on ~⅓ the nominal exceedance count, quantifying
   the reviewer's independence concern directly.
5. **Claim removal** — the abstract/introduction no longer state or imply that the
   pipeline achieves a specified false-alarm risk; calibration failure under drift is
   now a stated contribution (Contribution 3).

## Major Concern 3 — Causal vs. buffered reconciliation

**Response.** Fully addressed:

- **Timeline figure (new Fig. / `fig:timeline_regimes`)** shows the context window,
  64-step target horizon, score-emission point, buffer latency, and alarm attribution
  for both regimes on one axis, including an anomaly onset.
- **Separate reporting** — buffered ("smear") and strictly causal ("trailing") results
  appear in separate tables (Tables II–III) with regime labels on every number; the
  manuscript never mixes them.
- **Identical information access** — both regimes score the same windows; "trailing"
  emits the score at window completion (F_{t+S-1}-measurable), "smear" paints it back
  over the suspect region retrospectively. Alarm timestamping is now stated explicitly.
- **Matched-FPR operating points (DONE)** — `reports/matched_fpr_evaluation.csv`
  (11,100 cells; Supp. Table `matched_fpr`) evaluates all models at common held-out
  FPR budgets q* ∈ {0.001, 0.005, 0.01, 0.05} and reports event recall, precision,
  delay, and FA/1k at each point in both regimes. Key finding, now stated in the
  manuscript: under strictly causal streaming the ordering **partially reverses** —
  at q*=0.05 TS-JEPA reaches event recall 0.774 vs. 0.618 (TimesNet) with 180 vs. 262
  FA/1k, showing the Table III recall gap largely reflects TimesNet's miscalibrated
  threshold admitting a higher realized alarm rate, not an intrinsic sensitivity
  deficit. Under buffered attribution the ordering stays mixed (0.486–0.560 vs.
  0.583–0.612).
- **Pre-onset attribution sensitivity (new)** — the same script scores alarms under
  three conventions: strict (pre-onset = FP), grace-window (τ ∈ {16,64} steps), and
  smear attribution, quantifying how pre-onset alarms move between FP and early-detection
  credit.

## Major Concern 4 — Baseline fairness and transparency

**Response.**

- **Additional baselines (DONE)** — two training-free nonparametric references were run
  on all 19 non-GHL streams × 5 seeds under the identical protocol
  (`reports/heldout_nonparametric.csv`):
  `PCA-Recon` (PCA-subspace reconstruction) and `ZScoreMax` (per-channel max-z).
  Buffered results: PCA-Recon FPR 0.311 / F1 0.198 / PR-AUC 0.331 / recall 0.584 —
  statistically indistinguishable from TimesNet (0.300 / 0.197 / 0.329 / 0.583);
  ZScoreMax FPR 0.339 / F1 0.209 / PR-AUC 0.374 — the *best* ranking of any baseline
  yet the *worst* false-alarm rate. These results sharpen the manuscript's point:
  ranking quality and threshold conditioning are independent properties, and the
  reconstructive baselines' FPR failure is not a deficit of learned models per se.
  Linear-AR, empirical p99.5, and validation-max remain as explicit references.
  AnomalyTransformer and DCdetector in-house implementations exist in
  `src/models/baselines/` (results in `all_models_multidataset_benchmark.csv`); wiring
  them into the held-out protocol is available but matched-backbone controls were
  prioritized per the reviewer's experiment ordering.
- **Provenance statement (new in Sec. VI)** — TimesNet/TranAD are in-house
  re-implementations at published depths under the identical optimization budget;
  no per-model hyperparameter tuning; no method accesses held-out anomaly labels for
  thresholding or model selection.
- **Matched operating points** — see Concern 3; the headline comparison is now also
  reported at common FPR budgets, removing the "conservative detector looks calibrated"
  artifact the reviewer illustrates with the linear AR baseline (FPR 0.006 / F1 0.057).
- **Compute profile** — Supp. Table `computational_profile` reports params, peak VRAM,
  forward latency, throughput on RTX 5090 (batch 32, K=25, C+S=320); text now states
  these are workstation-GPU figures and does not claim embedded latency.

## Major Concern 5 — Statistical reconciliation

**Response.**

- **Complete test map (new Supp. longtable `test_map`)** — generated mechanically from
  the benchmark CSVs (`scripts/build_test_map.py`, `reports/test_map.csv`): 180 tests ×
  {endpoint, cohort (primary n=33 / non-GHL n=19), regime (buffered/causal), paired
  stream count, seed treatment, Wilcoxon statistic, raw p, BH-adjusted p, rank-biserial
  effect size, 95% bootstrap CI}. Every p-value quoted in the text maps to a row.
- **Seed handling stated** — seeds are averaged *within* each stream before pairing; the
  stream is the sole independent unit; nothing is double-counted. This is now stated in
  Sec. VI and in the test-map preamble.
- **No equivalence claims** — manuscript text now uniformly treats adjusted p>0.05 /
  CI-spanning-zero as inconclusive; we do not claim parity where tests are
  underpowered.
- **Per-dataset results** — Supp. Tables `all_datasets`/`dataset_prauc` give per-dataset
  metrics; the revised results text notes that conclusions do not rest on any single
  large dataset and cites the per-dataset breakdown.

## Major Concern 6 — Physics-inspired regularizers

**Response.** The revision reports these as null/domain-specific results and separates
*guaranteed* mathematical properties from *observed* empirical behavior:

- **Operator-Entropy** — text now states the barrier regularizes the transition
  operator only, not the encoder representation; collapse diagnostics and a
  regularizer-strength ablation (OpEntropy-lambda0, alpha→0) appear in the
  matched-backbone/ablation tables; Koopman-operator spectral diagnostics are in
  `operator_entropy_spectral_validation.csv`.
- **Reynolds-Stress** — we now state the instantaneous rank-one outer product is an
  auxiliary second-moment projection, not a temporal covariance; `heldout_reynolds_
  covmodes.csv` compares instant/temporal/batch estimators and shows the null result is
  robust to estimator choice. Reynolds-meanscore/alpha0 arms isolate the score vs. the
  regularizer weight.
- **Potential-Flow** — text explicitly says a smooth scalar-potential gradient flow
  cannot sustain nonconstant periodic orbits, a direct mismatch for cyclic telemetry
  (Daphnet, gait); PotentialFlow-noregime arm and the dynamical diagnostics in
  `potential_flow_*` files support this as a misspecified-prior finding rather than a
  tuning failure.
- **Framing** — Contribution 2 states the regularizers are exploratory domain-dependent
  hypotheses; no aggregate improvement is claimed anywhere.

## Major Concern 7 — Reproducibility and completeness

**Response.**

- **All `??` references resolved** — root cause: the submitted standalone was compiled
  before the supplement aux existed; the preamble already used `xr` +
  `\externaldocument`. Build order is now supplement-then-standalone (documented in
  README); the current PDF contains zero unresolved references. Verified mechanically.
- **Supplement completeness** — the supplement now contains: proofs (`sup:proofs`),
  protocol matrix (`tab:master_matrix`), per-dataset tables, gating detail (`sup:gating`),
  math diagnostics incl. the moved LP-shell construction (`sup:math_diagnostics`),
  operational tables (`sup:operational`), EVT detail (`sup:evt_detail`), the causal-vs-
  buffered case study (`tab:causal_vs_buffered`), the full statistical test map
  (`tab:test_map`), and [PENDING] residual-diagnostics + matched-FPR + drift tables.
- **Stream manifest** — `stream_manifest.csv` maps every stream to dataset, channel
  count, split lengths, seed coverage, and role (primary non-GHL / GHL / case study).
- **Reproduction map (new README section)** — each main table maps to its exact script,
  inputs, and output artifact.
- **Table-specific configs** — canonical benchmark = union of `reports/heldout_*.csv`
  files; we now document that `multiseed_heldout_benchmark.csv` contains legacy runs
  superseded by the canonical set and that `heldout_reynolds_covmodes.csv` is a
  standalone diagnostic excluded from headline aggregation.

## Editorial items

- **M1 (inconsistent recall)** — Root cause found and fixed: the abstract's "10.7% vs
  32.9%" were delay-budgeted `recall@64` figures from the causal case-study table
  (TS-JEPA vs TranAD), not causal event recall. Abstract/intro now report strictly
  causal event recall 46.6% vs 58.8% (TS-JEPA vs TimesNet, non-GHL) consistent with
  Table III, and label delay-budgeted recall separately.
- **M2 (terminology)** — the front-end is now called a smooth rFFT-domain dyadic-shell
  multiscale split throughout ("Littlewood–Paley–style" appears once, explicitly
  qualified as not an exact octave-localized decomposition); construction details moved
  to `sup:math_diagnostics`.
- **M3 (metric emphasis)** — the primary operating objective is now stated as
  false-alarm rate at preserved event sensitivity; matched-FPR analysis (Concern 3)
  provides the budgeted comparison.
- **M4 (dataset heterogeneity)** — manifest + per-dataset tables + cohort labels on
  every statistic (n=19 non-GHL at 5 seeds; n=33 incl. GHL at available coverage).
- **M5 (novelty framing)** — Contribution 1 now reads as an empirical characterization
  ("conservative low-false-alarm operating point, not universal dominance"); the
  contribution list states the architecture is a vehicle for the finding, not the
  finding itself.
- **M6 (deployment)** — compute table expanded/stated as GPU-workstation; causal regime
  latency is the honest per-window figure.
- **M7 (editing pass)** — full pass completed; page count reduced 13→12; references
  trimmed.

## Essential Experiments checklist

| Reviewer experiment | Status | Artifact |
| --- | --- | --- |
| 1 Matched-backbone representation study | DONE (5 arms, 19 streams × 5 seeds, primary protocol) | `reports/residual_diagnostics.csv` + Supp. Table |
| 2 Calibration under controlled drift | DONE (5 calibrators × 9 conditions × real residuals) | `reports/controlled_drift_calibration.csv` + Supp. Table |
| 3 Detection quality at matched FPR | DONE (4 budgets × 11 models × both regimes) | `reports/matched_fpr_evaluation.csv` + Supp. Table |
| 4 Component ablation + diagnostics | DONE (EMA/latent-recon arms integrated into MC1 analysis) | matched-backbone arms + ablation tables |
| 5 Strictly causal deployment | DONE | Table III + `tab:causal_vs_buffered` + timeline figure |

## Remaining gaps / honesty notes

- Calibration reliability is NOT established; the revision presents it as an open
  problem with a failure-mode decomposition and a controlled-shift taxonomy.
- The mechanism finding refines (and partially corrects) the original framing: the
  FPR gain is primarily a latent-*compression* effect; prediction supplies the
  anomaly-separation signal. The manuscript now says so.
- The kurtosis-adaptive calibrator is retained only as a (negative) ablation.
- Nonparametric references show reconstructive-baseline ranking is matched by a
  training-free PCA scorer; the differentiator is conditioning, not learned
  reconstruction.

## Remaining gaps / honesty notes

- Calibration reliability is NOT established; the revision presents it as an open
  problem with a failure-mode decomposition.
- Matched-FPR results are pending the dump run; numbers will be inserted mechanically
  from the generated CSVs.
- The kurtosis-adaptive calibrator is retained only as a (negative) ablation.
