# Author Response — Round 4 (Draft)

Manuscript: "On Latent Predictive Residuals and Automated Threshold Calibration for
Multivariate Time-Series Anomaly Detection: An Empirical Investigation"

We thank the referee for an exceptionally careful and constructive review. The report
correctly identifies that the original headline — a four-fold FPR reduction —
confounded score quality with the threshold operating point, that the "latent
conditioning" mechanism was confounded with bounded low-dimensional scoring, and that
several reported numbers were inconsistent. We have restructured the primary analysis
around operating characteristics, added the requested controls and baselines, corrected
the statistical treatment, reconciled all flagged numbers, and removed or qualified the
unsupported claims. The supplement is now included in this submission; the broken
cross-references resulted from a build-order issue (the supplement `.aux` was not
generated before the main document was compiled) and are resolved.

Below we respond point by point. "Sec./Tab./Fig." refer to the revised main text;
"S-" labels to the supplement.

---

## Major Concern 3.1 — FPR reduction is a threshold-operating-point effect

**Response.** Agreed and corrected. The primary analysis is now the recall–realized-FPR
operating curve (Sec. Results, Fig. 2): thresholds are fit on the calibration block
across a design-budget grid q ∈ [1e-4, 0.3] and all metrics are read on the test block,
so realized FPR — not the design value — is the comparison axis.

New results:

- **Design risk ≠ realized risk.** At q*=0.05, realized test FPR is 0.11–0.36
  (2–7× inflation), largest for observation-space baselines (0.26–0.30) and smallest
  for Potential-Flow (0.11) and the dimensionality-matched control (0.14). The text now
  states this explicitly; the prior wording implying "at FPR 0.05" described the
  *design* budget and was incorrect.
- **At matched budgets the causal ordering reverses.** Under strictly causal
  evaluation, the JEPA family occupies the upper envelope (partial AUC over budgets
  ≤0.10: 0.74–0.75 for all JEPA variants vs. 0.59–0.63 for every observation-space,
  nonparametric, and distance-based baseline). At q*=0.05, TS-JEPA attains event recall
  0.774 vs. 0.618 (TimesNet) at *lower* realized FPR (0.180 vs. 0.262). The abstract
  and Contribution 1 now lead with this matched-budget result.
- **Heterogeneity is stated.** The advantage is positive on only 6–10 of 18 streams
  (stream-clustered bootstrap p=0.05–0.25; CIs overlap zero) and disappears under
  buffered attribution. We claim a favorable shift of the strictly causal operating
  curve, not a uniform improvement.
- **Native operating point still reported** (Supp. Tab. S–causal benchmark), but now
  framed as operating-point evidence, not detector superiority.
- **Trivial-detection caveat added.** Models beat the constant-all-positive F1
  reference in only 18–50% of cells (Supp. `trivial_f1_comparison.csv`); the text now
  warns that low FPR on near-undetectable streams must not be read as favorable.

Artifacts: `reports/operating_characteristics.csv`, `oc_partial_auc.csv`,
`oc_model_curves.csv`, `oc_pairwise_tests.csv`; script
`scripts/run_operating_characteristic_analysis.py`.

## Major Concern 3.2 — Latent conditioning confounded with bounded, low-dim scoring

**Response.** Correct, and the revision now *demonstrates* this is largely the case.
Three new controls isolate boundedness (all under identical calibration):

| Control | What it matches | Result |
| --- | --- | --- |
| PCA-LinPred-LN | 32-d PCA embedding, ridge forecast, LayerNorm, identical saliency-gated norm — all in observation space | γ₂ = 1.16, SPOT FPR 0.052 (TS-JEPA: γ₂ 1.17, FPR 0.076) — conditioning reproduced almost exactly |
| IsoForest | bounded scores without learned encoder | γ₂ = −0.24, FPR 0.037 — conditioning reproduced |
| Obs-MAD | median/MAD robust normalization only | γ₂ = 1.30 but FPR 0.29 — robust normalization alone insufficient |
| TS-JEPA-NoGate | JEPA without saliency gate | recall 0.768 vs. 0.774, identical tails — gate contributes ~nothing |

The mechanism claim is revised to a two-factor decomposition: **bounded low-dimensional
scoring is sufficient for tail conditioning** (the learned encoder is not its cause),
**while the learned embedding supplies detection** (PCA-LinPred-LN reaches only 0.665
event recall vs. 0.774 for TS-JEPA at matched budgets). We no longer claim latent
prediction causes the conditioning; we do not claim EMA is the mechanism (p=0.17 is
reported as below detection threshold, not equivalence). Gating: the saliency gate and
dyadic front-end apply to JEPA arms only, now stated explicitly; the NoGate arm shows
the gate is empirically near-neutral.

**Conditioning → calibration link (requested).** Stream-level across all dumped cells:
KS distance between calibration and test nominal scores predicts realized SPOT FPR
(Spearman ρ=+0.38, p≈7e-38), while calibration-block kurtosis predicts *lower* FPR
(ρ=−0.32) — heavier tails make the GPD fit *more* conservative. Drift, not tail mass,
drives failure.

## Major Concern 3.3 — Premise vs. findings on EVT

**Response.** The narrative is reconciled: the introduction now states that inter-block
drift, not tail mass, is the dominant failure mechanism; JEPAs' ~50× miss of q is
stated alongside the baselines'.

- **Static POT vs. SPOT.** Terminology corrected throughout: the calibrator evaluated
  is a *static* POT threshold; conclusions no longer say "SPOT fails" where a static
  fit is meant. The streaming/rolling variant is evaluated separately
  (Supp. `controlled_drift_calibration.csv`): rolling calibrators control risk to 1–7%
  under step/variance/autocorrelation shifts but fail under continuous ramps
  (0.33–0.39 FPR) — adaptation absorbs regime jumps, not within-block drift.
- **Non-comparable calibrators.** The p99.5/validation-max comparison is now labeled
  as *not risk-matched*; the text states only that parametric GPD modeling confers no
  benefit at comparable realized FPR.
- **Kurtosis multiplier.** Relabeled explicitly as an ad-hoc sensitivity arm, not an
  EVT method; its failure carries no implication about GPD tail modeling.
- **Monotonicity anomaly (Q2) — resolved, not a bug.** Calibration-block excess
  kurtosis is ≤0 in ~40% of cells (median ≈0.23, min −2.0). The multiplier
  `1+tanh(γ₂)/6` therefore *lowers* the threshold in those cells, raising FPR.
  Eq. 13's "raises the threshold" wording assumed γ₂>0 and is corrected; the direction
  is now stated in the text.
- **Limited novelty acknowledged** — the calibration finding is presented as
  quantification of a known failure mode, not a new method.

## Major Concern 3.4 — Baselines

**Response.**

- **New baselines added:** IsolationForest, kNN-distance, Obs-MAD, PCA-LinPred-LN,
  TS-JEPA-NoGate — all under the identical split/calibration protocol
  (`reports/heldout_19stream_r4_arms.csv`). PCA-Recon and per-channel max-z were
  already present; we agree ZScoreMax's 0.374 PR-AUC indicates the benchmark
  discriminates weakly on ranking, which the operating-curve reframe is designed to
  address.
- **Provenance caveat added:** TimesNet/TranAD are in-house re-implementations at
  published depths under a shared optimization budget (AdamW, 20 epochs, W=320); the
  text states this is not a reproduction against published numbers on standard splits,
  and absolute performance comparisons to vendor implementations should be treated
  accordingly. We cannot verify against published splits within this revision's
  compute budget and now say so.
- **Trivial all-positive F1** is now reported per stream (supplement).

## Major Concern 3.5 — Statistical inference and cohort mixing

**Response.**

- **Dataset-clustered bootstrap** CIs are used for operating-curve comparisons;
  stream remains the unit of analysis with seeds averaged within stream.
- **GHL sensitivity.** The score dumps cover the 19 non-GHL streams; the operating-
  curve analysis is therefore the non-GHL cohort, now stated explicitly (the
  "nonghl"/"all" rows coincide). The 14 GHL streams' pathologies (near-identical
  streams, ~0 F1) are discussed; GHL is retained for continuity with prior work but
  flagged as an upper bound on detectability.
- **Zero-heavy Wilcoxon caveat** acknowledged; we report effect sizes + CIs and do
  not rest claims on raw p alone.
- **Multiplicity.** Raw p-values are labeled "unadjusted" where quoted in text
  (e.g., Potential-Flow p=0.033 unadjusted); the supplement test map carries BH-
  adjusted values.
- **Mixed cohorts/seeds.** The abstract no longer mixes the 33-stream means with
  19-stream operating-curve results; each number is tied to its cohort.
- **Primary endpoint declared:** the recall–realized-FPR curve / partial AUC is now
  primary (Sec. V-C revised); Dataset-Macro F1 is demoted to a secondary endpoint and
  the text discusses it.
- **Delay confound resolved.** Delay is now also reported on the *intersection* of
  events detected by both models (`delay_intersection.csv`): TS-JEPA ≈46–60 vs.
  TimesNet ≈57 — no systematic latency penalty; the raw gap was a detected-set
  artifact.

## Major Concern 3.6 — Physics-inspired components

**Response.**

- **Reynolds-Stress.** Retitled/described as a second-moment projection penalty; the
  "closure" language is softened.
- **Koopman/SMAP.** The SMAP PR-AUC 0.591 claim is now explicitly attributed to the
  shared architecture, not the Koopman prior (λ_ent=0 control shows the spectral term
  is irrelevant); TS-JEPA's SMAP number is reported alongside.
- **Potential-Flow.** The "unable to represent circulating orbits" claim is downgraded
  to conjecture. We tested the reviewer's inference-mismatch hypothesis directly —
  the model trains with z_noise~N(0,I) but inference queries the z_noise=0 mean path;
  noise-matched querying does *not* recover performance (Daphnet ER 0.71→0.21, others
  unchanged), so the mismatch is rejected as the explanation.
- **Propositions.** Prop. 1 is acknowledged as near-trivial and reframed as a
  consistency check; the K=0 edge case for ΔH is given a floor convention.
  Prop. 2 is labeled a textbook fact included for completeness, with the caveat it
  does not govern the trained flow-matching detector.
- **Gate identity — the reviewer made an arithmetic slip (stated politely).** With
  w_d = 1/D: D_sal = (1−α)‖e‖ + α·√(D·Σ_d w_d e_d²) = (1−α)‖e‖ + α·√(D·‖e‖²/D)
  = (1−α)‖e‖ + α‖e‖ = ‖e‖. The outer √D exactly cancels softmax's 1/D weights; the
  isotropic identity holds. The text now spells out the cancellation.
- **Empirical gate result:** TS-JEPA-NoGate is statistically indistinguishable
  (recall 0.768 vs. 0.774, identical tails) — the gate is near-neutral, now reported.

## Major Concern 3.7 — Early-warning claims

**Response.** "Early warning" is removed as a capability claim; buffered attribution is
described as detection with a fixed S=64 buffer, not prediction. The 180–204-step
pre-onset "leads" are explained as confounded by alarm rate (a detector emitting many
alarms trivially "leads" onsets) and are no longer cited as evidence of foresight. The
causal-vs-buffered event-recall similarity is explained: both mappings read the same
score stream; only the attribution differs, and the anomaly windows that survive
trailing attribution are largely the same ones. The FAR1k unit issue is fixed —
realized test FPR at q*=0.05 is 0.11–0.36, reported as such.

## Major Concern 3.8 — Dataset and benchmark issues

**Response.** Wu & Keogh (TKDE 2023) is now cited; SMAP/MSL label problems and
triviality are discussed, and their per-dataset numbers labeled upper bounds. The
entity-selection criteria (nominal-block length, stride-1 runtime; not labels) and
their selection-bias limitation are stated. "Seven physical domains" → seven *datasets*
(the domain count is corrected). SMD is clarified as outside the 45-stream manifest
(front-end ablation only). The reserved-stream count is corrected to 12.
Within-dataset-only scope is stated; no cross-dataset or unseen-fault claims remain.

## Section 4 — Numerical reconciliation

All flagged items reconciled against the canonical generated tables:

| Item | Resolution |
| --- | --- |
| Baseline nominal FPR (24–30/30.0–30.7/~22%) | Now stated per mapping/calibrator; buffered SPOT 0.300–0.307, causal ~0.22 |
| JEPA nominal FPR variants | Unified to the SPOT-calibrated values per table |
| Ablation p-values (0.375/0.58/0.62-0.41-0.048) | Canonical values from the test map; text updated |
| Op-Entropy/Reynolds/PF F1 | Text now matches Table I(a) (0.237/0.207/0.089) |
| Event recall non-GHL | Corrected to Table II values (0.583/0.481) |
| PR-AUC p (0.45 vs 0.39) | Unified |
| Calibrator for ablations | Stated: ablations use the kurtosis-multiplier arm; Table I(a) uses standard SPOT — the discrepancy is explained |
| "Physics-informed" vs "-inspired" | Fig. 1 and all text now consistently "physics-inspired" |

## Minor points

- Title/abstract "automated threshold calibration" — we retained the title (calibration
  *reliability* is the object of study) but the abstract no longer implies a new
  calibration method.
- Abstract simplified around the matched-budget result (235 words, one paragraph).
- Table I(b) dyadic shells labeled anecdotal (5 streams, one dominant SMD stream,
  +160k parameters).
- Per-stream results + manifest are in the supplement; stream/seed/gap counts are
  machine-verifiable via `reports/stream_manifest.csv`.
- Anonymous release contains checkpoints and the manifest generator (verified).

## Questions for the authors

1. Pooled recall–FPR curves: Fig. 2 + `oc_model_curves.csv`; dataset-clustered CIs.
2. Monotonicity: resolved — γ₂≤0 on ~40% of calibration blocks lowers the threshold.
3. Bounded low-dim obs control: PCA-LinPred-LN reproduces conditioning, not detection.
4. Gating applies to JEPA arms only; NoGate arm shows it is near-neutral.
5. 180–204-step leads are alarm-rate artifacts, not foresight; claim removed.
6. Re-implementations are not reproductions of published numbers; stated as a limit.
7. GHL: operating-curve cohort is the 19 non-GHL streams; GHL retained only in the
   native-point table with a pathologies caveat.

## Summary of changes

- Primary analysis reframed to operating characteristics (Fig. 2, new Sec. paragraph,
  abstract, Contribution 1, conclusion).
- 5 new arms: PCA-LinPred-LN, Obs-MAD, IsoForest, KNN-Dist, TS-JEPA-NoGate.
- Delay-intersection, trivial-F1, drift→FPR, kurtosis→FPR analyses.
- Static-POT terminology, kurtosis-arm relabel, non-risk-matched caveat.
- Early-warning removed; PF-mismatch tested and rejected; propositions qualified.
- ~15 numerical reconciliations; Wu & Keogh cited; dataset/entity caveats added.
- Supplement now included with resolved cross-references.
