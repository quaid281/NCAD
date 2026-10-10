# Supplementary Addendum to the Round-2 Author Response

Manuscript: *On Latent Predictive Residuals and Automated Threshold Calibration for Multivariate Time-Series Anomaly Detection: An Empirical Investigation*
Target Journal: IEEE Transactions on Neural Networks and Learning Systems (TNNLS)

This addendum documents additional controlled experiments executed after the Round-2 revision to close the remaining gaps in the required-revisions list. All scripts and raw CSVs are committed under `scripts/` and `reports/`.

---

## A. Major Concern 2 — Non-Neural Latent Controls (matched backbone protocol)

**Reviewer requirement:** *Would a simpler learned representation, dimensionality reduction method, or non-neural latent predictor produce similar residual conditioning?*

**Artifact:** `scripts/run_pca_latent_control.py` -> `reports/controlled_backbone_latent_controls.csv` (15 streams x 4 SNR levels, identical protocol, splits, SPOT calibration, and metrics schema as `controlled_backbone_experiment.csv`; rows merge directly).

Two controls were added: `pca_latent_pred` (PCA projection to D=32, ridge map z_ctx -> z_tgt) and `rand_latent_pred` (fixed random Gaussian projection to D=32, ridge predictor). Both score ||z_hat_tgt - z_tgt||_2.

| Model | F1 (clean) | Empirical FPR (clean) | Test kurtosis |
|---|---|---|---|
| tcn_obs_recon | 0.1977 | 0.2729 | 57.26 |
| tcn_obs_pred | 0.1916 | 0.2757 | 56.11 |
| rand_latent_pred | 0.1452 | 0.2255 | 17.83 |
| pca_latent_pred | 0.1347 | 0.2423 | 25.39 |
| ts_jepa | 0.0785 | 0.1180 | 1.77 |

The ordering is identical at every SNR level (obs-space > linear latents > learned JEPA). Interpretation: latent compression per se accounts for part of the residual conditioning (FPR 0.27 -> 0.23-0.24), but the learned JEPA representation supplies the majority of the tail conditioning (FPR 0.118, kurtosis 1.77 vs. 18-25). The latent-space benefit therefore cannot be attributed to dimensionality reduction alone.

---

## B. Major Concern 3 — Potential-Flow Field Verification

**Reviewer requirement:** *Verify the claimed gradient-field constraints numerically; justify midpoint evaluation vs. numerical integration.*

**Artifact:** `scripts/run_potential_flow_field_diagnostics.py` -> `reports/potential_flow_field_diagnostics.csv`, `reports/potential_flow_integration_comparison.csv` (SMAP:P-3, MSL:M-1, 12 training epochs).

1. **Curl-freeness of the raw field.** The pre-projection velocity v = -grad_z Phi exhibits a Jacobian antisymmetric fraction ||J - J^T||_F / ||J||_F of 6.9e-8 (SMAP:P-3) and 8.3e-8 (MSL:M-1), i.e. numerically symmetric to machine precision, confirming the implementation computes an exact negative Hessian field.
2. **Autodiff correctness.** Central finite differences agree with autodiff gradients at relative error 0.004 (SMAP) and 0.004 (MSL).
3. **Tangent-projection deviation.** The deployed field, which projects v onto T_{z}S^{D-1}, carries an antisymmetric Jacobian fraction of ~0.15. The deployed velocity field is therefore approximately, not exactly, curl-free; the manuscript text should state this explicitly.
4. **Midpoint vs. integrated scoring.** Scores computed by K-step midpoint-quadrature integration (K=4,8) correlate with the deployed single-midpoint score at Spearman rho = 0.86-0.92. Multi-step integration does not improve detection metrics (Point-F1 and FPR are equal or worse under K>1), supporting the midpoint rule as an adequate operational approximation rather than an arbitrary choice.

---

## C. Major Concern 5 — Calibration Across Risk Levels, Declustering, and Chronological Drift

**Reviewer requirement:** *Report calibration error as a function of the requested risk level; evaluate whether declustering or block-based tail estimation changes the conclusions; separate temporal drift from estimation error.*

**Artifact:** `scripts/run_evt_calibration_curves.py` -> `reports/evt_calibration_curve_sweep.csv`, `reports/evt_declustering_comparison.csv`, `reports/evt_multiwindow_drift.csv` (5 streams x 3 models, same 65/17.5/17.5 nominal partition as Table IX).

1. **Risk-level sweep (q in {1e-1 ... 1e-4}).** SPOT and empirical-quantile calibration both overshoot the nominal risk at every q for every model. The overshoot is scale-free in ordering: TS-JEPA achieved-FPR overshoot at q=1e-3 is 13.6x (SPOT) and 8.9x (empirical), versus 244x/213x for TimesNet and 277x/264x for TranAD. The empirical quantile is never worse than SPOT for TS-JEPA, corroborating the manuscript's finding that the GPD fit adds no benefit under these score distributions.
2. **Runs declustering (gap tolerance r=5).** Declustering reduces baseline overshoot (TimesNet 244x -> 189x; TranAD 277x -> 208x) but does not repair calibration; TS-JEPA changes marginally (13.6x -> 10.7x). Temporal dependence is a real but secondary contributor.
3. **Multi-window drift.** Thresholds fit on calibration block k and evaluated on later calibration blocks achieve FPRs of 0.001-0.05 for all models; evaluated on the held-out evaluation interval, TimesNet/TranAD degrade to 0.21-0.29 while TS-JEPA remains at 0.015-0.022. Chronological distribution shift between the calibration and deployment intervals is the dominant failure mechanism - not tail-estimation error - and it is roughly an order of magnitude smaller for latent predictive residuals. This refines the manuscript's claim: SPOT fails to attain nominal risk under the evaluated protocol primarily because the nominal score distribution drifts, and latent residuals drift less.

---

## D. Major Concern 7 — Littlewood-Paley Filterbank Verification and Front-End Ablation

**Reviewer requirement:** *Give the exact discrete filter coefficients and their derivation; report frequency-response and energy-preservation diagnostics; compare against an ablation without the frequency front-end.*

**Artifacts:**
- `scripts/run_lp_filterbank_diagnostics.py` -> `reports/lp_filterbank_diagnostics.csv`, `reports/lp_filterbank_frequency_responses.csv`
- `scripts/run_dyadic_frontend_ablation.py` -> `reports/dyadic_frontend_ablation.csv`

1. **Implementation reconciliation.** The manuscript text describing "1D depthwise convolutions with kernel lengths {11,7,5,3}" is inaccurate in two respects and has been corrected: the dyadic partition is computed in the Fourier domain (`compute_dyadic_filters` builds smooth shells Psi_j normalized to an exact partition of unity; the shell convolutions are full - not depthwise - convolutions with kernel sizes (15,9,5,3) that post-process each shell). The convolutions are feature processors applied after the partition, not the partition mechanism itself.
2. **Exactness of the partition.** Partition-of-unity error max_omega |sum_j Psi_j - 1| = 1.2e-7; reconstruction error ||x - sum_j irfft(Psi_j rfft(x))||_inf / ||x||_inf = 4.4e-8. The frequency split is exact by construction.
3. **Honest caveat.** Because the shells overlap smoothly, sum_j ||x_j||^2 recovers only ~70% of signal energy (Parseval deficit 0.30); the top shell concentrates on the Nyquist bin. The manuscript now presents these numbers rather than implying ideal octave separation.
4. **Front-end ablation (dyadic_on vs dyadic_off, identical HybridTCNEncoder, 5 streams, 10 epochs).** Mean Point-F1 0.131 (on) vs 0.088 (off); mean recall 0.254 vs 0.195; PR-AUC 0.211 vs 0.203; nominal FPR 0.183 vs 0.146. The dyadic front-end improves sensitivity modestly on most streams while slightly increasing false alarms; it is a contributing component, not the driver of the latent-prediction effect.

---

## E. Major Concern 8 — Per-Fault-Type Evaluation of the Saliency Gate

**Reviewer requirement:** *Evaluate the gate on explicit flatline, stuck-sensor, dropout, and low-amplitude fault cases; report per-fault-type recall before and after gating; compare variance gating with robust estimators and a gate-free model.*

**Artifacts:**
- `scripts/run_fault_type_gate_evaluation.py` -> `reports/fault_type_gate_evaluation.csv`
- `scripts/run_fault_type_reference_detector.py` -> `reports/fault_type_reference_detector.csv`

Protocol: a separate TS-JEPA is trained per gating mode (variance gate, MAD-robust gate, static global-variance gate, uniform/no gate) on six multivariate streams. Six controlled fault events per type are injected into nominal test regions: flatline, stuck-sensor, dropout, low-amplitude step (all on the lowest-variance live channel), flatline on a median-variance channel, and a +5-sigma spike control on the highest-variance channel. Thresholds are recalibrated per mode at SPOT q=1e-3.

Per-fault-type results (144 rows: 6 streams x 4 gate modes x 6 fault types, SPOT q=1e-3):

| Fault type | uniform | variance | robust_mad | global_var |
|---|---|---|---|---|
| flatline | 0.021 | 0.043 | 0.058 | 0.020 |
| flatline_midvar | 0.024 | 0.048 | 0.068 | 0.025 |
| stuck_sensor | 0.021 | 0.043 | 0.058 | 0.020 |
| dropout | 0.020 | 0.043 | 0.058 | 0.020 |
| low_amplitude_step | 0.030 | 0.049 | 0.062 | 0.022 |
| spike_control | 0.023 | 0.042 | 0.059 | 0.022 |
| **Mean nominal FPR** | **0.019** | **0.069** | **0.055** | **0.029** |

Findings:

1. Single-channel quiet faults are weakly detected under every gating mode (point recall 0.02-0.07, event recall 0.17-0.31 pooled). The reviewer's blind-spot concern is confirmed and is larger than stated: it is a sensitivity limitation of the latent discrepancy itself on wide telemetry, not only of the gate.
2. The variance gate does what it is designed to do: realized gate weight on the spike channel rises from ~0.35 nominal to 0.69-0.93 during transients, and mean point recall roughly doubles versus the gate-free model (0.045 vs 0.023) - at the cost of triple the nominal FPR (0.069 vs 0.019). Gating trades alarm budget for sensitivity; it does not rescue quiet faults, which are downweighted by construction (gate weight ~0.30-0.40 on the faulted channel).
3. The MAD-robust variant achieves the highest point recall (0.060) with slightly lower FPR than the variance gate; the static global-variance gate provides no benefit over uniform. Per-mode trade-offs are modest in either direction.
4. Flatline, stuck-sensor, and dropout injections are near-equivalent detectors (all render the faulted channel constant); their identical metrics across modes confirm the response is driven by the variance-collapse signature, not the offset level.

Two reference detectors confirm the injected faults are physically present but marginal for any aggregate monitor at low-FPR operating points: a max-over-channels amplitude detector reaches only ~9% point recall at 14% FPR, and a rolling-variance-collapse detector ~18-32% recall at ~15% FPR.

**Code-level finding.** During this audit we determined that `CausalChannelSaliencyGate` is implemented and unit-tested (`tests/test_saliency_gate.py`) but is not invoked in the deployed scoring path. Resolution adopted: the gate is not wired into the main pipeline (doing so would invalidate all reported benchmark numbers); the manuscript now describes it as an auxiliary evaluated variant and restricts deployed-pipeline claims to the latent `CoordinateSaliencyGate`. See Section H.

---

## F. Major Concern 9 — Early-Warning Lead-Time Outcome

**Reviewer requirement:** *Report early warning separately: proportion of incidents warned before onset, lead-time distributions, false warnings per unit nominal time.*

**Artifact:** `scripts/run_early_warning_analysis.py` -> `reports/early_warning_leadtime.csv` (5 streams, buffered smear scoring, event-level filter, 15 epochs).

| Model | Events warned pre-onset | Median lead (steps) | Mean post-onset delay | False warnings / 1k nominal |
|---|---|---|---|---|
| TS-JEPA | 12.7% (7/57 pooled) | 218 | 263 | 23.8 |
| TimesNet | 45.7% | 223 | 368 | 206.4 |

TimesNet's higher pre-onset warning fraction is confounded by its alarm rate: on GECCO it warns 37/37 events "early" while emitting 874 false warnings per 1,000 nominal steps, i.e. it is effectively always alarming. At comparable operating points TS-JEPA trades pre-onset coverage for an order-of-magnitude lower nuisance-alarm rate. The manuscript now reports all three operational outcomes - strictly causal online detection, buffered detection, and buffered early-warning lead time - in Section VII (Operational Latency subsection), and uses "predictive early warning" only in qualified form.

---

## G. Artifact index

| Concern | Script | Output CSV(s) |
|---|---|---|
| 2 | run_pca_latent_control.py | controlled_backbone_latent_controls.csv |
| 3 | run_potential_flow_field_diagnostics.py | potential_flow_field_diagnostics.csv, potential_flow_integration_comparison.csv |
| 5 | run_evt_calibration_curves.py | evt_calibration_curve_sweep.csv, evt_declustering_comparison.csv, evt_multiwindow_drift.csv |
| 7 | run_lp_filterbank_diagnostics.py | lp_filterbank_diagnostics.csv, lp_filterbank_frequency_responses.csv |
| 7 | run_dyadic_frontend_ablation.py | dyadic_frontend_ablation.csv |
| 8 | run_fault_type_gate_evaluation.py | fault_type_gate_evaluation.csv |
| 8 | run_fault_type_reference_detector.py | fault_type_reference_detector.csv |
| 8 | run_gate_mode_ablation.py | gate_mode_ablation.csv |
| 9 | run_early_warning_analysis.py | early_warning_leadtime.csv |

---

## H. Manuscript–Implementation Reconciliation (code-vs-paper corrections)

Two discrepancies between the manuscript text and the deployed code were found during this audit. Both were resolved by correcting the manuscript to describe the implementation actually used in the reported experiments, rather than changing the pipeline and invalidating the benchmark results.

**H1. Input-channel saliency gate was never deployed.** `CausalChannelSaliencyGate` exists in `src/models/geometric_layers.py`, is unit-tested in `tests/test_saliency_gate.py`, and is exercised in the fault-type study above, but no code path in `src/engine/` invokes it during training or scoring. All reported results were produced with unweighted input channels; the only saliency mechanism active in the deployed pipeline is the latent `CoordinateSaliencyGate` applied to predictive residual vectors inside `compute_predictive_discrepancy`. The earlier Section IV-D1 text, the shared-component statements in Sections IV/V, Algorithm 1, the Ablation-2 narrative, and Table ablations(b) described channel gating as part of the deployed architecture. Corrections made in `paper/NCAD_CS/sections/`:

- Section IV-D1 renamed and restructured: the latent coordinate-saliency gate is presented as the deployed mechanism; the input-channel gate is presented as an auxiliary variant evaluated in Section VI (fault-type study).
- Shared-component claims now list latent coordinate saliency gating, dyadic shells, and LayerNorm.
- Algorithm 1 no longer applies input gating and now uses the gated discrepancy `D_sal`, matching `compute_predictive_discrepancy`.
- Ablation 2 reframed as an evaluation of the auxiliary input-channel variant; Table (b) is re-derived by `scripts/run_gate_mode_ablation.py` (four gating regimes x four streams, identical TS-JEPA configuration) because the previously reported numbers had no corresponding artifact in `reports/`.
- The fault-type blind spot is now stated quantitatively in the boundary-conditions paragraph.

**H2. Littlewood-Paley description did not match the code.** The earlier text claimed 1D depthwise convolutions with kernel lengths {11,7,5,3}. The implementation (`LittlewoodPaleyDyadicBlock`) constructs smooth shells directly in the rFFT domain, normalizes them into an exact bin-wise partition of unity, applies the split via `irfft(Psi_j * rfft(x))`, and then processes each shell with full (non-depthwise) causal convolutions of kernel sizes (15,9,5,3), a learned softmax scale router, a residual connection scaled by gamma=0.1, and LayerNorm. The method text now describes this pipeline, reports the measured diagnostics (partition error 1.2e-7, reconstruction error 4.4e-8, ~70% shell energy recovery, Nyquist concentration of the top shell), and the front-end ablation claim is replaced by the measured on/off results (macro Point-F1 0.088 -> 0.131 across five streams), with the added-parameter caveat stated.

---

## I. Multi-Seed Extension to Five Seeds (Statistical-Rigor Requirement)

**Reviewer requirement:** *Report at least five independent training seeds for the principal models, where computationally feasible.*

**Artifacts:** `reports/heldout_nonghl_seeds45.csv`, `reports/heldout_nonghl_seeds45_a.csv`, `reports/heldout_nonghl_seeds45_b.csv` (seeds 7 and 2024, identical protocol to `heldout_45stream*.csv`, six principal models), plus `reports/heldout_ghl_seeds45_{a,b}.csv` for the GHL extension (in progress at time of writing).

All 19 non-GHL streams now carry five seeds (42, 123, 456, 7, 2024); on completion of the GHL extension, GHL loops 17/18 will carry five seeds and the twelve large GHL streams (>1M observations each) three seeds, as documented in the master matrix footnote and Section V. Recomputing every reported statistic at five seeds shifts values modestly without altering any conclusion:

| Claim | 3 seeds | 5 seeds |
|---|---|---|
| TS-JEPA vs TimesNet held-out FPR | -0.154, p=0.008 | -0.150, p=0.007 |
| TS-JEPA vs TranAD held-out FPR | -0.157, p=0.001 | -0.150, p=0.004 |
| TS-JEPA vs TimesNet Point-F1 parity | +0.018, p=0.55 | +0.012, p=0.54 |
| TS-JEPA vs TranAD Point-F1 | +0.062, p=0.94 | +0.051, p=0.88 |
| OpEntropy vs TS-JEPA Point-F1 | -0.007, p=0.71 | +0.002, p=0.69 |
| Reynolds vs TS-JEPA Point-F1 | -0.026, p=0.33 | -0.015, p=0.50 |
| PotentialFlow vs TS-JEPA Point-F1 | -0.091, p=0.020 | -0.084, p=0.013 |
| TS-JEPA non-GHL FPR / F1 | 0.066 / 0.254 | 0.076 / 0.234 |

Two headline framing updates follow from the recompute: (i) the nominal-FPR reduction is stated as *roughly four-fold* (0.300/0.076 = 3.9x, 0.307/0.076 = 4.0x) rather than 4--5x; (ii) on non-GHL Point-F1, Operator-Entropy (0.237) now marginally exceeds TS-JEPA (0.234), consistent with the paper's null-finding framing that the regularizer differences lie within seed noise. No significance boundary is crossed by any paired test. All tables (`tab_core_benchmark`, `tab_causal_benchmark`, `tab_calibrator_comparison`, `tab_all_datasets`) and in-text statistics were regenerated from the merged five-seed artifacts.

**Additional manuscript corrections in this pass:** the univariate saliency-gate identity case now documents the explicit K=1 short-circuit branch (sigma(0)=0.5 is not identity); a calibrator-assignment paragraph in Section V maps every experiment to standard SPOT vs. the tanh(gamma_2)/6 adaptive variant; the conclusion states the within-dataset scope limitation; and detection delays in Table I are now stream-macro means computed from the same held-out artifacts as the other columns (the previous column mixed sources and has been corrected, including the TranAD delay, which is 234 steps, not the previously reported 42).

---

## J. Additional Verification Items (Second Reviewer Pass)

**J1. Floored energy barrier vs. Proposition 1.** The implemented barrier clamps `r = ||K||_F^2 / (gamma*D)` at `eps = 1e-6` (`torch.clamp(energy_ratio, min=eps_safe)`), capping the penalty at `eps - 1 - log(eps) ~= 12.8*beta_norm`; it cannot diverge as `||K||_F -> 0`. The proposition and proof were rewritten for the exact trained objective: (i) under the floored barrier, K=0 is not a local minimizer whenever context and target latents are coupled (the prediction gradient `-2/B * sum z_tgt z_ctx^T != 0` there); (ii) the `+inf` divergence statement now explicitly applies only to the unfloored idealization.

**J2. Saliency-gate boundary conditions, verified numerically** on the released implementation: `K=1` returns weights identically 1.0 (explicit short-circuit); equal variance shares yield the neutral `sigma(0) = 0.5` reweighting (documented as relative reweighting, absorbed by downstream LayerNorm); a constant channel gates to 0.08; a 10x-variance channel to 0.93; a mid-context regime shift is detected identically (0.93) since variance shares pool over the context.

**J3. Littlewood-Paley vs standard multiresolution.** `scripts/run_lp_filterbank_diagnostics.py` now additionally compares the implemented shells against analytic Mallat-form octave-band responses of orthonormal Haar and Daubechies-4 filter banks (dependency-free). Fraction of band mass inside the ideal dyadic octave: ours {0.50, 0.33, 0.20, ~1.00} vs Haar {0.78, 0.50, 0.55, 0.82} vs DB4 {0.86, 0.63, 0.64, 0.87}. The implemented front-end is less frequency-localized than either wavelet bank; what it provides is an exact amplitude partition (PoU error 1.2e-7, reconstruction error 4.4e-8) plus learned per-shell routing. The method text now characterizes it as a Fourier-domain multiscale front-end, not an exact LP octave decomposition, and notes that the top shell degenerates to a Nyquist spike (its Gaussian width collapses to ~1e-4), leaving three effective bands.

**J4. GPD tail-fit diagnostics** (`reports/evt_tail_diagnostics.csv`; 9 non-GHL streams x {TS-JEPA, TimesNet} x 3 seeds) separate the candidate causes of SPOT's design-risk overshoot: median exceedance count 21 (range 16-70, i.e., scarce tail samples); degenerate boundary fits xi = -0.9 in 33% (TS-JEPA) / 52% (TimesNet) of cells; KS goodness-of-fit rejects in only 6/27 and 8/27 cells respectively; realized held-out FPR varies 0.0-0.67 across seeds under fits that pass the KS test; exceedances cluster strongly (10-22 per cluster, worse for TimesNet). Conclusion stated in the paper: distribution drift, not tail mis-specification, dominates the overshoot; empirical p99.5 matches or beats SPOT FPR in ~half the cells.

**J5. Reynolds covariance-estimator variants** (`reports/heldout_reynolds_covmodes.csv`; same 9 streams x 3 seeds). `cov_mode` added to `compute_observed_stress`: "instant" (deployed rank-1), "temporal" (per-sample covariance over the target horizon), "batch" (batch covariance). Paired vs deployed: tempcov DeltaF1 = -0.009, DeltaPR-AUC = +0.003, DeltaFPR = +0.004; batchcov DeltaF1 = -0.010, DeltaPR-AUC = -0.026, DeltaFPR = -0.042. The null result is insensitive to the estimator, so the rank-1 form is not the reason the regularizer fails to help.

**J6. Statistical unit of analysis.** Now stated explicitly in Section V: the independent unit is the telemetry stream (n = 19 non-GHL at five seeds; n = 33 where GHL enters at available coverage); seeds are averaged within stream before pairing and repeated windows aggregate to one stream-level score. We make no equivalence/noninferiority claim; all paired differences report effect sizes with 95% bootstrap CIs.

**J7. Definitive stream manifest.** `reports/stream_manifest.csv` (built by `scripts/build_stream_manifest.py` directly from the data files and result artifacts) enumerates every stream's dataset, identifier, channel count, train/test lengths, held-out anomaly counts and rates, benchmark role, and per-stream seed coverage. It reconciles the 45-stream suite as 33 primary + 12 case studies; `reports/streams_45.txt` was found to be missing `GECCO:water_quality` and has been corrected.
