# Response to the Fresh TNNLS Review (Round 2)

This round was answered with new experiments, and several results go against the manuscript's earlier claims. Where that happened, the manuscript was changed to match the data. Everything below traces to a CSV in `reports/` and a script in `scripts/`.

**New artifacts**
- `scripts/run_multiseed_heldout_benchmark.py` and `reports/multiseed_heldout_benchmark.csv`: 7 streams x 3 seeds x 10 model variants, thresholds fit on a held-out nominal calibration half, FPR measured on a separate held-out half with block-bootstrap intervals, both buffered and strictly causal mappings.
- `scripts/analyze_multiseed.py` and `reports/multiseed_summary.md`: paired stream-level tests.

**Corrections to our earlier response letter**
1. The TimesNet and TranAD rows of the computational table in the first letter were not backed by `computational_profile_benchmark.csv` and are withdrawn. The baselines are faster than the JEPAs (0.19 and 0.29 ms against 0.61 to 0.84 ms per window).
2. The first letter reported 5-seed means and Wilcoxon statistics (Concern 3.4). No artifact supported them and the manuscript's 45-stream results use one seed. Those numbers are withdrawn. The multi-seed results below replace them.
3. The first letter reported that EVT matched its design risk. The threshold in that experiment was fit on the same scores it was scored on. That agreement is in-sample and withdrawn.

## Headline result

| Model (buffered, SPOT u=0.98, q=1e-3) | Point-F1 | PR-AUC | Held-out nominal FPR | Event recall |
|:--|:-:|:-:|:-:|:-:|
| TS-JEPA | 0.032 | 0.137 | 0.047 | 0.099 |
| Operator-Entropy | 0.018 | 0.156 | 0.064 | 0.107 |
| Reynolds-Stress | 0.017 | 0.156 | 0.042 | 0.149 |
| Potential-Flow | 0.074 | 0.170 | 0.051 | 0.307 |
| TimesNet | 0.157 | 0.355 | 0.356 | 0.728 |
| TranAD | 0.161 | 0.390 | 0.449 | 0.691 |

The JEPA family has about one seventh of the baselines' nominal false-alarm rate, and substantially lower recall, Point-F1 and PR-AUC. TS-JEPA trails TimesNet by 0.125 Point-F1 (95% bootstrap interval [-0.205, -0.045], Wilcoxon p=0.031, TimesNet ahead on 7 of 7 streams). The F1 ordering and the "balanced ranking" statement in the 45-stream tables did not replicate under this protocol. We have not yet identified which protocol difference explains the gap (calibration source, smoothing and event filtering, epochs, stream selection). The manuscript now presents latent prediction as a conservative low-false-alarm operating point, not as a dominant detector.

## Point-by-point

**Major 1 (physical interpretation).** The manuscript now calls all three components physics-inspired inductive biases. The Reynolds-Stress section was rewritten to match the code that produced the results (rank-one stress outer product with a closure head, weight 0.1, stress term also in the score). The earlier text described a Cholesky temporal covariance that the code does not implement. Operator-Entropy uses lambda_ent=0.5, not the 10^-3 the earlier text stated. Matched controls (same architecture, term switched off) show no effect distinguishable from seed noise:
- Operator-Entropy vs lambda_ent=0: Point-F1 +0.001, interval [-0.006, +0.009], p=1.00.
- Reynolds-Stress vs stress loss off: -0.017 [-0.044, +0.003], p=0.375. Vs mean-only score: -0.024 [-0.056, +0.002], p=0.375.
- Potential-Flow vs no-regime control: +0.006 [-0.022, +0.032], p=0.58.
- Potential-Flow vs TS-JEPA: +0.043 [-0.024, +0.150], p=0.84. Operator-Entropy vs TS-JEPA: -0.014 [-0.029, -0.002].

We therefore make no claim that the entropy, stress, or potential-flow term causes any gain. The code also contains modules the paper did not describe (Hankel moment filter, Cohn-Elkies shell filter, shearing-wavelet pulses, resolvent purification, Grassmannian regime routing). The manuscript now says so for the Operator-Entropy and Reynolds-Stress models. The Potential-Flow section still needs a paragraph on regime routing and its conditioning on time and context.

**Major 2 (scoring protocol).** Algorithm 1 gives the timestamp-level procedure. The new benchmark reports buffered (64-step lookahead) and strictly causal mappings for every model. Under the causal mapping the JEPAs have 0.3 to 3% held-out FPR, event recall 0.25 to 0.50 and Point-F1 below 0.02, against Point-F1 near 0.18 for the baselines. Detection delay in physical units per domain is not yet reported.

**Major 3 (EVT).** The 98th vs 95th percentile discrepancy: the current manuscript source uses u=0.98 for every main-benchmark calibration and I found no u=0.95 description in the sources, so the reviewed PDF may predate the current text. The only u=0.95 runs are the sensitivity sweep. Held-out results with identical score streams:
- Plain SPOT at q=1e-3 gives 4 to 8% nominal FPR for the JEPAs and 36 to 45% for the baselines. The 95% interval excludes q in 67 to 100% of rows.
- The kurtosis multiplier raised TS-JEPA's held-out FPR from 0.047 to 0.131 and left Operator-Entropy unchanged (0.064 vs 0.063).
- A 99.5th percentile and the validation maximum perform like SPOT (TS-JEPA 0.049 and 0.047).

The kurtosis heuristic has no demonstrated benefit and SPOT shows no advantage over simpler rules on this data. The claim is withdrawn.

**Major 4 (mixed metrics, seeds, tests).** Paired stream-level tests with bootstrap intervals are in `reports/multiseed_summary.md`. Three seeds and seven streams give little power; this is stated in the manuscript. Reporting the 45-stream results with seeds is open work.

**Major 5 (ablations).** Done for the three physical terms (see Major 1). Gating and wavelet ablations are not repeated across seeds, and wavelet filter coefficients are not yet specified in the text.

**Major 6 (reproducibility).** Partly addressed: scripts, CSVs, seeds, split rule, and corrected model descriptions. Still missing: wavelet filter definitions, per-dataset split boundaries, and software versions.

**Major 7 (operational incidents).** Not re-done. The caveat that the earlier causal table calibrates on training residuals and omits recall is now in the manuscript.

## What remains open
1. Reconcile the 45-stream protocol with the held-out protocol and re-run the full benchmark with seeds.
2. Paper text for Potential-Flow regime routing and the shared latent modules.
3. Wavelet filter specification, split boundaries, software versions, detection delay in physical units, event-matching rules, matched-false-alarm-rate comparisons.
4. A matched-capacity unconstrained transition model, and generic entropy and covariance regularizers, as additional controls.

