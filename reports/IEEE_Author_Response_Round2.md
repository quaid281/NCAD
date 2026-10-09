# Response to the Fresh TNNLS Review (Round 2)

This round was answered with new experiments and verifiable code artifacts. Several empirical findings directly qualify or contradict the manuscript's earlier claims. Where the evidence did not support the original claims, the manuscript was revised to match the data. Everything below traces directly to scripts in `scripts/` and CSVs in `reports/`.

**New artifacts**
- `scripts/run_multiseed_heldout_benchmark.py`: Evaluates models under strictly held-out nominal threshold calibration (training on the first 65%, splitting the remaining 35% into calibration and evaluation halves separated by an $S=64$ gap).
- `scripts/analyze_multiseed.py`: Computes stream-level paired differences, moving-block bootstrap 95% confidence intervals, and Wilcoxon signed-rank tests.
- `reports/heldout_45stream*.csv` and `reports/heldout_38stream_summary.md`: Primary replication across 33 matched streams (19 non-GHL streams with 3 seeds: 42, 123, 456; 14 GHL streams).
- `scripts/run_protocol_ablation.py`, `scripts/analyze_protocol_ablation.py`, and `reports/protocol_ablation_summary.md`: 16-cell factorial ablation isolating calibration source (training vs. held-out), post-processing (raw vs. moving-average and event-filter), EVT thresholder (SPOT vs. SPOT+kurtosis), and training epochs (10 vs. 20).
- `reports/multiseed_heldout_benchmark.csv`: Controlled regularizer ablations on seven streams with matched-capacity controls (turning off spectral entropy, stress closure, and Grassmannian regime routing).

**Corrections to our earlier response letter**
1. The TimesNet and TranAD rows of the computational table in the first letter were not backed by `computational_profile_benchmark.csv` and are withdrawn. The baselines are faster than the JEPAs (0.19 and 0.29 ms against 0.61 to 0.84 ms per window).
2. The first letter reported 5-seed means and Wilcoxon statistics (Concern 3.4). No artifact supported them and the manuscript's original 45-stream results used one seed. Those numbers are withdrawn. The multi-seed held-out results below replace them.
3. The first letter reported that EVT matched its design risk. The threshold in that experiment was fit on the same scores it was evaluated on. That circular agreement is withdrawn.

## Headline result (33-Stream Replication)

Table I summarizes the held-out calibration replication across 33 telemetry streams under buffered mapping (64-step lookahead) with SPOT ($u=0.98, q=10^{-3}$) calibrated on held-out nominal telemetry.

| Model | F1 (non-GHL, 19 streams) | PR-AUC (non-GHL) | Held-out nominal FPR | Event recall (non-GHL) | F1 (dataset-macro) |
|:--|:-:|:-:|:-:|:-:|:-:|
| TS-JEPA | 0.254 | 0.333 | 0.066 | 0.498 | 0.153 |
| Operator-Entropy | 0.241 | 0.348 | 0.079 | 0.498 | 0.134 |
| Reynolds-Stress | 0.208 | 0.321 | 0.076 | 0.398 | 0.119 |
| Potential-Flow | 0.095 | 0.205 | 0.096 | 0.331 | 0.065 |
| TimesNet | 0.207 | 0.336 | 0.297 | 0.586 | 0.104 |
| TranAD | 0.131 | 0.291 | 0.309 | 0.481 | 0.070 |

Key empirical findings from paired stream-level tests (seeds averaged, $N=33$):
1. **False-alarm reduction:** The JEPA family achieves a statistically significant 4- to 5-fold reduction in held-out nominal false-positive rate compared to reconstructive baselines. The mean paired difference in held-out FPR is $-0.154$ against TimesNet (95% bootstrap CI $[-0.249, -0.071]$, Wilcoxon $p=0.008$) and $-0.157$ against TranAD ($[-0.247, -0.082]$, $p=0.001$).
2. **Point-F1 and PR-AUC parity:** Point-F1 and PR-AUC are statistically indistinguishable between TS-JEPA and the baselines. TS-JEPA minus TimesNet is $+0.018$ in F1 ($[-0.052, +0.089]$, $p=0.55$) and $+0.002$ in PR-AUC ($p=0.30$). TS-JEPA minus TranAD is $+0.062$ in F1 ($[-0.012, +0.150]$, $p=0.94$).
3. **No physical-variant benefit:** Over 33 streams, neither Operator-Entropy (F1 diff $-0.007$, $p=0.71$) nor Reynolds-Stress (F1 diff $-0.026$, $p=0.33$) outperforms the unconstrained TS-JEPA control. Potential-Flow performs significantly worse (F1 diff $-0.091$, $p=0.020$; PR-AUC diff $-0.076$, $p=0.039$). The regularizers do not provide a demonstrable empirical benefit over unconstrained latent prediction.
4. **Resolution of 7-stream vs. 33-stream discrepancy:** An earlier 7-stream subset showed much lower JEPA F1 (0.03 vs 0.16 for TimesNet). A 16-cell factorial ablation (`reports/protocol_ablation_summary.md`) confirmed that protocol choices (calibration source, moving-average smoothing, event filtering, kurtosis multiplier, epochs) do not alter the relative ordering on that subset. The divergence was driven by stream selection: TimesNet excels on Daphnet (F1 0.245 vs 0.155), whereas TS-JEPA leads on MSL (0.257 vs 0.159), SMAP (0.441 vs 0.304), and swan (0.191 vs 0.000).

## Point-by-point

**Major 1 (physical interpretation).** The manuscript now refers to all three formulations strictly as physics-inspired inductive biases without claiming demonstrated physical necessity. 
- Reynolds-Stress JEPA was rewritten in Section IV to reflect the code: rank-one stress outer-product with a closure head (loss weight 0.1), with the stress term entering both training and inference scoring.
- Potential-Flow JEPA was rewritten in Section IV to document the actual implementation: continuous velocity field parameterized via a scalar potential network with sinusoidal time embeddings, MovingTangentProjection, Harmonic Grassmannian regime routing ($K=4$), and optimal-transport flow matching.
- Shared latent filtering modules (Hankel moment filters, Littlewood--Paley dyadic frequency shells, Cohn--Elkies shell filters, resolvent purification) are explicitly documented in Section IV.
- Matched controls on seven streams (architecture held fixed, physical term switched off) show no measurable benefit beyond seed noise:
  - Operator-Entropy vs $\lambda_{\text{ent}}=0$: F1 diff $+0.001$, 95% CI $[-0.006, +0.009]$, $p=1.00$.
  - Reynolds-Stress vs stress loss off: $-0.017$ $[-0.044, +0.003]$, $p=0.375$. Vs mean-only score: $-0.024$ $[-0.056, +0.002]$, $p=0.375$.
  - Potential-Flow vs no-regime control ($K=1$): $+0.006$ $[-0.022, +0.032]$, $p=0.58$.

**Major 2 (scoring protocol).** Algorithm 1 explicitly details the streaming score aggregation and thresholding protocol. Both buffered ($S=64$ lookahead) and strictly causal (0 lookahead) mappings are reported for all models. Under causal streaming, JEPA held-out FPR drops to 0.5--1.1% (vs 20--22% for baselines), but causal Point-F1 reverses (TS-JEPA 0.113 vs TimesNet 0.198). This trade-off is explicitly stated.

**Major 3 (EVT calibration).** 
- The 98th vs 95th percentile discrepancy: The core benchmark exclusively uses $u=0.98$; the only $u=0.95$ runs were exploratory sensitivity checks.
- Held-out nominal FPR at design risk $q=10^{-3}$ settles at 5--10% for JEPAs and 24--30% for reconstructive baselines. The 95% bootstrap intervals exclude $q=10^{-3}$ in 57--87% of stream-seed runs.
- The kurtosis multiplier $1 + \frac{1}{6}\tanh\gamma_2$ raised held-out FPR (e.g., TS-JEPA from 0.052 to 0.188) without improving F1. Simple quantile rules (p99.5 or validation maximum) yield identical behavior to SPOT. The claim of adaptive EVT superiority is withdrawn.

**Major 4 (mixed metrics, seeds, tests).** Replicated across 33 streams with up to 3 seeds. Paired stream-level tests, bootstrap intervals, and seed standard deviations are reported in `reports/heldout_38stream_summary.md` and integrated into Section V-E of the manuscript.

**Major 5 (ablations).** Completed for all three physical regularizers and Grassmannian regime routing via matched controls. Shared modules are held constant to isolate the inductive priors.

**Major 6 (reproducibility).** All replication benchmarks are tracked in `scripts/run_multiseed_heldout_benchmark.py` and `scripts/run_protocol_ablation.py`. Output CSVs and summary markdown files are committed to the repository. The manuscript acknowledges that the original 45-stream CSV generator is absent from the repo and treats those legacy results as unreproduced.

**Major 7 (operational incidents).** Section V-D (causal vs. buffered detection) has been updated with explicit methodological caveats explaining that the initial 5-stream evaluation used training-residual calibration and that TS-JEPA's false-alarm reduction incurs a penalty in early event recall (10.7% vs 22.9% for TimesNet).

## What remains open
1. Expanding the 3-seed replication to the 12 single-seed GHL streams (currently 1 seed due to runtime).
2. Implementing an accelerated sliding-window inference path to evaluate large telemetry streams (GECCO and cicids) under held-out calibration.
3. Adding formal mathematical specifications of wavelet filter coefficients and software environment versions to the reproducibility appendix.
4. Matched-capacity unconstrained transition models and generic regularizers (Frobenius decay, trace penalties) as additional baseline controls.
