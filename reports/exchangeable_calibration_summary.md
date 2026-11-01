# Exchangeable / Shuffle Calibration Control Summary

Addresses Reviewer Concern A.2: Disentangling Inter-Block Distribution Drift from Finite-Sample Scarcity.

## Overview
- **Chronological:** Threshold calibrated on the nominal training tail, evaluated on the subsequent nominal block (subject to temporal drift).
- **Point Shuffle:** Calibration and evaluation sets drawn exchangeably by uniform random permutation of pooled nominal blocks (removes both temporal drift and serial autocorrelation).
- **Block Shuffle ($L=64$):** Calibration and evaluation sets drawn exchangeably by permuting 64-step contiguous blocks (eliminates macro drift between blocks while preserving intra-block autocorrelation).

## 1. Primary Design Point ($q = 10^{-3}$, Full Calibration Split)

### Strictly Causal Mapping (Trailing)
| Model | Chrono FPR | Point-Shuffle FPR | Block-Shuffle FPR | Chrono Overshoot | Point Overshoot | Block Overshoot |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| OpEntropy | 0.0065 | 0.0024 | 0.0046 | 6.49x | 2.36x | 4.57x |
| PotentialFlow | 0.0075 | 0.0018 | 0.0029 | 7.52x | 1.81x | 2.90x |
| Reynolds | 0.0022 | 0.0030 | 0.0037 | 2.24x | 2.95x | 3.72x |
| TCN-ObsPred-LN | 0.1422 | 0.0120 | 0.0577 | 142.17x | 12.03x | 57.70x |
| TS-JEPA | 0.0191 | 0.0017 | 0.0053 | 19.14x | 1.68x | 5.27x |
| TimesNet | 0.0872 | 0.0032 | 0.0298 | 87.15x | 3.23x | 29.84x |
| TranAD | 0.0741 | 0.0045 | 0.0150 | 74.12x | 4.53x | 15.05x |

### Buffered Mapping (Smear)
| Model | Chrono FPR | Point-Shuffle FPR | Block-Shuffle FPR | Chrono Overshoot | Point Overshoot | Block Overshoot |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| OpEntropy | 0.0192 | 0.0043 | 0.0049 | 19.18x | 4.34x | 4.88x |
| PotentialFlow | 0.0268 | 0.0034 | 0.0067 | 26.84x | 3.39x | 6.67x |
| Reynolds | 0.0034 | 0.0041 | 0.0068 | 3.44x | 4.13x | 6.77x |
| TCN-ObsPred-LN | 0.2065 | 0.0153 | 0.0431 | 206.55x | 15.31x | 43.06x |
| TS-JEPA | 0.0293 | 0.0026 | 0.0361 | 29.30x | 2.60x | 36.10x |
| TimesNet | 0.0582 | 0.0044 | 0.0232 | 58.22x | 4.40x | 23.18x |
| TranAD | 0.1028 | 0.0042 | 0.0220 | 102.77x | 4.17x | 21.96x |

## 2. Sweep across Design Risk $q \in \{10^{-3}, 10^{-2}, 5 \times 10^{-2}\}$ (Trailing, TS-JEPA)
| Design $q$ | Chrono FPR | Point-Shuffle FPR | Block-Shuffle FPR | Chrono Overshoot | Point Overshoot | Block Overshoot |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.001 | 0.0191 | 0.0017 | 0.0053 | 19.14x | 1.68x | 5.27x |
| 0.010 | 0.0264 | 0.0043 | 0.0117 | 2.64x | 0.43x | 1.17x |
| 0.050 | 0.6394 | 0.6052 | 0.5957 | 12.79x | 12.10x | 11.91x |

## 3. Calibration Horizon Sensitivity (Full vs. 50% Calibration Split, TS-JEPA, Trailing, $q=10^{-3}$)
| Calibration Split | Chrono FPR | Point-Shuffle FPR | Block-Shuffle FPR | Chrono Overshoot | Point Overshoot | Block Overshoot |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 50% N_cal | 0.0303 | 0.0051 | 0.0156 | 30.32x | 5.14x | 15.57x |
| 100% N_cal | 0.0191 | 0.0017 | 0.0053 | 19.14x | 1.68x | 5.27x |