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
| OpEntropy | 0.0185 | 0.0022 | 0.0073 | 18.48x | 2.20x | 7.28x |
| PotentialFlow | 0.0100 | 0.0017 | 0.0040 | 10.03x | 1.70x | 3.99x |
| Reynolds | 0.0068 | 0.0022 | 0.0035 | 6.79x | 2.22x | 3.55x |
| TCN-ObsPred-LN | 0.1449 | 0.0115 | 0.0598 | 144.86x | 11.54x | 59.85x |
| TS-JEPA | 0.0121 | 0.0019 | 0.0058 | 12.09x | 1.90x | 5.81x |
| TimesNet | 0.0872 | 0.0037 | 0.0204 | 87.15x | 3.73x | 20.40x |
| TranAD | 0.0741 | 0.0032 | 0.0067 | 74.12x | 3.19x | 6.66x |

### Buffered Mapping (Smear)
| Model | Chrono FPR | Point-Shuffle FPR | Block-Shuffle FPR | Chrono Overshoot | Point Overshoot | Block Overshoot |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| OpEntropy | 0.0150 | 0.0029 | 0.0101 | 14.96x | 2.91x | 10.11x |
| PotentialFlow | 0.0235 | 0.0030 | 0.0059 | 23.46x | 2.98x | 5.95x |
| Reynolds | 0.0099 | 0.0031 | 0.0260 | 9.88x | 3.12x | 25.96x |
| TCN-ObsPred-LN | 0.2008 | 0.0133 | 0.0690 | 200.85x | 13.26x | 68.97x |
| TS-JEPA | 0.0289 | 0.0032 | 0.0085 | 28.91x | 3.21x | 8.45x |
| TimesNet | 0.0582 | 0.0036 | 0.0222 | 58.22x | 3.65x | 22.21x |
| TranAD | 0.1028 | 0.0039 | 0.0219 | 102.77x | 3.86x | 21.86x |

## 2. Sweep across Design Risk $q \in \{10^{-3}, 10^{-2}, 5 \times 10^{-2}\}$ (Trailing, TS-JEPA)
| Design $q$ | Chrono FPR | Point-Shuffle FPR | Block-Shuffle FPR | Chrono Overshoot | Point Overshoot | Block Overshoot |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.001 | 0.0121 | 0.0019 | 0.0058 | 12.09x | 1.90x | 5.81x |
| 0.010 | 0.0203 | 0.0045 | 0.0062 | 2.03x | 0.45x | 0.62x |
| 0.050 | 0.6396 | 0.6130 | 0.6158 | 12.79x | 12.26x | 12.32x |

## 3. Calibration Horizon Sensitivity (Full vs. 50% Calibration Split, TS-JEPA, Trailing, $q=10^{-3}$)
| Calibration Split | Chrono FPR | Point-Shuffle FPR | Block-Shuffle FPR | Chrono Overshoot | Point Overshoot | Block Overshoot |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 50% N_cal | 0.0237 | 0.0082 | 0.0182 | 23.68x | 8.20x | 18.25x |
| 100% N_cal | 0.0121 | 0.0019 | 0.0058 | 12.09x | 1.90x | 5.81x |