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
| SolitonReynolds-JEPA | 0.0033 | 0.0025 | 0.0055 | 3.31x | 2.47x | 5.55x |
| SolitonReynolds-NoSoliton | 0.0036 | 0.0026 | 0.0034 | 3.58x | 2.62x | 3.45x |
| TCN-ObsPred-LN | 0.1165 | 0.0032 | 0.0446 | 116.52x | 3.22x | 44.59x |
| TS-JEPA | 0.0229 | 0.0047 | 0.0077 | 22.91x | 4.69x | 7.70x |
| TimesNet | 0.2302 | 0.0038 | 0.0521 | 230.16x | 3.76x | 52.06x |
| TranAD | 0.2103 | 0.0027 | 0.0529 | 210.27x | 2.67x | 52.91x |

### Buffered Mapping (Smear)
| Model | Chrono FPR | Point-Shuffle FPR | Block-Shuffle FPR | Chrono Overshoot | Point Overshoot | Block Overshoot |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| SolitonReynolds-JEPA | 0.1084 | 0.0020 | 0.0344 | 108.41x | 1.97x | 34.38x |
| SolitonReynolds-NoSoliton | 0.1216 | 0.0031 | 0.0393 | 121.56x | 3.15x | 39.29x |
| TCN-ObsPred-LN | 0.1934 | 0.0034 | 0.0595 | 193.36x | 3.44x | 59.55x |
| TS-JEPA | 0.0932 | 0.0024 | 0.0419 | 93.18x | 2.38x | 41.93x |
| TimesNet | 0.2994 | 0.0026 | 0.0754 | 299.44x | 2.63x | 75.37x |
| TranAD | 0.3070 | 0.0023 | 0.0545 | 307.04x | 2.28x | 54.54x |

## 2. Sweep across Design Risk $q \in \{10^{-3}, 10^{-2}, 5 \times 10^{-2}\}$ (Trailing, TS-JEPA)
| Design $q$ | Chrono FPR | Point-Shuffle FPR | Block-Shuffle FPR | Chrono Overshoot | Point Overshoot | Block Overshoot |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.001 | 0.0229 | 0.0047 | 0.0077 | 22.91x | 4.69x | 7.70x |
| 0.010 | 0.0391 | 0.0117 | 0.0260 | 3.91x | 1.17x | 2.60x |
| 0.050 | 0.0718 | 0.0470 | 0.0494 | 1.44x | 0.94x | 0.99x |

## 3. Calibration Horizon Sensitivity (Full vs. 50% Calibration Split, TS-JEPA, Trailing, $q=10^{-3}$)
| Calibration Split | Chrono FPR | Point-Shuffle FPR | Block-Shuffle FPR | Chrono Overshoot | Point Overshoot | Block Overshoot |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 50% N_cal | 0.0426 | 0.0075 | 0.0208 | 42.59x | 7.53x | 20.82x |
| 100% N_cal | 0.0229 | 0.0047 | 0.0077 | 22.91x | 4.69x | 7.70x |