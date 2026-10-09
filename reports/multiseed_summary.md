Streams: 7; seeds: [42, 123, 456]; rows: 1680

### Buffered (64-step lookahead), SPOT(u=0.98, q=1e-3) fit on held-out calibration half
| Model | Point-F1 (macro, mean ± seed SD) | PR-AUC | Held-out nominal FPR | Test FPR | Event recall |
|:--|:-:|:-:|:-:|:-:|:-:|
| TS-JEPA | 0.0317 ± 0.0176 | 0.1371 | 0.0467 | 0.0859 | 0.0992 |
| OpEntropy | 0.0178 ± 0.0187 | 0.1563 | 0.0637 | 0.0634 | 0.1071 |
| OpEntropy-lambda0 | 0.0168 ± 0.0146 | 0.1547 | 0.0801 | 0.0771 | 0.1310 |
| Reynolds | 0.0165 ± 0.0116 | 0.1563 | 0.0423 | 0.0888 | 0.1488 |
| Reynolds-meanscore | 0.0404 ± 0.0218 | 0.1580 | 0.0521 | 0.1017 | 0.2282 |
| Reynolds-alpha0 | 0.0333 ± 0.0349 | 0.1442 | 0.0422 | 0.0848 | 0.1607 |
| PotentialFlow | 0.0743 ± 0.0691 | 0.1698 | 0.0510 | 0.0725 | 0.3070 |
| PotentialFlow-noregime | 0.0680 ± 0.0818 | 0.1712 | 0.0491 | 0.0531 | 0.2200 |
| TimesNet | 0.1565 ± 0.0303 | 0.3546 | 0.3561 | 0.3378 | 0.7282 |
| TranAD | 0.1610 ± 0.0069 | 0.3899 | 0.4489 | 0.3242 | 0.6905 |

### Strictly causal (0 lookahead), SPOT(u=0.98, q=1e-3) fit on held-out calibration half
| Model | Point-F1 (macro, mean ± seed SD) | PR-AUC | Held-out nominal FPR | Test FPR | Event recall |
|:--|:-:|:-:|:-:|:-:|:-:|
| TS-JEPA | 0.0092 ± 0.0033 | 0.1356 | 0.0067 | 0.0147 | 0.3075 |
| OpEntropy | 0.0043 ± 0.0015 | 0.1411 | 0.0151 | 0.0231 | 0.3937 |
| OpEntropy-lambda0 | 0.0054 ± 0.0011 | 0.1391 | 0.0326 | 0.0258 | 0.3654 |
| Reynolds | 0.0177 ± 0.0123 | 0.1370 | 0.0081 | 0.0293 | 0.2857 |
| Reynolds-meanscore | 0.0118 ± 0.0096 | 0.1400 | 0.0111 | 0.0392 | 0.2812 |
| Reynolds-alpha0 | 0.0146 ± 0.0122 | 0.1354 | 0.0070 | 0.0298 | 0.2545 |
| PotentialFlow | 0.0156 ± 0.0016 | 0.1595 | 0.0032 | 0.0096 | 0.5006 |
| PotentialFlow-noregime | 0.0111 ± 0.0081 | 0.1617 | 0.0028 | 0.0083 | 0.3665 |
| TimesNet | 0.1792 ± 0.0389 | 0.2559 | 0.2673 | 0.2962 | 0.7534 |
| TranAD | 0.1759 ± 0.0016 | 0.2639 | 0.3153 | 0.2771 | 0.7619 |

### Paired differences (buffered, SPOT), stream-level, seeds averaged
| Comparison | Metric | Mean diff | 95% bootstrap CI | Wilcoxon p | Streams A>B |
|:--|:--|:-:|:-:|:-:|:-:|
| TS-JEPA vs TimesNet | f1 | -0.1248 | [-0.2046, -0.0451] | 0.031 | 0/7 |
| TS-JEPA vs TimesNet | pr_auc | -0.2175 | [-0.4695, -0.0399] | 0.047 | 1/7 |
| TS-JEPA vs TimesNet | hold_fpr | -0.3093 | [-0.5562, -0.0805] | 0.125 | 1/7 |
| TS-JEPA vs TranAD | f1 | -0.1292 | [-0.2330, -0.0379] | 0.094 | 1/7 |
| TS-JEPA vs TranAD | pr_auc | -0.2528 | [-0.5078, -0.0288] | 0.109 | 2/7 |
| TS-JEPA vs TranAD | hold_fpr | -0.4022 | [-0.5919, -0.1665] | 0.047 | 1/7 |
| OpEntropy vs TS-JEPA | f1 | -0.0139 | [-0.0293, -0.0015] | 0.125 | 0/7 |
| OpEntropy vs TS-JEPA | pr_auc | +0.0192 | [-0.0180, +0.0762] | 0.812 | 3/7 |
| OpEntropy vs TS-JEPA | hold_fpr | +0.0169 | [-0.0074, +0.0403] | 0.312 | 4/7 |
| OpEntropy vs OpEntropy-lambda0 | f1 | +0.0010 | [-0.0058, +0.0089] | 1.000 | 1/7 |
| OpEntropy vs OpEntropy-lambda0 | pr_auc | +0.0016 | [+0.0002, +0.0030] | 0.156 | 5/7 |
| OpEntropy vs OpEntropy-lambda0 | hold_fpr | -0.0164 | [-0.0561, +0.0083] | 0.812 | 2/7 |
| Reynolds vs TS-JEPA | f1 | -0.0152 | [-0.0478, +0.0200] | 0.625 | 2/7 |
| Reynolds vs TS-JEPA | pr_auc | +0.0192 | [-0.0226, +0.0736] | 1.000 | 3/7 |
| Reynolds vs TS-JEPA | hold_fpr | -0.0044 | [-0.0167, +0.0052] | 0.812 | 3/7 |
| Reynolds vs Reynolds-meanscore | f1 | -0.0238 | [-0.0562, +0.0016] | 0.375 | 1/7 |
| Reynolds vs Reynolds-meanscore | pr_auc | -0.0018 | [-0.0151, +0.0130] | 0.297 | 1/7 |
| Reynolds vs Reynolds-meanscore | hold_fpr | -0.0098 | [-0.0279, +0.0002] | 0.375 | 1/7 |
| Reynolds vs Reynolds-alpha0 | f1 | -0.0168 | [-0.0438, +0.0027] | 0.375 | 1/7 |
| Reynolds vs Reynolds-alpha0 | pr_auc | +0.0121 | [-0.0081, +0.0443] | 0.938 | 3/7 |
| Reynolds vs Reynolds-alpha0 | hold_fpr | +0.0002 | [-0.0245, +0.0197] | 1.000 | 3/7 |
| PotentialFlow vs TS-JEPA | f1 | +0.0426 | [-0.0244, +0.1499] | 0.844 | 3/7 |
| PotentialFlow vs TS-JEPA | pr_auc | +0.0326 | [-0.0113, +0.0958] | 0.578 | 4/7 |
| PotentialFlow vs TS-JEPA | hold_fpr | +0.0043 | [-0.0150, +0.0237] | 1.000 | 3/7 |
| PotentialFlow vs PotentialFlow-noregime | f1 | +0.0064 | [-0.0223, +0.0320] | 0.578 | 4/7 |
| PotentialFlow vs PotentialFlow-noregime | pr_auc | -0.0014 | [-0.0209, +0.0195] | 0.938 | 3/7 |
| PotentialFlow vs PotentialFlow-noregime | hold_fpr | +0.0020 | [-0.0215, +0.0269] | 1.000 | 3/7 |
| OpEntropy vs TimesNet | f1 | -0.1387 | [-0.2278, -0.0565] | 0.031 | 0/7 |
| OpEntropy vs TimesNet | pr_auc | -0.1983 | [-0.4814, +0.0200] | 0.156 | 1/7 |
| OpEntropy vs TimesNet | hold_fpr | -0.2924 | [-0.5275, -0.0639] | 0.125 | 1/7 |
| OpEntropy vs TranAD | f1 | -0.1432 | [-0.2550, -0.0356] | 0.062 | 1/7 |
| OpEntropy vs TranAD | pr_auc | -0.2336 | [-0.5299, +0.0027] | 0.156 | 2/7 |
| OpEntropy vs TranAD | hold_fpr | -0.3853 | [-0.5825, -0.1575] | 0.078 | 2/7 |

### Calibrator comparison on identical score streams (buffered)
| Model | Calibrator | Held-out nominal FPR (mean) | Share of rows whose 95% CI excludes q=1e-3 | Point-F1 |
|:--|:--|:-:|:-:|:-:|
| TS-JEPA | SPOT | 0.0467 | 0.81 | 0.0317 |
| TS-JEPA | SPOT+kurt(1/6) | 0.1313 | 0.90 | 0.0630 |
| TS-JEPA | p99.5 | 0.0486 | 0.81 | 0.0329 |
| TS-JEPA | max-val | 0.0466 | 0.76 | 0.0309 |
| OpEntropy | SPOT | 0.0637 | 0.71 | 0.0178 |
| OpEntropy | SPOT+kurt(1/6) | 0.0633 | 0.86 | 0.0273 |
| OpEntropy | p99.5 | 0.0756 | 0.71 | 0.0228 |
| OpEntropy | max-val | 0.0664 | 0.67 | 0.0184 |
| TimesNet | SPOT | 0.3561 | 1.00 | 0.1565 |
| TimesNet | SPOT+kurt(1/6) | 0.3470 | 1.00 | 0.1670 |
| TimesNet | p99.5 | 0.3727 | 1.00 | 0.1676 |
| TimesNet | max-val | 0.3707 | 1.00 | 0.1676 |
| TranAD | SPOT | 0.4489 | 0.95 | 0.1610 |
| TranAD | SPOT+kurt(1/6) | 0.4104 | 1.00 | 0.1759 |
| TranAD | p99.5 | 0.4658 | 0.86 | 0.1673 |
| TranAD | max-val | 0.4642 | 0.95 | 0.1697 |

### Calibration gap: SPOT design risk 1e-3 vs measured held-out FPR, by model (buffered)
| model                  |   hold_fpr |   hold_fpr_lo |   hold_fpr_hi |
|:-----------------------|-----------:|--------------:|--------------:|
| OpEntropy              |     0.0637 |        0.0117 |        0.1247 |
| OpEntropy-lambda0      |     0.0801 |        0.0272 |        0.142  |
| PotentialFlow          |     0.051  |        0.015  |        0.0915 |
| PotentialFlow-noregime |     0.0491 |        0.0097 |        0.0998 |
| Reynolds               |     0.0423 |        0.0073 |        0.084  |
| Reynolds-alpha0        |     0.0422 |        0.011  |        0.0829 |
| Reynolds-meanscore     |     0.0521 |        0.0129 |        0.1073 |
| TS-JEPA                |     0.0467 |        0.0068 |        0.0938 |
| TimesNet               |     0.3561 |        0.2104 |        0.4996 |
| TranAD                 |     0.4489 |        0.2377 |        0.6644 |