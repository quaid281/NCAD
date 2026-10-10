Streams: 33; seeds: [np.int64(7), np.int64(42), np.int64(123), np.int64(456), np.int64(2024)]; rows: 4008

### Buffered (64-step lookahead), SPOT(u=0.98, q=1e-3) fit on held-out calibration half
| Model | Point-F1 (macro, mean ± seed SD) | PR-AUC | Held-out nominal FPR | Test FPR | Event recall |
|:--|:-:|:-:|:-:|:-:|:-:|
| TS-JEPA | 0.1279 ± 0.0477 | 0.1332 | 0.0621 | 0.4273 | 0.5745 |
| OpEntropy | 0.1461 ± 0.0378 | 0.1582 | 0.0146 | 0.3414 | 0.5002 |
| Reynolds | 0.1047 ± 0.0360 | 0.1500 | 0.0325 | 0.3364 | 0.3453 |
| PotentialFlow | 0.0507 ± 0.0384 | 0.1419 | 0.1204 | 0.2621 | 0.3179 |
| TimesNet | 0.1981 ± 0.0165 | 0.2856 | 0.1922 | 0.2158 | 0.5013 |
| TranAD | 0.1196 ± 0.0215 | 0.2124 | 0.1582 | 0.2440 | 0.4612 |

### Strictly causal (0 lookahead), SPOT(u=0.98, q=1e-3) fit on held-out calibration half
| Model | Point-F1 (macro, mean ± seed SD) | PR-AUC | Held-out nominal FPR | Test FPR | Event recall |
|:--|:-:|:-:|:-:|:-:|:-:|
| TS-JEPA | 0.0717 ± 0.0243 | 0.1317 | 0.0101 | 0.2456 | 0.6508 |
| OpEntropy | 0.0794 ± 0.0328 | 0.1561 | 0.0047 | 0.2013 | 0.5553 |
| Reynolds | 0.0316 ± 0.0133 | 0.1430 | 0.0064 | 0.1919 | 0.4507 |
| PotentialFlow | 0.0347 ± 0.0243 | 0.1481 | 0.0158 | 0.0827 | 0.5330 |
| TimesNet | 0.1713 ± 0.0168 | 0.2916 | 0.1875 | 0.2049 | 0.5174 |
| TranAD | 0.1124 ± 0.0232 | 0.2177 | 0.0988 | 0.2240 | 0.5002 |

### Paired differences (buffered, SPOT), stream-level, seeds averaged
| Comparison | Metric | Mean diff | 95% bootstrap CI | Wilcoxon p | Streams A>B |
|:--|:--|:-:|:-:|:-:|:-:|
| TS-JEPA vs TimesNet | f1 | +0.0143 | [-0.0544, +0.0846] | 0.537 | 10/33 |
| TS-JEPA vs TimesNet | pr_auc | +0.0012 | [-0.0982, +0.0958] | 0.304 | 22/33 |
| TS-JEPA vs TimesNet | hold_fpr | -0.1522 | [-0.2476, -0.0685] | 0.010 | 11/33 |
| TS-JEPA vs TranAD | f1 | +0.0586 | [-0.0146, +0.1475] | 0.894 | 10/33 |
| TS-JEPA vs TranAD | pr_auc | +0.0275 | [-0.0744, +0.1346] | 0.304 | 21/33 |
| TS-JEPA vs TranAD | hold_fpr | -0.1538 | [-0.2440, -0.0776] | 0.002 | 9/33 |
| OpEntropy vs TS-JEPA | f1 | -0.0037 | [-0.0256, +0.0170] | 0.777 | 9/33 |
| OpEntropy vs TS-JEPA | pr_auc | +0.0092 | [-0.0119, +0.0357] | 0.313 | 11/33 |
| OpEntropy vs TS-JEPA | hold_fpr | +0.0045 | [-0.0265, +0.0411] | 0.309 | 12/33 |
| Reynolds vs TS-JEPA | f1 | -0.0251 | [-0.0697, +0.0038] | 0.313 | 9/33 |
| Reynolds vs TS-JEPA | pr_auc | -0.0090 | [-0.0235, +0.0078] | 0.012 | 9/33 |
| Reynolds vs TS-JEPA | hold_fpr | +0.0029 | [-0.0266, +0.0393] | 0.852 | 16/33 |
| PotentialFlow vs TS-JEPA | f1 | -0.0892 | [-0.1748, -0.0179] | 0.016 | 5/33 |
| PotentialFlow vs TS-JEPA | pr_auc | -0.0755 | [-0.1418, -0.0188] | 0.023 | 11/33 |
| PotentialFlow vs TS-JEPA | hold_fpr | +0.0159 | [-0.0137, +0.0493] | 0.859 | 16/33 |
| OpEntropy vs TimesNet | f1 | +0.0106 | [-0.0608, +0.0838] | 0.421 | 10/33 |
| OpEntropy vs TimesNet | pr_auc | +0.0104 | [-0.0951, +0.1168] | 0.416 | 20/33 |
| OpEntropy vs TimesNet | hold_fpr | -0.1476 | [-0.2353, -0.0688] | 0.001 | 7/33 |
| OpEntropy vs TranAD | f1 | +0.0549 | [-0.0133, +0.1364] | 0.614 | 9/33 |
| OpEntropy vs TranAD | pr_auc | +0.0368 | [-0.0677, +0.1479] | 0.348 | 19/33 |
| OpEntropy vs TranAD | hold_fpr | -0.1493 | [-0.2467, -0.0637] | 0.003 | 8/33 |

### Calibrator comparison on identical score streams (buffered)
| Model | Calibrator | Held-out nominal FPR (mean) | Share of rows whose 95% CI excludes q=1e-3 | Point-F1 |
|:--|:--|:-:|:-:|:-:|
| TS-JEPA | SPOT | 0.0549 | 0.59 | 0.1807 |
| TS-JEPA | SPOT+kurt(1/6) | 0.2168 | 0.85 | 0.1894 |
| TS-JEPA | p99.5 | 0.0692 | 0.49 | 0.1830 |
| TS-JEPA | max-val | 0.0563 | 0.60 | 0.1807 |
| OpEntropy | SPOT | 0.0562 | 0.67 | 0.1783 |
| OpEntropy | SPOT+kurt(1/6) | 0.1231 | 0.76 | 0.1615 |
| OpEntropy | p99.5 | 0.0701 | 0.61 | 0.1847 |
| OpEntropy | max-val | 0.0626 | 0.64 | 0.1807 |
| TimesNet | SPOT | 0.2334 | 0.84 | 0.1651 |
| TimesNet | SPOT+kurt(1/6) | 0.2449 | 0.85 | 0.1563 |
| TimesNet | p99.5 | 0.2426 | 0.87 | 0.1663 |
| TimesNet | max-val | 0.2374 | 0.84 | 0.1658 |
| TranAD | SPOT | 0.2339 | 0.81 | 0.1048 |
| TranAD | SPOT+kurt(1/6) | 0.2691 | 0.81 | 0.1055 |
| TranAD | p99.5 | 0.2490 | 0.86 | 0.1080 |
| TranAD | max-val | 0.2414 | 0.82 | 0.1069 |

### Calibration gap: SPOT design risk 1e-3 vs measured held-out FPR, by model (buffered)
| model         |   hold_fpr |   hold_fpr_lo |   hold_fpr_hi |
|:--------------|-----------:|--------------:|--------------:|
| OpEntropy     |     0.0562 |        0.0249 |        0.0966 |
| PotentialFlow |     0.0817 |        0.0387 |        0.1336 |
| Reynolds      |     0.0568 |        0.0228 |        0.1    |
| TS-JEPA       |     0.0549 |        0.0141 |        0.1092 |
| TimesNet      |     0.2334 |        0.1248 |        0.3416 |
| TranAD        |     0.2339 |        0.1254 |        0.3505 |