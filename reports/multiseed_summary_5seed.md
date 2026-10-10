Streams: 33; seeds: [np.int64(7), np.int64(42), np.int64(123), np.int64(456), np.int64(2024)]; rows: 6648

### Buffered (64-step lookahead), SPOT(u=0.98, q=1e-3) fit on held-out calibration half
| Model | Point-F1 (macro, mean ± seed SD) | PR-AUC | Held-out nominal FPR | Test FPR | Event recall |
|:--|:-:|:-:|:-:|:-:|:-:|
| TS-JEPA | 0.2118 ± 0.0272 | 0.2924 | 0.0690 | 0.1846 | 0.4353 |
| OpEntropy | 0.2144 ± 0.0114 | 0.2930 | 0.0784 | 0.2166 | 0.4574 |
| Reynolds | 0.1887 ± 0.0462 | 0.2801 | 0.0706 | 0.1774 | 0.4024 |
| PotentialFlow | 0.0803 ± 0.0323 | 0.1826 | 0.0825 | 0.1372 | 0.2739 |
| TimesNet | 0.1786 ± 0.0158 | 0.2978 | 0.2765 | 0.2991 | 0.6229 |
| TranAD | 0.1180 ± 0.0036 | 0.2601 | 0.2815 | 0.2998 | 0.5384 |

### Strictly causal (0 lookahead), SPOT(u=0.98, q=1e-3) fit on held-out calibration half
| Model | Point-F1 (macro, mean ± seed SD) | PR-AUC | Held-out nominal FPR | Test FPR | Event recall |
|:--|:-:|:-:|:-:|:-:|:-:|
| TS-JEPA | 0.1006 ± 0.0371 | 0.2408 | 0.0124 | 0.0764 | 0.4190 |
| OpEntropy | 0.0984 ± 0.0076 | 0.2282 | 0.0174 | 0.0888 | 0.4223 |
| Reynolds | 0.0805 ± 0.0331 | 0.2457 | 0.0103 | 0.0761 | 0.4207 |
| PotentialFlow | 0.0210 ± 0.0149 | 0.1603 | 0.0089 | 0.0262 | 0.3086 |
| TimesNet | 0.1884 ± 0.0238 | 0.2639 | 0.2235 | 0.2713 | 0.5993 |
| TranAD | 0.1268 ± 0.0137 | 0.2236 | 0.2030 | 0.2546 | 0.5700 |

### Paired differences (buffered, SPOT), stream-level, seeds averaged
| Comparison | Metric | Mean diff | 95% bootstrap CI | Wilcoxon p | Streams A>B |
|:--|:--|:-:|:-:|:-:|:-:|
| TS-JEPA vs TimesNet | f1 | +0.0118 | [-0.0528, +0.0748] | 0.537 | 10/33 |
| TS-JEPA vs TimesNet | pr_auc | -0.0012 | [-0.0986, +0.0920] | 0.447 | 20/33 |
| TS-JEPA vs TimesNet | hold_fpr | -0.1528 | [-0.2474, -0.0713] | 0.002 | 11/33 |
| TS-JEPA vs TranAD | f1 | +0.0515 | [-0.0135, +0.1289] | 0.765 | 10/33 |
| TS-JEPA vs TranAD | pr_auc | +0.0227 | [-0.0775, +0.1241] | 0.208 | 22/33 |
| TS-JEPA vs TranAD | hold_fpr | -0.1506 | [-0.2472, -0.0703] | 0.003 | 11/33 |
| OpEntropy vs TS-JEPA | f1 | +0.0021 | [-0.0245, +0.0290] | 0.709 | 9/33 |
| OpEntropy vs TS-JEPA | pr_auc | +0.0017 | [-0.0196, +0.0233] | 0.860 | 14/33 |
| OpEntropy vs TS-JEPA | hold_fpr | +0.0090 | [-0.0128, +0.0331] | 0.751 | 16/33 |
| Reynolds vs TS-JEPA | f1 | -0.0145 | [-0.0419, +0.0074] | 0.614 | 11/33 |
| Reynolds vs TS-JEPA | pr_auc | -0.0073 | [-0.0249, +0.0111] | 0.130 | 12/33 |
| Reynolds vs TS-JEPA | hold_fpr | +0.0005 | [-0.0231, +0.0265] | 0.916 | 16/33 |
| PotentialFlow vs TS-JEPA | f1 | -0.0810 | [-0.1597, -0.0176] | 0.033 | 7/33 |
| PotentialFlow vs TS-JEPA | pr_auc | -0.0673 | [-0.1316, -0.0126] | 0.584 | 17/33 |
| PotentialFlow vs TS-JEPA | hold_fpr | +0.0112 | [-0.0120, +0.0354] | 0.321 | 21/33 |
| OpEntropy vs TimesNet | f1 | +0.0139 | [-0.0483, +0.0746] | 0.911 | 12/33 |
| OpEntropy vs TimesNet | pr_auc | +0.0005 | [-0.0949, +0.0921] | 0.257 | 20/33 |
| OpEntropy vs TimesNet | hold_fpr | -0.1438 | [-0.2333, -0.0643] | 0.006 | 11/33 |
| OpEntropy vs TranAD | f1 | +0.0536 | [-0.0074, +0.1267] | 0.985 | 11/33 |
| OpEntropy vs TranAD | pr_auc | +0.0244 | [-0.0705, +0.1247] | 0.280 | 20/33 |
| OpEntropy vs TranAD | hold_fpr | -0.1417 | [-0.2402, -0.0543] | 0.006 | 12/33 |

### Calibrator comparison on identical score streams (buffered)
| Model | Calibrator | Held-out nominal FPR (mean) | Share of rows whose 95% CI excludes q=1e-3 | Point-F1 |
|:--|:--|:-:|:-:|:-:|
| TS-JEPA | SPOT | 0.0526 | 0.61 | 0.1577 |
| TS-JEPA | SPOT+kurt(1/6) | 0.1768 | 0.82 | 0.1694 |
| TS-JEPA | p99.5 | 0.0682 | 0.55 | 0.1598 |
| TS-JEPA | max-val | 0.0553 | 0.65 | 0.1591 |
| OpEntropy | SPOT | 0.0617 | 0.65 | 0.1599 |
| OpEntropy | SPOT+kurt(1/6) | 0.1229 | 0.76 | 0.1441 |
| OpEntropy | p99.5 | 0.0758 | 0.62 | 0.1658 |
| OpEntropy | max-val | 0.0656 | 0.64 | 0.1627 |
| TimesNet | SPOT | 0.2217 | 0.86 | 0.1395 |
| TimesNet | SPOT+kurt(1/6) | 0.2281 | 0.82 | 0.1352 |
| TimesNet | p99.5 | 0.2338 | 0.87 | 0.1413 |
| TimesNet | max-val | 0.2263 | 0.86 | 0.1410 |
| TranAD | SPOT | 0.2217 | 0.82 | 0.0936 |
| TranAD | SPOT+kurt(1/6) | 0.2453 | 0.84 | 0.0940 |
| TranAD | p99.5 | 0.2343 | 0.84 | 0.0962 |
| TranAD | max-val | 0.2270 | 0.84 | 0.0952 |

### Calibration gap: SPOT design risk 1e-3 vs measured held-out FPR, by model (buffered)
| model         |   hold_fpr |   hold_fpr_lo |   hold_fpr_hi |
|:--------------|-----------:|--------------:|--------------:|
| OpEntropy     |     0.0617 |        0.0277 |        0.1037 |
| PotentialFlow |     0.0645 |        0.0304 |        0.1054 |
| Reynolds      |     0.0534 |        0.0241 |        0.0904 |
| TS-JEPA       |     0.0526 |        0.0168 |        0.0996 |
| TimesNet      |     0.2217 |        0.118  |        0.3228 |
| TranAD        |     0.2217 |        0.1178 |        0.3294 |