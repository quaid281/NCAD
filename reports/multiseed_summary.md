Streams: 33; seeds: [np.int64(7), np.int64(42), np.int64(123), np.int64(456), np.int64(2024)]; rows: 5304

### Buffered (64-step lookahead), SPOT(u=0.98, q=1e-3) fit on held-out calibration half
| Model | Point-F1 (macro, mean ± seed SD) | PR-AUC | Held-out nominal FPR | Test FPR | Event recall |
|:--|:-:|:-:|:-:|:-:|:-:|
| TS-JEPA | 0.2341 ± 0.0300 | 0.3229 | 0.0761 | 0.2035 | 0.4811 |
| OpEntropy | 0.2369 ± 0.0126 | 0.3233 | 0.0862 | 0.2393 | 0.5056 |
| Reynolds | 0.2075 ± 0.0515 | 0.3071 | 0.0777 | 0.1956 | 0.4343 |
| PotentialFlow | 0.0887 ± 0.0357 | 0.2012 | 0.0908 | 0.1514 | 0.3028 |
| TimesNet | 0.1970 ± 0.0175 | 0.3290 | 0.3001 | 0.2491 | 0.5832 |
| TranAD | 0.1299 ± 0.0041 | 0.2872 | 0.3070 | 0.2531 | 0.4898 |

### Strictly causal (0 lookahead), SPOT(u=0.98, q=1e-3) fit on held-out calibration half
| Model | Point-F1 (macro, mean ± seed SD) | PR-AUC | Held-out nominal FPR | Test FPR | Event recall |
|:--|:-:|:-:|:-:|:-:|:-:|
| TS-JEPA | 0.1118 ± 0.0412 | 0.2672 | 0.0136 | 0.0844 | 0.4655 |
| OpEntropy | 0.1093 ± 0.0085 | 0.2528 | 0.0189 | 0.0986 | 0.4692 |
| Reynolds | 0.0876 ± 0.0367 | 0.2700 | 0.0111 | 0.0842 | 0.4563 |
| PotentialFlow | 0.0233 ± 0.0166 | 0.1775 | 0.0097 | 0.0289 | 0.3428 |
| TimesNet | 0.2091 ± 0.0265 | 0.2931 | 0.2427 | 0.2163 | 0.5882 |
| TranAD | 0.1405 ± 0.0154 | 0.2483 | 0.2219 | 0.2027 | 0.5334 |

### Paired differences (buffered, SPOT), stream-level, seeds averaged
| Comparison | Metric | Mean diff | 95% bootstrap CI | Wilcoxon p | Streams A>B |
|:--|:--|:-:|:-:|:-:|:-:|
| TS-JEPA vs TimesNet | f1 | +0.0120 | [-0.0526, +0.0749] | 0.537 | 10/33 |
| TS-JEPA vs TimesNet | pr_auc | -0.0003 | [-0.0979, +0.0922] | 0.386 | 21/33 |
| TS-JEPA vs TimesNet | hold_fpr | -0.1496 | [-0.2445, -0.0674] | 0.007 | 12/33 |
| TS-JEPA vs TranAD | f1 | +0.0512 | [-0.0148, +0.1285] | 0.877 | 10/33 |
| TS-JEPA vs TranAD | pr_auc | +0.0233 | [-0.0766, +0.1245] | 0.155 | 22/33 |
| TS-JEPA vs TranAD | hold_fpr | -0.1499 | [-0.2456, -0.0695] | 0.004 | 11/33 |
| OpEntropy vs TS-JEPA | f1 | +0.0019 | [-0.0250, +0.0287] | 0.687 | 8/33 |
| OpEntropy vs TS-JEPA | pr_auc | -0.0002 | [-0.0214, +0.0203] | 0.491 | 11/33 |
| OpEntropy vs TS-JEPA | hold_fpr | +0.0041 | [-0.0181, +0.0291] | 0.769 | 14/33 |
| Reynolds vs TS-JEPA | f1 | -0.0149 | [-0.0425, +0.0070] | 0.502 | 10/33 |
| Reynolds vs TS-JEPA | pr_auc | -0.0112 | [-0.0282, +0.0071] | 0.014 | 10/33 |
| Reynolds vs TS-JEPA | hold_fpr | -0.0015 | [-0.0254, +0.0241] | 0.805 | 16/33 |
| PotentialFlow vs TS-JEPA | f1 | -0.0837 | [-0.1619, -0.0214] | 0.013 | 5/33 |
| PotentialFlow vs TS-JEPA | pr_auc | -0.0715 | [-0.1356, -0.0175] | 0.060 | 12/33 |
| PotentialFlow vs TS-JEPA | hold_fpr | +0.0067 | [-0.0165, +0.0314] | 0.957 | 17/33 |
| OpEntropy vs TimesNet | f1 | +0.0138 | [-0.0485, +0.0746] | 0.940 | 12/33 |
| OpEntropy vs TimesNet | pr_auc | -0.0006 | [-0.0960, +0.0894] | 0.242 | 21/33 |
| OpEntropy vs TimesNet | hold_fpr | -0.1455 | [-0.2355, -0.0676] | 0.003 | 9/33 |
| OpEntropy vs TranAD | f1 | +0.0531 | [-0.0087, +0.1264] | 0.829 | 11/33 |
| OpEntropy vs TranAD | pr_auc | +0.0231 | [-0.0719, +0.1224] | 0.288 | 19/33 |
| OpEntropy vs TranAD | hold_fpr | -0.1458 | [-0.2454, -0.0602] | 0.002 | 8/33 |

### Calibrator comparison on identical score streams (buffered)
| Model | Calibrator | Held-out nominal FPR (mean) | Share of rows whose 95% CI excludes q=1e-3 | Point-F1 |
|:--|:--|:-:|:-:|:-:|
| TS-JEPA | SPOT | 0.0651 | 0.64 | 0.1968 |
| TS-JEPA | SPOT+kurt(1/6) | 0.2205 | 0.84 | 0.2113 |
| TS-JEPA | p99.5 | 0.0838 | 0.58 | 0.1995 |
| TS-JEPA | max-val | 0.0686 | 0.65 | 0.1985 |
| OpEntropy | SPOT | 0.0733 | 0.66 | 0.1992 |
| OpEntropy | SPOT+kurt(1/6) | 0.1506 | 0.78 | 0.1795 |
| OpEntropy | p99.5 | 0.0891 | 0.64 | 0.2060 |
| OpEntropy | max-val | 0.0790 | 0.65 | 0.2027 |
| TimesNet | SPOT | 0.2609 | 0.87 | 0.1685 |
| TimesNet | SPOT+kurt(1/6) | 0.2748 | 0.87 | 0.1634 |
| TimesNet | p99.5 | 0.2695 | 0.89 | 0.1706 |
| TimesNet | max-val | 0.2651 | 0.87 | 0.1702 |
| TranAD | SPOT | 0.2655 | 0.84 | 0.1119 |
| TranAD | SPOT+kurt(1/6) | 0.2962 | 0.84 | 0.1130 |
| TranAD | p99.5 | 0.2786 | 0.88 | 0.1148 |
| TranAD | max-val | 0.2725 | 0.85 | 0.1139 |

### Calibration gap: SPOT design risk 1e-3 vs measured held-out FPR, by model (buffered)
| model         |   hold_fpr |   hold_fpr_lo |   hold_fpr_hi |
|:--------------|-----------:|--------------:|--------------:|
| OpEntropy     |     0.0733 |        0.0332 |        0.1223 |
| PotentialFlow |     0.0771 |        0.0371 |        0.1247 |
| Reynolds      |     0.0659 |        0.0298 |        0.111  |
| TS-JEPA       |     0.0651 |        0.021  |        0.1229 |
| TimesNet      |     0.2609 |        0.1406 |        0.3763 |
| TranAD        |     0.2655 |        0.1426 |        0.3918 |