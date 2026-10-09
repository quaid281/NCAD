Streams: 33; seeds: [42, 123, 456]; rows: 3528

### Buffered (64-step lookahead), SPOT(u=0.98, q=1e-3) fit on held-out calibration half
| Model | Point-F1 (macro, mean ± seed SD) | PR-AUC | Held-out nominal FPR | Test FPR | Event recall |
|:--|:-:|:-:|:-:|:-:|:-:|
| TS-JEPA | 0.2296 ± 0.0132 | 0.3019 | 0.0594 | 0.1798 | 0.4504 |
| OpEntropy | 0.2183 ± 0.0126 | 0.3151 | 0.0721 | 0.2009 | 0.4505 |
| Reynolds | 0.1882 ± 0.0642 | 0.2909 | 0.0686 | 0.1788 | 0.3599 |
| PotentialFlow | 0.0864 ± 0.0234 | 0.1857 | 0.0874 | 0.1520 | 0.2999 |
| TimesNet | 0.1874 ± 0.0125 | 0.3044 | 0.2727 | 0.2918 | 0.6253 |
| TranAD | 0.1192 ± 0.0043 | 0.2634 | 0.2829 | 0.2999 | 0.5300 |

### Strictly causal (0 lookahead), SPOT(u=0.98, q=1e-3) fit on held-out calibration half
| Model | Point-F1 (macro, mean ± seed SD) | PR-AUC | Held-out nominal FPR | Test FPR | Event recall |
|:--|:-:|:-:|:-:|:-:|:-:|
| TS-JEPA | 0.1132 ± 0.0447 | 0.2495 | 0.0047 | 0.0624 | 0.4281 |
| OpEntropy | 0.1018 ± 0.0041 | 0.2422 | 0.0095 | 0.0794 | 0.3937 |
| Reynolds | 0.0697 ± 0.0406 | 0.2492 | 0.0090 | 0.0885 | 0.4007 |
| PotentialFlow | 0.0275 ± 0.0161 | 0.1608 | 0.0113 | 0.0294 | 0.3587 |
| TimesNet | 0.1983 ± 0.0194 | 0.2698 | 0.2202 | 0.2640 | 0.5797 |
| TranAD | 0.1321 ± 0.0164 | 0.2274 | 0.2014 | 0.2492 | 0.5563 |

### Paired differences (buffered, SPOT), stream-level, seeds averaged
| Comparison | Metric | Mean diff | 95% bootstrap CI | Wilcoxon p | Streams A>B |
|:--|:--|:-:|:-:|:-:|:-:|
| TS-JEPA vs TimesNet | f1 | +0.0177 | [-0.0518, +0.0890] | 0.550 | 10/33 |
| TS-JEPA vs TimesNet | pr_auc | +0.0015 | [-0.0982, +0.0970] | 0.304 | 22/33 |
| TS-JEPA vs TimesNet | hold_fpr | -0.1538 | [-0.2493, -0.0708] | 0.008 | 10/33 |
| TS-JEPA vs TranAD | f1 | +0.0617 | [-0.0122, +0.1501] | 0.943 | 10/33 |
| TS-JEPA vs TranAD | pr_auc | +0.0272 | [-0.0745, +0.1349] | 0.304 | 21/33 |
| TS-JEPA vs TranAD | hold_fpr | -0.1570 | [-0.2471, -0.0823] | 0.001 | 8/33 |
| OpEntropy vs TS-JEPA | f1 | -0.0069 | [-0.0276, +0.0119] | 0.711 | 9/33 |
| OpEntropy vs TS-JEPA | pr_auc | +0.0079 | [-0.0127, +0.0335] | 0.280 | 11/33 |
| OpEntropy vs TS-JEPA | hold_fpr | +0.0061 | [-0.0238, +0.0415] | 0.393 | 12/33 |
| Reynolds vs TS-JEPA | f1 | -0.0260 | [-0.0695, +0.0036] | 0.332 | 11/33 |
| Reynolds vs TS-JEPA | pr_auc | -0.0094 | [-0.0241, +0.0074] | 0.013 | 9/33 |
| Reynolds vs TS-JEPA | hold_fpr | +0.0032 | [-0.0259, +0.0397] | 0.940 | 16/33 |
| PotentialFlow vs TS-JEPA | f1 | -0.0911 | [-0.1775, -0.0186] | 0.020 | 5/33 |
| PotentialFlow vs TS-JEPA | pr_auc | -0.0756 | [-0.1415, -0.0187] | 0.039 | 12/33 |
| PotentialFlow vs TS-JEPA | hold_fpr | +0.0160 | [-0.0145, +0.0504] | 0.815 | 16/33 |
| OpEntropy vs TimesNet | f1 | +0.0107 | [-0.0600, +0.0836] | 0.421 | 10/33 |
| OpEntropy vs TimesNet | pr_auc | +0.0094 | [-0.0964, +0.1155] | 0.426 | 20/33 |
| OpEntropy vs TimesNet | hold_fpr | -0.1478 | [-0.2356, -0.0686] | 0.001 | 7/33 |
| OpEntropy vs TranAD | f1 | +0.0548 | [-0.0128, +0.1360] | 0.614 | 10/33 |
| OpEntropy vs TranAD | pr_auc | +0.0351 | [-0.0692, +0.1466] | 0.321 | 19/33 |
| OpEntropy vs TranAD | hold_fpr | -0.1509 | [-0.2501, -0.0645] | 0.003 | 8/33 |

### Calibrator comparison on identical score streams (buffered)
| Model | Calibrator | Held-out nominal FPR (mean) | Share of rows whose 95% CI excludes q=1e-3 | Point-F1 |
|:--|:--|:-:|:-:|:-:|
| TS-JEPA | SPOT | 0.0517 | 0.57 | 0.1929 |
| TS-JEPA | SPOT+kurt(1/6) | 0.1880 | 0.84 | 0.1947 |
| TS-JEPA | p99.5 | 0.0657 | 0.47 | 0.1942 |
| TS-JEPA | max-val | 0.0524 | 0.60 | 0.1922 |
| OpEntropy | SPOT | 0.0615 | 0.65 | 0.1835 |
| OpEntropy | SPOT+kurt(1/6) | 0.1098 | 0.76 | 0.1666 |
| OpEntropy | p99.5 | 0.0752 | 0.60 | 0.1903 |
| OpEntropy | max-val | 0.0691 | 0.63 | 0.1869 |
| TimesNet | SPOT | 0.2388 | 0.87 | 0.1614 |
| TimesNet | SPOT+kurt(1/6) | 0.2439 | 0.85 | 0.1550 |
| TimesNet | p99.5 | 0.2488 | 0.91 | 0.1626 |
| TimesNet | max-val | 0.2430 | 0.87 | 0.1623 |
| TranAD | SPOT | 0.2459 | 0.84 | 0.1039 |
| TranAD | SPOT+kurt(1/6) | 0.2659 | 0.83 | 0.1047 |
| TranAD | p99.5 | 0.2616 | 0.88 | 0.1062 |
| TranAD | max-val | 0.2535 | 0.85 | 0.1054 |

### Calibration gap: SPOT design risk 1e-3 vs measured held-out FPR, by model (buffered)
| model         |   hold_fpr |   hold_fpr_lo |   hold_fpr_hi |
|:--------------|-----------:|--------------:|--------------:|
| OpEntropy     |     0.0615 |        0.0282 |        0.1053 |
| PotentialFlow |     0.0744 |        0.0326 |        0.1247 |
| Reynolds      |     0.0582 |        0.0257 |        0.1013 |
| TS-JEPA       |     0.0517 |        0.0128 |        0.1019 |
| TimesNet      |     0.2388 |        0.1259 |        0.348  |
| TranAD        |     0.2459 |        0.1304 |        0.3655 |