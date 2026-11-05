Streams: 33; seeds: [np.int64(7), np.int64(42), np.int64(123), np.int64(456), np.int64(2024)]; rows: 5132

### Buffered (64-step lookahead), SPOT(u=0.98, q=1e-3) fit on held-out calibration half
| Model | Point-F1 (macro, mean ± seed SD) | PR-AUC | Held-out nominal FPR | Test FPR | Event recall |
|:--|:-:|:-:|:-:|:-:|:-:|
| TimesNet | 0.1638 ± 0.0183 | 0.2312 | 0.2112 | 0.3339 | 0.6057 |
| TranAD | 0.1000 ± 0.0159 | 0.1877 | 0.2123 | 0.3883 | 0.5885 |
| TS-JEPA | 0.1067 ± 0.0639 | 0.1154 | 0.0753 | 0.2371 | 0.3503 |
| OpEntropy | 0.0970 ± 0.0253 | 0.1140 | 0.0701 | 0.3134 | 0.4467 |
| PotentialFlow | 0.0734 ± 0.0239 | 0.1299 | 0.1104 | 0.2046 | 0.3716 |
| Reynolds | 0.1088 ± 0.0131 | 0.1772 | 0.0318 | 0.0796 | 0.2461 |
| SolitonReynolds-JEPA | 0.1475 ± 0.0389 | 0.3313 | 0.0783 | 0.1664 | 0.4302 |
| SolitonReynolds-NoSoliton | 0.1487 ± 0.0375 | 0.3297 | 0.0830 | 0.1733 | 0.4622 |

### Strictly causal (0 lookahead), SPOT(u=0.98, q=1e-3) fit on held-out calibration half
| Model | Point-F1 (macro, mean ± seed SD) | PR-AUC | Held-out nominal FPR | Test FPR | Event recall |
|:--|:-:|:-:|:-:|:-:|:-:|
| TimesNet | 0.1460 ± 0.0194 | 0.2269 | 0.1993 | 0.3259 | 0.5104 |
| TranAD | 0.1099 ± 0.0386 | 0.1864 | 0.1579 | 0.3421 | 0.5695 |
| TS-JEPA | 0.0519 ± 0.0354 | 0.1108 | 0.0290 | 0.1260 | 0.4016 |
| OpEntropy | 0.0456 ± 0.0158 | 0.1099 | 0.0160 | 0.1566 | 0.4988 |
| PotentialFlow | 0.0272 ± 0.0068 | 0.1193 | 0.0127 | 0.0509 | 0.4560 |
| Reynolds | 0.0580 ± 0.0170 | 0.1597 | 0.0120 | 0.0376 | 0.2900 |
| SolitonReynolds-JEPA | 0.0503 ± 0.0176 | 0.2589 | 0.0056 | 0.0198 | 0.4781 |
| SolitonReynolds-NoSoliton | 0.0549 ± 0.0189 | 0.2612 | 0.0061 | 0.0202 | 0.4393 |

### Paired differences (strictly causal, SPOT), stream-level, seeds averaged
| Comparison | Metric | Mean diff | 95% bootstrap CI | Wilcoxon p | Streams A>B |
|:--|:--|:-:|:-:|:-:|:-:|
| SolitonReynolds-JEPA vs Reynolds | f1 | -0.0360 | [-0.0772, -0.0048] | 0.156 | 2/9 |
| SolitonReynolds-JEPA vs Reynolds | pr_auc | -0.0090 | [-0.1016, +0.0767] | 0.496 | 3/9 |
| SolitonReynolds-JEPA vs Reynolds | event_recall | +0.0748 | [-0.0572, +0.2049] | 0.312 | 4/9 |
| SolitonReynolds-JEPA vs Reynolds | hold_fpr | -0.0061 | [-0.0128, -0.0005] | 0.301 | 4/9 |
| SolitonReynolds-JEPA vs OpEntropy | f1 | -0.0236 | [-0.0474, -0.0027] | 0.094 | 1/9 |
| SolitonReynolds-JEPA vs OpEntropy | pr_auc | +0.0408 | [+0.0090, +0.0798] | 0.098 | 6/9 |
| SolitonReynolds-JEPA vs OpEntropy | event_recall | -0.1340 | [-0.3064, +0.0011] | 0.219 | 1/9 |
| SolitonReynolds-JEPA vs OpEntropy | hold_fpr | -0.0100 | [-0.0229, -0.0012] | 0.020 | 2/9 |
| SolitonReynolds-JEPA vs TS-JEPA | f1 | -0.0300 | [-0.0673, +0.0051] | 0.156 | 1/9 |
| SolitonReynolds-JEPA vs TS-JEPA | pr_auc | +0.0399 | [-0.0017, +0.0854] | 0.250 | 5/9 |
| SolitonReynolds-JEPA vs TS-JEPA | event_recall | -0.0368 | [-0.0864, +0.0151] | 0.312 | 2/9 |
| SolitonReynolds-JEPA vs TS-JEPA | hold_fpr | -0.0231 | [-0.0642, -0.0005] | 0.074 | 2/9 |
| TS-JEPA vs TimesNet | f1 | -0.0940 | [-0.2274, +0.0292] | 0.375 | 2/9 |
| TS-JEPA vs TimesNet | pr_auc | -0.1161 | [-0.2690, +0.0020] | 0.129 | 3/9 |
| TS-JEPA vs TimesNet | event_recall | -0.1088 | [-0.3846, +0.1156] | 0.578 | 2/9 |
| TS-JEPA vs TimesNet | hold_fpr | -0.1702 | [-0.3725, -0.0074] | 0.203 | 4/9 |
| TS-JEPA vs TranAD | f1 | -0.0579 | [-0.1535, +0.0347] | 0.250 | 2/9 |
| TS-JEPA vs TranAD | pr_auc | -0.0756 | [-0.1874, +0.0142] | 0.734 | 5/9 |
| TS-JEPA vs TranAD | event_recall | -0.1679 | [-0.4868, +0.1487] | 0.383 | 3/9 |
| TS-JEPA vs TranAD | hold_fpr | -0.1289 | [-0.3303, -0.0019] | 0.496 | 5/9 |
| OpEntropy vs TS-JEPA | f1 | -0.0064 | [-0.0364, +0.0167] | 1.000 | 3/9 |
| OpEntropy vs TS-JEPA | pr_auc | -0.0009 | [-0.0298, +0.0350] | 0.910 | 4/9 |
| OpEntropy vs TS-JEPA | event_recall | +0.0972 | [-0.0000, +0.2253] | 0.219 | 5/9 |
| OpEntropy vs TS-JEPA | hold_fpr | -0.0131 | [-0.0433, +0.0058] | 0.844 | 4/9 |
| Reynolds vs TS-JEPA | f1 | +0.0061 | [-0.0560, +0.0781] | 0.844 | 2/9 |
| Reynolds vs TS-JEPA | pr_auc | +0.0489 | [-0.0042, +0.1185] | 0.074 | 7/9 |
| Reynolds vs TS-JEPA | event_recall | -0.1116 | [-0.2670, +0.0358] | 0.219 | 2/9 |
| Reynolds vs TS-JEPA | hold_fpr | -0.0170 | [-0.0605, +0.0081] | 0.910 | 4/9 |
| PotentialFlow vs TS-JEPA | f1 | -0.0248 | [-0.0622, +0.0001] | 0.219 | 2/9 |
| PotentialFlow vs TS-JEPA | pr_auc | +0.0085 | [-0.0159, +0.0322] | 0.301 | 6/9 |
| PotentialFlow vs TS-JEPA | event_recall | +0.0545 | [-0.0589, +0.2092] | 0.844 | 3/9 |
| PotentialFlow vs TS-JEPA | hold_fpr | -0.0163 | [-0.0420, +0.0015] | 0.734 | 5/9 |
| OpEntropy vs TimesNet | f1 | -0.1004 | [-0.2177, +0.0023] | 0.219 | 2/9 |
| OpEntropy vs TimesNet | pr_auc | -0.1170 | [-0.2814, +0.0288] | 0.164 | 3/9 |
| OpEntropy vs TimesNet | event_recall | -0.0115 | [-0.3184, +0.2456] | 0.938 | 3/9 |
| OpEntropy vs TimesNet | hold_fpr | -0.1833 | [-0.4195, -0.0093] | 0.109 | 3/9 |
| OpEntropy vs TranAD | f1 | -0.0643 | [-0.1488, +0.0035] | 0.109 | 1/9 |
| OpEntropy vs TranAD | pr_auc | -0.0765 | [-0.2018, +0.0385] | 0.496 | 4/9 |
| OpEntropy vs TranAD | event_recall | -0.0707 | [-0.3925, +0.2181] | 0.945 | 4/9 |
| OpEntropy vs TranAD | hold_fpr | -0.1420 | [-0.3453, -0.0126] | 0.109 | 3/9 |

### Paired differences (buffered, SPOT), stream-level, seeds averaged
| Comparison | Metric | Mean diff | 95% bootstrap CI | Wilcoxon p | Streams A>B |
|:--|:--|:-:|:-:|:-:|:-:|
| SolitonReynolds-JEPA vs Reynolds | f1 | -0.0120 | [-0.0777, +0.0373] | 0.812 | 5/9 |
| SolitonReynolds-JEPA vs Reynolds | pr_auc | -0.0198 | [-0.1410, +0.0778] | 0.652 | 4/9 |
| SolitonReynolds-JEPA vs Reynolds | event_recall | +0.0500 | [-0.0159, +0.1170] | 0.219 | 4/9 |
| SolitonReynolds-JEPA vs Reynolds | hold_fpr | +0.0275 | [-0.0277, +0.0969] | 0.570 | 5/9 |
| SolitonReynolds-JEPA vs OpEntropy | f1 | -0.0002 | [-0.0458, +0.0569] | 0.578 | 2/9 |
| SolitonReynolds-JEPA vs OpEntropy | pr_auc | +0.0434 | [+0.0111, +0.0832] | 0.074 | 6/9 |
| SolitonReynolds-JEPA vs OpEntropy | event_recall | -0.1506 | [-0.3384, -0.0249] | 0.078 | 2/9 |
| SolitonReynolds-JEPA vs OpEntropy | hold_fpr | -0.0108 | [-0.0589, +0.0481] | 0.426 | 3/9 |
| SolitonReynolds-JEPA vs TS-JEPA | f1 | -0.0099 | [-0.0889, +0.0904] | 0.688 | 3/9 |
| SolitonReynolds-JEPA vs TS-JEPA | pr_auc | +0.0420 | [-0.0045, +0.0942] | 0.301 | 5/9 |
| SolitonReynolds-JEPA vs TS-JEPA | event_recall | -0.0542 | [-0.2093, +0.0843] | 0.688 | 3/9 |
| SolitonReynolds-JEPA vs TS-JEPA | hold_fpr | -0.0160 | [-0.0954, +0.0419] | 0.734 | 6/9 |
| TS-JEPA vs TimesNet | f1 | -0.0572 | [-0.1813, +0.0484] | 0.547 | 3/9 |
| TS-JEPA vs TimesNet | pr_auc | -0.1158 | [-0.2744, +0.0047] | 0.250 | 3/9 |
| TS-JEPA vs TimesNet | event_recall | -0.2554 | [-0.5775, +0.0656] | 0.242 | 2/9 |
| TS-JEPA vs TimesNet | hold_fpr | -0.1358 | [-0.3559, +0.0398] | 0.250 | 3/9 |
| TS-JEPA vs TranAD | f1 | +0.0067 | [-0.0501, +0.0637] | 0.547 | 2/9 |
| TS-JEPA vs TranAD | pr_auc | -0.0723 | [-0.1746, +0.0162] | 0.426 | 4/9 |
| TS-JEPA vs TranAD | event_recall | -0.2382 | [-0.5704, +0.1121] | 0.367 | 3/9 |
| TS-JEPA vs TranAD | hold_fpr | -0.1369 | [-0.3763, +0.0642] | 0.641 | 4/9 |
| OpEntropy vs TS-JEPA | f1 | -0.0097 | [-0.0587, +0.0370] | 0.938 | 5/9 |
| OpEntropy vs TS-JEPA | pr_auc | -0.0014 | [-0.0356, +0.0360] | 0.820 | 4/9 |
| OpEntropy vs TS-JEPA | event_recall | +0.0963 | [-0.0201, +0.2121] | 0.219 | 6/9 |
| OpEntropy vs TS-JEPA | hold_fpr | -0.0052 | [-0.0654, +0.0469] | 0.734 | 6/9 |
| Reynolds vs TS-JEPA | f1 | +0.0021 | [-0.0754, +0.0803] | 1.000 | 3/9 |
| Reynolds vs TS-JEPA | pr_auc | +0.0618 | [-0.0064, +0.1536] | 0.203 | 6/9 |
| Reynolds vs TS-JEPA | event_recall | -0.1043 | [-0.3086, +0.0813] | 0.562 | 2/9 |
| Reynolds vs TS-JEPA | hold_fpr | -0.0435 | [-0.1212, +0.0155] | 0.375 | 3/9 |
| PotentialFlow vs TS-JEPA | f1 | -0.0332 | [-0.0723, +0.0003] | 0.219 | 3/9 |
| PotentialFlow vs TS-JEPA | pr_auc | +0.0145 | [-0.0173, +0.0498] | 0.426 | 6/9 |
| PotentialFlow vs TS-JEPA | event_recall | +0.0213 | [-0.1334, +0.1880] | 1.000 | 3/9 |
| PotentialFlow vs TS-JEPA | hold_fpr | +0.0351 | [-0.0427, +0.0944] | 0.129 | 8/9 |
| OpEntropy vs TimesNet | f1 | -0.0669 | [-0.1822, +0.0295] | 0.570 | 4/9 |
| OpEntropy vs TimesNet | pr_auc | -0.1172 | [-0.2776, +0.0292] | 0.164 | 3/9 |
| OpEntropy vs TimesNet | event_recall | -0.1590 | [-0.5204, +0.1744] | 0.625 | 5/9 |
| OpEntropy vs TimesNet | hold_fpr | -0.1411 | [-0.3796, +0.0479] | 0.496 | 4/9 |
| OpEntropy vs TranAD | f1 | -0.0030 | [-0.0501, +0.0557] | 0.652 | 3/9 |
| OpEntropy vs TranAD | pr_auc | -0.0737 | [-0.2017, +0.0458] | 0.496 | 4/9 |
| OpEntropy vs TranAD | event_recall | -0.1419 | [-0.4768, +0.1612] | 0.789 | 5/9 |
| OpEntropy vs TranAD | hold_fpr | -0.1422 | [-0.3672, +0.0384] | 0.496 | 4/9 |

### Calibrator comparison on identical score streams (buffered)
| Model | Calibrator | Held-out nominal FPR (mean) | Share of rows whose 95% CI excludes q=1e-3 | Point-F1 |
|:--|:--|:-:|:-:|:-:|
| TS-JEPA | SPOT | 0.0753 | 0.74 | 0.1067 |
| TS-JEPA | SPOT+kurt(1/6) | 0.2605 | 0.89 | 0.0938 |
| TS-JEPA | p99.5 | 0.0840 | 0.59 | 0.1089 |
| TS-JEPA | max-val | 0.0799 | 0.74 | 0.1072 |
| OpEntropy | SPOT | 0.0701 | 0.59 | 0.0970 |
| OpEntropy | SPOT+kurt(1/6) | 0.1930 | 0.63 | 0.0959 |
| OpEntropy | p99.5 | 0.0782 | 0.52 | 0.1023 |
| OpEntropy | max-val | 0.0692 | 0.52 | 0.0983 |
| TimesNet | SPOT | 0.2112 | 0.85 | 0.1638 |
| TimesNet | SPOT+kurt(1/6) | 0.2160 | 0.85 | 0.1509 |
| TimesNet | p99.5 | 0.2212 | 0.89 | 0.1648 |
| TimesNet | max-val | 0.2162 | 0.89 | 0.1637 |
| TranAD | SPOT | 0.2123 | 0.81 | 0.1000 |
| TranAD | SPOT+kurt(1/6) | 0.2438 | 0.78 | 0.0983 |
| TranAD | p99.5 | 0.2328 | 0.85 | 0.1037 |
| TranAD | max-val | 0.2215 | 0.85 | 0.1018 |

### Calibration gap: SPOT design risk 1e-3 vs measured held-out FPR, by model (buffered)
| model                     |   hold_fpr |   hold_fpr_lo |   hold_fpr_hi |
|:--------------------------|-----------:|--------------:|--------------:|
| OpEntropy                 |     0.0701 |        0.0145 |        0.1517 |
| PotentialFlow             |     0.1104 |        0.0427 |        0.1952 |
| Reynolds                  |     0.0318 |        0.0012 |        0.0778 |
| SolitonReynolds-JEPA      |     0.055  |        0.0163 |        0.1012 |
| SolitonReynolds-NoSoliton |     0.0583 |        0.018  |        0.1065 |
| TS-JEPA                   |     0.0753 |        0.0431 |        0.1174 |
| TimesNet                  |     0.2112 |        0.1564 |        0.2808 |
| TranAD                    |     0.2123 |        0.168  |        0.2812 |