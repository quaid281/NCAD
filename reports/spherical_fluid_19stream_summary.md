Streams: 19; seeds: [np.int64(42), np.int64(123), np.int64(456)]; rows: 9540

### Buffered (64-step lookahead), SPOT(u=0.98, q=1e-3) fit on held-out calibration half
| Model | Point-F1 (macro, mean ± seed SD) | PR-AUC | Held-out nominal FPR | Test FPR | Event recall |
|:--|:-:|:-:|:-:|:-:|:-:|
| SolitonReynolds-JEPA (v1) | 0.1223 ± 0.0567 | 0.2803 | 0.0414 | 0.1017 | 0.4124 |
| EddyDelta-JEPA (v1) | 0.1202 ± 0.0079 | 0.2784 | 0.0333 | 0.1051 | 0.3647 |
| AnchoredStokes-JEPA | 0.1329 ± 0.0290 | 0.2842 | 0.0416 | 0.1239 | 0.3681 |
| Alien-DualTower-JEPA | 0.1123 ± 0.0665 | 0.2926 | 0.0490 | 0.1258 | 0.3187 |
| TS-JEPA | 0.0981 ± 0.0171 | 0.2399 | 0.0403 | 0.1191 | 0.2658 |
| HelmholtzSoliton-JEPA | 0.1919 ± 0.0161 | 0.3655 | 0.1026 | 0.1094 | 0.4648 |
| EddyDelta-JEPA | 0.1715 ± 0.0155 | 0.3727 | 0.0874 | 0.0989 | 0.4692 |
| SolitonReynolds-JEPA | 0.1971 ± 0.0243 | 0.3806 | 0.1085 | 0.1052 | 0.4749 |
| Speculative-JEPA | 0.1751 ± 0.0253 | 0.3699 | 0.1102 | 0.1048 | 0.4253 |
| Speculative-Alien-JEPA | 0.1894 ± 0.0359 | 0.3771 | 0.1260 | 0.1159 | 0.4949 |
| TimesNet | 0.2038 ± 0.0114 | 0.3364 | 0.2964 | 0.2425 | 0.5764 |
| TranAD | 0.1314 ± 0.0048 | 0.2909 | 0.3089 | 0.2525 | 0.4805 |
| PotentialFlow | 0.1350 ± 0.0624 | 0.2353 | 0.0758 | 0.1557 | 0.4329 |
| OpEntropy | 0.2099 ± 0.0238 | 0.3442 | 0.0786 | 0.1557 | 0.4442 |
| Reynolds | 0.1278 ± 0.0275 | 0.3071 | 0.1765 | 0.1321 | 0.4499 |

### Strictly causal (0 lookahead), SPOT(u=0.98, q=1e-3) fit on held-out calibration half
| Model | Point-F1 (macro, mean ± seed SD) | PR-AUC | Held-out nominal FPR | Test FPR | Event recall |
|:--|:-:|:-:|:-:|:-:|:-:|
| SolitonReynolds-JEPA (v1) | 0.0214 ± 0.0084 | 0.2188 | 0.0102 | 0.0165 | 0.5115 |
| EddyDelta-JEPA (v1) | 0.0242 ± 0.0066 | 0.2253 | 0.0084 | 0.0253 | 0.5186 |
| AnchoredStokes-JEPA | 0.0599 ± 0.0207 | 0.2381 | 0.0067 | 0.0159 | 0.5105 |
| Alien-DualTower-JEPA | 0.0301 ± 0.0054 | 0.2298 | 0.0092 | 0.0156 | 0.4417 |
| TS-JEPA | 0.0461 ± 0.0160 | 0.2072 | 0.0075 | 0.0246 | 0.3183 |
| HelmholtzSoliton-JEPA | 0.0525 ± 0.0104 | 0.3000 | 0.0049 | 0.0082 | 0.4684 |
| EddyDelta-JEPA | 0.0463 ± 0.0238 | 0.3003 | 0.0019 | 0.0059 | 0.4367 |
| SolitonReynolds-JEPA | 0.0521 ± 0.0100 | 0.2941 | 0.0038 | 0.0091 | 0.5516 |
| Speculative-JEPA | 0.0108 ± 0.0030 | 0.2878 | 0.0031 | 0.0061 | 0.4564 |
| Speculative-Alien-JEPA | 0.0235 ± 0.0122 | 0.2988 | 0.0026 | 0.0071 | 0.4772 |
| TimesNet | 0.2209 ± 0.0195 | 0.2997 | 0.2406 | 0.2090 | 0.5906 |
| TranAD | 0.1465 ± 0.0183 | 0.2524 | 0.2208 | 0.1958 | 0.5255 |
| PotentialFlow | 0.0245 ± 0.0123 | 0.1898 | 0.0088 | 0.0296 | 0.4873 |
| OpEntropy | 0.1202 ± 0.0422 | 0.2724 | 0.0132 | 0.0579 | 0.4461 |
| Reynolds | 0.0573 ± 0.0081 | 0.2490 | 0.0724 | 0.0508 | 0.4944 |

### Paired differences (strictly causal, SPOT), stream-level, seeds averaged
| Comparison | Metric | Mean diff | 95% bootstrap CI | Wilcoxon p | Streams A>B |
|:--|:--|:-:|:-:|:-:|:-:|
| SolitonReynolds-JEPA vs Reynolds | f1 | -0.0052 | [-0.0463, +0.0360] | 0.642 | 8/19 |
| SolitonReynolds-JEPA vs Reynolds | pr_auc | +0.0450 | [-0.0136, +0.1107] | 0.045 | 14/19 |
| SolitonReynolds-JEPA vs Reynolds | event_recall | +0.0572 | [-0.0467, +0.1478] | 0.139 | 8/19 |
| SolitonReynolds-JEPA vs Reynolds | hold_fpr | -0.0686 | [-0.1232, -0.0248] | 0.093 | 8/19 |
| HelmholtzSoliton-JEPA vs Reynolds | f1 | -0.0048 | [-0.0620, +0.0718] | 0.349 | 8/19 |
| HelmholtzSoliton-JEPA vs Reynolds | pr_auc | +0.0510 | [-0.0094, +0.1129] | 0.049 | 15/19 |
| HelmholtzSoliton-JEPA vs Reynolds | event_recall | -0.0260 | [-0.1726, +0.1150] | 0.943 | 10/19 |
| HelmholtzSoliton-JEPA vs Reynolds | hold_fpr | -0.0675 | [-0.1192, -0.0213] | 0.125 | 7/19 |
| EddyDelta-JEPA vs Reynolds | f1 | -0.0110 | [-0.0475, +0.0234] | 0.379 | 8/19 |
| EddyDelta-JEPA vs Reynolds | pr_auc | +0.0513 | [+0.0014, +0.1040] | 0.036 | 15/19 |
| EddyDelta-JEPA vs Reynolds | event_recall | -0.0577 | [-0.1797, +0.0560] | 0.504 | 5/19 |
| EddyDelta-JEPA vs Reynolds | hold_fpr | -0.0704 | [-0.1215, -0.0255] | 0.015 | 6/19 |
| SolitonReynolds-JEPA vs OpEntropy | f1 | -0.0681 | [-0.1084, -0.0332] | 0.003 | 3/19 |
| SolitonReynolds-JEPA vs OpEntropy | pr_auc | +0.0217 | [-0.0575, +0.1061] | 0.798 | 9/19 |
| SolitonReynolds-JEPA vs OpEntropy | event_recall | +0.1055 | [-0.0271, +0.2270] | 0.176 | 8/19 |
| SolitonReynolds-JEPA vs OpEntropy | hold_fpr | -0.0095 | [-0.0224, -0.0003] | 0.145 | 6/19 |
| HelmholtzSoliton-JEPA vs OpEntropy | f1 | -0.0677 | [-0.1182, -0.0227] | 0.025 | 6/19 |
| HelmholtzSoliton-JEPA vs OpEntropy | pr_auc | +0.0276 | [-0.0462, +0.1018] | 0.441 | 12/19 |
| HelmholtzSoliton-JEPA vs OpEntropy | event_recall | +0.0223 | [-0.1286, +0.1607] | 0.530 | 8/19 |
| HelmholtzSoliton-JEPA vs OpEntropy | hold_fpr | -0.0083 | [-0.0213, +0.0018] | 0.136 | 4/19 |
| EddyDelta-JEPA vs OpEntropy | f1 | -0.0739 | [-0.1187, -0.0333] | 0.003 | 3/19 |
| EddyDelta-JEPA vs OpEntropy | pr_auc | +0.0279 | [-0.0415, +0.0943] | 0.352 | 11/19 |
| EddyDelta-JEPA vs OpEntropy | event_recall | -0.0094 | [-0.1642, +0.1274] | 0.861 | 6/19 |
| EddyDelta-JEPA vs OpEntropy | hold_fpr | -0.0113 | [-0.0250, -0.0024] | 0.011 | 3/19 |
| SolitonReynolds-JEPA vs Speculative-Alien-JEPA | f1 | +0.0286 | [+0.0019, +0.0685] | 0.070 | 12/19 |
| SolitonReynolds-JEPA vs Speculative-Alien-JEPA | pr_auc | -0.0047 | [-0.0221, +0.0138] | 0.465 | 9/19 |
| SolitonReynolds-JEPA vs Speculative-Alien-JEPA | event_recall | +0.0744 | [-0.0232, +0.1806] | 0.158 | 8/19 |
| SolitonReynolds-JEPA vs Speculative-Alien-JEPA | hold_fpr | +0.0012 | [-0.0005, +0.0029] | 0.125 | 9/19 |
| HelmholtzSoliton-JEPA vs Speculative-Alien-JEPA | f1 | +0.0290 | [-0.0040, +0.0796] | 0.381 | 10/19 |
| HelmholtzSoliton-JEPA vs Speculative-Alien-JEPA | pr_auc | +0.0013 | [-0.0090, +0.0111] | 0.922 | 10/19 |
| HelmholtzSoliton-JEPA vs Speculative-Alien-JEPA | event_recall | -0.0088 | [-0.0784, +0.0564] | 0.695 | 5/19 |
| HelmholtzSoliton-JEPA vs Speculative-Alien-JEPA | hold_fpr | +0.0023 | [-0.0000, +0.0054] | 0.245 | 8/19 |
| EddyDelta-JEPA vs Speculative-Alien-JEPA | f1 | +0.0228 | [-0.0010, +0.0526] | 0.460 | 8/19 |
| EddyDelta-JEPA vs Speculative-Alien-JEPA | pr_auc | +0.0016 | [-0.0122, +0.0169] | 0.738 | 11/19 |
| EddyDelta-JEPA vs Speculative-Alien-JEPA | event_recall | -0.0405 | [-0.1489, +0.0736] | 0.286 | 4/19 |
| EddyDelta-JEPA vs Speculative-Alien-JEPA | hold_fpr | -0.0007 | [-0.0020, +0.0008] | 0.112 | 4/19 |
| HelmholtzSoliton-JEPA vs Alien-DualTower-JEPA | f1 | +0.0224 | [-0.0166, +0.0885] | 0.983 | 8/19 |
| HelmholtzSoliton-JEPA vs Alien-DualTower-JEPA | pr_auc | +0.0702 | [+0.0079, +0.1390] | 0.040 | 13/19 |
| HelmholtzSoliton-JEPA vs Alien-DualTower-JEPA | event_recall | +0.0266 | [-0.0847, +0.1398] | 0.432 | 6/19 |
| HelmholtzSoliton-JEPA vs Alien-DualTower-JEPA | hold_fpr | -0.0043 | [-0.0140, +0.0038] | 0.113 | 3/19 |
| SolitonReynolds-JEPA vs Alien-DualTower-JEPA | f1 | +0.0220 | [-0.0056, +0.0598] | 0.286 | 11/19 |
| SolitonReynolds-JEPA vs Alien-DualTower-JEPA | pr_auc | +0.0642 | [+0.0052, +0.1334] | 0.169 | 10/19 |
| SolitonReynolds-JEPA vs Alien-DualTower-JEPA | event_recall | +0.1099 | [+0.0074, +0.2105] | 0.068 | 9/19 |
| SolitonReynolds-JEPA vs Alien-DualTower-JEPA | hold_fpr | -0.0054 | [-0.0145, +0.0011] | 0.379 | 7/19 |
| EddyDelta-JEPA vs Alien-DualTower-JEPA | f1 | +0.0162 | [-0.0144, +0.0535] | 0.528 | 8/19 |
| EddyDelta-JEPA vs Alien-DualTower-JEPA | pr_auc | +0.0705 | [+0.0130, +0.1335] | 0.040 | 11/19 |
| EddyDelta-JEPA vs Alien-DualTower-JEPA | event_recall | -0.0050 | [-0.1184, +0.1072] | 0.816 | 7/19 |
| EddyDelta-JEPA vs Alien-DualTower-JEPA | hold_fpr | -0.0072 | [-0.0164, -0.0015] | 0.008 | 3/19 |
| AnchoredStokes-JEPA vs Alien-DualTower-JEPA | f1 | +0.0298 | [-0.0088, +0.0797] | 0.248 | 10/19 |
| AnchoredStokes-JEPA vs Alien-DualTower-JEPA | pr_auc | +0.0083 | [-0.0088, +0.0273] | 0.210 | 13/19 |
| AnchoredStokes-JEPA vs Alien-DualTower-JEPA | event_recall | +0.0688 | [-0.0224, +0.1609] | 0.157 | 8/19 |
| AnchoredStokes-JEPA vs Alien-DualTower-JEPA | hold_fpr | -0.0024 | [-0.0062, +0.0005] | 0.326 | 7/19 |
| HelmholtzSoliton-JEPA vs TS-JEPA | f1 | +0.0064 | [-0.0460, +0.0669] | 0.981 | 10/19 |
| HelmholtzSoliton-JEPA vs TS-JEPA | pr_auc | +0.0928 | [+0.0432, +0.1456] | 0.001 | 16/19 |
| HelmholtzSoliton-JEPA vs TS-JEPA | event_recall | +0.1500 | [+0.0691, +0.2451] | 0.005 | 12/19 |
| HelmholtzSoliton-JEPA vs TS-JEPA | hold_fpr | -0.0026 | [-0.0083, +0.0019] | 0.233 | 5/19 |
| SolitonReynolds-JEPA vs TS-JEPA | f1 | +0.0060 | [-0.0352, +0.0476] | 0.836 | 8/19 |
| SolitonReynolds-JEPA vs TS-JEPA | pr_auc | +0.0868 | [+0.0413, +0.1349] | 0.011 | 13/19 |
| SolitonReynolds-JEPA vs TS-JEPA | event_recall | +0.2333 | [+0.1291, +0.3400] | 0.003 | 12/19 |
| SolitonReynolds-JEPA vs TS-JEPA | hold_fpr | -0.0037 | [-0.0091, +0.0002] | 0.234 | 5/19 |
| EddyDelta-JEPA vs TS-JEPA | f1 | +0.0002 | [-0.0166, +0.0204] | 0.532 | 6/19 |
| EddyDelta-JEPA vs TS-JEPA | pr_auc | +0.0931 | [+0.0467, +0.1459] | 0.001 | 15/19 |
| EddyDelta-JEPA vs TS-JEPA | event_recall | +0.1184 | [+0.0314, +0.2189] | 0.040 | 7/19 |
| EddyDelta-JEPA vs TS-JEPA | hold_fpr | -0.0055 | [-0.0109, -0.0015] | 0.010 | 3/19 |
| AnchoredStokes-JEPA vs TS-JEPA | f1 | +0.0139 | [-0.0059, +0.0391] | 0.586 | 10/19 |
| AnchoredStokes-JEPA vs TS-JEPA | pr_auc | +0.0309 | [-0.0030, +0.0660] | 0.182 | 11/19 |
| AnchoredStokes-JEPA vs TS-JEPA | event_recall | +0.1922 | [+0.0860, +0.3178] | 0.003 | 12/19 |
| AnchoredStokes-JEPA vs TS-JEPA | hold_fpr | -0.0007 | [-0.0046, +0.0028] | 0.811 | 9/19 |
| Alien-DualTower-JEPA vs TS-JEPA | f1 | -0.0159 | [-0.0557, +0.0153] | 0.679 | 9/19 |
| Alien-DualTower-JEPA vs TS-JEPA | pr_auc | +0.0226 | [-0.0140, +0.0596] | 0.241 | 12/19 |
| Alien-DualTower-JEPA vs TS-JEPA | event_recall | +0.1234 | [+0.0297, +0.2212] | 0.023 | 11/19 |
| Alien-DualTower-JEPA vs TS-JEPA | hold_fpr | +0.0017 | [-0.0027, +0.0068] | 0.554 | 9/19 |
| SolitonReynolds-JEPA vs SolitonReynolds-JEPA (v1) | f1 | +0.0307 | [-0.0077, +0.0798] | 0.796 | 7/19 |
| SolitonReynolds-JEPA vs SolitonReynolds-JEPA (v1) | pr_auc | +0.0752 | [+0.0194, +0.1388] | 0.096 | 11/19 |
| SolitonReynolds-JEPA vs SolitonReynolds-JEPA (v1) | event_recall | +0.0402 | [-0.0805, +0.1598] | 0.479 | 7/19 |
| SolitonReynolds-JEPA vs SolitonReynolds-JEPA (v1) | hold_fpr | -0.0064 | [-0.0146, -0.0001] | 0.157 | 6/19 |
| EddyDelta-JEPA vs EddyDelta-JEPA (v1) | f1 | +0.0221 | [-0.0112, +0.0609] | 0.906 | 7/19 |
| EddyDelta-JEPA vs EddyDelta-JEPA (v1) | pr_auc | +0.0751 | [+0.0154, +0.1486] | 0.123 | 11/19 |
| EddyDelta-JEPA vs EddyDelta-JEPA (v1) | event_recall | -0.0818 | [-0.2313, +0.0390] | 0.346 | 4/19 |
| EddyDelta-JEPA vs EddyDelta-JEPA (v1) | hold_fpr | -0.0064 | [-0.0156, -0.0006] | 0.022 | 2/19 |
| TS-JEPA vs TimesNet | f1 | -0.1729 | [-0.2650, -0.0886] | 0.002 | 2/18 |
| TS-JEPA vs TimesNet | pr_auc | -0.1013 | [-0.1948, -0.0239] | 0.167 | 8/18 |
| TS-JEPA vs TimesNet | event_recall | -0.2916 | [-0.4409, -0.1527] | 0.002 | 2/18 |
| TS-JEPA vs TimesNet | hold_fpr | -0.2327 | [-0.3528, -0.1190] | 0.010 | 7/18 |
| TS-JEPA vs TranAD | f1 | -0.0985 | [-0.1859, -0.0229] | 0.047 | 4/18 |
| TS-JEPA vs TranAD | pr_auc | -0.0541 | [-0.1350, +0.0297] | 0.523 | 9/18 |
| TS-JEPA vs TranAD | event_recall | -0.2265 | [-0.3951, -0.0538] | 0.020 | 5/18 |
| TS-JEPA vs TranAD | hold_fpr | -0.2129 | [-0.3299, -0.1068] | 0.010 | 7/18 |
| OpEntropy vs TS-JEPA | f1 | +0.0741 | [+0.0206, +0.1351] | 0.013 | 14/19 |
| OpEntropy vs TS-JEPA | pr_auc | +0.0652 | [+0.0043, +0.1331] | 0.104 | 12/19 |
| OpEntropy vs TS-JEPA | event_recall | +0.1278 | [+0.0127, +0.2625] | 0.136 | 8/19 |
| OpEntropy vs TS-JEPA | hold_fpr | +0.0058 | [-0.0010, +0.0149] | 0.278 | 9/19 |
| Reynolds vs TS-JEPA | f1 | +0.0112 | [-0.0175, +0.0418] | 0.756 | 8/19 |
| Reynolds vs TS-JEPA | pr_auc | +0.0418 | [-0.0015, +0.0894] | 0.312 | 10/19 |
| Reynolds vs TS-JEPA | event_recall | +0.1761 | [+0.0334, +0.3371] | 0.059 | 8/19 |
| Reynolds vs TS-JEPA | hold_fpr | +0.0649 | [+0.0161, +0.1170] | 0.134 | 9/19 |
| PotentialFlow vs TS-JEPA | f1 | -0.0235 | [-0.0645, +0.0112] | 0.233 | 7/18 |
| PotentialFlow vs TS-JEPA | pr_auc | -0.0086 | [-0.0459, +0.0232] | 0.865 | 8/18 |
| PotentialFlow vs TS-JEPA | event_recall | +0.1883 | [+0.0571, +0.3238] | 0.023 | 10/18 |
| PotentialFlow vs TS-JEPA | hold_fpr | +0.0009 | [-0.0078, +0.0106] | 0.799 | 10/18 |
| OpEntropy vs TimesNet | f1 | -0.0972 | [-0.2087, -0.0038] | 0.136 | 6/18 |
| OpEntropy vs TimesNet | pr_auc | -0.0318 | [-0.1433, +0.0810] | 0.766 | 9/18 |
| OpEntropy vs TimesNet | event_recall | -0.1567 | [-0.3577, +0.0477] | 0.124 | 5/18 |
| OpEntropy vs TimesNet | hold_fpr | -0.2268 | [-0.3524, -0.1145] | 0.009 | 6/18 |
| OpEntropy vs TranAD | f1 | -0.0227 | [-0.1587, +0.1063] | 0.723 | 8/18 |
| OpEntropy vs TranAD | pr_auc | +0.0155 | [-0.1054, +0.1515] | 0.966 | 9/18 |
| OpEntropy vs TranAD | event_recall | -0.0917 | [-0.3102, +0.1315] | 0.438 | 8/18 |
| OpEntropy vs TranAD | hold_fpr | -0.2071 | [-0.3251, -0.1040] | 0.009 | 6/18 |

### Paired differences (buffered, SPOT), stream-level, seeds averaged
| Comparison | Metric | Mean diff | 95% bootstrap CI | Wilcoxon p | Streams A>B |
|:--|:--|:-:|:-:|:-:|:-:|
| SolitonReynolds-JEPA vs Reynolds | f1 | +0.0693 | [-0.0271, +0.1732] | 0.169 | 11/19 |
| SolitonReynolds-JEPA vs Reynolds | pr_auc | +0.0735 | [-0.0677, +0.1917] | 0.040 | 14/19 |
| SolitonReynolds-JEPA vs Reynolds | event_recall | +0.0250 | [-0.1408, +0.1843] | 0.705 | 10/19 |
| SolitonReynolds-JEPA vs Reynolds | hold_fpr | -0.0681 | [-0.1632, +0.0304] | 0.233 | 6/19 |
| HelmholtzSoliton-JEPA vs Reynolds | f1 | +0.0640 | [-0.0181, +0.1570] | 0.441 | 10/19 |
| HelmholtzSoliton-JEPA vs Reynolds | pr_auc | +0.0584 | [-0.0789, +0.1752] | 0.073 | 14/19 |
| HelmholtzSoliton-JEPA vs Reynolds | event_recall | +0.0149 | [-0.1828, +0.1915] | 0.569 | 9/19 |
| HelmholtzSoliton-JEPA vs Reynolds | hold_fpr | -0.0739 | [-0.1739, +0.0233] | 0.125 | 5/19 |
| EddyDelta-JEPA vs Reynolds | f1 | +0.0437 | [-0.0224, +0.1318] | 0.557 | 10/19 |
| EddyDelta-JEPA vs Reynolds | pr_auc | +0.0656 | [-0.0617, +0.1807] | 0.045 | 14/19 |
| EddyDelta-JEPA vs Reynolds | event_recall | +0.0194 | [-0.1229, +0.1427] | 0.638 | 8/19 |
| EddyDelta-JEPA vs Reynolds | hold_fpr | -0.0891 | [-0.1848, +0.0031] | 0.061 | 5/19 |
| SolitonReynolds-JEPA vs OpEntropy | f1 | -0.0128 | [-0.1284, +0.1041] | 0.709 | 8/19 |
| SolitonReynolds-JEPA vs OpEntropy | pr_auc | +0.0364 | [-0.1019, +0.1656] | 0.441 | 11/19 |
| SolitonReynolds-JEPA vs OpEntropy | event_recall | +0.0307 | [-0.1388, +0.1902] | 0.733 | 7/19 |
| SolitonReynolds-JEPA vs OpEntropy | hold_fpr | +0.0299 | [-0.0416, +0.1073] | 0.501 | 8/19 |
| HelmholtzSoliton-JEPA vs OpEntropy | f1 | -0.0180 | [-0.1225, +0.0902] | 0.541 | 7/19 |
| HelmholtzSoliton-JEPA vs OpEntropy | pr_auc | +0.0212 | [-0.1161, +0.1507] | 0.374 | 13/19 |
| HelmholtzSoliton-JEPA vs OpEntropy | event_recall | +0.0207 | [-0.1637, +0.2076] | 0.816 | 8/19 |
| HelmholtzSoliton-JEPA vs OpEntropy | hold_fpr | +0.0241 | [-0.0585, +0.1127] | 0.955 | 7/19 |
| EddyDelta-JEPA vs OpEntropy | f1 | -0.0384 | [-0.1375, +0.0620] | 0.441 | 7/19 |
| EddyDelta-JEPA vs OpEntropy | pr_auc | +0.0285 | [-0.1144, +0.1519] | 0.169 | 13/19 |
| EddyDelta-JEPA vs OpEntropy | event_recall | +0.0251 | [-0.1390, +0.1852] | 0.955 | 7/19 |
| EddyDelta-JEPA vs OpEntropy | hold_fpr | +0.0089 | [-0.0711, +0.0937] | 0.836 | 7/19 |
| SolitonReynolds-JEPA vs Speculative-Alien-JEPA | f1 | +0.0077 | [-0.0293, +0.0469] | 0.879 | 9/19 |
| SolitonReynolds-JEPA vs Speculative-Alien-JEPA | pr_auc | +0.0035 | [-0.0361, +0.0502] | 0.623 | 7/19 |
| SolitonReynolds-JEPA vs Speculative-Alien-JEPA | event_recall | -0.0200 | [-0.1466, +0.1046] | 0.583 | 4/19 |
| SolitonReynolds-JEPA vs Speculative-Alien-JEPA | hold_fpr | -0.0175 | [-0.0480, +0.0074] | 0.532 | 8/19 |
| HelmholtzSoliton-JEPA vs Speculative-Alien-JEPA | f1 | +0.0025 | [-0.0163, +0.0233] | 0.605 | 6/19 |
| HelmholtzSoliton-JEPA vs Speculative-Alien-JEPA | pr_auc | -0.0117 | [-0.0288, +0.0047] | 0.293 | 7/19 |
| HelmholtzSoliton-JEPA vs Speculative-Alien-JEPA | event_recall | -0.0300 | [-0.0656, +0.0006] | 0.110 | 3/19 |
| HelmholtzSoliton-JEPA vs Speculative-Alien-JEPA | hold_fpr | -0.0233 | [-0.0403, -0.0092] | 0.006 | 2/19 |
| EddyDelta-JEPA vs Speculative-Alien-JEPA | f1 | -0.0179 | [-0.0825, +0.0351] | 0.913 | 9/19 |
| EddyDelta-JEPA vs Speculative-Alien-JEPA | pr_auc | -0.0044 | [-0.0248, +0.0137] | 0.709 | 12/19 |
| EddyDelta-JEPA vs Speculative-Alien-JEPA | event_recall | -0.0256 | [-0.0968, +0.0470] | 0.271 | 3/19 |
| EddyDelta-JEPA vs Speculative-Alien-JEPA | hold_fpr | -0.0385 | [-0.0662, -0.0145] | 0.009 | 3/19 |
| HelmholtzSoliton-JEPA vs Alien-DualTower-JEPA | f1 | +0.0796 | [-0.0189, +0.2078] | 0.210 | 12/19 |
| HelmholtzSoliton-JEPA vs Alien-DualTower-JEPA | pr_auc | +0.0729 | [-0.0155, +0.1735] | 0.087 | 14/19 |
| HelmholtzSoliton-JEPA vs Alien-DualTower-JEPA | event_recall | +0.1461 | [-0.0131, +0.3123] | 0.129 | 11/19 |
| HelmholtzSoliton-JEPA vs Alien-DualTower-JEPA | hold_fpr | +0.0536 | [-0.0317, +0.1462] | 0.845 | 7/19 |
| SolitonReynolds-JEPA vs Alien-DualTower-JEPA | f1 | +0.0848 | [-0.0256, +0.2106] | 0.472 | 10/19 |
| SolitonReynolds-JEPA vs Alien-DualTower-JEPA | pr_auc | +0.0880 | [-0.0084, +0.1836] | 0.134 | 12/19 |
| SolitonReynolds-JEPA vs Alien-DualTower-JEPA | event_recall | +0.1562 | [+0.0162, +0.3047] | 0.072 | 11/19 |
| SolitonReynolds-JEPA vs Alien-DualTower-JEPA | hold_fpr | +0.0595 | [-0.0184, +0.1422] | 0.327 | 10/19 |
| EddyDelta-JEPA vs Alien-DualTower-JEPA | f1 | +0.0592 | [-0.0199, +0.1508] | 0.170 | 12/19 |
| EddyDelta-JEPA vs Alien-DualTower-JEPA | pr_auc | +0.0801 | [-0.0141, +0.1761] | 0.029 | 14/19 |
| EddyDelta-JEPA vs Alien-DualTower-JEPA | event_recall | +0.1505 | [+0.0023, +0.2932] | 0.055 | 12/19 |
| EddyDelta-JEPA vs Alien-DualTower-JEPA | hold_fpr | +0.0384 | [-0.0379, +0.1236] | 0.913 | 7/19 |
| AnchoredStokes-JEPA vs Alien-DualTower-JEPA | f1 | +0.0206 | [-0.0326, +0.0817] | 0.496 | 9/19 |
| AnchoredStokes-JEPA vs Alien-DualTower-JEPA | pr_auc | -0.0084 | [-0.0290, +0.0092] | 0.891 | 9/19 |
| AnchoredStokes-JEPA vs Alien-DualTower-JEPA | event_recall | +0.0494 | [-0.0437, +0.1299] | 0.211 | 7/19 |
| AnchoredStokes-JEPA vs Alien-DualTower-JEPA | hold_fpr | -0.0074 | [-0.0296, +0.0173] | 0.301 | 7/19 |
| HelmholtzSoliton-JEPA vs TS-JEPA | f1 | +0.0937 | [+0.0169, +0.1908] | 0.163 | 9/19 |
| HelmholtzSoliton-JEPA vs TS-JEPA | pr_auc | +0.1255 | [+0.0535, +0.2028] | 0.002 | 16/19 |
| HelmholtzSoliton-JEPA vs TS-JEPA | event_recall | +0.1990 | [+0.0233, +0.3649] | 0.070 | 9/19 |
| HelmholtzSoliton-JEPA vs TS-JEPA | hold_fpr | +0.0624 | [-0.0136, +0.1562] | 0.831 | 7/19 |
| SolitonReynolds-JEPA vs TS-JEPA | f1 | +0.0990 | [+0.0123, +0.2014] | 0.133 | 11/19 |
| SolitonReynolds-JEPA vs TS-JEPA | pr_auc | +0.1407 | [+0.0680, +0.2191] | 0.003 | 15/19 |
| SolitonReynolds-JEPA vs TS-JEPA | event_recall | +0.2091 | [+0.0735, +0.3527] | 0.017 | 11/19 |
| SolitonReynolds-JEPA vs TS-JEPA | hold_fpr | +0.0682 | [-0.0008, +0.1446] | 0.177 | 9/19 |
| EddyDelta-JEPA vs TS-JEPA | f1 | +0.0734 | [+0.0117, +0.1591] | 0.071 | 11/19 |
| EddyDelta-JEPA vs TS-JEPA | pr_auc | +0.1328 | [+0.0605, +0.2054] | 0.001 | 17/19 |
| EddyDelta-JEPA vs TS-JEPA | event_recall | +0.2034 | [+0.0523, +0.3572] | 0.031 | 10/19 |
| EddyDelta-JEPA vs TS-JEPA | hold_fpr | +0.0472 | [-0.0205, +0.1291] | 0.795 | 6/19 |
| AnchoredStokes-JEPA vs TS-JEPA | f1 | +0.0347 | [-0.0100, +0.0862] | 0.379 | 10/19 |
| AnchoredStokes-JEPA vs TS-JEPA | pr_auc | +0.0443 | [-0.0026, +0.0884] | 0.080 | 13/19 |
| AnchoredStokes-JEPA vs TS-JEPA | event_recall | +0.1023 | [+0.0108, +0.2050] | 0.064 | 10/19 |
| AnchoredStokes-JEPA vs TS-JEPA | hold_fpr | +0.0014 | [-0.0232, +0.0284] | 0.687 | 8/19 |
| Alien-DualTower-JEPA vs TS-JEPA | f1 | +0.0142 | [-0.0325, +0.0611] | 0.717 | 8/19 |
| Alien-DualTower-JEPA vs TS-JEPA | pr_auc | +0.0526 | [+0.0065, +0.0988] | 0.073 | 13/19 |
| Alien-DualTower-JEPA vs TS-JEPA | event_recall | +0.0529 | [-0.0088, +0.1208] | 0.169 | 8/19 |
| Alien-DualTower-JEPA vs TS-JEPA | hold_fpr | +0.0088 | [-0.0215, +0.0389] | 0.528 | 10/19 |
| SolitonReynolds-JEPA vs SolitonReynolds-JEPA (v1) | f1 | +0.0749 | [-0.0050, +0.1740] | 0.472 | 10/19 |
| SolitonReynolds-JEPA vs SolitonReynolds-JEPA (v1) | pr_auc | +0.1003 | [+0.0145, +0.1950] | 0.049 | 12/19 |
| SolitonReynolds-JEPA vs SolitonReynolds-JEPA (v1) | event_recall | +0.0625 | [-0.0959, +0.2208] | 0.409 | 8/19 |
| SolitonReynolds-JEPA vs SolitonReynolds-JEPA (v1) | hold_fpr | +0.0671 | [-0.0109, +0.1520] | 0.177 | 10/19 |
| EddyDelta-JEPA vs EddyDelta-JEPA (v1) | f1 | +0.0513 | [-0.0433, +0.1464] | 0.306 | 10/19 |
| EddyDelta-JEPA vs EddyDelta-JEPA (v1) | pr_auc | +0.0943 | [-0.0006, +0.1976] | 0.066 | 14/19 |
| EddyDelta-JEPA vs EddyDelta-JEPA (v1) | event_recall | +0.1045 | [-0.0687, +0.2747] | 0.218 | 10/19 |
| EddyDelta-JEPA vs EddyDelta-JEPA (v1) | hold_fpr | +0.0542 | [-0.0059, +0.1210] | 0.266 | 9/19 |
| TS-JEPA vs TimesNet | f1 | -0.1056 | [-0.1988, -0.0255] | 0.068 | 5/19 |
| TS-JEPA vs TimesNet | pr_auc | -0.0965 | [-0.2269, +0.0249] | 0.241 | 7/19 |
| TS-JEPA vs TimesNet | event_recall | -0.3106 | [-0.5270, -0.0858] | 0.015 | 5/19 |
| TS-JEPA vs TimesNet | hold_fpr | -0.2562 | [-0.3910, -0.1214] | 0.020 | 8/19 |
| TS-JEPA vs TranAD | f1 | -0.0332 | [-0.1318, +0.0668] | 0.407 | 6/19 |
| TS-JEPA vs TranAD | pr_auc | -0.0510 | [-0.1840, +0.0727] | 0.465 | 7/19 |
| TS-JEPA vs TranAD | event_recall | -0.2147 | [-0.4311, +0.0339] | 0.068 | 7/19 |
| TS-JEPA vs TranAD | hold_fpr | -0.2687 | [-0.4074, -0.1393] | 0.007 | 7/19 |
| OpEntropy vs TS-JEPA | f1 | +0.1118 | [+0.0484, +0.1831] | 0.004 | 14/19 |
| OpEntropy vs TS-JEPA | pr_auc | +0.1043 | [+0.0173, +0.2067] | 0.018 | 14/19 |
| OpEntropy vs TS-JEPA | event_recall | +0.1784 | [+0.0700, +0.2915] | 0.010 | 13/19 |
| OpEntropy vs TS-JEPA | hold_fpr | +0.0383 | [-0.0047, +0.0965] | 0.266 | 11/19 |
| Reynolds vs TS-JEPA | f1 | +0.0297 | [-0.0153, +0.0750] | 0.528 | 9/19 |
| Reynolds vs TS-JEPA | pr_auc | +0.0672 | [-0.0277, +0.1874] | 0.490 | 10/19 |
| Reynolds vs TS-JEPA | event_recall | +0.1841 | [+0.0547, +0.3396] | 0.049 | 10/19 |
| Reynolds vs TS-JEPA | hold_fpr | +0.1363 | [+0.0345, +0.2446] | 0.136 | 10/19 |
| PotentialFlow vs TS-JEPA | f1 | +0.0369 | [-0.0167, +0.0817] | 0.096 | 13/19 |
| PotentialFlow vs TS-JEPA | pr_auc | -0.0046 | [-0.0609, +0.0454] | 0.768 | 9/19 |
| PotentialFlow vs TS-JEPA | event_recall | +0.1671 | [+0.0448, +0.3038] | 0.032 | 12/19 |
| PotentialFlow vs TS-JEPA | hold_fpr | +0.0356 | [-0.0034, +0.0846] | 0.225 | 11/19 |
| OpEntropy vs TimesNet | f1 | +0.0061 | [-0.1079, +0.1345] | 0.913 | 9/19 |
| OpEntropy vs TimesNet | pr_auc | +0.0078 | [-0.1700, +0.1809] | 0.984 | 9/19 |
| OpEntropy vs TimesNet | event_recall | -0.1322 | [-0.3552, +0.1036] | 0.155 | 5/19 |
| OpEntropy vs TimesNet | hold_fpr | -0.2179 | [-0.3547, -0.0824] | 0.025 | 6/19 |
| OpEntropy vs TranAD | f1 | +0.0785 | [-0.0494, +0.2156] | 0.446 | 9/19 |
| OpEntropy vs TranAD | pr_auc | +0.0533 | [-0.1290, +0.2384] | 0.860 | 10/19 |
| OpEntropy vs TranAD | event_recall | -0.0364 | [-0.2618, +0.1913] | 0.586 | 7/19 |
| OpEntropy vs TranAD | hold_fpr | -0.2304 | [-0.3710, -0.0925] | 0.031 | 7/19 |

### Calibrator comparison on identical score streams (buffered)
| Model | Calibrator | Held-out nominal FPR (mean) | Share of rows whose 95% CI excludes q=1e-3 | Point-F1 |
|:--|:--|:-:|:-:|:-:|
| TS-JEPA | SPOT | 0.0403 | 0.58 | 0.0981 |
| TS-JEPA | SPOT+kurt(1/6) | 0.1736 | 0.81 | 0.1161 |
| TS-JEPA | p99.5 | 0.0632 | 0.61 | 0.1104 |
| TS-JEPA | max-val | 0.0502 | 0.60 | 0.0990 |
| OpEntropy | SPOT | 0.0786 | 0.65 | 0.2099 |
| OpEntropy | SPOT+kurt(1/6) | 0.1311 | 0.75 | 0.1723 |
| OpEntropy | p99.5 | 0.1072 | 0.63 | 0.2093 |
| OpEntropy | max-val | 0.0949 | 0.61 | 0.2025 |
| TimesNet | SPOT | 0.2964 | 0.91 | 0.2038 |
| TimesNet | SPOT+kurt(1/6) | 0.3077 | 0.89 | 0.1952 |
| TimesNet | p99.5 | 0.3030 | 0.95 | 0.2080 |
| TimesNet | max-val | 0.3010 | 0.93 | 0.2076 |
| TranAD | SPOT | 0.3089 | 0.86 | 0.1314 |
| TranAD | SPOT+kurt(1/6) | 0.3404 | 0.86 | 0.1330 |
| TranAD | p99.5 | 0.3233 | 0.88 | 0.1343 |
| TranAD | max-val | 0.3182 | 0.86 | 0.1334 |

### Calibration gap: SPOT design risk 1e-3 vs measured held-out FPR, by model (buffered)
| model                     |   hold_fpr |   hold_fpr_lo |   hold_fpr_hi |
|:--------------------------|-----------:|--------------:|--------------:|
| Alien-DualTower-JEPA      |     0.049  |        0.008  |        0.0992 |
| AnchoredStokes-JEPA       |     0.0416 |        0.0069 |        0.0806 |
| EddyDelta-JEPA            |     0.0874 |        0.0344 |        0.1518 |
| EddyDelta-JEPA (v1)       |     0.0333 |        0.0041 |        0.0705 |
| HelmholtzSoliton-JEPA     |     0.1026 |        0.0395 |        0.1784 |
| OpEntropy                 |     0.0786 |        0.0229 |        0.1512 |
| PotentialFlow             |     0.0758 |        0.0207 |        0.1472 |
| Reynolds                  |     0.1765 |        0.0754 |        0.2816 |
| SolitonReynolds-JEPA      |     0.1085 |        0.0445 |        0.1857 |
| SolitonReynolds-JEPA (v1) |     0.0414 |        0.0094 |        0.0806 |
| Speculative-Alien-JEPA    |     0.126  |        0.0576 |        0.2093 |
| Speculative-JEPA          |     0.1102 |        0.0451 |        0.1873 |
| TS-JEPA                   |     0.0403 |        0.0032 |        0.0881 |
| TimesNet                  |     0.2964 |        0.1581 |        0.4286 |
| TranAD                    |     0.3089 |        0.1671 |        0.4545 |