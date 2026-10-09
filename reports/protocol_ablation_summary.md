# Protocol ablation (7 streams x 3 seeds, stream-macro means)


## epochs=10 calib=heldout post=ma_evt thr=SPOT

| model | F1 | test FPR | event recall |
|:--|:-:|:-:|:-:|
| OpEntropy | 0.029 | 0.106 | 0.200 |
| PotentialFlow | 0.047 | 0.072 | 0.308 |
| TS-JEPA | 0.038 | 0.085 | 0.109 |
| TimesNet | 0.154 | 0.342 | 0.734 |
| TranAD | 0.161 | 0.323 | 0.690 |

JEPA mean F1 0.038 vs Recon 0.158; FPR 0.087 vs 0.333

## epochs=10 calib=heldout post=ma_evt thr=SPOT+K

| model | F1 | test FPR | event recall |
|:--|:-:|:-:|:-:|
| OpEntropy | 0.029 | 0.106 | 0.200 |
| PotentialFlow | 0.046 | 0.071 | 0.308 |
| TS-JEPA | 0.038 | 0.085 | 0.109 |
| TimesNet | 0.154 | 0.342 | 0.728 |
| TranAD | 0.161 | 0.323 | 0.690 |

JEPA mean F1 0.038 vs Recon 0.158; FPR 0.087 vs 0.332

## epochs=10 calib=heldout post=raw thr=SPOT

| model | F1 | test FPR | event recall |
|:--|:-:|:-:|:-:|
| OpEntropy | 0.028 | 0.092 | 0.200 |
| PotentialFlow | 0.043 | 0.066 | 0.308 |
| TS-JEPA | 0.038 | 0.080 | 0.109 |
| TimesNet | 0.149 | 0.334 | 0.728 |
| TranAD | 0.161 | 0.324 | 0.690 |

JEPA mean F1 0.036 vs Recon 0.155; FPR 0.079 vs 0.329

## epochs=10 calib=heldout post=raw thr=SPOT+K

| model | F1 | test FPR | event recall |
|:--|:-:|:-:|:-:|
| OpEntropy | 0.028 | 0.092 | 0.200 |
| PotentialFlow | 0.042 | 0.066 | 0.308 |
| TS-JEPA | 0.038 | 0.080 | 0.109 |
| TimesNet | 0.149 | 0.334 | 0.728 |
| TranAD | 0.161 | 0.324 | 0.690 |

JEPA mean F1 0.036 vs Recon 0.155; FPR 0.079 vs 0.329

## epochs=10 calib=train post=ma_evt thr=SPOT

| model | F1 | test FPR | event recall |
|:--|:-:|:-:|:-:|
| OpEntropy | 0.061 | 0.114 | 0.255 |
| PotentialFlow | 0.001 | 0.007 | 0.007 |
| TS-JEPA | 0.034 | 0.079 | 0.075 |
| TimesNet | 0.224 | 0.193 | 0.485 |
| TranAD | 0.239 | 0.097 | 0.387 |

JEPA mean F1 0.032 vs Recon 0.232; FPR 0.067 vs 0.145

## epochs=10 calib=train post=ma_evt thr=SPOT+K

| model | F1 | test FPR | event recall |
|:--|:-:|:-:|:-:|
| OpEntropy | 0.061 | 0.115 | 0.255 |
| PotentialFlow | 0.001 | 0.007 | 0.007 |
| TS-JEPA | 0.034 | 0.080 | 0.075 |
| TimesNet | 0.234 | 0.194 | 0.478 |
| TranAD | 0.241 | 0.096 | 0.387 |

JEPA mean F1 0.032 vs Recon 0.237; FPR 0.067 vs 0.145

## epochs=10 calib=train post=raw thr=SPOT

| model | F1 | test FPR | event recall |
|:--|:-:|:-:|:-:|
| OpEntropy | 0.054 | 0.110 | 0.231 |
| PotentialFlow | 0.001 | 0.003 | 0.007 |
| TS-JEPA | 0.034 | 0.062 | 0.075 |
| TimesNet | 0.229 | 0.201 | 0.478 |
| TranAD | 0.234 | 0.095 | 0.387 |

JEPA mean F1 0.030 vs Recon 0.232; FPR 0.058 vs 0.148

## epochs=10 calib=train post=raw thr=SPOT+K

| model | F1 | test FPR | event recall |
|:--|:-:|:-:|:-:|
| OpEntropy | 0.054 | 0.110 | 0.231 |
| PotentialFlow | 0.001 | 0.003 | 0.007 |
| TS-JEPA | 0.022 | 0.041 | 0.027 |
| TimesNet | 0.124 | 0.199 | 0.336 |
| TranAD | 0.129 | 0.093 | 0.244 |

JEPA mean F1 0.026 vs Recon 0.126; FPR 0.051 vs 0.146

## epochs=20 calib=heldout post=ma_evt thr=SPOT

| model | F1 | test FPR | event recall |
|:--|:-:|:-:|:-:|
| OpEntropy | 0.065 | 0.093 | 0.181 |
| PotentialFlow | 0.053 | 0.130 | 0.256 |
| TS-JEPA | 0.026 | 0.054 | 0.101 |
| TimesNet | 0.151 | 0.339 | 0.686 |
| TranAD | 0.158 | 0.326 | 0.690 |

JEPA mean F1 0.048 vs Recon 0.154; FPR 0.092 vs 0.333

## epochs=20 calib=heldout post=ma_evt thr=SPOT+K

| model | F1 | test FPR | event recall |
|:--|:-:|:-:|:-:|
| OpEntropy | 0.065 | 0.092 | 0.181 |
| PotentialFlow | 0.053 | 0.130 | 0.256 |
| TS-JEPA | 0.026 | 0.054 | 0.101 |
| TimesNet | 0.151 | 0.339 | 0.683 |
| TranAD | 0.158 | 0.326 | 0.690 |

JEPA mean F1 0.048 vs Recon 0.154; FPR 0.092 vs 0.333

## epochs=20 calib=heldout post=raw thr=SPOT

| model | F1 | test FPR | event recall |
|:--|:-:|:-:|:-:|
| OpEntropy | 0.065 | 0.071 | 0.133 |
| PotentialFlow | 0.048 | 0.109 | 0.256 |
| TS-JEPA | 0.027 | 0.051 | 0.149 |
| TimesNet | 0.153 | 0.338 | 0.686 |
| TranAD | 0.162 | 0.329 | 0.690 |

JEPA mean F1 0.047 vs Recon 0.157; FPR 0.077 vs 0.333

## epochs=20 calib=heldout post=raw thr=SPOT+K

| model | F1 | test FPR | event recall |
|:--|:-:|:-:|:-:|
| OpEntropy | 0.065 | 0.071 | 0.133 |
| PotentialFlow | 0.048 | 0.109 | 0.256 |
| TS-JEPA | 0.027 | 0.051 | 0.149 |
| TimesNet | 0.153 | 0.338 | 0.686 |
| TranAD | 0.162 | 0.329 | 0.690 |

JEPA mean F1 0.047 vs Recon 0.157; FPR 0.077 vs 0.333

## epochs=20 calib=train post=ma_evt thr=SPOT

| model | F1 | test FPR | event recall |
|:--|:-:|:-:|:-:|
| OpEntropy | 0.076 | 0.100 | 0.283 |
| PotentialFlow | 0.019 | 0.035 | 0.030 |
| TS-JEPA | 0.031 | 0.076 | 0.172 |
| TimesNet | 0.210 | 0.322 | 0.731 |
| TranAD | 0.224 | 0.122 | 0.390 |

JEPA mean F1 0.042 vs Recon 0.217; FPR 0.070 vs 0.222

## epochs=20 calib=train post=ma_evt thr=SPOT+K

| model | F1 | test FPR | event recall |
|:--|:-:|:-:|:-:|
| OpEntropy | 0.078 | 0.099 | 0.235 |
| PotentialFlow | 0.019 | 0.034 | 0.030 |
| TS-JEPA | 0.031 | 0.077 | 0.172 |
| TimesNet | 0.210 | 0.323 | 0.731 |
| TranAD | 0.223 | 0.119 | 0.390 |

JEPA mean F1 0.043 vs Recon 0.217; FPR 0.070 vs 0.221

## epochs=20 calib=train post=raw thr=SPOT

| model | F1 | test FPR | event recall |
|:--|:-:|:-:|:-:|
| OpEntropy | 0.072 | 0.086 | 0.235 |
| PotentialFlow | 0.005 | 0.024 | 0.006 |
| TS-JEPA | 0.031 | 0.067 | 0.172 |
| TimesNet | 0.197 | 0.312 | 0.585 |
| TranAD | 0.185 | 0.107 | 0.343 |

JEPA mean F1 0.036 vs Recon 0.191; FPR 0.059 vs 0.210

## epochs=20 calib=train post=raw thr=SPOT+K

| model | F1 | test FPR | event recall |
|:--|:-:|:-:|:-:|
| OpEntropy | 0.073 | 0.095 | 0.235 |
| PotentialFlow | 0.005 | 0.024 | 0.006 |
| TS-JEPA | 0.031 | 0.070 | 0.172 |
| TimesNet | 0.207 | 0.336 | 0.731 |
| TranAD | 0.081 | 0.102 | 0.200 |

JEPA mean F1 0.036 vs Recon 0.144; FPR 0.063 vs 0.219