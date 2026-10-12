# Latent Representation Health & Collapse Diagnostics

Evaluation across diverse telemetry streams (Daphnet, SMAP, MSL, swan) at 3 seeds.
Latent dimension $d = 32$. Effective rank computed via exponential spectral entropy $\exp(H(p))$.

| Model Architecture | Context Effective Rank ($z_{\text{ctx}}$) | Target Effective Rank ($z_{\text{tgt}}$) | Context Feature Variance | Target Feature Variance | Min Eigenvalue ($\lambda_{\min}$) | Top-1 Share |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| JEPA-CtxRecon | 5.62 / 32 | 8.54 / 32 | 0.0085 | 0.0288 | 1.14e-09 | 53.5% |
| JEPA-NoEMA | 5.68 / 32 | 13.97 / 32 | 0.0054 | 0.1237 | 1.31e-10 | 53.1% |
| TS-JEPA | 6.12 / 32 | 9.55 / 32 | 0.0031 | 0.0312 | 1.05e-10 | 52.9% |

### Key Empirical Findings:
1. **Subspace Compression without Total Collapse:** Neither TS-JEPA nor its controls suffer complete rank-1 point collapse. Effective rank spans 5.6–6.1 for context and 8.5–14.0 for target representations out of 32 dimensions. However, representations exhibit substantial spectral concentration: the leading principal component accounts for ~53% of variance, and the participation ratio occupies 4–7 effective dimensions.
2. **Representation Parity Between Prediction and Context Autoencoding:** TS-JEPA and JEPA-CtxRecon exhibit almost identical representation geometry (context rank 6.12 vs. 5.62; target rank 9.55 vs. 8.54; top-1 share 52.9% vs. 53.5%). Because JEPA-CtxRecon matches or exceeds TS-JEPA on recall (0.513 vs. 0.473) and FPR without forecasting future states, low false-positive rates are consistent with conservative subspace projection and bounded error geometry rather than an exclusive property of predictive temporal dynamics.
3. **Role of EMA Target Updates:** Removing the EMA target encoder (JEPA-NoEMA) causes target variance to expand (0.1237 vs. 0.0312) and target rank to increase to 13.97, reflecting higher unregularized representation drift across optimization steps.
4. **Conservatism Mechanism:** Compact feature variance (0.003–0.031) and small minimum eigenvalues ($\sim 10^{-10}$) explain why latent predictive scoring acts as a conservative filter under causal streaming: residual excursions remain tightly bounded, suppressing false alarms while capping sensitivity on subtle deviations.