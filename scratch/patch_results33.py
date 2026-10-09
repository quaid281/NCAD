import io
p = r"c:\Users\andre\OneDrive\Desktop\NCAD_CS\paper\NCAD_CS\sections\10_results.tex"
L = io.open(p, encoding="utf-8", newline="").read().split("\r\n")
# 1-indexed lines 299, 304, 326-334 (verified before patching)
assert L[298].startswith("The 45-stream benchmark above fits"), L[298][:40]
assert L[325].startswith("Table~\\ref{tab:multiseed_heldout} supports"), L[325][:40]
assert L[333].startswith("\\paragraph{Causal mapping.}"), L[333][:40]
assert "Held-out calibration replication (7 streams" in L[303]

intro = (
    "The 45-stream benchmark above fits each threshold on residuals of windows the model was trained on, uses one seed, "
    "and has no matched-capacity controls. Reviewers correctly noted that each of these can favor one model family. "
    "We therefore re-ran the benchmark with a stricter protocol (\\texttt{scripts/run\\_multiseed\\_heldout\\_benchmark.py}). "
    "Models train on the first 65\\% of each nominal training sequence. The remaining 35\\% is split chronologically into a "
    "calibration half, which fits every threshold, and an evaluation half (separated by a 64-step gap) on which the nominal "
    "false-positive rate (FPR) is measured. All models use the same split, the same SPOT setting ($u=0.98$, $q=10^{-3}$), "
    "raw thresholding with no smoothing or event filter, and 10 training epochs. Of the 45 streams, GECCO and the six cicids "
    "streams were not run (stride-1 inference over 10$^5$--10$^6$ test points was too slow), and CalIt2, room-occupancy and "
    "some MSL streams were skipped because the held-out block was too short for a calibration and an evaluation half. "
    "The 33 streams that completed are 6 Daphnet, 14 GHL, 4 MSL, 6 SMAP, Genesis, metro and swan. The 19 non-GHL streams and "
    "the first two GHL streams use three seeds (42, 123, 456); the other 12 GHL streams use seed 42 only because each takes about "
    "an hour per seed. Paired tests use streams as the unit after averaging seeds, and intervals are 95\\% bootstrap intervals "
    "over streams. Control variants (a regularizer switched off with the architecture held fixed) were run only on the seven "
    "streams SMAP \\texttt{P-3} and \\texttt{A-1}, MSL \\texttt{M-1} and \\texttt{C-1}, SMD \\texttt{machine-1-2} and "
    "\\texttt{machine-1-3}, and Daphnet \\texttt{S01R01E1}, with three seeds."
)

new_table = r"""\begin{table}[t]
\centering
\small
\caption{Held-out calibration benchmark, buffered mapping, SPOT ($q=10^{-3}$) fit on a held-out nominal half. ``non-GHL'' is the stream-macro mean over 19 streams (Daphnet, MSL, SMAP, Genesis, metro, swan), seed-averaged. ``Dataset-macro'' averages the seven dataset means. ``FPR'' is measured on nominal data the threshold never saw. GHL has 14 streams, 12 of them with one seed. Source: \texttt{reports/heldout\_45stream*.csv}, \texttt{reports/heldout\_38stream\_summary.md}.}
\label{tab:heldout_33}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lccccccc@{}}
\toprule
\textbf{Model} & \textbf{F1 non-GHL} & \textbf{PR-AUC non-GHL} & \textbf{FPR non-GHL} & \textbf{Event recall non-GHL} & \textbf{F1 dataset-macro} & \textbf{F1 GHL} & \textbf{FPR GHL} \\
\midrule
TS-JEPA & 0.254 & 0.333 & 0.066 & 0.498 & 0.153 & 0.000 & 0.010 \\
Operator-Entropy & 0.241 & 0.348 & 0.079 & 0.498 & 0.134 & 0.001 & 0.005 \\
Reynolds-Stress & 0.208 & 0.321 & 0.076 & 0.398 & 0.119 & 0.001 & 0.004 \\
Potential-Flow & 0.095 & 0.205 & 0.096 & 0.331 & 0.065 & 0.000 & 0.005 \\
TimesNet & 0.207 & 0.336 & 0.297 & 0.586 & 0.104 & 0.022 & 0.058 \\
TranAD & 0.131 & 0.291 & 0.309 & 0.481 & 0.070 & 0.021 & 0.049 \\
\bottomrule
\end{tabular}%
}
\end{table}
"""

after_main = r"""Tables~\ref{tab:heldout_33} and~\ref{tab:multiseed_heldout} support the following statements, and they contradict several claims made earlier in this paper.

\paragraph{False alarms versus ranking and F1.} The JEPA family keeps the measured nominal FPR near 5--10\% on non-GHL streams, while TimesNet and TranAD reach about 30\%. Across all 33 streams the paired difference in held-out FPR is $-0.154$ against TimesNet (bootstrap interval $[-0.249, -0.071]$, Wilcoxon $p=0.008$) and $-0.157$ against TranAD ($[-0.247, -0.082]$, $p=0.001$). This is the one comparison where the latent-prediction residual is clearly better. Point-F1 and PR-AUC are not distinguishable from the baselines: TS-JEPA minus TimesNet is $+0.018$ in F1 ($[-0.052, +0.089]$, $p=0.55$) and $+0.002$ in PR-AUC ($p=0.30$), and TS-JEPA minus TranAD is $+0.062$ in F1 ($[-0.012, +0.150]$, $p=0.94$). The result depends on the dataset (Table~\ref{tab:heldout_33}). TimesNet leads on Daphnet (F1 $0.245$ against $0.155$), the JEPAs lead on MSL, SMAP and swan, and every model is near zero F1 on GHL and metro. The baselines also keep a much higher PR-AUC on SMAP ($0.51$--$0.55$ against $0.05$--$0.12$) and a higher event recall. The latent residual is therefore a more conservative detector with similar overall F1, and not a better ranker. The earlier seven-stream subset (Table~\ref{tab:multiseed_heldout}) showed much lower JEPA F1 (below $0.08$ against $0.16$); that subset does not represent the 33 streams, and it should not be used to compare model families. A factorial re-evaluation on those seven streams (\texttt{scripts/run\_protocol\_ablation.py}; calibration on training or held-out residuals, raw or 12-step moving average with event filter, SPOT with or without the kurtosis multiplier, 10 or 20 epochs; 16 cells) did not reverse the ordering there, so on that subset the gap is not caused by those protocol choices. The earlier 45-stream tables were produced by a script that is not in the repository and are not reproduced here; we treat the held-out results above as the supported ones.

\paragraph{Calibration.} None of the calibrators reaches its stated risk. With $q=10^{-3}$, plain SPOT gives held-out nominal FPR of $0.052$ (TS-JEPA), $0.062$ (Operator-Entropy), $0.074$ (Potential-Flow) and $0.058$ (Reynolds-Stress) against $0.24$ for TimesNet and TranAD. The 95\% interval excludes $q$ in 57--87\% of stream-model-seed rows. The kurtosis multiplier $1+\tfrac{1}{6}\tanh\gamma_2$ does not repair this: it raised TS-JEPA's held-out FPR from $0.052$ to $0.188$ and Operator-Entropy's from $0.062$ to $0.110$, with no F1 gain. A 99.5th percentile ($0.066$ and $0.075$) and the validation maximum ($0.052$ and $0.069$) behave like SPOT, so SPOT gives no measurable advantage over these simpler rules on this data.

\paragraph{Physical variants against the unconstrained control.} Over 33 streams, Operator-Entropy does not differ from TS-JEPA in F1 ($-0.007$, $[-0.028, +0.012]$, $p=0.71$) or PR-AUC ($+0.008$, $p=0.28$). Reynolds-Stress has a lower PR-AUC ($-0.009$, $p=0.013$) and a lower F1 that is not significant ($-0.026$, $p=0.33$). Potential-Flow is worse in F1 ($-0.091$, $[-0.178, -0.019]$, $p=0.020$) and PR-AUC ($-0.076$, $p=0.039$). None of the three physical variants improves on TS-JEPA, so the data do not support the claim that the regularizers provide a benefit.

\paragraph{Regularizer controls (seven streams).} Each physical term was switched off while the architecture was held fixed. Removing the entropy term from Operator-Entropy changes Point-F1 by $+0.001$ (interval $[-0.006, +0.009]$, $p=1.00$). Switching off the stress loss in Reynolds-Stress, or scoring with the mean term only, does not change F1 beyond seed noise ($p \geq 0.375$ for both). Removing the Grassmannian regime routing from Potential-Flow changes F1 by $+0.006$ (interval $[-0.022, +0.032]$, $p=0.58$). We therefore cannot attribute any measured effect to the spectral entropy, the stress closure, or the potential-flow structure. Whatever separates these models from TS-JEPA comes from other shared or model-specific modules (Hankel, Cohn--Elkies, shearing-wavelet and resolvent layers, regime routing), not from the physical interpretation of the regularizer. These controls were not repeated on the 33-stream set.

\paragraph{Causal mapping.} Under the strictly causal timestamp mapping, measured held-out FPR for the JEPAs falls to 0.5--1.1\%, against 20--22\% for TimesNet and TranAD, but Point-F1 reverses: TS-JEPA reaches $0.113$, Operator-Entropy $0.102$, Reynolds-Stress $0.070$ and Potential-Flow $0.028$, against $0.198$ for TimesNet and $0.132$ for TranAD (21 streams with three seeds). The false-alarm reduction reported in Table~\ref{tab:causal_vs_buffered} should therefore be read together with this recall loss, and with the fact that that table calibrates on training residuals."""

old_caption = L[303]
L[303] = old_caption.replace("Held-out calibration replication (7 streams, 3 seeds).", "Seven-stream subset with control variants (3 seeds).")
L[325:334] = after_main.split("\n")
L[298] = intro
# insert new table after intro paragraph (line index 299 is blank) 
L[299:299] = [""] + new_table.rstrip("\n").split("\n")
io.open(p, "w", encoding="utf-8", newline="").write("\r\n".join(L))
print("patched")
