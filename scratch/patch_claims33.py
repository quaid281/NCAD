import io
base = r"c:\Users\andre\OneDrive\Desktop\NCAD_CS\paper\NCAD_CS\sections"


def sub(fn, old, new):
    p = base + "\\" + fn
    t = io.open(p, encoding="utf-8", newline="").read()
    assert old in t, (fn, old[:50])
    io.open(p, "w", encoding="utf-8", newline="").write(t.replace(old, new))


sub("03_abstract.tex",
    "A stricter replication (seven streams, three seeds, thresholds fit on held-out nominal data, matched controls) qualifies these results. The JEPA family cuts the measured nominal false-positive rate from 36--45\\% (TimesNet, TranAD) to 4--8\\%, but at the cost of lower recall, Point-F1 and PR-AUC, and no model reaches the $10^{-3}$ SPOT design risk. Switching off the spectral-entropy, stress-closure, and regime-routing terms changes Point-F1 by no more than seed noise, so we cannot attribute the gains to the physical regularizers. We therefore present latent prediction as a conservative, low-false-alarm operating point and describe the regularizers as physics-inspired inductive biases without demonstrated benefit.",
    "These 45-stream numbers come from a single-seed pipeline that fits thresholds on training residuals, and we could not reproduce them under a stricter protocol. That protocol (33 streams, thresholds fit on held-out nominal data, up to three seeds) gives a narrower result. The unconstrained TS-JEPA lowers the measured nominal false-positive rate from about 24--30\\% (TimesNet, TranAD) to 5--7\\% (paired difference about $-0.15$, $p\\leq 0.008$), with Point-F1 and PR-AUC not significantly different from the baselines and strong dependence on the dataset. No model reaches the $10^{-3}$ SPOT design risk. None of the three physical variants improves on TS-JEPA, and switching off the entropy, stress and regime terms changes Point-F1 by no more than seed noise on a seven-stream subset, so we cannot attribute any gain to the physical regularizers. We therefore present latent prediction as a conservative, low-false-alarm operating point and describe the regularizers as physics-inspired inductive biases without demonstrated benefit.")

sub("04_intro.tex",
    "Under thresholds fit on held-out nominal data, this comes with lower recall and Point-F1, so the contribution is a trade-off and not a dominance result.",
    "Under thresholds fit on held-out nominal data (33 streams), the false-positive rate is about 4--5 times lower, while Point-F1 and PR-AUC are statistically indistinguishable from the baselines and event recall is lower, so the contribution is a trade-off and not a dominance result.")

sub("11_future.tex",
    "A held-out-calibration replication with matched controls (Section~\\ref{subsec:controlled_ablations}) limits these conclusions. The JEPA family lowers the nominal false-positive rate roughly sevenfold relative to TimesNet and TranAD, but its Point-F1, PR-AUC and event recall are lower in that experiment, and no calibrator reaches its design risk. Removing the entropy, stress or regime terms does not change Point-F1 beyond seed noise, so the physical regularizers remain motivation and not demonstrated mechanism. The kurtosis multiplier raised held-out FPR for TS-JEPA and offers no advantage over a 99.5th percentile rule. The F1 ordering in the 45-stream tables did not replicate under the stricter protocol, and reconciling the two protocols is open work.",
    "A held-out-calibration replication on 33 streams with matched controls (Section~\\ref{subsec:controlled_ablations}) limits these conclusions. The latent-prediction family lowers the nominal false-positive rate by about a factor of four to five relative to TimesNet and TranAD, with Point-F1 and PR-AUC not significantly different and a clear dependence on the dataset, and no calibrator reaches its design risk. None of the three physical variants beats the unconstrained TS-JEPA, and removing the entropy, stress or regime terms does not change Point-F1 beyond seed noise on the seven streams where controls were run, so the physical regularizers remain motivation and not demonstrated mechanism. The kurtosis multiplier raised held-out FPR for TS-JEPA and offers no advantage over a 99.5th percentile rule. The 45-stream tables above could not be reproduced under the stricter protocol and the script that generated them is not in the repository. Open work includes repeating the controls on all 33 streams, running GECCO and cicids with a faster inference path, and giving the 12 single-seed GHL streams additional seeds.")
print("ok")
