#!/usr/bin/env python3
"""Summarize reports/protocol_ablation_e{10,20}.csv -> reports/protocol_ablation_summary.md"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.concat([pd.read_csv(ROOT / "reports" / f"protocol_ablation_e{e}.csv") for e in (10, 20)])
# average seeds within stream, then streams
g = df.groupby(["epochs", "calib_source", "postproc", "thresholder", "model", "dataset", "channel"], as_index=False).mean(numeric_only=True)
t = g.groupby(["epochs", "calib_source", "postproc", "thresholder", "model"], as_index=False)[
    ["f1", "test_fpr", "event_recall", "pr_auc"]].mean()
lines = ["# Protocol ablation (7 streams x 3 seeds, stream-macro means)\n"]
fam = {"TS-JEPA": "JEPA", "OpEntropy": "JEPA", "PotentialFlow": "JEPA", "TimesNet": "Recon", "TranAD": "Recon"}
for (e, cs, po, th), s in t.groupby(["epochs", "calib_source", "postproc", "thresholder"]):
    lines.append(f"\n## epochs={e} calib={cs} post={po} thr={th}\n")
    lines.append("| model | F1 | test FPR | event recall |\n|:--|:-:|:-:|:-:|")
    for _, r in s.iterrows():
        lines.append(f"| {r.model} | {r.f1:.3f} | {r.test_fpr:.3f} | {r.event_recall:.3f} |")
    j, b = s[s.model.map(fam) == "JEPA"], s[s.model.map(fam) == "Recon"]
    lines.append(f"\nJEPA mean F1 {j.f1.mean():.3f} vs Recon {b.f1.mean():.3f}; FPR {j.test_fpr.mean():.3f} vs {b.test_fpr.mean():.3f}")
(ROOT / "reports" / "protocol_ablation_summary.md").write_text("\n".join(lines), encoding="utf-8")
cmp = t.assign(fam=t.model.map(fam)).groupby(["epochs", "calib_source", "postproc", "thresholder", "fam"])[["f1", "test_fpr", "event_recall"]].mean().round(3)
print(cmp.unstack("fam").to_string())
