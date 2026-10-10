#!/usr/bin/env python3
"""Analyze expanded regularizer ablations across 19 non-GHL benchmark streams.

Combines:
  - reports/expanded_ablations_19streams.csv (the ablation controls: OpEntropy-lambda0, Reynolds-alpha0, PotentialFlow-noregime)
  - reports/heldout_45stream.csv and reports/heldout_45stream_b.csv (the main models: TS-JEPA, OpEntropy, Reynolds, PotentialFlow, TimesNet, TranAD)
Computes paired differences, 95% bootstrap intervals, Wilcoxon signed-rank tests,
and generates the revised Table IV(a) LaTeX block.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]

def load_data():
    ablation_file = ROOT / "reports" / "expanded_ablations_19streams.csv"
    if not ablation_file.exists():
        print(f"Error: {ablation_file} not found.")
        return None
    
    df_abl = pd.read_csv(ablation_file)
    
    # Load main models for matching streams
    main_files = [ROOT / "reports" / "heldout_45stream.csv", ROOT / "reports" / "heldout_45stream_b.csv"]
    df_mains = [pd.read_csv(f) for f in main_files if f.exists()]
    df_main = pd.concat(df_mains, ignore_index=True)
    
    # Filter to matching (dataset, channel) in the 19 streams
    streams = df_abl[["dataset", "channel"]].drop_duplicates()
    df_main_sub = df_main.merge(streams, on=["dataset", "channel"])
    
    df = pd.concat([df_main_sub, df_abl], ignore_index=True)
    return df

def analyze():
    df = load_data()
    if df is None:
        return
    
    rng = np.random.default_rng(42)
    # Focus on buffered inference, SPOT calibrator
    d0 = df[(df.mapping == "smear") & (df.calibrator == "SPOT")]
    
    print("=== SUMMARY METRICS ACROSS 19 STREAMS (SEEDS 42, 123, 456) ===")
    models = [
        "TS-JEPA", "OpEntropy", "OpEntropy-lambda0",
        "Reynolds", "Reynolds-alpha0",
        "PotentialFlow", "PotentialFlow-noregime",
        "TimesNet", "TranAD"
    ]
    
    summary_rows = []
    for m in models:
        dm = d0[d0.model == m]
        if len(dm) == 0:
            continue
        # Per-seed stream means
        per_seed_f1 = dm.groupby(["seed", "dataset", "channel"]).f1.mean().groupby("seed").mean()
        f1_mean = per_seed_f1.mean()
        f1_sd = per_seed_f1.std(ddof=1) if len(per_seed_f1) > 1 else 0.0
        
        pr_auc = dm.groupby(["dataset", "channel"]).pr_auc.mean().mean()
        hold_fpr = dm.groupby(["dataset", "channel"]).hold_fpr.mean().mean()
        event_recall = dm.groupby(["dataset", "channel"]).event_recall.mean().mean()
        
        summary_rows.append({
            "model": m,
            "f1_mean": f1_mean,
            "f1_sd": f1_sd,
            "pr_auc": pr_auc,
            "hold_fpr": hold_fpr,
            "event_recall": event_recall,
            "n_streams": dm.groupby(["dataset", "channel"]).ngroups,
        })
        print(f"{m:25s} | F1: {f1_mean:.4f} ± {f1_sd:.4f} | PR-AUC: {pr_auc:.4f} | Held-FPR: {hold_fpr:.4f} | EventRec: {event_recall:.4f} (streams={dm.groupby(['dataset', 'channel']).ngroups})")
    
    print("\n=== PAIRED DIFFERENCE HYPOTHESIS TESTS (WILCOXON & 95% BOOTSTRAP CI) ===")
    pairs = [
        ("OpEntropy", "OpEntropy-lambda0"),
        ("Reynolds", "Reynolds-alpha0"),
        ("PotentialFlow", "PotentialFlow-noregime"),
        ("TS-JEPA", "OpEntropy"),
        ("TS-JEPA", "Reynolds"),
        ("TS-JEPA", "PotentialFlow"),
    ]
    
    # Compute stream means averaging seeds
    stream_m = d0.groupby(["dataset", "channel", "model"]).f1.mean().unstack("model")
    
    for a, b in pairs:
        if a not in stream_m or b not in stream_m:
            continue
        diff = (stream_m[a] - stream_m[b]).dropna()
        if len(diff) < 3:
            continue
        boots = [rng.choice(diff.values, len(diff)).mean() for _ in range(2000)]
        ci_lo, ci_hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
        pv = wilcoxon(diff.values).pvalue if np.any(diff.values != 0) else 1.0
        print(f"{a} vs {b}: Mean Diff = {diff.mean():+.4f}, 95% CI = [{ci_lo:+.4f}, {ci_hi:+.4f}], Wilcoxon p = {pv:.4f}, Streams A>B = {(diff > 0).sum()}/{len(diff)}")

if __name__ == "__main__":
    analyze()
