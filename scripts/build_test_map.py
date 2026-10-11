#!/usr/bin/env python3
"""Build the complete statistical test map (Reviewer Major Concern 5).

Enumerates every paired stream-level comparison reported in the manuscript:
cohort, inference regime, endpoint, model pair, number of paired streams,
seed treatment (averaged within stream), Wilcoxon signed-rank statistic,
raw p-value, Benjamini-Hochberg adjusted p-value (within each cohort x
regime family), median paired difference with 95% bootstrap CI, and
rank-biserial effect size.

Cohorts:
  * nonGHL19 : the 19 non-GHL primary streams (5 seeds each) -- powers all
               paired-inference claims.
  * primary33: all 33 primary-benchmark streams (non-GHL 5 seeds; GHL 1-3
               seeds) -- used for the headline means cited in the abstract.

Inputs : reports/heldout_*.csv (canonical multi-seed benchmark files;
         heldout_reynolds_covmodes.csv excluded -- separate diagnostic)
Outputs: reports/test_map.csv, paper/NCAD_CS/sections/tab_test_map.tex
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]

ENDPOINTS = ["hold_fpr", "f1", "pr_auc", "event_recall", "mean_delay"]
ENDPOINT_LABEL = {
    "hold_fpr": "Held-out nominal FPR",
    "f1": "Point-F1",
    "pr_auc": "PR-AUC",
    "event_recall": "Event recall",
    "mean_delay": "Detection delay (steps)",
}
PAIRS = [
    ("TS-JEPA", "TimesNet"),
    ("TS-JEPA", "TranAD"),
    ("OpEntropy", "TS-JEPA"),
    ("Reynolds", "TS-JEPA"),
    ("PotentialFlow", "TS-JEPA"),
    ("OpEntropy", "TimesNet"),
    ("Reynolds", "TimesNet"),
    ("PotentialFlow", "TimesNet"),
    ("TimesNet", "TranAD"),
]
REGIMES = {"smear": "Buffered", "trailing": "Causal"}
NONGHL = {l.strip() for l in (ROOT / "reports" / "streams_19_nonghl.txt").read_text().splitlines() if l.strip()}


def load_canonical() -> pd.DataFrame:
    # heldout_reynolds_covmodes.csv is a separate cov_mode diagnostic whose
    # 'Reynolds' rows collide with the canonical benchmark cells; exclude it.
    files = [f for f in sorted(glob.glob(str(ROOT / "reports" / "heldout_*.csv")))
             if "covmodes" not in f]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(
        subset=["dataset", "channel", "seed", "model", "mapping", "calibrator"], keep="last"
    )
    df["key"] = df["dataset"] + ":" + df["channel"]
    return df


def paired_stats(x: np.ndarray, rng: np.random.Generator):
    """x = per-stream paired differences (a - b), seeds already averaged."""
    n = len(x)
    if n < 3:
        return None
    if np.all(x == 0):
        return dict(n=n, W=0.0, p=1.0, med=0.0, lo=0.0, hi=0.0, rbc=np.nan)
    W, p = wilcoxon(x, zero_method="wilcox", alternative="two-sided")
    med = float(np.median(x))
    boots = np.array([rng.choice(x, n).mean() for _ in range(4000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    pos = np.sum(x > 0)
    neg = np.sum(x < 0)
    rbc = (pos - neg) / max(pos + neg, 1)
    return dict(n=n, W=float(W), p=float(p), med=med,
                lo=float(lo), hi=float(hi), rbc=float(rbc))


def main():
    rng = np.random.default_rng(0)
    df = load_canonical()
    man = pd.read_csv(ROOT / "reports" / "stream_manifest.csv")
    prim = set(man.loc[man.role == "primary_benchmark", "dataset"]
               + ":" + man.loc[man.role == "primary_benchmark", "stream"])
    cohorts = {"non-GHL (n=19)": NONGHL, "primary (n=33)": prim}

    rows = []
    for cohort_name, keys in cohorts.items():
        for mapping, regime in REGIMES.items():
            d = df[(df.mapping == mapping) & (df.calibrator == "SPOT") & df.key.isin(keys)]
            for ep in ENDPOINTS:
                sm = d.groupby(["key", "model"])[ep].mean().unstack("model")
                for a, b in PAIRS:
                    if a not in sm or b not in sm:
                        continue
                    pair = sm[[a, b]].dropna()
                    x = (pair[a] - pair[b]).values
                    st = paired_stats(x, rng)
                    if st is None:
                        continue
                    rows.append(dict(
                        cohort=cohort_name, regime=regime, endpoint=ep,
                        endpoint_label=ENDPOINT_LABEL[ep],
                        comparison=f"{a} - {b}", n_streams=st["n"],
                        seed_treatment="averaged", W=st["W"], p_raw=st["p"],
                        med_diff=st["med"], ci_lo=st["lo"], ci_hi=st["hi"],
                        rank_biserial=st["rbc"],
                    ))
    out = pd.DataFrame(rows)
    out["p_adj"] = np.nan
    for (coh, regime), idx in out.groupby(["cohort", "regime"]).groups.items():
        p = out.loc[idx, "p_raw"].values
        order = np.argsort(p)
        q = np.empty_like(p)
        prev = 1.0
        for i in range(len(p) - 1, -1, -1):
            prev = min(prev, p[order[i]] * len(p) / (i + 1))
            q[order[i]] = prev
        out.loc[idx, "p_adj"] = q
    out.to_csv(ROOT / "reports" / "test_map.csv", index=False)

    def fmt(v, nd=4):
        return "n/a" if pd.isna(v) else f"{v:.{nd}f}"

    def fmtp(v):
        return "$<$0.0001" if v < 1e-4 else f"{v:.4f}"

    lines = [
        "\\section{Complete Statistical Test Map}\\label{sup:test_map}",
        "",
        "Table~\\ref{tab:test_map} enumerates every paired stream-level comparison "
        "reported in the manuscript, resolving Reviewer Concern~5. Two cohorts are "
        "reported separately: the 19 non-GHL streams evaluated at five seeds "
        "(all paired-inference claims) and the full 33-stream primary benchmark "
        "(headline means cited in the abstract). Multiple seeds are averaged within "
        "a stream before pairing; the independent statistical unit is the telemetry "
        "stream. $p_{\\mathrm{adj}}$ applies Benjamini--Hochberg FDR control within "
        "each cohort $\\times$ regime family (45 tests per family). The effect column "
        "reports the median paired difference with a 95\\% bootstrap CI over the "
        "stream-level differences; $r_b$ is the rank-biserial correlation.",
        "",
        "{\\scriptsize",
        "\\begin{longtable}{@{}llllccccc@{}}",
        "\\caption{Complete statistical test map for paired stream-level comparisons "
        "(Wilcoxon signed-rank on stream-macro means, seeds averaged within stream).}"
        "\\label{tab:test_map}\\\\",
        "\\toprule",
        "Cohort & Regime & Endpoint & Comparison & $n$ & $p$ & $p_{\\mathrm{adj}}$ & "
        "Median diff. [CI] & $r_b$ \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\multicolumn{9}{l}{\\emph{(continued)}}\\\\",
        "\\toprule",
        "Cohort & Regime & Endpoint & Comparison & $n$ & $p$ & $p_{\\mathrm{adj}}$ & "
        "Median diff. [CI] & $r_b$ \\\\",
        "\\midrule",
        "\\endhead",
        "\\midrule\\multicolumn{9}{r}{\\emph{continued}}\\\\",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
    ]
    prev_key = None
    for _, r in out.iterrows():
        key = (r.cohort, r.regime)
        if prev_key is not None and key != prev_key:
            lines.append("\\midrule")
        cell = f"{fmt(r.med_diff)} [{fmt(r.ci_lo)}, {fmt(r.ci_hi)}]"
        lines.append(
            f"{r.cohort if key != prev_key else ''} & "
            f"{r.regime if key != prev_key else ''} & "
            f"{r.endpoint_label} & {r.comparison} & {int(r.n_streams)} & "
            f"{fmtp(r.p_raw)} & {fmtp(r.p_adj)} & {cell} & {fmt(r.rank_biserial,2)} \\\\"
        )
        prev_key = key
    lines += ["\\end{longtable}", "}", ""]
    (ROOT / "paper" / "NCAD_CS" / "sections" / "sup_test_map.tex").write_text("\n".join(lines))
    print(f"wrote reports/test_map.csv ({len(out)} tests) and sections/sup_test_map.tex")


if __name__ == "__main__":
    main()
