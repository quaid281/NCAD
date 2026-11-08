#!/usr/bin/env python3
"""Uncertainty and dataset robustness of the 141-stream circuit-breaker benchmark (Table V).

Seeds are averaged within stream first. Reports, for causal (trailing) and buffered (smear) mappings:

* headline means with 95% stream-bootstrap and dataset-cluster-bootstrap CIs;
* paired differences of Speculative-224Bit-Breaker vs. 1D baselines: stream-level Wilcoxon,
  wins/losses (|dF1| <= 1e-4 counted as ties), dataset-level exact sign-flip permutation test on the
  12 per-dataset mean differences, and leave-one-dataset-out (LODO) means;
* the same paired test restricted to the 118 streams never used to tune the panel.

Usage: python -m scripts.reviewer_analyses.benchmark_141_uncertainty [--out-prefix PATH]
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from scripts.circuit_breaker_19stream.common import DEVELOPMENT_STREAMS, ROOT

PANEL = "Speculative-224Bit-Breaker"
BASELINES = ("Speculative-Alien-JEPA", "TS-JEPA")
METRICS = ("f1", "event_recall", "mean_delay", "hold_fpr", "pr_auc")
N_BOOT = 2000
TIE_TOL = 1e-4  # |dF1| below this counts as a tie, matching the manuscript win counts


def load_stream_means() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "reports" / "heldout_tier2_141stream_circuit_breaker.csv")
    is_panel = d.model.str.contains("224Bit")
    d = d[(is_panel & (d.calibrator == "CircuitBreaker")) | (~is_panel & (d.calibrator == "SPOT"))]
    d = d.assign(ds=d.dataset.str.lower(), ch=d.channel.astype(str))
    return d.groupby(["mapping", "model", "ds", "ch"], as_index=False)[list(METRICS)].mean()


def boot_ci(values: np.ndarray, groups: np.ndarray | None, rng: np.random.Generator) -> tuple[float, float]:
    """95% percentile CI of the mean, resampling streams or whole dataset clusters."""
    if groups is None:
        stats = [rng.choice(values, len(values)).mean() for _ in range(N_BOOT)]
    else:
        uniq = np.unique(groups)
        idx = {g: np.flatnonzero(groups == g) for g in uniq}
        stats = [values[np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq))])].mean() for _ in range(N_BOOT)]
    return tuple(np.nanpercentile(stats, [2.5, 97.5]))


def sign_flip_p(cluster_means: np.ndarray) -> float:
    """Exact two-sided sign-flip permutation p-value for the mean of cluster-level differences."""
    obs = abs(cluster_means.mean())
    flips = np.array(list(itertools.product((-1.0, 1.0), repeat=len(cluster_means))))
    return float((np.abs((flips * cluster_means).mean(axis=1)) >= obs - 1e-12).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default=str(ROOT / "reports" / "benchmark_141"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    sm = load_stream_means()
    head_rows, pair_rows, lodo_rows = [], [], []

    for mapping in ("trailing", "smear"):
        sub = sm[sm.mapping == mapping]
        for model, g in sub.groupby("model"):
            for m in METRICS:
                v = g[m].to_numpy(float)
                ok = np.isfinite(v)
                lo_s, hi_s = boot_ci(v[ok], None, rng)
                lo_d, hi_d = boot_ci(v[ok], g.ds.to_numpy()[ok], rng)
                head_rows.append(dict(mapping=mapping, model=model, metric=m, mean=v[ok].mean(), n=int(ok.sum()),
                                      ci_stream_lo=lo_s, ci_stream_hi=hi_s, ci_dataset_lo=lo_d, ci_dataset_hi=hi_d))

        wide = sub.pivot_table(index=["ds", "ch"], columns="model", values="f1")
        for base in BASELINES:
            for cohort, mask in (("all141", np.ones(len(wide), bool)),
                                 ("untouched118", ~wide.index.isin(list(DEVELOPMENT_STREAMS)))):
                w = wide[mask][[PANEL, base]].dropna()
                diff = (w[PANEL] - w[base]).to_numpy()
                ds = w.index.get_level_values("ds").to_numpy()
                ds_means = pd.Series(diff).groupby(ds).mean()
                lo, hi = boot_ci(diff, None, rng)
                lo_d, hi_d = boot_ci(diff, ds, rng)
                pair_rows.append(dict(
                    mapping=mapping, comparison=f"{PANEL} - {base}", cohort=cohort, n_streams=len(diff),
                    n_datasets=len(ds_means), mean_diff=diff.mean(), ci_stream_lo=lo, ci_stream_hi=hi,
                    ci_dataset_lo=lo_d, ci_dataset_hi=hi_d, wins=int((diff > TIE_TOL).sum()), losses=int((diff < -TIE_TOL).sum()),
                    ties=int((np.abs(diff) <= TIE_TOL).sum()), p_wilcoxon=float(wilcoxon(diff).pvalue),
                    datasets_positive=int((ds_means > 0).sum()), p_signflip_dataset=sign_flip_p(ds_means.to_numpy()),
                ))
                if cohort == "all141":
                    for drop in sorted(set(ds)):
                        keep = ds != drop
                        lodo_rows.append(dict(mapping=mapping, comparison=f"{PANEL} - {base}", dropped=drop,
                                              n_streams=int(keep.sum()), mean_diff=diff[keep].mean(),
                                              p_wilcoxon=float(wilcoxon(diff[keep]).pvalue)))

    head, pairs, lodo = pd.DataFrame(head_rows), pd.DataFrame(pair_rows), pd.DataFrame(lodo_rows)
    head.to_csv(f"{args.out_prefix}_headline_ci.csv", index=False)
    pairs.to_csv(f"{args.out_prefix}_paired_tests.csv", index=False)
    lodo.to_csv(f"{args.out_prefix}_lodo.csv", index=False)

    pd.set_option("display.width", 250)
    print("=== HEADLINE (224Bit-Breaker vs 1D baselines) ===")
    keep = head.model.isin((PANEL,) + BASELINES)
    print(head[keep].round(4).to_string(index=False))
    print("\n=== PAIRED POINT-F1 TESTS ===")
    print(pairs.round(5).to_string(index=False))
    print("\n=== LODO RANGE OF MEAN DIFF (min / max, max p) ===")
    print(lodo.groupby(["mapping", "comparison"]).agg(min_diff=("mean_diff", "min"), max_diff=("mean_diff", "max"),
                                                       max_p=("p_wilcoxon", "max")).round(5).to_string())
    print("\nLargest-influence drops:")
    print(lodo.loc[lodo.groupby(["mapping", "comparison"]).mean_diff.idxmin()].round(5).to_string(index=False))


if __name__ == "__main__":
    main()
