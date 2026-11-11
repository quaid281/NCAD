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
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from scripts.circuit_breaker_19stream.common import DEVELOPMENT_STREAMS, ROOT
from scripts.run_tier2_exhaustive_benchmark import atomic_save_csv

PANEL = "Speculative-224Bit-Breaker"
BASELINES = ("Speculative-Alien-JEPA", "TS-JEPA")
REQUIRED_MODELS = ("TS-JEPA", "Speculative-JEPA", "Speculative-Alien-JEPA",
                   "Speculative-224Bit-Strict", PANEL)
SEEDS = (42, 123, 456)
METRICS = ("f1", "event_recall", "mean_delay", "hold_fpr", "pr_auc")
N_BOOT = 2000
TIE_TOL = 1e-4  # |dF1| below this counts as a tie, matching the manuscript win counts


def load_stream_means(input_path: Path | None = None) -> pd.DataFrame:
    d = pd.read_csv(input_path or ROOT / "reports" / "heldout_tier2_141stream_circuit_breaker.csv")
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


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values for one explicitly defined family."""
    p = np.asarray(p_values, dtype=float)
    if not len(p):
        return p.copy()
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    adjusted = np.empty_like(q)
    adjusted[order] = np.clip(q, 0.0, 1.0)
    return adjusted


def validate_complete_archive(rows: pd.DataFrame, manifest: set[tuple[str, str]]) -> None:
    panel = rows.model.isin(REQUIRED_MODELS[-2:])
    baselines = rows.model.isin(REQUIRED_MODELS[:3])
    selected = rows[(panel & (rows.calibrator == "CircuitBreaker")) |
                    (baselines & (rows.calibrator == "SPOT"))]
    actual = set(zip(selected.dataset, selected.channel))
    groups = selected.groupby(["dataset", "channel", "model", "seed", "mapping"]).size()
    expected = len(manifest) * len(REQUIRED_MODELS) * len(SEEDS) * 2
    if (actual != manifest or len(groups) != expected or not groups.eq(1).all() or
            set(selected.seed) != set(SEEDS) or set(selected.mapping) != {"trailing", "smear"}):
        raise ValueError(f"Incomplete or duplicated archive: {len(actual)} streams, {len(groups)}/{expected} model-seed-mapping cells")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "reports" / "heldout_tier2_141stream_circuit_breaker.csv")
    ap.add_argument("--out-prefix", default=str(ROOT / "reports" / "benchmark_141"))
    args = ap.parse_args()
    targets = [Path(f"{args.out_prefix}_{suffix}.csv") for suffix in ("headline_ci", "paired_tests", "lodo")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Choose a new --out-prefix to preserve existing benchmark statistics")
    manifest = {tuple(line.split(":", 1)) for line in
                (ROOT / "reports" / "streams_141_tier2.txt").read_text(encoding="utf-8").splitlines() if line}
    validate_complete_archive(pd.read_csv(args.input), manifest)
    rng = np.random.default_rng(0)
    sm = load_stream_means(args.input)
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
    focal = (pairs.mapping == "trailing") & (pairs.cohort == "untouched118")
    pairs["p_wilcoxon_bh"] = np.nan
    pairs["p_signflip_dataset_bh"] = np.nan
    for raw, adjusted in (("p_wilcoxon", "p_wilcoxon_bh"),
                          ("p_signflip_dataset", "p_signflip_dataset_bh")):
        pairs.loc[focal, adjusted] = bh_adjust(pairs.loc[focal, raw].to_numpy())
    for frame, path in zip((head, pairs, lodo), targets):
        atomic_save_csv(frame, path)

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
