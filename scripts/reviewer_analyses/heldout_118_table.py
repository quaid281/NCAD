#!/usr/bin/env python3
"""Held-out 118-stream view of the archive-wide circuit-breaker benchmark (Table V).

The panel constants were tuned on 23 development streams (``DEVELOPMENT_STREAMS``); the remaining
118 streams of the 141-stream benchmark are the held-out test of the frozen panel. This script
reports, for both the held-out 118 and the full 141 cohorts (seeds averaged within stream first):

* per-model means of Point-F1, PR-AUC, event recall, held-out nominal FPR, test FPR, and delay,
  for causal (trailing) and buffered (smear) mappings;
* per-dataset Point-F1 and held-out FPR on the 118 streams;
* held-out FPR binned by calibration-set size ``n_cal`` (short-stream reliability).

Usage: python -m scripts.reviewer_analyses.heldout_118_table [--out-prefix PATH]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from scripts.circuit_breaker_19stream.common import DEVELOPMENT_STREAMS, ROOT

METRICS = ("f1", "pr_auc", "event_recall", "hold_fpr", "test_fpr", "mean_delay", "n_cal")
MODELS = (
    "TS-JEPA",
    "Speculative-JEPA",
    "Speculative-Alien-JEPA",
    "Speculative-224Bit-Strict",
    "Speculative-224Bit-Breaker",
)
NCAL_BINS = (0, 250, 500, 1000, 2500, np.inf)


def load_stream_means() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "reports" / "heldout_tier2_141stream_circuit_breaker.csv")
    is_panel = d.model.str.contains("224Bit")
    d = d[(is_panel & (d.calibrator == "CircuitBreaker")) | (~is_panel & (d.calibrator == "SPOT"))]
    d = d.assign(ds=d.dataset.str.lower(), ch=d.channel.astype(str))
    sm = d.groupby(["mapping", "model", "ds", "ch"], as_index=False)[list(METRICS)].mean()
    keys = list(zip(sm.ds, sm.ch))
    return sm.assign(heldout=[k not in DEVELOPMENT_STREAMS for k in keys])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default=str(ROOT / "reports" / "benchmark_118"))
    args = ap.parse_args()
    sm = load_stream_means()
    pd.set_option("display.width", 250)

    cohorts = {"heldout118": sm[sm.heldout], "all141": sm}
    head = pd.concat(
        [
            g.groupby(["mapping", "model"])[list(METRICS)].mean().assign(
                cohort=name, n=g.groupby(["mapping", "model"]).size()
            )
            for name, g in cohorts.items()
        ]
    ).reset_index()
    head.to_csv(f"{args.out_prefix}_headline.csv", index=False)
    print("=== HEADLINE MEANS ===")
    print(head.round(4).to_string(index=False))

    h = sm[sm.heldout]
    fam = h.pivot_table(index=["mapping", "ds"], columns="model", values=["f1", "hold_fpr"], aggfunc="mean")
    fam["n"] = h[h.model == MODELS[-1]].groupby(["mapping", "ds"]).size()
    fam.to_csv(f"{args.out_prefix}_per_dataset.csv")
    print("\n=== HELD-OUT 118: PER DATASET ===")
    print(fam.round(3).to_string())

    h = h.assign(ncal_bin=pd.cut(h.n_cal, NCAL_BINS, right=False))
    by_ncal = h.groupby(["mapping", "model", "ncal_bin"], observed=True).agg(
        n=("f1", "size"), hold_fpr=("hold_fpr", "mean"), hold_fpr_median=("hold_fpr", "median"), f1=("f1", "mean")
    )
    by_ncal.to_csv(f"{args.out_prefix}_by_ncal.csv")
    print("\n=== HELD-OUT 118: FPR BY CALIBRATION SIZE ===")
    print(by_ncal.round(4).to_string())


if __name__ == "__main__":
    main()
