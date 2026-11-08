#!/usr/bin/env python3
"""Calibration-set-size sensitivity of the 224-bit panel on cached 19-stream residuals.

For each target calibration size n (in windows, stride 1) the nominal validation block of every run
is cut into non-overlapping sub-streams of 2n windows. Each sub-stream is processed exactly as a
short deployed stream: dial thresholds and quorums are calibrated on its first half, and the
held-out nominal FPR is measured on its second half after the S-step isolation gap. Replicates
across sub-streams give the variability of the fitted quorum K_224 and of held-out FPR. A 1D SPOT
score on the same backbone is calibrated identically for reference.

The effective sample size n_eff = n / tau_int of the calibration dials (integrated autocorrelation
time truncated at the first non-positive lag) is reported for the full calibration block (median
over dials). A matched summary restricted to streams long enough for every size is also printed.

Note: Cohort-2 caches store whitened residuals only, so Eq. (3) is not applied (as in Table II).

Usage: NCAD_PANEL_CACHE=<dir> python -m scripts.reviewer_analyses.calibration_size_sweep [--out CSV]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from scripts.circuit_breaker_19stream.common import C, ROOT, S, SEEDS, W, fit_spot, load_cached, load_streams_19
from src.scoring.circuit_breaker_panel import MultiPerspectiveCircuitBreakerPanel, build_224_perspective_dials
from src.scoring.event_fusion import aggregate_window_scores

N_CAL = (100, 250, 500, 1000)
MAX_REPLICATES = 8


def integrated_neff(va_ew: np.ndarray, n_cal_w: int) -> float:
    """Median over dials of n / tau_int, with tau_int = 1 + 2 sum_k rho_k truncated at the first rho_k <= 0."""
    d = build_224_perspective_dials(va_ew[:n_cal_w])
    d = d - d.mean(axis=0, keepdims=True)
    n = len(d)
    spec = np.fft.rfft(d, n=2 * n, axis=0)
    acov = np.fft.irfft(spec * np.conj(spec), axis=0)[:n]
    var = acov[0]
    ok = var > 1e-12
    rho = acov[:, ok] / var[ok]
    first_nonpos = np.argmax(rho[1:] <= 0, axis=0) + 1  # index of first non-positive lag
    first_nonpos[~(rho[1:] <= 0).any(axis=0)] = n
    mask = np.arange(n)[:, None] < first_nonpos[None, :]
    tau = 1.0 + 2.0 * np.where(mask, rho, 0.0)[1:].sum(axis=0)
    return float(np.median(n / np.maximum(tau, 1.0)))


def run_substream(panel, ew: np.ndarray, s_base: np.ndarray, mapping: str) -> dict:
    """Deployed protocol on one short nominal sub-stream (first half calibrates, second half is held out)."""
    n_win = len(ew)
    len_va = n_win + W - 1
    bits = panel.calibrate_and_trip_windows(ew, ew[:1], n_win // 2)
    dec = panel.evaluate_stream_decisions(bits, len_va, W, mapping)
    vp, vm = aggregate_window_scores(s_base, len_va, C, S, step=1, reducer="mean", mapping_method=mapping)
    v = vp[vm]
    half = len(v) // 2
    th = fit_spot(v[:half])
    return dict(
        fpr_breaker=float(dec.hold_pred_balanced.mean()),
        fpr_strict=float(dec.hold_pred_strict.mean()),
        fpr_spot1d=float((v[half + S :] > th).mean()),
        k224_breaker=dec.k_gate_balanced,
        n_hold=int(len(dec.hold_pred_balanced)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports" / "calibration_size_sweep_19stream.csv"))
    args = ap.parse_args()
    panel = MultiPerspectiveCircuitBreakerPanel(context_size=C, suspect_size=S)
    rows, neff_rows = [], []

    for ds, ch in load_streams_19():
        for seed in SEEDS:
            data = load_cached(ds, ch, seed)
            va_ew, va_s = data["va_ew"], data["va_s_base"]
            n_full = len(va_ew) // 2
            neff_rows.append(dict(stream=f"{ds}:{ch}", seed=seed, n_cal=n_full, n_eff=integrated_neff(va_ew, n_full)))
            for mapping in ("trailing", "smear"):
                full = run_substream(panel, va_ew, va_s, mapping)
                rows.append(dict(stream=f"{ds}:{ch}", seed=seed, mapping=mapping, n_cal="full", rep=0, **full))
                for n in N_CAL:
                    seg = 2 * n
                    for rep, start in enumerate(range(0, len(va_ew) - seg + 1, seg)):
                        if rep >= MAX_REPLICATES:
                            break
                        r = run_substream(panel, va_ew[start : start + seg], va_s[start : start + seg], mapping)
                        rows.append(dict(stream=f"{ds}:{ch}", seed=seed, mapping=mapping, n_cal=str(n), rep=rep, **r))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    neff = pd.DataFrame(neff_rows)
    neff.to_csv(args.out.replace(".csv", "_neff.csv"), index=False)

    # Per-stream means first (replicates and seeds), then across streams.
    per_stream = df.groupby(["mapping", "n_cal", "stream"]).agg(
        fpr_breaker=("fpr_breaker", "mean"), fpr_strict=("fpr_strict", "mean"), fpr_spot1d=("fpr_spot1d", "mean"),
        k_cv=("k224_breaker", lambda k: np.std(k) / max(np.mean(k), 1e-9)), reps=("rep", "size"),
    )
    summary = per_stream.groupby(["mapping", "n_cal"]).agg(
        n_streams=("reps", "size"), fpr_breaker_mean=("fpr_breaker", "mean"), fpr_breaker_median=("fpr_breaker", "median"),
        fpr_breaker_p90=("fpr_breaker", lambda x: np.percentile(x, 90)), fpr_strict_mean=("fpr_strict", "mean"),
        fpr_spot1d_mean=("fpr_spot1d", "mean"), k224_cv_median=("k_cv", "median"),
    )
    pd.set_option("display.width", 250)
    print(summary.round(4).to_string())

    # Matched comparison: only streams that admit at least one replicate at every size.
    sizes = [str(n) for n in N_CAL] + ["full"]
    ps = per_stream.reset_index()
    counts = ps.groupby(["mapping", "stream"]).n_cal.nunique()
    for mapping in ("trailing", "smear"):
        keep = counts.loc[mapping][counts.loc[mapping] == len(sizes)].index
        m = ps[(ps.mapping == mapping) & ps.stream.isin(keep)]
        print(f"\nMatched {mapping} ({len(keep)} streams):")
        print(m.groupby("n_cal")[["fpr_breaker", "fpr_strict", "fpr_spot1d", "k_cv"]].mean().loc[sizes].round(4).to_string())

    print("\nIntegrated-autocorrelation effective sample size of calibration dials (full block):")
    print(f"  median n_cal={neff.n_cal.median():.0f}, median n_eff={neff.n_eff.median():.1f}, "
          f"median ratio={np.median(neff.n_eff / neff.n_cal):.4f}")


if __name__ == "__main__":
    main()
