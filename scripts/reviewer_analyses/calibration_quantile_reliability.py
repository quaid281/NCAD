#!/usr/bin/env python3
"""Per-quantile calibration reliability and stream-specific autocorrelation of the 224-bit panel.

Addresses reviewer concerns on calibration statistics (19 cached streams x 3 seeds, causal/trailing
window-to-timestamp mapping, so each window contributes exactly one value per statistic).

Calibration statistics (all computed on the nominal validation split, cached whitened residuals):
  * ``z_dial``  : the 224 robust-standardized perspective dials z_j (tripwire level q = 0.985);
  * ``N_224``   : per-window count of tripped 224 dials (Balanced quorum at q = 0.995);
  * ``N_128``   : per-window count of tripped 128 sub-block dials (q = 0.995);
  * ``N_edge``  : per-window count of fired onset relays (K_edge at q = 0.999);
  * ``s_base``  : the 1D scalar predictive discrepancy (1D baseline, SPOT at q = 0.999).

Task 1 (``reports/calibration_quantile_reliability.csv``): for q in {0.985, 0.995, 0.999} and target
N_eff in {100, 250, 500, 1000, 2500} (plus the full protocol block ``full``), the calibration sample
is the first N_cal = round(N_eff * tau_int) windows of the protocol calibration block [0, H) where
tau_int is the stream-seed's own Sokal integrated autocorrelation time of that statistic. Cells with
N_cal > H are marked unsupported (no resampling). Realized exceedance is measured on (A) the protocol
held-out nominal block [H + S, M_val) and (B) test windows whose entire suspect span is labeled nominal.
Thresholds: empirical quantile (all statistics), Harrell-Davis (continuous statistics), and GPD tail
extrapolation above u = q_0.98 (probability-weighted moments for dials; SPOT for ``s_base`` at
q = 0.999). The 95% moving-block-bootstrap (L = 64, B = 200) CI relative half-width of each threshold
is reported. For the trip counts the deployed quorum (multiplier and floor) is also evaluated.

Task 2 (``reports/tau_int_per_stream.csv`` and ``reports/mbb_block_length_sensitivity.csv``): Sokal
automatic-window (c = 5) and batch-means tau_int per stream, and bootstrap-implied N_eff for block
lengths L in {1, 16, 32, 64, 128}.

Usage: python -m scripts.reviewer_analyses.calibration_quantile_reliability [--n-jobs 8] [--boot 200]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import betainc

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.circuit_breaker_19stream.common import C, S, SEEDS, fit_spot, load_cached, load_streams_19  # noqa: E402
from src.scoring.circuit_breaker_panel import (  # noqa: E402
    MultiPerspectiveCircuitBreakerPanel,
    build_224_perspective_dials,
)

W = C + S
Q_LEVELS = (0.985, 0.995, 0.999)
N_EFF_TARGETS = (25, 50, 100, 250, 500, 1000, 2500)
N_CAL_TARGETS = (250, 500, 1000, 2000, 4000)  # raw window-count grid, independent of tau_int
BLOCK_LENS = (1, 16, 32, 64, 128)
# Deployed quorum rules (Balanced): K = max(mult * q, floor)
DEPLOYED = {"N_224": (0.995, 1.12, 5.0), "N_128": (0.995, 1.12, 3.5), "N_edge": (0.999, 1.25, 2.5)}
CONTINUOUS = ("z_dial", "s_base")


# ----------------------------------------------------------------------------- autocorrelation
def acf_fft(x: np.ndarray) -> np.ndarray:
    """Normalized autocorrelation function of a 1D series (biased estimator, FFT)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    if n < 4 or np.var(x) < 1e-14:
        return np.r_[1.0, np.zeros(max(n - 1, 0))]
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    return acf / acf[0]


def tau_sokal(x: np.ndarray, c: float = 5.0) -> float:
    """Sokal automatic-window integrated autocorrelation time: smallest M with M >= c * tau(M)."""
    rho = acf_fft(x)
    n = len(rho)
    if n < 4 or not np.any(rho[1:]):
        return 1.0
    taus = 1.0 + 2.0 * np.cumsum(rho[1:])
    m = np.arange(1, n)
    ok = m >= c * taus
    tau = taus[np.argmax(ok)] if ok.any() else taus[-1]
    return float(max(tau, 1.0))


def tau_batch_means(x: np.ndarray, batch: int) -> float:
    """Non-overlapping batch-means estimate tau = b * Var(batch means) / Var(x)."""
    x = np.asarray(x, float)
    nb = len(x) // batch
    if nb < 4 or np.var(x) < 1e-14:
        return float("nan")
    bm = x[: nb * batch].reshape(nb, batch).mean(axis=1)
    return float(max(batch * np.var(bm, ddof=1) / np.var(x, ddof=1), 0.0))


# ----------------------------------------------------------------------------- quantile estimators
def q_empirical(x: np.ndarray, q: float) -> np.ndarray:
    return np.quantile(x, q, axis=0)


def q_harrell_davis(x: np.ndarray, q: float) -> np.ndarray:
    """Harrell-Davis quantile (Beta-weighted average of all order statistics), column-wise."""
    n = len(x)
    a, b = (n + 1) * q, (n + 1) * (1 - q)
    grid = np.arange(n + 1) / n
    w = np.diff(betainc(a, b, grid))
    xs = np.sort(x, axis=0)
    return np.tensordot(w, xs, axes=(0, 0))


def _gpd_pwm_1d(x: np.ndarray, q: float, u_level: float = 0.98) -> float:
    """POT tail extrapolation: GPD fitted by probability-weighted moments above u = q_{u_level}."""
    n = len(x)
    u = np.quantile(x, u_level)
    y = np.sort(x[x > u] - u)
    nu = len(y)
    if nu < 5 or y.mean() <= 1e-12:
        return float(np.quantile(x, q))
    p = (np.arange(1, nu + 1) - 0.35) / nu
    a0, a1 = y.mean(), np.mean((1.0 - p) * y)
    denom = a0 - 2.0 * a1
    if denom <= 1e-12:
        return float(np.quantile(x, q))
    sigma = 2.0 * a0 * a1 / denom
    xi = float(np.clip(2.0 - a0 / denom, -0.5, 0.49))  # Hosking k = a0/denom - 2, xi = -k
    ratio = (nu / n) / (1.0 - q)
    if abs(xi) < 1e-6:
        return float(u + sigma * np.log(ratio))
    return float(u + sigma / xi * (ratio**xi - 1.0))


def q_gpd(x: np.ndarray, q: float, use_spot: bool = False) -> np.ndarray:
    if x.ndim == 1:
        return np.asarray(fit_spot(x) if use_spot else _gpd_pwm_1d(x, q))
    return np.array([_gpd_pwm_1d(x[:, j], q) for j in range(x.shape[1])])


def estimate(x: np.ndarray, q: float, method: str) -> np.ndarray:
    if method == "empirical":
        return q_empirical(x, q)
    if method == "harrell_davis":
        return q_harrell_davis(x, q)
    if method == "gpd_pot":
        return q_gpd(x, q)
    if method == "spot":
        return q_gpd(x, q, use_spot=True)
    raise ValueError(method)


def mbb_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap index vector of length n (block = 1 gives the iid bootstrap)."""
    block = max(1, min(block, n))
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=nb)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def bootstrap_thresholds(x: np.ndarray, q: float, method: str, block: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([estimate(x[mbb_indices(len(x), block, rng)], q, method) for _ in range(n_boot)])


def rel_halfwidth(boot: np.ndarray, point: np.ndarray) -> np.ndarray:
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rhw = 0.5 * (hi - lo) / np.abs(point)
    return np.where(np.abs(point) > 1e-9, rhw, np.nan)


# ----------------------------------------------------------------------------- panel statistics
def test_nominal_mask(y: np.ndarray, n_win: int) -> np.ndarray:
    """True for test windows whose whole S-step suspect span [i + C, i + W) is labeled nominal."""
    cs = np.r_[0, np.cumsum(np.asarray(y) != 0)]
    i = np.arange(n_win)
    end = np.minimum(i + W, len(y))
    return (cs[end] - cs[np.minimum(i + C, len(y))]) == 0


def panel_statistics(panel, va_ew, te_ew, n_cal: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-window statistics on (validation, test) for a panel calibrated on va windows [0, n_cal)."""
    bits = panel.calibrate_and_trip_windows(va_ew, te_ew, n_cal)
    va224, te224 = build_224_perspective_dials(va_ew), build_224_perspective_dials(te_ew)
    med = np.median(va224[:n_cal], axis=0, keepdims=True)
    mad = 1.4826 * np.median(np.abs(va224[:n_cal] - med), axis=0, keepdims=True) + 1e-4
    return {
        "z_dial": (np.maximum(0.0, (va224 - med) / mad), np.maximum(0.0, (te224 - med) / mad)),
        "N_224": (bits["bits224_va"].sum(1), bits["bits224_te"].sum(1)),
        "N_128": (bits["bits128_va"].sum(1), bits["bits128_te"].sum(1)),
        "N_edge": (bits["fast_edge_va"].sum(1), bits["fast_edge_te"].sum(1)),
        "_bits224": (bits["bits224_va"], bits["bits224_te"]),
    }


def methods_for(stat: str, q: float) -> tuple[str, ...]:
    if stat not in CONTINUOUS:
        return ("empirical",)
    out = ["empirical", "harrell_davis"]
    if q > 0.98:
        out.append("gpd_pot")
    if stat == "s_base" and q == 0.999:
        out.append("spot")
    return tuple(out)


def exceed(x: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """Exceedance fraction per column (or scalar) of x above threshold thr."""
    if len(x) == 0:
        return np.full(np.shape(thr), np.nan)
    return (x > thr).mean(axis=0)


def summarize_dials(v: np.ndarray, prefix: str) -> dict[str, float]:
    v = np.atleast_1d(v)
    return {f"{prefix}": float(np.nanmedian(v))} if v.size == 1 else {
        f"{prefix}": float(np.nanmedian(v)),
        f"{prefix}_dial_q25": float(np.nanquantile(v, 0.25)),
        f"{prefix}_dial_q75": float(np.nanquantile(v, 0.75)),
    }


# ----------------------------------------------------------------------------- per stream-seed work
def run_stream_seed(ds: str, ch: str, seed: int, n_boot: int) -> tuple[list[dict], dict, list[dict]]:
    panel = MultiPerspectiveCircuitBreakerPanel(context_size=C, suspect_size=S)
    d = load_cached(ds, ch, seed)
    va_ew, te_ew, y = d["va_ew"], d["te_ew"], d["y"]
    m_val = len(va_ew)
    h = m_val // 2
    te_nom = test_nominal_mask(y, len(te_ew))
    stream = f"{ds}:{ch}"

    # --- Task 2: autocorrelation on the full protocol calibration block [0, H)
    full = panel_statistics(panel, va_ew, te_ew, h)
    full["s_base"] = (d["va_s_base"], d["te_s_base"])
    tau = {}
    tau_row = dict(stream=stream, dataset=ds, seed=seed, M_val=m_val, H=h, n_test_nominal=int(te_nom.sum()))
    for stat in ("z_dial", "N_224", "N_128", "N_edge", "s_base"):
        x = full[stat][0][:h]
        if x.ndim == 2:
            t_dials = np.array([tau_sokal(x[:, j]) for j in range(x.shape[1])])
            tau[stat] = float(np.median(t_dials))
            tau_row.update({f"tau_sokal_{stat}": tau[stat], f"tau_sokal_{stat}_dial_q25": np.quantile(t_dials, 0.25),
                            f"tau_sokal_{stat}_dial_q75": np.quantile(t_dials, 0.75),
                            f"tau_sokal_{stat}_dial_max": t_dials.max()})
            bm_sq = [tau_batch_means(x[:, j], max(2, int(np.sqrt(h)))) for j in range(0, x.shape[1], 4)]
            bm_10 = [tau_batch_means(x[:, j], max(2, h // 10)) for j in range(0, x.shape[1], 4)]
            tau_row[f"tau_bm_sqrtN_{stat}"] = float(np.nanmedian(bm_sq))
            tau_row[f"tau_bm_10batch_{stat}"] = float(np.nanmedian(bm_10))
        else:
            tau[stat] = tau_sokal(x)
            tau_row[f"tau_sokal_{stat}"] = tau[stat]
            tau_row[f"tau_bm_sqrtN_{stat}"] = tau_batch_means(x, max(2, int(np.sqrt(h))))
            tau_row[f"tau_bm_10batch_{stat}"] = tau_batch_means(x, max(2, h // 10))
        tau_row[f"n_eff_fullH_{stat}"] = h / tau[stat]
        tau_row[f"n_cal_for_neff300_{stat}"] = 300.0 * tau[stat]
        tau_row[f"neff300_supported_{stat}"] = bool(300.0 * tau[stat] <= h)
    # pooled-dial Bartlett estimator used for the manuscript's single tau_int = 3.23
    from scripts.reviewer_analyses.calibration_size_sweep import integrated_acf_time

    tau_row["tau_bartlett_pooled_legacy"] = integrated_acf_time(build_224_perspective_dials(va_ew)[: max(64, m_val // 2)])

    # --- Task 2b: block-length sensitivity at the full block (bootstrap-implied N_eff)
    bl_rows = []
    for stat, q in (("z_dial", 0.985), ("N_224", 0.995), ("N_edge", 0.999), ("s_base", 0.995), ("s_base", 0.999)):
        x = full[stat][0][:h]
        point = q_empirical(x, q)
        var_by_l = {}
        for lb in BLOCK_LENS:
            boot = bootstrap_thresholds(x, q, "empirical", lb, n_boot, seed + lb)
            var_by_l[lb] = np.var(boot, axis=0, ddof=1)
            rhw = rel_halfwidth(boot, point)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(var_by_l[lb] > 1e-14, var_by_l[1] / var_by_l[lb], np.nan)
            bl_rows.append(dict(stream=stream, dataset=ds, seed=seed, statistic=stat, q=q, block_len=lb, n_cal=h,
                                ci_rel_halfwidth=float(np.nanmedian(rhw)),
                                var_ratio_iid_over_L=float(np.nanmedian(ratio)),
                                n_eff_bootstrap=float(h * np.nanmedian(ratio)),
                                degenerate_frac=float(np.mean(var_by_l[lb] <= 1e-14))))

    # --- Task 1: per-quantile reliability across calibration sizes
    rows = []
    targets = [(str(n), n) for n in N_EFF_TARGETS] + [(f"ncal_{n}", -n) for n in N_CAL_TARGETS] + [("full", None)]
    stats_cache: dict[int, dict] = {}
    for stat in ("z_dial", "N_224", "N_128", "N_edge", "s_base"):
        for label, n_eff_t in targets:
            if n_eff_t is None:
                n_cal = h
            elif n_eff_t < 0:  # fixed window count
                n_cal = -n_eff_t
            else:
                n_cal = int(round(n_eff_t * tau[stat]))
            supported = n_cal <= h and n_cal >= 50
            base = dict(stream=stream, dataset=ds, seed=seed, statistic=stat, n_eff_target=label,
                        tau_int=tau[stat], n_cal=n_cal, H=h, supported=supported)
            if not supported:
                for q in Q_LEVELS:
                    for meth in methods_for(stat, q):
                        rows.append({**base, "q": q, "method": meth})
                continue
            if n_cal not in stats_cache:
                st = panel_statistics(panel, va_ew, te_ew, n_cal)
                st["s_base"] = (d["va_s_base"], d["te_s_base"])
                stats_cache[n_cal] = st
            va_x, te_x = stats_cache[n_cal][stat]
            cal, hold, test_nom = va_x[:n_cal], va_x[h + S:], te_x[te_nom]
            for q in Q_LEVELS:
                for meth in methods_for(stat, q):
                    thr = estimate(cal, q, meth)
                    boot = bootstrap_thresholds(cal, q, meth, 64, n_boot, seed + n_cal)
                    r = {**base, "q": q, "method": meth, "n_eff_actual": n_cal / tau[stat],
                         "expected_tail_count_neff": n_cal / tau[stat] * (1 - q),
                         "expected_tail_count_ncal": n_cal * (1 - q), "nominal_rate": 1 - q,
                         "n_hold": len(hold), "n_test_nominal": len(test_nom)}
                    r.update(summarize_dials(exceed(hold, thr), "exc_hold"))
                    r.update(summarize_dials(exceed(test_nom, thr), "exc_test_nominal"))
                    r.update(summarize_dials(exceed(cal, thr), "exc_cal_insample"))
                    r.update(summarize_dials(rel_halfwidth(boot, thr), "ci_rel_halfwidth"))
                    r["threshold_median"] = float(np.median(thr))
                    if np.ndim(thr) == 1:  # per-dial: fraction of dials with realized > 2(1-q)
                        r["frac_dials_hold_gt_2x"] = float(np.mean(exceed(hold, thr) > 2 * (1 - q)))
                    if stat in DEPLOYED and meth == "empirical" and q == DEPLOYED[stat][0]:
                        _, mult, floor = DEPLOYED[stat]
                        k_dep = max(mult * float(thr), floor)
                        kb = np.maximum(mult * boot, floor)
                        r.update(k_deployed=k_dep, floor_active=bool(mult * float(thr) < floor),
                                 exc_hold_deployed=float(exceed(hold, k_dep)),
                                 exc_test_nominal_deployed=float(exceed(test_nom, k_dep)),
                                 ci_rel_halfwidth_deployed=float(0.5 * np.diff(np.quantile(kb, [0.025, 0.975]))[0] / k_dep))
                    rows.append(r)
            if stat == "z_dial":  # deployed tripwire (q_0.985 x hijack guard) realized trip rate
                bits_va, bits_te = stats_cache[n_cal]["_bits224"]
                rows.append({**base, "q": 0.985, "method": "deployed_guarded_tripwire", "nominal_rate": 0.015,
                             "n_eff_actual": n_cal / tau[stat], "n_hold": len(hold), "n_test_nominal": len(test_nom),
                             **summarize_dials(bits_va[h + S:].mean(0), "exc_hold"),
                             **summarize_dials(bits_te[te_nom].mean(0) if te_nom.any() else np.nan, "exc_test_nominal"),
                             **summarize_dials(bits_va[:n_cal].mean(0), "exc_cal_insample"),
                             "frac_dials_hold_gt_2x": float(np.mean(bits_va[h + S:].mean(0) > 0.03))})
    print(f"done {stream} seed {seed}", flush=True)
    return rows, tau_row, bl_rows


# ----------------------------------------------------------------------------- summaries
def summarize_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stream-seed cells into stream means, then median/IQR across streams per cell."""
    ok = df[df.supported.astype(bool)].copy()
    keys = ["statistic", "q", "method", "n_eff_target"]
    cols = ["exc_hold", "exc_test_nominal", "ci_rel_halfwidth", "n_eff_actual", "n_cal",
            "expected_tail_count_neff", "exc_hold_deployed", "exc_test_nominal_deployed", "ci_rel_halfwidth_deployed"]
    cols = [c for c in cols if c in ok]
    sm = ok.groupby(keys + ["stream"], as_index=False)[cols].mean()
    out = []
    n_streams_total = df.stream.nunique()
    for k, g in sm.groupby(keys):
        nom = 1 - k[1]
        rec = dict(zip(keys, k), nominal_rate=nom, n_streams=len(g), n_streams_total=n_streams_total)
        for c in cols:
            v = g[c].dropna()
            rec[f"{c}_median"] = v.median() if len(v) else np.nan
            rec[f"{c}_q25"] = v.quantile(0.25) if len(v) else np.nan
            rec[f"{c}_q75"] = v.quantile(0.75) if len(v) else np.nan
        cell = ok.set_index(keys).loc[k]
        rec["frac_cells_hold_gt_2x"] = float((cell.exc_hold > 2 * nom).mean())
        rec["frac_cells_testnom_gt_2x"] = float((cell.exc_test_nominal > 2 * nom).mean())
        if "frac_dials_hold_gt_2x" in cell:
            rec["frac_dials_hold_gt_2x_mean"] = float(cell.frac_dials_hold_gt_2x.mean())
        e = rec["exc_hold_median"]
        rec["within_2x_hold"] = bool(nom / 2 <= e <= 2 * nom) if np.isfinite(e) else False
        rec["ci_le_10pct"] = bool(rec.get("ci_rel_halfwidth_median", np.nan) <= 0.10)
        out.append(rec)
    return pd.DataFrame(out)


def minimum_neff(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (stat, q, meth), g in summary.groupby(["statistic", "q", "method"]):
        g = g[g.n_eff_target.str.isdigit()].assign(n=lambda t: t.n_eff_target.astype(int)).sort_values("n")
        good = g[g.within_2x_hold & g.ci_le_10pct]
        rows.append(dict(statistic=stat, q=q, method=meth,
                         min_neff_supported=int(good.n.min()) if len(good) else None,
                         evaluated=",".join(f"{n}({s})" for n, s in zip(g.n, g.n_streams))))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--streams", type=int, default=0, help="debug: limit number of streams")
    args = ap.parse_args()
    streams = load_streams_19()[: args.streams or None]
    jobs = [(ds, ch, s) for ds, ch in streams for s in SEEDS]
    # import by module path so loky workers can unpickle the function when run via ``-m``
    from scripts.reviewer_analyses.calibration_quantile_reliability import run_stream_seed as worker

    res = Parallel(n_jobs=args.n_jobs, verbose=0)(delayed(worker)(ds, ch, s, args.boot) for ds, ch, s in jobs)
    rel = pd.DataFrame([r for rows, _, _ in res for r in rows])
    tau = pd.DataFrame([t for _, t, _ in res])
    bl = pd.DataFrame([b for _, _, bls in res for b in bls])
    rep = ROOT / "reports"
    rel.to_csv(rep / "calibration_quantile_reliability_cells.csv", index=False)
    summ = summarize_reliability(rel)
    summ.to_csv(rep / "calibration_quantile_reliability.csv", index=False)
    mn = minimum_neff(summ)
    mn.to_csv(rep / "calibration_quantile_min_neff.csv", index=False)

    tau_stream = tau.groupby(["stream", "dataset"], as_index=False).mean(numeric_only=True)
    tau_stream.to_csv(rep / "tau_int_per_stream.csv", index=False)
    bl.to_csv(rep / "mbb_block_length_sensitivity_cells.csv", index=False)
    bl_sum = (bl.groupby(["statistic", "q", "block_len", "stream"], as_index=False).mean(numeric_only=True)
              .groupby(["statistic", "q", "block_len"])[["ci_rel_halfwidth", "n_eff_bootstrap", "var_ratio_iid_over_L"]]
              .median().reset_index())
    bl_sum.to_csv(rep / "mbb_block_length_sensitivity.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    show = ["statistic", "q", "method", "n_eff_target", "n_streams", "exc_hold_median", "exc_hold_q25", "exc_hold_q75",
            "exc_test_nominal_median", "frac_cells_hold_gt_2x", "ci_rel_halfwidth_median", "expected_tail_count_neff_median"]
    print(summ[show].round(4).to_string(index=False))
    print(mn.to_string(index=False))
    tcols = [c for c in tau_stream if c.startswith("tau_sokal_") and "dial_" not in c] + ["tau_bartlett_pooled_legacy"]
    print(tau_stream[["stream"] + tcols].round(2).to_string(index=False))
    print(tau_stream[tcols].describe(percentiles=[0.25, 0.5, 0.75]).round(2).to_string())
    print(bl_sum.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
