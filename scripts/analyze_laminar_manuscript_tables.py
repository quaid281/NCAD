#!/usr/bin/env python3
"""Aggregate & Paired Statistical Analysis for Laminar-JEPA and Laminar-JEPA-Lite Manuscript Tables.

Computes exact stream-averaged (after within-stream seed averaging) metrics, medians,
stream-bootstrap and dataset-cluster-bootstrap 95% CIs, paired Wilcoxon tests, exact 2^D
dataset sign-flip permutation p-values, Benjamini-Hochberg adjustments, and 12-fold LODO
audits across:
  1. `dev33` (19 Non-GHL + 14 GHL, 7 datasets) -> Table III(c), Table IV (`tab:core_benchmark`), Fig. 4(b)
  2. `tier2_141` under disjoint selection (`X_sel _|_ X_cal`, 118 untouched / 9 datasets & 141 all / 12 datasets) -> Table VII (`tab:calibrator_comparison`)
  3. `horizon_sweep` (`S in {1, 4, 16, 32, 64}`, 5 streams x 3 seeds) -> Fig. 4(a)
  4. `contextual_events` (NASA 5 contextual / 8 point & 19-stream |z|<=5 / |z|>5) -> Table VI (`tab:point_vs_contextual`)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.circuit_breaker_19stream.common import DEVELOPMENT_STREAMS
from scripts.reviewer_analyses.benchmark_141_uncertainty import bh_adjust, boot_ci, sign_flip_p
from scripts.run_tier2_exhaustive_benchmark import atomic_save_csv

TIE_TOL = 1e-4


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{os.getpid()}.json")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def analyze_dev33() -> dict[str, Any]:
    """Summarize Laminar-JEPA and Laminar-JEPA-Lite on the 33-stream development suite."""
    out: dict[str, Any] = {}
    lite_p = ROOT / "reports" / "laminar_lite_dev33_multiseed.csv"
    dev19_p = ROOT / "reports" / "laminar_dev19_full_multiseed.csv"
    frames = []
    if lite_p.exists():
        frames.append(pd.read_csv(lite_p))
    if dev19_p.exists():
        frames.append(pd.read_csv(dev19_p))
    if not frames:
        return out

    df = pd.concat(frames, ignore_index=True)
    df["ds_low"] = df["dataset"].str.lower()
    df["ch_str"] = df["channel"].astype(str)
    num_cols = [
        c for c in (
            "hold_fpr", "test_fpr", "f1", "precision", "recall", "pr_auc",
            "event_recall", "mean_delay", "cal_kurtosis", "ks_drift",
            "er_q05", "er_f01", "er_f05", "trainable_params",
        )
        if c in df.columns
    ]
    sm = df.groupby(["model", "mapping", "dataset", "ds_low", "ch_str"], as_index=False)[num_cols].mean()
    rng = np.random.default_rng(0)

    nonghl = sm[sm["ds_low"] != "ghl"]
    ghl = sm[sm["ds_low"] == "ghl"]

    dev_rows = []
    for (model, mapping), g in nonghl.groupby(["model", "mapping"]):
        row: dict[str, Any] = {"cohort": "dev19_nonghl", "model": model, "mapping": mapping, "n": len(g)}
        for col in num_cols:
            v = g[col].dropna().to_numpy(float)
            if len(v):
                lo, hi = boot_ci(v, None, rng)
                row[col] = float(v.mean())
                row[f"{col}_med"] = float(np.median(v))
                row[f"{col}_ci_lo"] = float(lo)
                row[f"{col}_ci_hi"] = float(hi)
        dev_rows.append(row)

    for (model, mapping), g in ghl.groupby(["model", "mapping"]):
        dev_rows.append({
            "cohort": "dev14_ghl",
            "model": model,
            "mapping": mapping,
            "n": len(g),
            "hold_fpr": float(g["hold_fpr"].mean()),
            "test_fpr": float(g["test_fpr"].mean()),
            "f1": float(g["f1"].mean()),
            "event_recall": float(g["event_recall"].mean()),
        })

    per_ds = (
        sm.groupby(["model", "mapping", "ds_low"], as_index=False)[["f1", "hold_fpr", "test_fpr", "pr_auc", "event_recall"]]
        .mean()
    )
    ds_macro = per_ds.groupby(["model", "mapping"], as_index=False)[["f1", "hold_fpr"]].mean()

    out["headline"] = dev_rows
    out["per_dataset"] = per_ds.round(4).to_dict("records")
    out["dataset_macro_7ds"] = ds_macro.round(4).to_dict("records")

    dev4_p = ROOT / "reports" / "laminar_dev19_4block.csv"
    if dev4_p.exists():
        d4 = pd.read_csv(dev4_p)
        d4_cols = [c for c in ("hold_fpr", "test_fpr", "f1", "pr_auc", "pr_auc_unw224", "event_recall", "mean_delay") if c in d4.columns]
        d4_sm = d4.groupby(["model", "mapping", "dataset", "channel"], as_index=False)[d4_cols].mean()
        out["dev19_4block"] = d4_sm.groupby(["model", "mapping"], as_index=False)[d4_cols].mean().round(4).to_dict("records")
    return out


def analyze_tier2_141(csv_5block: Path | None = None) -> dict[str, Any]:
    """Paired analysis on 118 untouched held-out and 141 archive streams."""
    if csv_5block is not None:
        if not csv_5block.exists():
            return {}
        comb = pd.read_csv(csv_5block)
    else:
        lam_path = ROOT / "reports" / "laminar_tier2_141stream_disjoint.csv"
        base_path = ROOT / "reports" / "heldout_tier2_141stream_independent_select.csv"
        if not lam_path.exists() or not base_path.exists():
            return {}
        lam = pd.read_csv(lam_path)
        base = pd.read_csv(base_path)
        base = base[base["model"].isin(["TS-JEPA", "Speculative-Alien-JEPA"]) & (base["calibrator"] == "SPOT")].copy()
        base["model"] = base["model"].replace({
            "Speculative-Alien-JEPA": "Multi-Horizon-JEPA-1D",
            "TS-JEPA": "TS-JEPA-1D",
        })
        metrics_0 = ["f1", "pr_auc", "event_recall", "mean_delay", "hold_fpr", "test_fpr", "n_cal"]
        common_cols = ["dataset", "channel", "seed", "model", "mapping"] + [
            m for m in metrics_0 if m in lam.columns and m in base.columns
        ]
        comb = pd.concat([lam[common_cols], base[common_cols]], ignore_index=True)

    metrics = [
        m for m in ("f1", "pr_auc", "pr_auc_unw224", "pr_auc_unw224_edge", "event_recall", "mean_delay", "hold_fpr", "test_fpr", "n_cal", "t_nom")
        if m in comb.columns
    ]
    comb["ds"] = comb["dataset"].str.lower()
    comb["ch"] = comb["channel"].astype(str)

    sm = comb.groupby(["mapping", "model", "ds", "ch"], as_index=False)[metrics].mean()
    dev_keys = {(d.lower(), str(c)) for d, c in DEVELOPMENT_STREAMS}
    sm["is_untouched"] = [((d, c) not in dev_keys) for d, c in zip(sm["ds"], sm["ch"])]

    rng = np.random.default_rng(0)
    head_rows, pair_rows, lodo_rows = [], [], []

    cohorts = {
        "untouched118": lambda d: d["is_untouched"].to_numpy(bool),
        "non_exathlon101": lambda d: d["is_untouched"].to_numpy(bool) & (d["ds"] != "exathlon").to_numpy(bool),
        "all141": lambda d: np.ones(len(d), dtype=bool),
    }
    if "t_nom" in sm.columns:
        cohorts["short_tnom_lt2500"] = lambda d: (d["t_nom"] < 2500).to_numpy(bool)
        cohorts["untouched_short_lt2500"] = lambda d: d["is_untouched"].to_numpy(bool) & (d["t_nom"] < 2500).to_numpy(bool)

    for mapping in ("trailing", "smear"):
        sub = sm[sm["mapping"] == mapping].copy()
        for cname, mask_fn in cohorts.items():
            csub = sub[mask_fn(sub)]
            for model, g in csub.groupby("model"):
                rec: dict[str, Any] = {
                    "cohort": cname,
                    "mapping": mapping,
                    "model": model,
                    "n_streams": len(g),
                    "n_datasets": int(g["ds"].nunique()),
                }
                for m in ("f1", "pr_auc", "pr_auc_unw224", "pr_auc_unw224_edge", "event_recall", "mean_delay", "hold_fpr", "test_fpr"):
                    if m not in g.columns:
                        continue
                    v = g[m].to_numpy(float)
                    ok = np.isfinite(v)
                    if not ok.any():
                        continue
                    lo_s, hi_s = boot_ci(v[ok], None, rng)
                    lo_d, hi_d = boot_ci(v[ok], g["ds"].to_numpy()[ok], rng)
                    rec[m] = float(v[ok].mean())
                    rec[f"{m}_med"] = float(np.median(v[ok]))
                    rec[f"{m}_stream_ci"] = [float(lo_s), float(hi_s)]
                    rec[f"{m}_cluster_ci"] = [float(lo_d), float(hi_d)]
                head_rows.append(rec)

        for metric_name in ("f1", "event_recall", "hold_fpr", "pr_auc"):
            wide = sub.pivot_table(index=["ds", "ch"], columns="model", values=metric_name)
            wide_untouched = np.array([(idx not in dev_keys) for idx in wide.index], dtype=bool)
            wide_nonex = wide_untouched & (wide.index.get_level_values("ds") != "exathlon")
            comparisons = [
                ("Laminar-JEPA-Lite", "Multi-Horizon-JEPA-1D"),
                ("Laminar-JEPA-Lite", "TS-JEPA-1D"),
                ("Laminar-JEPA", "Multi-Horizon-JEPA-1D"),
                ("Laminar-JEPA", "TS-JEPA-1D"),
                ("Laminar-JEPA", "Laminar-JEPA-Unq"),
                ("Laminar-JEPA-Lite", "Laminar-JEPA"),
                ("Laminar-JEPA-Lite", "Lite-MaxCoordZ"),
                ("Laminar-JEPA-Lite", "Lite-RawTopK"),
                ("Laminar-JEPA-Lite", "Lite-ConformalL2"),
            ]
            for m_a, m_b in comparisons:
                if m_a not in wide.columns or m_b not in wide.columns:
                    continue
                for cname, cmask in (
                    ("untouched118", wide_untouched),
                    ("non_exathlon101", wide_nonex),
                    ("all141", np.ones(len(wide), dtype=bool)),
                ):
                    w = wide[cmask][[m_a, m_b]].dropna()
                    if len(w) == 0:
                        continue
                    diff = (w[m_a] - w[m_b]).to_numpy(float)
                    ds_arr = w.index.get_level_values("ds").to_numpy()
                    ds_means = pd.Series(diff).groupby(ds_arr).mean()
                    lo_s, hi_s = boot_ci(diff, None, rng)
                    lo_d, hi_d = boot_ci(diff, ds_arr, rng)
                    nz = diff[np.abs(diff) > 1e-12]
                    p_wilc = float(wilcoxon(diff).pvalue) if len(nz) > 0 else 1.0
                    pair_rows.append({
                        "metric": metric_name,
                        "mapping": mapping,
                        "cohort": cname,
                        "comparison": f"{m_a} - {m_b}",
                        "n_streams": len(diff),
                        "n_datasets": len(ds_means),
                        "mean_diff": float(diff.mean()),
                        "ci_stream_lo": float(lo_s),
                        "ci_stream_hi": float(hi_s),
                        "ci_cluster_lo": float(lo_d),
                        "ci_cluster_hi": float(hi_d),
                        "wins": int((diff > TIE_TOL).sum()),
                        "losses": int((diff < -TIE_TOL).sum()),
                        "ties": int((np.abs(diff) <= TIE_TOL).sum()),
                        "p_wilcoxon": p_wilc,
                        "datasets_pos": int((ds_means > 0).sum()),
                        "datasets_neg": int((ds_means < 0).sum()),
                        "p_signflip": sign_flip_p(ds_means.to_numpy()),
                    })
                    if metric_name == "f1" and cname == "all141" and m_b in ("Multi-Horizon-JEPA-1D", "TS-JEPA-1D"):
                        for drop_ds in sorted(set(ds_arr)):
                            keep = ds_arr != drop_ds
                            lodo_rows.append({
                                "mapping": mapping,
                                "comparison": f"{m_a} - {m_b}",
                                "dropped": drop_ds,
                                "n_streams": int(keep.sum()),
                                "mean_diff": float(diff[keep].mean()),
                                "p_wilcoxon": float(wilcoxon(diff[keep]).pvalue),
                            })

    pairs_df = pd.DataFrame(pair_rows)
    if len(pairs_df):
        for arm in ("Laminar-JEPA-Lite", "Laminar-JEPA"):
            for m_focal in ("f1", "event_recall"):
                focal = (
                    (pairs_df["metric"] == m_focal)
                    & (pairs_df["mapping"] == "trailing")
                    & (pairs_df["cohort"] == "untouched118")
                    & (pairs_df["comparison"].isin([f"{arm} - Multi-Horizon-JEPA-1D", f"{arm} - TS-JEPA-1D"]))
                )
                if focal.any():
                    pairs_df.loc[focal, "p_signflip_bh"] = bh_adjust(pairs_df.loc[focal, "p_signflip"].to_numpy())
                    pairs_df.loc[focal, "p_wilcoxon_bh"] = bh_adjust(pairs_df.loc[focal, "p_wilcoxon"].to_numpy())

    # 12-fold LODO model-selection rule on all141 trailing
    sub_tr = sm[sm["mapping"] == "trailing"]
    cand_models = [m for m in ("TS-JEPA-1D", "Multi-Horizon-JEPA-1D", "Laminar-JEPA", "Laminar-JEPA-Lite") if m in sub_tr["model"].unique()]
    lodo_winners = {}
    for drop_ds in sorted(sub_tr["ds"].unique()):
        keep_df = sub_tr[(sub_tr["ds"] != drop_ds) & (sub_tr["model"].isin(cand_models))]
        agg_m = keep_df.groupby("model")[["f1", "hold_fpr"]].mean()
        feas = agg_m[agg_m["hold_fpr"] <= 0.06]
        winner = str((feas if len(feas) else agg_m)["f1"].idxmax())
        lodo_winners[drop_ds] = winner

    # Per-dataset table on the 6 multi-stream untouched cohorts
    unt = sm[sm["is_untouched"]]
    multi_ds = ["smap", "smd", "exathlon", "daphnet", "msl", "opportunity"]
    per_ds_unt = (
        unt[unt["ds"].isin(multi_ds)]
        .groupby(["mapping", "ds", "model"], as_index=False)
        .agg(n=("ch", "count"), f1=("f1", "mean"), pr_auc=("pr_auc", "mean"), er=("event_recall", "mean"), hold_fpr=("hold_fpr", "mean"), delay=("mean_delay", "mean"))
    )

    # N_cal bins on untouched 118
    ncal_bins = []
    unt_tr = unt[unt["mapping"] == "trailing"]
    for model, g in unt_tr.groupby("model"):
        small = g[g["n_cal"] < 250]
        large = g[g["n_cal"] >= 2500]
        ncal_bins.append({
            "model": model,
            "n_small": len(small),
            "fpr_small": float(small["hold_fpr"].mean()) if len(small) else float("nan"),
            "n_large": len(large),
            "fpr_large": float(large["hold_fpr"].mean()) if len(large) else float("nan"),
        })

    atomic_save_csv(pd.DataFrame(head_rows), ROOT / "reports" / "laminar_141_headline_ci.csv")
    atomic_save_csv(pairs_df, ROOT / "reports" / "laminar_141_paired_tests.csv")
    atomic_save_csv(per_ds_unt, ROOT / "reports" / "laminar_118_per_dataset.csv")
    if lodo_rows:
        atomic_save_csv(pd.DataFrame(lodo_rows), ROOT / "reports" / "laminar_141_lodo.csv")

    return {
        "headline": head_rows,
        "paired_tests": pairs_df.round(6).to_dict("records"),
        "per_dataset_untouched": per_ds_unt.round(4).to_dict("records"),
        "ncal_bins": ncal_bins,
        "lodo_winners": lodo_winners,
    }


def analyze_horizon_and_contextual() -> dict[str, Any]:
    """Summarize horizon S in {1,4,16,32,64} and contextual vs. point evaluations."""
    out: dict[str, Any] = {}
    h_base = ROOT / "reports" / "horizon_224bit_sweep.csv"
    h_lite = ROOT / "reports" / "horizon_laminar_lite_sweep.csv"
    h_frames = []
    if h_base.exists():
        b = pd.read_csv(h_base)
        b = b[b["variant"].isin(["Laminar-JEPA-224Bit", "1D-Scalar-SPOT"])].copy()
        h_frames.append(b)
    if h_lite.exists():
        h_frames.append(pd.read_csv(h_lite))
    if h_frames:
        hdf = pd.concat(h_frames, ignore_index=True)
        h_sm = hdf.groupby(["variant", "mapping", "horizon_S", "stream"], as_index=False)[
            [c for c in ("f1", "event_recall", "fpr", "hold_fpr", "pr_auc", "delay") if c in hdf.columns]
        ].mean()
        h_agg = h_sm.groupby(["variant", "mapping", "horizon_S"], as_index=False).mean(numeric_only=True)
        out["horizon_sweep"] = h_agg.round(4).to_dict("records")

    ctx_p = ROOT / "reports" / "laminar_contextual_summary.json"
    if ctx_p.exists():
        out["contextual"] = json.loads(ctx_p.read_text(encoding="utf-8"))
    return out


def main() -> None:
    p5 = ROOT / "reports" / "laminar_tier2_141stream_5block.csv"
    summary = {
        "dev33": analyze_dev33(),
        "tier2_141_4block": analyze_tier2_141(None),
        "tier2_141_5block": analyze_tier2_141(p5) if p5.exists() else {},
        "supplementary": analyze_horizon_and_contextual(),
    }
    summary["tier2_141"] = summary["tier2_141_5block"] or summary["tier2_141_4block"]
    out_json = ROOT / "reports" / "laminar_manuscript_summary.json"
    _atomic_json(summary, out_json)
    print(f"Wrote comprehensive summary -> {out_json}")

    for label, key in (("5-BLOCK", "tier2_141_5block"), ("4-BLOCK", "tier2_141_4block")):
        if not summary[key]:
            continue
        h = pd.DataFrame(summary[key]["headline"])
        cols = ["cohort", "mapping", "model", "n_streams", "f1", "pr_auc", "pr_auc_unw224", "event_recall", "hold_fpr", "hold_fpr_med", "test_fpr", "mean_delay"]
        print(f"\n=== TIER-2 118/141 HEADLINE ({label}) ===")
        print(h[[c for c in cols if c in h.columns]].round(4).to_string(index=False))

        p = pd.DataFrame(summary[key]["paired_tests"])
        f1p = p[(p["metric"].isin(["f1", "event_recall", "hold_fpr"])) & (p["cohort"].isin(["untouched118", "all141"]))]
        print(f"\n=== TIER-2 PAIRED TESTS ({label}) ===")
        print(f1p[["metric", "mapping", "cohort", "comparison", "mean_diff", "wins", "losses", "p_wilcoxon", "datasets_pos", "n_datasets", "p_signflip", "p_signflip_bh"]].round(5).to_string(index=False))

    if "dev19_4block" in summary["dev33"]:
        print("\n=== DEV-19 4-BLOCK DISJOINT ADAPTATION SENSITIVITY ===")
        print(pd.DataFrame(summary["dev33"]["dev19_4block"]).to_string(index=False))


if __name__ == "__main__":
    main()
