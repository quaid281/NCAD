#!/usr/bin/env python3
"""Generate Figure 2: 3-panel composite empirical figure for IEEE TNNLS manuscript.

Panels:
  (a) Mechanism Isolation (Calibration Excess Kurtosis gamma_2 vs.
      Calibration-to-Holdout KS Drift D_KS, colored by Held-Out Causal FPR).
  (b) Matched Realized Test-FPR Operating Curves (Event Recall vs. Realized Test
      FPR f* in [0.01, 0.50] across 19 non-GHL streams, strictly causal trailing,
      with 95% bootstrap confidence bands for TS-JEPA and TimesNet).
  (c) Multi-Risk Calibration Reliability Diagram (Target Design Risk q in
      [10^-4, 10^{-1}] vs. Realized Held-Out Causal FPR on log-log axes, showing
      median + IQR bands, stream-macro means, ideal y=x diagonal, and the
      exchangeable permutation control curve for TS-JEPA).

Outputs:
  paper/NCAD_CS/figures/mechanism_operating_calibration.pdf
  paper/NCAD_CS/figures/mechanism_operating_calibration.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
SKILL_SCRIPTS = ROOT / ".agents" / "skills" / "beautiful-research-figures" / "scripts"
if str(SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS))

from run_multiseed_heldout_benchmark import event_stats  # noqa: E402
from src.scoring.evt_calibrator import fit_gpd  # noqa: E402
from ieee_vector_style import (  # noqa: E402
    save_true_vector_figure,
    vectorize_figure_collections,
    is_bbox_outside_axes,
)

REPORTS = ROOT / "reports"
DUMP_DIR = REPORTS / "score_dumps"
FIG_DIR = ROOT / "paper" / "NCAD_CS" / "figures"


def compute_panel_a_data() -> pd.DataFrame:
    """Compute stream-macro Kurtosis, KS Drift, and Held-Out Causal FPR across arms."""
    dfs = []
    for fn in [
        "heldout_19stream_r3.csv",
        "heldout_bounded_obs_tcn.csv",
        "heldout_19stream_r4_arms.csv",
        "heldout_nonparametric.csv",
        "matched_fpr_evaluation.csv",
    ]:
        p = REPORTS / fn
        if p.exists() and fn != "matched_fpr_evaluation.csv":
            dfs.append(pd.read_csv(p))
    h = pd.concat(dfs, ignore_index=True).drop_duplicates(
        subset=["dataset", "channel", "seed", "model", "mapping", "calibrator"]
    )
    h = h[~h.dataset.str.contains("GHL")]
    spot_tr = (
        h[(h.calibrator == "SPOT") & (h.mapping == "trailing")]
        .groupby(["dataset", "channel", "model"], as_index=False)
        .mean(numeric_only=True)
    )
    m_tr = spot_tr.groupby("model")[["cal_kurtosis", "hold_fpr", "f1", "pr_auc", "event_recall"]].mean()

    # Calibration-to-holdout KS drift from dumped score arrays
    ks_rows = []
    for fn in sorted(DUMP_DIR.glob("*__trailing.npz")):
        ds, ch, seed, model, _ = fn.stem.split("__")
        if "GHL" in ds:
            continue
        d = np.load(fn)
        cal, hold = d["cal"], d["hold"]
        ks_val = float(stats.ks_2samp(cal, hold).statistic) if len(hold) > 10 else np.nan
        ks_rows.append(dict(dataset=ds, channel=ch, seed=int(seed), model=model, ks_drift=ks_val))
    df_ks = (
        pd.DataFrame(ks_rows)
        .groupby(["dataset", "channel", "model"], as_index=False)
        .mean(numeric_only=True)
        .groupby("model")[["ks_drift"]]
        .mean()
    )
    return m_tr.join(df_ks)


def compute_panel_b_data():
    """Compute direct matched realized test-FPR operating curves across 19 non-GHL streams."""
    f_grid = np.array([0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
    models = [
        "TS-JEPA",
        "OpEntropy",
        "JEPA-CtxRecon",
        "PCA-LinPred-LN",
        "TCN-ObsPred-LN",
        "TimesNet",
        "TCN-ObsPred",
        "TranAD",
    ]

    rows = []
    for fn in sorted(DUMP_DIR.glob("*__trailing.npz")):
        ds, ch, seed, model, _ = fn.stem.split("__")
        if model not in models or "GHL" in ds:
            continue
        d = np.load(fn)
        ts, tl = d["test_scores"], d["test_labels"]
        if tl.sum() == 0:
            continue
        if not np.isfinite(ts).all():
            ts = np.where(np.isfinite(ts), ts, np.inf)
        nom = ts[tl == 0]
        for f in f_grid:
            th = float(np.quantile(nom, 1.0 - f))
            pred = (ts > th).astype(int)
            er, _ = event_stats(tl, pred)
            rows.append(dict(dataset=ds, channel=ch, seed=int(seed), model=model, f_star=f, er=er))

    df = pd.DataFrame(rows)
    sm = df.groupby(["dataset", "channel", "model", "f_star"], as_index=False)["er"].mean()
    curves = sm.groupby(["model", "f_star"])["er"].mean().unstack("f_star")

    # Stream-level bootstrap 95% CI bands (1000 resamples of the 19 streams)
    rng = np.random.default_rng(2024)
    ci_bands = {}
    for m in ["TS-JEPA", "TimesNet"]:
        sub = sm[sm.model == m].pivot(index=["dataset", "channel"], columns="f_star", values="er")
        arr = sub.to_numpy()
        n_s = len(arr)
        boots = np.empty((1000, len(f_grid)))
        for b in range(1000):
            idx = rng.choice(n_s, size=n_s, replace=True)
            boots[b] = np.nanmean(arr[idx], axis=0)
        ci_bands[m] = (np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0))

    return f_grid, curves, ci_bands


def compute_panel_c_data():
    """Compute multi-risk calibration curves and exchangeable permutation control."""
    sw = pd.read_csv(REPORTS / "evt_calibration_curve_sweep.csv")
    ex = pd.read_csv(REPORTS / "exchangeable_calibration_control.csv")

    q_levels = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    stats_dict = {}
    for m in ["TS-JEPA", "TimesNet", "TranAD"]:
        sub = sw[sw.model == m]
        meds, q25s, q75s, means = [], [], [], []
        for q in q_levels:
            vals = sub[np.isclose(sub.risk_q, q)]["achieved_fpr"].to_numpy()
            meds.append(float(np.median(vals)))
            q25s.append(float(np.quantile(vals, 0.25)))
            q75s.append(float(np.quantile(vals, 0.75)))
            means.append(float(np.mean(vals)))
        stats_dict[m] = dict(
            q=q_levels,
            median=np.array(meds),
            q25=np.array(q25s),
            q75=np.array(q75s),
            mean=np.array(means),
        )

    rng = np.random.default_rng(42)
    ex_rows = []
    for fn in sorted(DUMP_DIR.glob("*__trailing.npz")):
        ds, ch, seed, model, _ = fn.stem.split("__")
        if model != "TS-JEPA":
            continue
        d = np.load(fn)
        cal, hold = d["cal"], d["hold"]
        if len(cal) < 100 or len(hold) < 100:
            continue
        pooled = np.concatenate([cal, hold])
        idx = rng.permutation(len(pooled))
        cal_p, hold_p = pooled[idx[: len(cal)]], pooled[idx[len(cal) :]]
        u = float(np.quantile(cal_p, 0.98))
        exc = cal_p[cal_p > u] - u
        fit = fit_gpd(exc) if len(exc) >= 5 else None
        for q in q_levels:
            th_emp = float(np.quantile(cal_p, 1.0 - q))
            if q < 0.02 and fit is not None:
                r = (len(cal_p) / len(exc)) * q
                th_spot = (
                    u + (fit.sigma / fit.gamma) * (r ** (-fit.gamma) - 1.0)
                    if abs(fit.gamma) > 1e-6
                    else u - fit.sigma * np.log(r)
                )
            else:
                th_spot = th_emp
            ex_rows.append(dict(q=q, fpr_spot=float((hold_p > th_spot).mean()), fpr_emp=float((hold_p > th_emp).mean())))
    df_ex = pd.DataFrame(ex_rows)
    ex_curve = df_ex.groupby("q")["fpr_spot"].mean().reindex(q_levels).to_numpy()
    ex_sub = ex[(ex.cal_fraction == 1.0) & (ex.mapping == "trailing") & (ex.model == "TS-JEPA")]
    for i, q in enumerate(q_levels):
        m_rows = ex_sub[np.isclose(ex_sub.design_q, q)]
        if len(m_rows) and q <= 0.01:
            ex_curve[i] = float(m_rows["fpr_point_shuffle"].mean())

    return q_levels, stats_dict, ex_curve


def main():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 7.5,
        "axes.labelsize": 7.8,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 4.8,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    df_a = compute_panel_a_data()
    f_grid, curves_b, ci_bands_b = compute_panel_b_data()
    q_levels, stats_c, ex_curve_c = compute_panel_c_data()

    fig, axes = plt.subplots(
        1, 3, figsize=(7.16, 2.75),
        gridspec_kw={"width_ratios": [1.14, 1.0, 1.0]}
    )
    fig.subplots_adjust(top=0.88, bottom=0.24, left=0.08, right=0.98, wspace=0.32)

    # Subtle dual-hierarchy grids across all panels (Principle 4)
    for ax_p in axes:
        ax_p.grid(True, which="major", linestyle="-", linewidth=0.55, alpha=0.22)
        ax_p.minorticks_on()
        ax_p.grid(True, which="minor", linestyle="-", linewidth=0.20, alpha=0.10)
        ax_p.set_axisbelow(True)

    # =========================================================================
    # PANEL (a): Mechanism Isolation (Kurtosis vs. KS Drift)
    # =========================================================================
    ax = axes[0]
    arms_a = [
        ("TS-JEPA", "TS-JEPA", "o"),
        ("OpEntropy", "OpEntropy", "s"),
        ("JEPA-CtxRecon", "JEPA-CtxRecon", "P"),
        ("JEPA-NoEMA", "JEPA-NoEMA", "v"),
        ("TS-JEPA-NoGate", "JEPA-NoGate", "<"),
        ("PCA-LinPred-LN", "PCA-LinPred-LN", "h"),
        ("TCN-ObsPred-LN", "TCN-ObsPred-LN", "s"),
        ("IsoForest", "IsoForest", "X"),
        ("TCN-ObsPred", "TCN-ObsPred", "o"),
        ("TCN-ObsRecon", "TCN-ObsRecon", "^"),
        ("TimesNet", "TimesNet", "D"),
        ("TranAD", "TranAD", "v"),
    ]

    # Monotonic luminance, colorblind- and grayscale-safe colormap (Principle 5)
    cmap = plt.cm.cividis
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=0.26)

    for key, label, marker in arms_a:
        if key not in df_a.index:
            continue
        row = df_a.loc[key]
        x, y, fpr = row["cal_kurtosis"], row["ks_drift"], row["hold_fpr"]
        ax.scatter(
            x, y, c=[fpr], cmap=cmap, norm=norm, marker=marker, s=32,
            edgecolors="#111111", linewidths=0.55, zorder=4
        )

    # Shaded regime box for latent / bounded predictors
    ax.axhspan(0.108, 0.258, xmin=0.0, xmax=0.55, color="#1d4e89", alpha=0.08, zorder=1)
    ax.text(
        0.40, 0.112,
        "Latent / Bounded Pred.\n($D_{\\mathrm{KS}}\\approx 0.17\\text{--}0.23$, FPR $0.9\\text{--}2.8\\%$)",
        fontsize=4.9, color="#0b4f8a", weight="bold"
    )

    # Mechanism arrows
    x1, y1 = df_a.loc["TCN-ObsPred", ["cal_kurtosis", "ks_drift"]]
    x2, y2 = df_a.loc["TCN-ObsPred-LN", ["cal_kurtosis", "ks_drift"]]
    ax.annotate(
        "", xy=(x2 + 0.16, y2 + 0.003), xytext=(x1 - 0.16, y1 - 0.003),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.05, ls="--"), zorder=3
    )
    ax.text(
        2.75, 0.450,
        "(1) LayerNorm\n($\\gamma_2: 6.10\\!\\to\\!0.91$)",
        fontsize=4.9, color="#333333", ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#cccccc", alpha=0.88)
    )

    x3, y3 = df_a.loc["OpEntropy", ["cal_kurtosis", "ks_drift"]]
    ax.annotate(
        "", xy=(x3, y3 + 0.015), xytext=(x2, y2 - 0.015),
        arrowprops=dict(arrowstyle="->", color="#1d4e89", lw=1.25), zorder=3
    )
    ax.text(
        2.35, 0.300,
        "(2) Latent proj.\n($D_{\\mathrm{KS}}: 0.376\\!\\to\\!0.209$)",
        fontsize=4.9, color="#0b4f8a", ha="center",
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#99c2ff", alpha=0.9)
    )

    leader_kw = dict(arrowstyle="-", lw=0.6, color="0.4", shrinkA=1, shrinkB=3)
    ax.annotate(
        "TCN-ObsPred (24.7%)\nTCN-ObsRecon (24.5%)",
        xy=(6.10, 0.463), xytext=(6.78, 0.476),
        fontsize=4.8, ha="right", va="bottom", color="#111111",
        arrowprops=leader_kw, zorder=5
    )
    ax.annotate(
        "TimesNet (24.3%)\nTranAD (22.2%)",
        xy=(5.54, 0.438), xytext=(6.78, 0.380),
        fontsize=4.8, ha="right", va="center", color="#111111",
        arrowprops=leader_kw, zorder=5
    )
    ax.annotate(
        "TCN-ObsPred-LN (11.7%)",
        xy=(0.91, 0.376), xytext=(1.45, 0.358),
        fontsize=4.8, ha="left", va="center", color="#111111",
        arrowprops=leader_kw, zorder=5
    )
    ax.annotate(
        "IsoForest (3.9%)",
        xy=(0.70, 0.373), xytext=(0.38, 0.475),
        fontsize=4.8, ha="left", va="bottom", color="#111111",
        arrowprops=leader_kw, zorder=5
    )
    ax.annotate(
        "PCA-LinPred-LN (2.1%)",
        xy=(0.95, 0.233), xytext=(1.55, 0.266),
        fontsize=4.8, ha="left", va="center", color="#111111",
        arrowprops=leader_kw, zorder=5
    )
    ax.annotate(
        "TS-JEPA / OpEntropy\n(2.4--2.8%)",
        xy=(1.33, 0.214), xytext=(2.95, 0.232),
        fontsize=4.8, ha="left", va="center", color="#111111",
        arrowprops=leader_kw, zorder=5
    )
    ax.annotate(
        "NoEMA / NoGate (0.9--1.8%)",
        xy=(1.16, 0.167), xytext=(1.55, 0.165),
        fontsize=4.8, ha="left", va="center", color="#111111",
        arrowprops=leader_kw, zorder=5
    )
    ax.annotate(
        "JEPA-CtxRecon (1.7%)",
        xy=(2.59, 0.203), xytext=(3.20, 0.182),
        fontsize=4.8, ha="left", va="center", color="#111111",
        arrowprops=leader_kw, zorder=5
    )

    sm_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm_map.set_array([])
    cbar = fig.colorbar(sm_map, ax=ax, pad=0.015, aspect=25)
    cbar.solids.set_rasterized(False)  # True vector colorbar (Principle 1)
    cbar.set_label("Held-out causal FPR", fontsize=6.4, labelpad=2)
    cbar.ax.tick_params(labelsize=5.8)

    ax.set_xlim(0.3, 6.85)
    ax.set_ylim(0.104, 0.542)
    ax.set_xlabel(r"Calibration excess kurtosis ($\gamma_2$)", labelpad=2)
    ax.set_ylabel(r"Cal$\to$Hold KS drift ($D_{\mathrm{KS}}$)", labelpad=2)
    ax.set_title("(a) Mechanism Isolation", fontweight="bold", pad=4)
    ax.text(
        0.5, -0.24, "Marker: 12 backbone arms  |  Color: causal FPR",
        transform=ax.transAxes, ha="center", va="top", fontsize=4.8, color="#555555"
    )

    # =========================================================================
    # PANEL (b): Matched Realized Test-FPR Operating Curves
    # =========================================================================
    ax = axes[1]
    # Curated academic palette (high-contrast, distinct luminance & stroke; Principle 5)
    styles_b = {
        "TS-JEPA":        ("#1d4e89", "-",  "o", "TS-JEPA"),
        "OpEntropy":      ("#5c2d91", "-",  "s", "OpEntropy"),
        "JEPA-CtxRecon":  ("#00706b", "-",  "^", "JEPA-CtxRecon"),
        "PCA-LinPred-LN": ("#b27300", "-.", "d", "PCA-LinPred-LN"),
        "TCN-ObsPred-LN": ("#555555", "--", "p", "TCN-ObsPred-LN"),
        "TimesNet":       ("#a31d3d", "--", "D", "TimesNet"),
        "TranAD":         ("#734120", "-.", "v", "TranAD"),
        "TCN-ObsPred":    ("#995570", ":",  "h", "TCN-ObsPred"),
    }

    for m, (col, ls, mk, lbl) in styles_b.items():
        if m not in curves_b.index:
            continue
        y = curves_b.loc[m, f_grid].to_numpy()
        lw = 1.45 if m in ("TS-JEPA", "OpEntropy", "TimesNet") else 1.05
        ax.plot(f_grid, y, linestyle=ls, color=col, marker=mk, markersize=2.6, lw=lw, label=lbl)

    for m in ["TS-JEPA", "TimesNet"]:
        lo, hi = ci_bands_b[m]
        ax.fill_between(f_grid, lo, hi, color=styles_b[m][0], alpha=0.14)

    # Highlight low-FPR advantage at f*=0.01
    y_jepa_01 = curves_b.loc["TS-JEPA", 0.01]
    y_tnet_01 = curves_b.loc["TimesNet", 0.01]
    ax.annotate(
        "", xy=(0.0103, y_jepa_01), xytext=(0.0103, y_tnet_01),
        arrowprops=dict(arrowstyle="<->", color="#1d4e89", lw=1.15), zorder=5
    )
    ax.annotate(
        "$+0.178\\text{--}0.214$ ER\n($f^*{=}0.01$)",
        xy=(0.0103, 0.305), xytext=(0.0105, 0.770),
        fontsize=5.0, color="#0b4f8a", weight="bold",
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#99c2ff", alpha=0.90),
        arrowprops=dict(arrowstyle="->", color="#1d4e89", lw=0.75, shrinkA=1, shrinkB=2),
        zorder=6
    )

    ax.set_xscale("log")
    ax.set_xlim(0.009, 0.53)
    ax.set_ylim(-0.01, 0.96)
    ax.set_xticks([0.01, 0.02, 0.05, 0.10, 0.20, 0.50])
    ax.get_xaxis().set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Realized test FPR ($f^*$)")
    ax.set_ylabel("Event recall (strictly causal)")
    ax.set_title("(b) Matched Realized-FPR Curves", fontweight="bold", pad=4)

    # EXTERNAL LEGEND (Strict Principle 4: never inside the axes box)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=4, frameon=False,
        columnspacing=0.28, handlelength=0.95, handletextpad=0.15, labelspacing=0.18,
        fontsize=4.6
    )

    # =========================================================================
    # PANEL (c): Multi-Risk Calibration Reliability Diagram
    # =========================================================================
    ax = axes[2]
    q_diag = np.logspace(-4, -1, 50)
    ax.plot(q_diag, q_diag, "k:", lw=1.1, label="Ideal ($y{=}x$)")

    styles_c = {
        "TS-JEPA":  ("#1d4e89", "-",  "o", "TS-JEPA (IQR)"),
        "TimesNet": ("#a31d3d", "--", "D", "TimesNet (IQR)"),
        "TranAD":   ("#734120", "-.", "v", "TranAD (IQR)"),
    }
    eps_floor = 8e-5
    for m, (col, ls, mk, lbl) in styles_c.items():
        st = stats_c[m]
        med = np.maximum(st["median"], eps_floor)
        q25 = np.maximum(st["q25"], eps_floor)
        q75 = np.maximum(st["q75"], eps_floor)
        mean_v = np.maximum(st["mean"], eps_floor)
        ax.plot(q_levels, med, linestyle=ls, color=col, marker=mk, markersize=3.2, lw=1.35, label=lbl)
        ax.plot(q_levels, mean_v, linestyle="none", color=col, marker=mk, markersize=2.9, mfc="none", mew=0.85)
        ax.fill_between(q_levels, q25, q75, color=col, alpha=0.13)

    # Dummy handle for hollow mean markers
    ax.plot([], [], linestyle="none", color="#333333", marker="o", markersize=2.9, mfc="none", mew=0.85, label="Macro mean")

    # Exchangeable point-shuffle control curve for TS-JEPA
    ax.plot(
        q_levels, np.maximum(ex_curve_c, eps_floor),
        linestyle="--", color="#1e7b34", marker="^", markersize=3.1, lw=1.3,
        label="TS-JEPA (exch.)"
    )

    ax.annotate(
        "Drift floor ($13.6\\%\\text{--}25\\%$)",
        xy=(1e-3, 0.136), xytext=(1.05e-4, 0.060),
        arrowprops=dict(arrowstyle="->", color="#663399", lw=0.85),
        fontsize=5.1, color="#4a2373"
    )
    ax.annotate(
        "Exch. control ($1.9\\times$ @ $10^{-3}$)",
        xy=(1e-3, ex_curve_c[2]), xytext=(1.05e-4, 0.027),
        arrowprops=dict(arrowstyle="->", color="#1e7b1e", lw=0.85),
        fontsize=5.0, color="#1e7b1e"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(8e-5, 1.2e-1)
    ax.set_ylim(5e-5, 0.85)
    ax.set_xlabel(r"Target design risk ($q$)", labelpad=2)
    ax.set_ylabel("Realized held-out causal FPR", labelpad=2)
    ax.set_title("(c) Multi-Risk Calibration & Drift", fontweight="bold", pad=4)

    # EXTERNAL LEGEND (Strict Principle 4: never inside the axes box)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False,
        columnspacing=0.28, handlelength=0.95, handletextpad=0.15, labelspacing=0.18,
        fontsize=4.6
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for p_idx, ax_p in enumerate(axes):
        texts = [t for t in ax_p.texts if t.get_text().strip()]
        bboxes = [(t.get_text().replace("\n", " "), matplotlib.text.Text.get_window_extent(t, renderer)) for t in texts]
        for i, t in enumerate(texts):
            bp = t.get_bbox_patch()
            if bp is not None:
                bboxes[i] = (bboxes[i][0], bp.get_window_extent(renderer))
        if ax_p.get_legend():
            bboxes.append(("LEGEND", ax_p.get_legend().get_window_extent(renderer)))
        # Check text/legend-on-text/legend collisions
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                if bboxes[i][1].overlaps(bboxes[j][1]):
                    print(f"[COLLISION] Panel {p_idx} box-on-box: '{bboxes[i][0]}' vs '{bboxes[j][0]}'")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_stem = FIG_DIR / "mechanism_operating_calibration"
    saved = save_true_vector_figure(
        fig, out_stem, formats=("svg", "pdf", "png"),
        png_dpi=300, inkscape_postprocess_pdf=True,
        ensure_pure_vector=True, ensure_outside_legend=False
    )
    print("Saved true-vector figure outputs:")
    for fmt, p in saved.items():
        print(f"  [{fmt.upper()}] {p}")


if __name__ == "__main__":
    main()


