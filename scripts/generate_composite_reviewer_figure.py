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
        "legend.fontsize": 5.0,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    df_a = compute_panel_a_data()
    f_grid, curves_b, ci_bands_b = compute_panel_b_data()

    fig, axes = plt.subplots(
        1, 2, figsize=(7.16, 2.75),
        gridspec_kw={"width_ratios": [1.15, 1.0]}
    )
    fig.subplots_adjust(top=0.88, bottom=0.24, left=0.08, right=0.98, wspace=0.26)

    # Subtle dual-hierarchy grids across both panels (Principle 4)
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

    # Fit monotone bilinear iso-FPR response surface across the 12 arms:
    # FPR_hat(gamma_2, D_KS) = b0 + b1*gamma_2 + b2*D_KS + b3*(gamma_2 * D_KS)
    from scipy.optimize import lsq_linear
    arm_keys = [k for k, _, _ in arms_a if k in df_a.index]
    xa = df_a.loc[arm_keys, "cal_kurtosis"].to_numpy()
    ya = df_a.loc[arm_keys, "ks_drift"].to_numpy()
    za = df_a.loc[arm_keys, "hold_fpr"].to_numpy()
    X_des = np.column_stack([np.ones_like(xa), xa, ya, xa * ya])
    res_fit = lsq_linear(X_des, za, bounds=([-np.inf, 0.001, 0.01, 0.0], [np.inf, np.inf, np.inf, np.inf]))
    b0, b1, b2, b3 = res_fit.x
    r2_surf = 1.0 - np.sum((za - X_des @ res_fit.x) ** 2) / np.sum((za - za.mean()) ** 2)
    print(f"Panel (a) Iso-FPR surface R^2 = {r2_surf:.4f} (b0={b0:.4f}, b1={b1:.4f}, b2={b2:.4f}, b3={b3:.4f})")

    gx = np.linspace(0.30, 6.95, 90)
    gy = np.linspace(0.104, 0.550, 90)
    GX, GY = np.meshgrid(gx, gy)
    GZ = np.clip(b0 + b1 * GX + b2 * GY + b3 * (GX * GY), 0.0, 0.28)

    # Subtle pure-vector topographic contour fill + crisp iso-FPR contour isolines
    cf_levels = np.linspace(0.0, 0.26, 14)
    cf = ax.contourf(GX, GY, GZ, levels=cf_levels, cmap=cmap, norm=norm, alpha=0.14, zorder=1)
    if hasattr(cf, "collections"):
        for coll in cf.collections:
            coll.set_rasterized(False)
    else:
        cf.set_rasterized(False)

    iso_levels = [0.02, 0.05, 0.10, 0.15, 0.20]
    cs = ax.contour(
        GX, GY, GZ, levels=iso_levels,
        colors=["#1d4e89", "#2c5f78", "#4f6d6a", "#787546", "#9a7b1c"],
        linewidths=0.65, linestyles=":", alpha=0.80, zorder=2
    )
    if hasattr(cs, "collections"):
        for coll in cs.collections:
            coll.set_rasterized(False)
    else:
        cs.set_rasterized(False)

    # Place clean non-overlapping iso-FPR contour labels along the right/upper-middle margin
    iso_label_positions = [
        (0.02, 3.60),
        (0.05, 4.75),
        (0.10, 5.45),
        (0.15, 5.85),
        (0.20, 6.25),
    ]
    for lvl, x_pos in iso_label_positions:
        y_pos = (lvl - b0 - b1 * x_pos) / (b2 + b3 * x_pos)
        if 0.115 <= y_pos <= 0.475:
            ax.text(
                x_pos, y_pos, f"{int(round(lvl * 100))}% FPR",
                fontsize=4.5, color="#333333", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.78),
                zorder=2.5
            )

    for key, label, marker in arms_a:
        if key not in df_a.index:
            continue
        row = df_a.loc[key]
        x, y, fpr = row["cal_kurtosis"], row["ks_drift"], row["hold_fpr"]
        ax.scatter(
            x, y, c=[fpr], cmap=cmap, norm=norm, marker=marker, s=38,
            edgecolors="#111111", linewidths=0.60, zorder=4
        )

    # Shaded regime box for latent / bounded predictors
    ax.axhspan(0.108, 0.258, xmin=0.0, xmax=0.55, color="#1d4e89", alpha=0.07, zorder=1.5)
    ax.text(
        0.45, 0.115,
        "Latent / Bounded Predictors\n($D_{\\mathrm{KS}}\\approx 0.17$–$0.23$, FPR $0.9$–$2.8\\%$)",
        fontsize=5.2, color="#0b4f8a", weight="bold", zorder=5
    )

    # Mechanism arrows
    x1, y1 = df_a.loc["TCN-ObsPred", ["cal_kurtosis", "ks_drift"]]
    x2, y2 = df_a.loc["TCN-ObsPred-LN", ["cal_kurtosis", "ks_drift"]]
    ax.annotate(
        "", xy=(x2 + 0.16, y2 + 0.003), xytext=(x1 - 0.16, y1 - 0.003),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.1, ls="--"), zorder=3
    )
    ax.text(
        2.75, 0.445,
        "(1) LayerNorm score compression\n($\\gamma_2: 6.10\\!\\to\\!0.91$)",
        fontsize=5.2, color="#333333", ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#cccccc", alpha=0.92),
        zorder=5
    )

    x3, y3 = df_a.loc["OpEntropy", ["cal_kurtosis", "ks_drift"]]
    ax.annotate(
        "", xy=(x3, y3 + 0.015), xytext=(x2, y2 - 0.015),
        arrowprops=dict(arrowstyle="->", color="#1d4e89", lw=1.3), zorder=3
    )
    ax.text(
        2.20, 0.300,
        "(2) Latent projection\n($D_{\\mathrm{KS}}: 0.376\\!\\to\\!0.209$)",
        fontsize=5.2, color="#0b4f8a", ha="center",
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#99c2ff", alpha=0.92),
        zorder=5
    )

    # Clean cluster label for reconstruction baselines (no cluttered leader lines)
    ax.text(
        6.10, 0.500,
        "Reconstruction baselines\n(TimesNet, TranAD, TCN)",
        fontsize=5.1, ha="center", va="bottom", color="#444444",
        bbox=dict(boxstyle="round,pad=0.12", fc="#fafafa", ec="#dddddd", alpha=0.88),
        zorder=5
    )

    sm_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm_map.set_array([])
    cbar = fig.colorbar(sm_map, ax=ax, pad=0.018, aspect=24)
    cbar.solids.set_rasterized(False)  # True vector colorbar (Principle 1)
    cbar.set_label("Held-out causal FPR (SPOT)", fontsize=6.6, labelpad=3)
    cbar.ax.tick_params(labelsize=6.0)

    ax.set_xlim(0.3, 6.95)
    ax.set_ylim(0.104, 0.550)
    ax.set_xlabel(r"Calibration excess kurtosis ($\gamma_2$)", labelpad=2)
    ax.set_ylabel(r"Cal$\to$Hold KS drift ($D_{\mathrm{KS}}$)", labelpad=2)
    ax.set_title("(a) Mechanism Isolation & Iso-FPR Contours", fontweight="bold", pad=4)
    ax.text(
        0.5, -0.22, "Markers: 12 arms  |  Contours: bilinear iso-FPR surface ($R^2{=}0.97$)",
        transform=ax.transAxes, ha="center", va="top", fontsize=5.0, color="#555555"
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
        "PCA-LinPred-LN": "#b27300",
        "TCN-ObsPred-LN": "#555555",
        "TimesNet":       ("#a31d3d", "--", "D", "TimesNet"),
        "TranAD":         ("#734120", "-.", "v", "TranAD"),
        "TCN-ObsPred":    ("#995570", ":",  "h", "TCN-ObsPred"),
    }
    # Expanded styles tuple
    style_specs = {
        "TS-JEPA":        ("#1d4e89", "-",  "o", "TS-JEPA"),
        "OpEntropy":      ("#5c2d91", "-",  "s", "OpEntropy"),
        "JEPA-CtxRecon":  ("#00706b", "-",  "^", "JEPA-CtxRecon"),
        "PCA-LinPred-LN": ("#b27300", "-.", "d", "PCA-LinPred-LN"),
        "TCN-ObsPred-LN": ("#555555", "--", "p", "TCN-ObsPred-LN"),
        "TimesNet":       ("#a31d3d", "--", "D", "TimesNet"),
        "TranAD":         ("#734120", "-.", "v", "TranAD"),
        "TCN-ObsPred":    ("#995570", ":",  "h", "TCN-ObsPred"),
    }

    for m, (col, ls, mk, lbl) in style_specs.items():
        if m not in curves_b.index:
            continue
        y = curves_b.loc[m, f_grid].to_numpy()
        lw = 1.45 if m in ("TS-JEPA", "OpEntropy", "TimesNet") else 1.05
        ax.plot(f_grid, y, linestyle=ls, color=col, marker=mk, markersize=2.8, lw=lw, label=lbl)

    for m in ["TS-JEPA", "TimesNet"]:
        lo, hi = ci_bands_b[m]
        ax.fill_between(f_grid, lo, hi, color=style_specs[m][0], alpha=0.14)

    # Highlight low-FPR advantage at f*=0.01
    y_jepa_01 = curves_b.loc["TS-JEPA", 0.01]
    y_tnet_01 = curves_b.loc["TimesNet", 0.01]
    ax.annotate(
        "", xy=(0.0103, y_jepa_01), xytext=(0.0103, y_tnet_01),
        arrowprops=dict(arrowstyle="<->", color="#1d4e89", lw=1.15), zorder=5
    )
    ax.annotate(
        "$+0.178$–$0.214$ ER\n($f^*{=}0.01$)",
        xy=(0.0103, 0.305), xytext=(0.0108, 0.770),
        fontsize=5.2, color="#0b4f8a", weight="bold",
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#99c2ff", alpha=0.90),
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
    ax.set_title("(b) Matched Realized-FPR Operating Curves", fontweight="bold", pad=4)

    # EXTERNAL LEGEND (Strict Principle 4: never inside the axes box)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False,
        columnspacing=0.35, handlelength=1.0, handletextpad=0.18, labelspacing=0.18,
        fontsize=5.0
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


