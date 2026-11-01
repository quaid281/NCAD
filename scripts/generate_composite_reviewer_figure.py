#!/usr/bin/env python3
"""Generate Figure 2: 2-panel composite empirical figure for IEEE TNNLS manuscript.

Panels:
  (a) Mechanism Isolation (Calibration Excess Kurtosis gamma_2 vs.
      Calibration-to-Holdout KS Drift D_KS, colored by Held-Out Causal FPR).
  (b) Matched Realized Test-FPR Operating Curves (Event Recall vs. Realized Test
      FPR f* in [0.01, 0.50] across 19 non-GHL streams, strictly causal trailing,
      with 95% bootstrap confidence bands for TS-JEPA and TimesNet).

Outputs:
  paper/NCAD_CS/figures/mechanism_operating_calibration.pdf
  paper/NCAD_CS/figures/mechanism_operating_calibration.svg
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
)

REPORTS = ROOT / "reports"
DUMP_DIR = REPORTS / "score_dumps"
FIG_DIR = ROOT / "paper" / "NCAD_CS" / "figures"


def compute_panel_a_data() -> pd.DataFrame:
    """Compute stream-macro Kurtosis, KS Drift, and Held-Out Causal FPR across arms."""
    s19 = set(
        tuple(l.strip().split(":"))
        for l in (REPORTS / "streams_19_nonghl.txt").read_text().splitlines()
        if l.strip()
    )
    dfs = []
    for fn in [
        "heldout_19stream_r3.csv",
        "heldout_bounded_obs_tcn.csv",
        "heldout_19stream_r4_arms.csv",
        "heldout_nonparametric.csv",
        "expanded_ablations_19streams.csv",
    ]:
        p = REPORTS / fn
        if p.exists():
            dfs.append(pd.read_csv(p))
    h = pd.concat(dfs, ignore_index=True).drop_duplicates(
        subset=["dataset", "channel", "seed", "model", "mapping", "calibrator"], keep="first"
    )
    h = h[h.apply(lambda r: (r["dataset"], str(r["channel"])) in s19, axis=1)]
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
        if (ds, ch) not in s19:
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
    s19 = set(
        tuple(l.strip().split(":"))
        for l in (REPORTS / "streams_19_nonghl.txt").read_text().splitlines()
        if l.strip()
    )
    f_grid = np.array([0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
    models = [
        "TS-JEPA",
        "OpEntropy",
        "Reynolds",
        "Reynolds-alpha0",
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
        if model not in models or (ds, ch) not in s19:
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


def main():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 7.8,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.2,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 6.4,
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
    fig.subplots_adjust(top=0.88, bottom=0.25, left=0.08, right=0.98, wspace=0.26)

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
        ("Reynolds-alpha0", "Reynolds (a=0)", "p"),
        ("Reynolds", "Reynolds", "*"),
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

    # Fit monotone bilinear iso-FPR response surface across the 14 arms
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
                fontsize=5.8, color="#333333", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.78),
                zorder=2.5
            )

    for key, label, marker in arms_a:
        if key not in df_a.index:
            continue
        row = df_a.loc[key]
        x, y, fpr = row["cal_kurtosis"], row["ks_drift"], row["hold_fpr"]
        sz = 48 if marker == "*" else 38
        ax.scatter(
            x, y, c=[fpr], cmap=cmap, norm=norm, marker=marker, s=sz,
            edgecolors="#111111", linewidths=0.60, zorder=4
        )

    # Shaded regime box for latent / bounded predictors
    ax.axhspan(0.108, 0.278, xmin=0.0, xmax=0.55, color="#1d4e89", alpha=0.07, zorder=1.5)
    ax.text(
        0.45, 0.115,
        "Latent / Bounded Predictors\n($D_{\\mathrm{KS}}\\approx 0.17$–$0.27$, FPR $0.9$–$7.3\\%$)",
        fontsize=6.2, color="#0b4f8a", weight="bold", zorder=5
    )

    # Mechanism arrows
    x1, y1 = df_a.loc["TCN-ObsPred", ["cal_kurtosis", "ks_drift"]]
    x2, y2 = df_a.loc["TCN-ObsPred-LN", ["cal_kurtosis", "ks_drift"]]
    ax.annotate(
        "", xy=(x2 + 0.16, y2 + 0.003), xytext=(x1 - 0.16, y1 - 0.003),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.1, ls="--"), zorder=3
    )
    ax.text(
        2.95, 0.445,
        "(1) LayerNorm score compression\n($\\gamma_2: 6.10\\!\\to\\!0.91$)",
        fontsize=6.0, color="#333333", ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#cccccc", alpha=0.92),
        zorder=5
    )

    x3, y3 = df_a.loc["Reynolds-alpha0", ["cal_kurtosis", "ks_drift"]]
    ax.annotate(
        "", xy=(x3, y3 + 0.015), xytext=(x2, y2 - 0.015),
        arrowprops=dict(arrowstyle="->", color="#1d4e89", lw=1.3), zorder=3
    )
    ax.text(
        2.35, 0.305,
        "(2) Latent projection\n($D_{\\mathrm{KS}}: 0.376\\!\\to\\!0.193$)",
        fontsize=6.0, color="#0b4f8a", ha="center",
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#99c2ff", alpha=0.92),
        zorder=5
    )

    # Clean cluster label for reconstruction baselines
    ax.text(
        6.00, 0.500,
        "Reconstruction baselines\n(TimesNet, TranAD, TCN)",
        fontsize=6.0, ha="center", va="bottom", color="#444444",
        bbox=dict(boxstyle="round,pad=0.12", fc="#fafafa", ec="#dddddd", alpha=0.88),
        zorder=5
    )

    sm_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm_map.set_array([])
    cbar = fig.colorbar(sm_map, ax=ax, pad=0.018, aspect=24)
    cbar.solids.set_rasterized(False)
    cbar.set_label("Held-out causal FPR (SPOT)", fontsize=7.2, labelpad=3)
    cbar.ax.tick_params(labelsize=6.6)

    ax.set_xlim(0.3, 6.95)
    ax.set_ylim(0.104, 0.550)
    ax.set_xlabel(r"Calibration excess kurtosis ($\gamma_2$)", labelpad=2)
    ax.set_ylabel(r"Cal$\to$Hold KS drift ($D_{\mathrm{KS}}$)", labelpad=2)
    ax.set_title("(a) Mechanism Isolation & Iso-FPR Contours", fontweight="bold", pad=4)
    ax.text(
        0.5, -0.23, f"Markers: 14 arms  |  Contours: bilinear iso-FPR surface ($R^2{{=}}{r2_surf:.2f}$)",
        transform=ax.transAxes, ha="center", va="top", fontsize=6.2, color="#555555"
    )

    # =========================================================================
    # PANEL (b): Matched Realized Test-FPR Operating Curves
    # =========================================================================
    ax = axes[1]
    style_specs = {
        "TS-JEPA":         ("#1d4e89", "-",  "o", "TS-JEPA"),
        "OpEntropy":       ("#5c2d91", "-",  "s", "OpEntropy"),
        "Reynolds-alpha0": ("#2b7a78", "-",  "p", "Reynolds ($\\alpha{=}0$)"),
        "Reynolds":        ("#378d94", "--", "*", "Reynolds"),
        "JEPA-CtxRecon":   ("#00706b", ":",  "^", "JEPA-CtxRecon"),
        "PCA-LinPred-LN":  ("#b27300", "-.", "d", "PCA-LinPred-LN"),
        "TCN-ObsPred-LN":  ("#555555", "--", "p", "TCN-ObsPred-LN"),
        "TimesNet":        ("#a31d3d", "--", "D", "TimesNet"),
        "TranAD":          ("#734120", "-.", "v", "TranAD"),
        "TCN-ObsPred":     ("#995570", ":",  "h", "TCN-ObsPred"),
    }

    for m, (col, ls, mk, lbl) in style_specs.items():
        if m not in curves_b.index:
            continue
        y = curves_b.loc[m, f_grid].to_numpy()
        lw = 1.45 if m in ("TS-JEPA", "OpEntropy", "Reynolds-alpha0", "TimesNet") else 1.05
        ax.plot(f_grid, y, linestyle=ls, color=col, marker=mk, markersize=2.8, lw=lw, label=lbl)

    for m in ["TS-JEPA", "TimesNet"]:
        lo, hi = ci_bands_b[m]
        ax.fill_between(f_grid, lo, hi, color=style_specs[m][0], alpha=0.14)

    # Highlight low-FPR advantage at f*=0.01
    y_rey0_01 = curves_b.loc["Reynolds-alpha0", 0.01]
    y_tnet_01 = curves_b.loc["TimesNet", 0.01]
    ax.annotate(
        "", xy=(0.0103, y_rey0_01), xytext=(0.0103, y_tnet_01),
        arrowprops=dict(arrowstyle="<->", color="#1d4e89", lw=1.15), zorder=5
    )
    ax.annotate(
        "$+0.178$–$0.240$ ER\n($f^*{=}0.01$)",
        xy=(0.0103, 0.340), xytext=(0.0108, 0.775),
        fontsize=6.0, color="#0b4f8a", weight="bold",
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
        loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=5, frameon=False,
        columnspacing=0.35, handlelength=1.0, handletextpad=0.18, labelspacing=0.18,
        fontsize=5.8
    )

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
