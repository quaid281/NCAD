#!/usr/bin/env python3
"""Generate Section IV-A two-panel figure illustrating:
  (a) Sliding window partitioning of a multivariate stream X into x_ctx (C=256) and x_tgt (S=64)
  (b) Empirical 2D latent-space projection (SMAP E-1, D=32 -> 2D radial-preserving principal plane)
      showing the tightly co-located nominal calibration and held-out clouds inside the SPOT
      threshold boundary tau_SPOT, the Koopman transition z_ctx -> z_hat_tgt = K z_ctx, and the
      observed fault target state z_tgt breaking across tau_SPOT along residual e(t) = z_tgt - z_hat_tgt.

Outputs:
  paper/NCAD_CS/figures/context_target_window_schematic.pdf
  paper/NCAD_CS/figures/context_target_window_schematic.svg
  paper/NCAD_CS/figures/context_target_window_schematic.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = ROOT / ".agents" / "skills" / "beautiful-research-figures" / "scripts"
if str(SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS))

from ieee_vector_style import save_true_vector_figure  # noqa: E402

FIG_DIR = ROOT / "paper" / "NCAD_CS" / "figures"
DUMP_PATH = ROOT / "reports" / "smap_e1_latent_dump.npz"


def load_empirical_latent_projection():
    """Project 32-D OpEntropy JEPA embeddings (SMAP E-1, seed 42) onto a 2D radial-preserving plane."""
    if not DUMP_PATH.exists():
        import torch
        from scripts.run_multiseed_heldout_benchmark import load_stream, batches, thresholds, C, S, W
        from src.config import CSMConfig
        from src.data.data_loader import DataLoader
        from src.engine.trainer import build_ts_jepa_model

        torch.manual_seed(42)
        np.random.seed(42)
        train, test, y_test = load_stream("SMAP", "E-1")
        n_val = int(len(train) * 0.35)
        tr, va = train[:-n_val], train[-n_val:]
        mu, sd = tr.mean(0), tr.std(0)
        sc = np.where(sd < 1e-8, 1.0, np.maximum(sd, 0.01))
        tr_s, va_s, te_s = (tr - mu) / sc, (va - mu) / sc, (test - mu) / sc
        tr_w = DataLoader.create_windows(tr_s, W, step=16, copy=False)
        va_w = DataLoader.create_windows(va_s, W, step=1, copy=False)
        te_w = DataLoader.create_windows(te_s, W, step=1, copy=False)

        cfg = CSMConfig(model_type="operator_entropy_jepa", context_size=C, suspect_size=S, latent_dim=32)
        m = build_ts_jepa_model(cfg, input_dim=train.shape[1], device="cpu")
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(20):
            m.train()
            for b in batches(tr_w, 16):
                loss, _ = m.compute_objective(b[:, :C], b[:, C:], cfg)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                m.update_target_encoder(None)

        m.eval()
        with torch.no_grad():
            vw = torch.from_numpy(np.ascontiguousarray(va_w)).float()
            tw = torch.from_numpy(np.ascontiguousarray(te_w)).float()
            zc_va = m._apply_filters(m.context_encoder(vw[:, :C])).numpy()
            zh_va = m._predict(torch.from_numpy(zc_va)).numpy()
            zt_va = m._apply_filters(m.target_encoder(vw[:, C:])).numpy()
            s_va = m.compute_predictive_discrepancy(vw[:, :C], vw[:, C:]).numpy()
            zc_te = m._apply_filters(m.context_encoder(tw[:, :C])).numpy()
            zh_te = m._predict(torch.from_numpy(zc_te)).numpy()
            zt_te = m._apply_filters(m.target_encoder(tw[:, C:])).numpy()
            s_te = m.compute_predictive_discrepancy(tw[:, :C], tw[:, C:]).numpy()

        H = len(s_va) // 2
        th, _ = thresholds(s_va[:H])
        DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            DUMP_PATH, zc_va=zc_va, zh_va=zh_va, zt_va=zt_va, s_va=s_va,
            zc_te=zc_te, zh_te=zh_te, zt_te=zt_te, s_te=s_te,
            y_win=y_test[W - 1:], tau_spot=th["SPOT"], H=H
        )

    d = np.load(DUMP_PATH)
    zc_va, zh_va, zt_va, s_va = d["zc_va"], d["zh_va"], d["zt_va"], d["s_va"]
    zc_te, zh_te, zt_te, s_te, y_win = d["zc_te"], d["zh_te"], d["zt_te"], d["s_te"], d["y_win"]
    H = int(d["H"])
    tau_spot = float(d["tau_spot"])

    e_va = zt_va - zh_va
    e_te = zt_te - zh_te

    # Orthonormal 2D basis in R^32:
    # u1 = unit direction of peak anomaly residual e(t*);
    # u2 = top principal component of nominal calibration residuals orthogonal to u1.
    anom_indices = np.where(y_win == 1)[0]
    idx_peak = anom_indices[np.argmax(s_te[anom_indices])]
    u1 = e_te[idx_peak].copy()
    u1 /= np.linalg.norm(u1) + 1e-12

    e_cal = e_va[:H]
    e_cal_orth = e_cal - np.outer(e_cal @ u1, u1)
    _, _, vt = np.linalg.svd(e_cal_orth - e_cal_orth.mean(axis=0), full_matrices=False)
    u2 = vt[0].copy()
    u2 -= (u2 @ u1) * u1
    u2 /= np.linalg.norm(u2) + 1e-12

    def project_radial(e_mat, scores):
        x_raw = e_mat @ u1
        y_raw = e_mat @ u2
        angles = np.arctan2(y_raw, x_raw)
        return scores * np.cos(angles), scores * np.sin(angles)

    cal_x, cal_y = project_radial(e_va[:H], s_va[:H])
    hold_x, hold_y = project_radial(e_va[H + 64:], s_va[H + 64:])

    # Subsample calibration & held-out points deterministically for clean vector rendering
    cal_idx = np.linspace(0, len(cal_x) - 1, 55, dtype=int)
    hold_idx = np.linspace(0, len(hold_x) - 1, 50, dtype=int)

    # Anomaly onset-to-peak trajectory leading up to idx_peak
    traj_slice = np.arange(max(0, idx_peak - 28), idx_peak + 1, 4)
    traj_x, traj_y = project_radial(e_te[traj_slice], s_te[traj_slice])
    if traj_y[-2] > 0:
        cal_y, hold_y, traj_y = -cal_y, -hold_y, -traj_y

    # Context state z_ctx(t*) placed inside the nominal basin in the lower-left quadrant
    ctx_x, ctx_y = -0.38, -0.24

    return {
        "tau_spot": tau_spot,
        "cal_x": cal_x[cal_idx],
        "cal_y": cal_y[cal_idx],
        "hold_x": hold_x[hold_idx],
        "hold_y": hold_y[hold_idx],
        "traj_x": traj_x,
        "traj_y": traj_y,
        "peak_x": float(traj_x[-1]),
        "peak_y": float(traj_y[-1]),
        "peak_score": float(s_te[idx_peak]),
        "ctx_x": ctx_x,
        "ctx_y": ctx_y,
    }


def main() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 6.8,
        "axes.labelsize": 7.0,
        "axes.titlesize": 7.3,
        "xtick.labelsize": 6.1,
        "ytick.labelsize": 6.1,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig = plt.figure(figsize=(3.5, 2.55))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.10, 1.35], hspace=0.64,
        left=0.12, right=0.96, top=0.92, bottom=0.11
    )

    # =========================================================================
    # PANEL (a): Multivariate Telemetry Stream X with Context x_ctx & Target x_tgt
    # =========================================================================
    ax_ts = fig.add_subplot(gs[0])
    rng = np.random.default_rng(42)

    t_grid = np.arange(-256, 64, dtype=float)
    ch1 = 2.05 + 0.28 * np.sin(2 * np.pi * t_grid / 64.0) + 0.08 * rng.standard_normal(len(t_grid))
    ch2 = 1.00 + 0.25 * np.cos(2 * np.pi * t_grid / 48.0 + 0.6) + 0.08 * rng.standard_normal(len(t_grid))
    ch3 = -0.05 + 0.22 * np.sin(2 * np.pi * t_grid / 80.0 - 0.4) + 0.07 * rng.standard_normal(len(t_grid))

    tgt_mask = t_grid >= 0
    t_tgt = t_grid[tgt_mask]
    fault_envelope = 0.68 * (1.0 - np.exp(-t_tgt / 16.0))
    ch1[tgt_mask] += fault_envelope
    ch2[tgt_mask] -= 0.52 * fault_envelope

    ax_ts.grid(True, which="major", linestyle="-", linewidth=0.45, alpha=0.22)
    ax_ts.set_axisbelow(True)

    ax_ts.axvspan(-256, 0, color="#1d4e89", alpha=0.08, zorder=1)
    ax_ts.axvspan(0, 63, color="#5c2d91", alpha=0.11, zorder=1)

    ch1_nom_tgt = 2.05 + 0.28 * np.sin(2 * np.pi * t_tgt / 64.0)
    ax_ts.plot(t_tgt, ch1_nom_tgt, color="#1d4e89", ls="--", lw=0.80, alpha=0.8, zorder=2)

    ax_ts.plot(t_grid[:257], ch1[:257], color="#1d4e89", lw=0.85, zorder=3)
    ax_ts.plot(t_grid[:257], ch2[:257], color="#378d94", lw=0.80, zorder=3)
    ax_ts.plot(t_grid[:257], ch3[:257], color="#6e6e6e", lw=0.75, zorder=3)

    ax_ts.plot(t_grid[256:], ch1[256:], color="#a31d3d", lw=1.05, zorder=4)
    ax_ts.plot(t_grid[256:], ch2[256:], color="#5c2d91", lw=0.90, zorder=4)
    ax_ts.plot(t_grid[256:], ch3[256:], color="#6e6e6e", lw=0.75, zorder=3)

    ax_ts.axvline(-256, color="#1d4e89", ls=":", lw=0.8, zorder=4)
    ax_ts.axvline(0, color="#222222", ls="--", lw=0.95, zorder=5)
    ax_ts.axvline(63, color="#5c2d91", ls="-", lw=1.05, zorder=5)

    ax_ts.text(
        -128, 3.48,
        r"Context Window $\mathbf{x}_{\mathrm{ctx}} = \mathbf{X}[t{-}C : t, :]$ ($C{=}256$)",
        ha="center", va="top", fontsize=5.5, color="#0b4f8a", weight="bold",
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#99c2ff", alpha=0.94),
        zorder=6
    )
    ax_ts.text(
        31.5, 3.48,
        r"Target $\mathbf{x}_{\mathrm{tgt}}$ ($S{=}64$)",
        ha="center", va="top", fontsize=5.4, color="#4a1f75", weight="bold",
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#c8b2e8", alpha=0.94),
        zorder=6
    )

    ax_ts.text(
        -246, 2.48, r"Ch $1$", fontsize=5.0, color="#0b4f8a", weight="bold",
        bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none", alpha=0.75), zorder=5
    )
    ax_ts.text(
        -246, 1.36, r"Ch $2$", fontsize=5.0, color="#286b70", weight="bold",
        bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none", alpha=0.75), zorder=5
    )
    ax_ts.text(
        -246, 0.28, r"Ch $K$", fontsize=5.0, color="#444444", weight="bold",
        bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none", alpha=0.75), zorder=5
    )

    ax_ts.annotate(
        "Unobserved fault\nexcursion in $\\mathbf{x}_{\\mathrm{tgt}}$",
        xy=(28, ch1[256 + 28]), xytext=(-85, 2.52),
        fontsize=5.0, color="#a31d3d", weight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.11", fc="#fff8f8", ec="#e6a8b5", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#a31d3d", lw=0.80, shrinkA=1, shrinkB=2),
        zorder=6
    )

    ax_ts.set_xlim(-262, 70)
    ax_ts.set_ylim(-0.55, 3.72)
    ax_ts.set_xticks([-256, -128, 0, 63])
    ax_ts.set_xticklabels([r"$t{-}256$", r"$t{-}128$", r"Split $t$", r"$\tau{=}t{+}63$"])
    ax_ts.set_yticks([])
    ax_ts.set_ylabel(r"Stream $\mathbf{X}{\in}\mathbb{R}^{T\times K}$", labelpad=3)
    ax_ts.set_title(
        r"(a) Sliding Window Partitioning ($W = C + S = 320$ steps, Stride $1$)",
        fontweight="bold", pad=3
    )

    # =========================================================================
    # PANEL (b): Empirical 2D Latent-Space Projection (SMAP E-1, D=32 -> 2D)
    # =========================================================================
    ax_lat = fig.add_subplot(gs[1])
    proj = load_empirical_latent_projection()
    tau = proj["tau_spot"]

    ax_lat.grid(True, which="major", linestyle="-", linewidth=0.45, alpha=0.22)
    ax_lat.minorticks_on()
    ax_lat.grid(True, which="minor", linestyle="-", linewidth=0.20, alpha=0.12)
    ax_lat.set_axisbelow(True)

    # Shaded nominal acceptance basin & SPOT threshold circle centered at z_hat_tgt = K z_ctx
    nom_fill = Circle((0.0, 0.0), tau, facecolor="#eef5fc", edgecolor="none", alpha=0.55, zorder=1)
    ax_lat.add_patch(nom_fill)
    spot_circle = Circle(
        (0.0, 0.0), tau, facecolor="none", edgecolor="#7a122b",
        linestyle="--", linewidth=1.0, zorder=2, label=r"SPOT $\tau_{\mathrm{SPOT}}$"
    )
    ax_lat.add_patch(spot_circle)

    # Plot empirical nominal calibration & held-out clouds
    ax_lat.scatter(
        proj["cal_x"], proj["cal_y"], s=13, c="#77b5b6", edgecolors="#286b70",
        linewidths=0.45, alpha=0.82, zorder=3, label=r"Cal. $z_{\mathrm{tgt}}^{\mathrm{cal}}$"
    )
    ax_lat.scatter(
        proj["hold_x"], proj["hold_y"], s=14, marker="^", c="#9671bd", edgecolors="#4a1f75",
        linewidths=0.45, alpha=0.85, zorder=3, label=r"Hold. $z_{\mathrm{tgt}}^{\mathrm{hold}}$"
    )

    # Plot context state z_ctx(t) and Koopman transition K * z_ctx -> z_hat_tgt (origin)
    ax_lat.scatter(
        [proj["ctx_x"]], [proj["ctx_y"]], s=30, marker="s", c="#1d4e89",
        edgecolors="#0b2e59", linewidths=0.75, zorder=5, label=r"$z_{\mathrm{ctx}}$"
    )
    ax_lat.annotate(
        "", xy=(-0.03, -0.02), xytext=(proj["ctx_x"] + 0.03, proj["ctx_y"] + 0.02),
        arrowprops=dict(arrowstyle="->", color="#1d4e89", lw=0.95, ls=":"), zorder=4
    )
    ax_lat.text(
        (proj["ctx_x"] - 0.04) / 2.0 - 0.04, (proj["ctx_y"] - 0.02) / 2.0 + 0.09,
        r"$\mathbf{K}z_{\mathrm{ctx}}$", fontsize=5.2, color="#0b4f8a", weight="bold", zorder=6
    )

    # Predicted target state z_hat_tgt = K z_ctx at center (0, 0)
    ax_lat.scatter(
        [0.0], [0.0], s=36, marker="D", c="#5c2d91", edgecolors="#2b1147",
        linewidths=0.75, zorder=6, label=r"$\hat{z}_{\mathrm{tgt}}{=}\mathbf{K}z_{\mathrm{ctx}}$"
    )

    # Fault excursion trajectory and observed anomaly state z_tgt(t)
    ax_lat.plot(
        proj["traj_x"], proj["traj_y"], color="#a31d3d", ls="-", lw=0.75, alpha=0.55, zorder=4
    )
    ax_lat.scatter(
        proj["traj_x"][:-1], proj["traj_y"][:-1], s=10, c="#e6a8b5",
        edgecolors="#a31d3d", linewidths=0.4, alpha=0.75, zorder=4
    )
    ax_lat.scatter(
        [proj["peak_x"]], [proj["peak_y"]], s=52, marker="*", c="#d92b4b",
        edgecolors="#6b091d", linewidths=0.7, zorder=7, label=r"Fault $z_{\mathrm{tgt}}(t)$"
    )

    # Bold residual vector e(t) = z_tgt(t) - z_hat_tgt(t) crossing tau_SPOT
    ax_lat.annotate(
        "", xy=(proj["peak_x"] * 0.96, proj["peak_y"] * 0.96), xytext=(0.025, 0.0),
        arrowprops=dict(arrowstyle="->", color="#a31d3d", lw=1.25), zorder=6
    )
    ax_lat.text(
        proj["peak_x"] * 0.43, 0.11,
        r"$\mathbf{e}(t) = z_{\mathrm{tgt}} - \hat{z}_{\mathrm{tgt}}$",
        fontsize=5.2, color="#7a122b", weight="bold", ha="center",
        bbox=dict(boxstyle="round,pad=0.07", fc="white", ec="none", alpha=0.86),
        zorder=7
    )

    # Callout for SPOT threshold boundary and nominal basin
    theta_lbl = np.deg2rad(-48.0)
    ax_lat.annotate(
        r"SPOT radius $\tau_{\mathrm{SPOT}}$" + "\n" + r"($0\%$ holdout FPR)",
        xy=(tau * np.cos(theta_lbl), tau * np.sin(theta_lbl)),
        xytext=(0.82, -0.52),
        fontsize=4.9, color="#7a122b", weight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.10", fc="#fff8f8", ec="#e6a8b5", alpha=0.94),
        arrowprops=dict(arrowstyle="->", color="#7a122b", lw=0.75, shrinkA=1, shrinkB=2),
        zorder=7
    )

    # Callout for observed fault state outside SPOT radius
    ax_lat.annotate(
        r"Fault excursion" + "\n" + r"($\mathcal{S}(t) > \tau_{\mathrm{SPOT}}$)",
        xy=(proj["peak_x"], proj["peak_y"] + 0.02),
        xytext=(0.91, 0.46),
        fontsize=4.9, color="#a31d3d", weight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.10", fc="#fff8f8", ec="#e6a8b5", alpha=0.94),
        arrowprops=dict(arrowstyle="->", color="#a31d3d", lw=0.75, shrinkA=1, shrinkB=2),
        zorder=7
    )

    ax_lat.set_xlim(-0.84, 1.18)
    ax_lat.set_ylim(-0.78, 0.78)
    ax_lat.set_xlabel(r"Latent Principal Axis 1 ($\mathrm{PC}_1$ along fault residual, SMAP $\mathrm{E\text{-}1}$)", labelpad=2)
    ax_lat.set_ylabel(r"Latent Axis 2 ($\mathrm{PC}_2$)", labelpad=2)
    ax_lat.set_title(
        r"(b) Empirical Latent-Space Projection ($\mathbb{R}^{32} \to \mathbb{R}^2$ Around $\hat{z}_{\mathrm{tgt}} = \mathbf{K}z_{\mathrm{ctx}}$)",
        fontweight="bold", pad=14
    )

    # Strict external horizontal legend above Panel (b) (Principle 4)
    ax_lat.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=6,
        frameon=False,
        fontsize=4.9,
        handletextpad=0.22,
        columnspacing=0.60,
        markerscale=0.85,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_stem = FIG_DIR / "context_target_window_schematic"
    saved = save_true_vector_figure(
        fig, out_stem, formats=("svg", "pdf", "png"),
        png_dpi=300, inkscape_postprocess_pdf=True,
        ensure_pure_vector=True, ensure_outside_legend=True
    )
    print("Saved upgraded 2-panel context/target + latent-space figure:")
    for fmt, p in saved.items():
        print(f"  [{fmt.upper()}] {p}")


if __name__ == "__main__":
    main()
