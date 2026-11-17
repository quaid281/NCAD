#!/usr/bin/env python3
"""Generate Figure 4: Clean 2-Column Circuit-Breaker Interpretability & Onset-Boundary Case Studies.

Focuses on two complementary continuous-telemetry regimes across a wide 2-column IEEE layout
(7.16 in, zero interior callout boxes):
  - Col (a) Abrupt Sensor Fault (`GECCO` water-chemistry spike): continuous sensor jump triggers all
    four breaker groups (`Short S1=16`, `Mid S2=32`, `Long S3=64`, and `Mean Drift |e_bar|`), driving
    `N_128` to 4.9x quorum `K_Lite` while `1D Scalar SPOT` barely grazes threshold (`1.04x`).
  - Col (b) Slow Process Ramp (`GHL-44` real heating-loop thermal ramp): continuous temperature climbs
    gradually over ~860 steps while context-target co-movement keeps `N_128 = 0` until the abrupt
    reset at the end; the complementary `Frozen-Reference Drift` monitor crosses threshold inside
    the ramp.

Three synchronized rows:
  - Row 1: Continuous Sensor Waveforms (`z`-score) with ground-truth fault span.
  - Row 2: Stacked Active Circuit Breakers by Timescale (`S1=16`, `S2=32`, `S3=64`, `Drift`), whose top
           edge equals total count `N_128(tau)` plotted against dashed `Quorum K_Lite`.
  - Row 3: Score-to-Threshold Ratio (`Score / Threshold`, alarming above `1x`) comparing
           `Lite quorum`, `1D Scalar SPOT`, and `Frozen-Ref. Drift`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SKILL_SCRIPTS = ROOT / ".agents" / "skills" / "beautiful-research-figures" / "scripts"
for p in (ROOT, SKILL_SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ieee_vector_style import save_true_vector_figure  # noqa: E402
from scripts.run_laminar_paired_benchmarks import (
    C,
    CSMConfig,
    S,
    W,
    _score_1d_windows,
    _split_nominal_dev_4block,
    _standardize_arrays,
    _train_fixed_epochs,
    load_stream,
    thresholds,
)
from scripts.tnnls_revision.common_eval import FrozenLatentDriftMonitor
from src.engine.builder import build_encoder
from src.models.jepa.alien_jepas import SpeculativeJEPAModel
from src.models.laminar_lite import LaminarJEPALiteModel, LaminarLiteEncoder
from src.scoring.circuit_breaker_panel import aggregate_vector_stream, stabilize_whitened_subblocks
from src.scoring.laminar_lite_panel import PositionDriftLitePanel

FIGDIR = ROOT / "paper" / "NCAD_CS" / "figures"
OUT_STEM = FIGDIR / "panel_interpretability_case_studies"

# Harmonious, grayscale-safe Churkin / Okabe-Ito publication palette
C_SENS1 = "#1d4e89"    # Primary continuous sensor (Deep Navy)
C_SENS2 = "#6a408d"    # Secondary continuous sensor (Churkin Purple)
C_S1 = "#8ecae6"       # Short horizon S1=16 stack (Sky Teal)
C_S2 = "#219ebc"       # Medium horizon S2=32 stack (Cerulean)
C_S3 = "#5e548e"       # Long horizon S3=64 stack (Deep Violet)
C_DRIFT = "#b51749"    # Persistent drift |e_bar| stack (Crimson)
C_LITE = "#0d7377"     # Laminar-JEPA-Lite curve (Deep Teal)
C_1D = "#3d5a80"       # 1D Scalar SPOT curve (Slate Navy)
C_LONG = "#c85a17"     # Frozen-Ref Long-Horizon Drift (Burnt Ochre)
C_THRESH = "#222222"   # Threshold line


def _set_ieee_style() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.labelsize": 8.4,
            "axes.titlesize": 8.8,
            "legend.fontsize": 7.6,
            "figure.dpi": 300,
        }
    )


def _alarm_index(series: np.ndarray, baseline_ref: float, threshold: float) -> np.ndarray:
    """Monotone affine rescaling: `0 = healthy baseline`, `1.0 = exact calibrated alarm threshold`."""
    denom = max(float(threshold) - float(baseline_ref), 1e-6)
    return np.maximum(-0.10, (np.asarray(series, dtype=np.float64) - float(baseline_ref)) / denom)


def _evaluate_case(
    ds: str,
    ch: str,
    view_valid_slice: tuple[int, int],
    sensor_channels: list[int],
    device: torch.device,
    epochs: int = 15,
    include_drift_hybrid: bool = False,
) -> dict[str, Any]:
    """Train Lite + 1D Scalar SPOT (+ optional Frozen-Ref Drift) and extract synchronized slice."""
    train, test, y = load_stream(ds, ch)
    tr_w, q_w, va_w, te_w, len_va = _split_nominal_dev_4block(train, test)
    k_ch = int(train.shape[1])
    n_cal_w = len(va_w) // 2

    # 1. Laminar-JEPA-Lite (128 dials)
    torch.manual_seed(42)
    np.random.seed(42)
    cfg_l = CSMConfig(model_type="laminar_jepa_lite", context_size=C, suspect_size=S, latent_dim=32)
    enc = LaminarLiteEncoder(k_ch, latent_dim=32, filters=48, tcn_layers=4, proj_dim=64)
    m_lite = LaminarJEPALiteModel(enc, latent_dim=32, predictor_hidden_dim=64, num_horizons=3).to(device)
    _train_fixed_epochs(m_lite, tr_w, cfg_l, epochs=epochs, device=device)
    m_lite.fit_resolvent_whitening(tr_w, context_len=C, batch_size=512)
    va_r, va_w_res = m_lite.extract_raw_and_whitened_residuals(va_w, context_len=C, batch_size=512)
    te_r, te_w_res = m_lite.extract_raw_and_whitened_residuals(te_w, context_len=C, batch_size=512)
    q_r, q_w_res = m_lite.extract_raw_and_whitened_residuals(q_w, context_len=C, batch_size=512)

    va_ew, te_ew = stabilize_whitened_subblocks(va_w_res, te_w_res, va_r, te_r, n_cal_w, q_w_res, q_r)
    panel = PositionDriftLitePanel(context_size=C, suspect_size=S)
    bits = panel.calibrate_and_trip_windows(va_ew, te_ew, n_cal_w)
    dec_l = panel.evaluate_stream_decisions(bits, len_va, len(test), "trailing")

    i0, i1 = view_valid_slice
    i0 = max(0, i0)

    dec_d = None
    if include_drift_hybrid:
        drift_mon = FrozenLatentDriftMonitor(ewma_alpha=0.02, safety=1.15, q=0.995)
        f_fit = drift_mon.extract_pre_ln_context_summaries(m_lite, tr_w, context_len=C, batch_size=512)
        f_val = drift_mon.extract_pre_ln_context_summaries(m_lite, va_w, context_len=C, batch_size=512)
        f_te = drift_mon.extract_pre_ln_context_summaries(m_lite, te_w, context_len=C, batch_size=512)
        drift_mon.fit_reference(f_fit)
        dec_d = drift_mon.calibrate_and_score(f_val, f_te, n_cal_w=n_cal_w, suspect_size=S)
    del m_lite

    # 2. Matched 1D Multi-Horizon JEPA (SPOT)
    torch.manual_seed(42)
    np.random.seed(42)
    cfg_1 = CSMConfig(model_type="ts_jepa", context_size=C, suspect_size=S, latent_dim=32)
    m_1d = SpeculativeJEPAModel(
        context_encoder=build_encoder(cfg_1, k_ch, device),
        latent_dim=32,
        predictor_hidden_dim=64,
        ema_decay=0.995,
        dropout=cfg_1.dropout,
    ).to(device)
    _train_fixed_epochs(m_1d, tr_w, cfg_1, epochs=epochs, device=device)
    m_1d.fit_alien_bank(tr_w, context_len=C, batch_size=512)
    vs1 = _score_1d_windows(m_1d, va_w, device, use_bank=True)
    ts1 = _score_1d_windows(m_1d, te_w, device, use_bank=True)
    del m_1d

    vp1, vm1 = aggregate_vector_stream(vs1[:, None], len_va, C, S, "trailing")
    tp1, tm1 = aggregate_vector_stream(ts1[:, None], len(test), C, S, "trailing")
    cal_1d = vp1[vm1, 0][: len(vp1[vm1, 0]) // 2]
    th_1d = float(thresholds(cal_1d)[0]["SPOT"])

    n_fit = len(train) - int(len(train) * 0.35) - max(256, int(len(train) * 0.15))
    _, test_std = _standardize_arrays(train[:n_fit], test)
    raw_valid = test_std[dec_l.valid_mask]
    y_valid = y[dec_l.valid_mask]
    i1 = min(len(y_valid), i1)

    b_te = np.asarray(bits["bits_pd_te"], dtype=np.float32)
    c_s1 = b_te[i0:i1, 0:32].sum(axis=1)
    c_s2 = b_te[i0:i1, 32:64].sum(axis=1)
    c_s3 = b_te[i0:i1, 64:96].sum(axis=1)
    c_dr = b_te[i0:i1, 96:128].sum(axis=1)

    idx_lite = dec_l.test_switch_counts[i0:i1] / max(float(dec_l.k_gate_balanced), 1e-6)
    idx_1d = tp1[tm1, 0][i0:i1] / max(float(th_1d), 1e-6)
    idx_long = None
    if dec_d is not None:
        pre_s = dec_d.test_scores[i0 : i0 + 180]
        ref_d = min(float(np.median(pre_s)), float(dec_d.threshold) * 0.85)
        idx_long = 0.15 + 0.85 * _alarm_index(dec_d.test_scores[i0:i1], ref_d, dec_d.threshold)

    return {
        "t": np.arange(i1 - i0),
        "sensors": [raw_valid[i0:i1, c] for c in sensor_channels],
        "y": y_valid[i0:i1],
        "c_s1": c_s1,
        "c_s2": c_s2,
        "c_s3": c_s3,
        "c_dr": c_dr,
        "n_tot": dec_l.test_switch_counts[i0:i1],
        "k_quorum": float(dec_l.k_gate_balanced),
        "idx_lite": idx_lite,
        "idx_1d": idx_1d,
        "idx_long": idx_long,
    }


def main() -> None:
    _set_ieee_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # (a) Abrupt Sensor Spike: GECCO water quality (continuous chemical sensors, seg=[3876, 4129))
    tr_g, te_g, _ = load_stream("GECCO", "water_quality")
    _, te_g_std = _standardize_arrays(tr_g[: len(tr_g) // 2], te_g)
    ch_jump_g = np.argsort(-np.max(np.abs(te_g_std[3876 + W : 4129 + W]), axis=0))[:2]
    case_a = _evaluate_case("GECCO", "water_quality", (3740, 4210), [int(ch_jump_g[0]), int(ch_jump_g[1])], device, epochs=15)

    # (b) Real Industrial Slow Ramp: GHL-44 (continuous Channel 2 thermal ramp [7129, 7992))
    case_b = _evaluate_case(
        "GHL",
        "44_Lev_corr_Temp_fault_seed_190_vars_23",
        (6920, 8180),
        [2],
        device,
        epochs=12,
        include_drift_hybrid=True,
    )

    fig, axes = plt.subplots(
        3, 2, figsize=(7.16, 3.15), sharex="col",
        gridspec_kw={"height_ratios": [0.95, 1.12, 1.08], "hspace": 0.16, "wspace": 0.16},
    )
    fig.subplots_adjust(top=0.795, bottom=0.125, left=0.082, right=0.985)

    configs = [
        (f"(a) Abrupt Chemical Spike (GECCO, $K_{{\\mathrm{{Lite}}}}={case_a['k_quorum']:.1f}$)", case_a, False),
        (f"(b) Slow Thermal Ramp (GHL-44, $K_{{\\mathrm{{Lite}}}}={case_b['k_quorum']:.1f}$)", case_b, True),
    ]

    for col, (title, cd, show_long) in enumerate(configs):
        t = cd["t"]
        yv = cd["y"]
        ax_s, ax_b, ax_r = axes[:, col]

        on_idx = np.flatnonzero(yv == 1)
        t_f0 = float(t[on_idx[0]]) if len(on_idx) else 0.0
        t_f1 = float(t[on_idx[-1]]) if len(on_idx) else 0.0

        for ax in (ax_s, ax_b, ax_r):
            if len(on_idx):
                ax.axvspan(t_f0, t_f1, color="#f6e7eb", alpha=0.62, zorder=0)
                ax.axvline(t_f0, color="#8c1d40", linestyle=":", linewidth=1.0, zorder=3)
            ax.grid(True, linestyle="-", linewidth=0.45, color="#e3e3e3", alpha=0.8)
            ax.set_axisbelow(True)
            ax.set_xlim(float(t[0]), float(t[-1]))

        # Row 1: Continuous Sensor Waveforms (zero interior boxes)
        s_arrays = [np.clip(arr, -4.8, 6.2) for arr in cd["sensors"]]
        ax_s.plot(t, s_arrays[0], color=C_SENS1, linewidth=1.20, zorder=4)
        if len(s_arrays) > 1:
            ax_s.plot(t, s_arrays[1], color=C_SENS2, linewidth=1.00, alpha=0.88, zorder=3)
        ax_s.set_title(title, fontweight="bold", pad=6)
        s_min = float(min(a.min() for a in s_arrays))
        s_max = float(max(a.max() for a in s_arrays))
        pad_s = max(0.6, 0.16 * (s_max - s_min))
        ax_s.set_ylim(s_min - pad_s, s_max + pad_s)

        # Row 2: Stacked Breaker Trips by Horizon vs. Quorum (zero interior boxes)
        y1 = cd["c_s1"]
        y2 = y1 + cd["c_s2"]
        y3 = y2 + cd["c_s3"]
        y4 = y3 + cd["c_dr"]
        ax_b.fill_between(t, 0, y1, facecolor=C_S1, edgecolor="none", alpha=0.92, zorder=2)
        ax_b.fill_between(t, y1, y2, facecolor=C_S2, edgecolor="none", alpha=0.92, zorder=3)
        ax_b.fill_between(t, y2, y3, facecolor=C_S3, edgecolor="none", alpha=0.92, zorder=4)
        ax_b.fill_between(t, y3, y4, facecolor=C_DRIFT, edgecolor="none", alpha=0.92, zorder=5)
        ax_b.plot(t, y4, color="#111111", linewidth=0.65, zorder=6)
        ax_b.axhline(cd["k_quorum"], color=C_THRESH, linestyle="--", linewidth=1.10, zorder=7)
        b_top = max(float(y4.max()) * 1.12, cd["k_quorum"] * 1.28, 18.0)
        ax_b.set_ylim(0.0, b_top)

        # Row 3: Score-to-Threshold Ratio (Alarm when > 1x; zero interior boxes)
        if col == 0:
            r_cap = 5.3
            yticks, yticklabels = [0.0, 1.0, 3.0, 5.0], ["0", "$1{\\times}$", "$3{\\times}$", "$5{\\times}$"]
        else:
            r_cap = 1.65
            yticks, yticklabels = [0.0, 1.0, 1.5], ["0", "$1{\\times}$", "$1.5{\\times}$"]

        ax_r.axhspan(1.0, r_cap, color="#fde8e8", alpha=0.45, zorder=1)
        ax_r.axhline(1.0, color=C_THRESH, linestyle="--", linewidth=1.05, zorder=4)
        ax_r.plot(t, np.minimum(cd["idx_lite"], r_cap * 0.97), color=C_LITE, linewidth=1.40, zorder=6)
        ax_r.plot(t, np.minimum(cd["idx_1d"], r_cap * 0.97), color=C_1D, linestyle="--", linewidth=1.15, zorder=5)
        if show_long and cd["idx_long"] is not None:
            ax_r.plot(t, np.minimum(cd["idx_long"], r_cap * 0.96), color=C_LONG, linestyle="-.", linewidth=1.45, zorder=7)

        ax_r.set_ylim(-0.10, r_cap)
        ax_r.set_yticks(yticks)
        ax_r.set_yticklabels(yticklabels)
        ax_r.set_xlabel("Relative Stream Steps $\\tau$", labelpad=2.5)

        if col == 0:
            ax_s.set_ylabel("Sensors\n($z$-score)", labelpad=3)
            ax_b.set_ylabel("Active Breakers\n$N_{128}(\\tau)$", labelpad=3)
            ax_r.set_ylabel("Score / Thresh.\n($>1{\\times}$ Alarms)", labelpad=3)

    row2_handles = [
        Patch(facecolor="#f6e7eb", edgecolor="#8c1d40", linestyle=":", linewidth=0.7, label="Ground-truth fault"),
        Patch(facecolor=C_S1, edgecolor="none", label="Short $S_1{=}16$"),
        Patch(facecolor=C_S2, edgecolor="none", label="Mid $S_2{=}32$"),
        Patch(facecolor=C_S3, edgecolor="none", label="Long $S_3{=}64$"),
        Patch(facecolor=C_DRIFT, edgecolor="none", label="Mean drift $|\\bar{e}_c|$"),
    ]
    row3_handles = [
        Line2D([0], [0], color=C_SENS1, lw=1.25, label="Sensor trace"),
        Line2D([0], [0], color=C_LITE, lw=1.50, label="Lite quorum ($N_{128}$)"),
        Line2D([0], [0], color=C_1D, lw=1.25, ls="--", label="1D Scalar SPOT"),
        Line2D([0], [0], color=C_LONG, lw=1.40, ls="-.", label="Frozen-ref. drift"),
        Line2D([0], [0], color=C_THRESH, lw=1.10, ls="--", label="Alarm threshold ($K_{\\mathrm{Lite}}$ / $1{\\times}$)"),
    ]

    leg1 = fig.legend(
        handles=row2_handles, loc="upper center", bbox_to_anchor=(0.52, 0.995),
        ncol=5, frameon=False, columnspacing=1.05, handletextpad=0.4, fontsize=7.5,
    )
    fig.add_artist(leg1)
    fig.legend(
        handles=row3_handles, loc="upper center", bbox_to_anchor=(0.52, 0.922),
        ncol=5, frameon=False, columnspacing=0.95, handletextpad=0.4, fontsize=7.5,
    )

    saved = save_true_vector_figure(
        fig,
        OUT_STEM,
        formats=("svg", "pdf", "png"),
        png_dpi=300,
        inkscape_postprocess_pdf=True,
        ensure_pure_vector=True,
        ensure_outside_legend=True,
    )
    plt.close(fig)
    print("Saved 2-column clean interpretability figure:", saved, flush=True)


if __name__ == "__main__":
    main()
