"""Audit multi-horizon whitening structure vs. horizon-preserving ZCA operators (TNNLS R3 Item 1).

Evaluates on identical trained `LaminarJEPALiteModel` checkpoints:
  1. 19 development streams x 3 seeds (`dev19_4block`, 57 runs)
  2. 118 untouched archive streams (`tier2_141_5block`, seed 42)

Compares five Stage-3 whitening operators on identical network residuals:
  - `Full_MD_ZCA`: 96x96 symmetric ZCA square root (current implementation)
  - `Shared_Kronecker_ZCA`: shared 32x32 ZCA (`I_M (x) Omega_D^{1/2}`), exact horizon & drift preservation
  - `BlockDiag_PerHorizon_ZCA`: per-horizon 32x32 ZCA (`(+)_{m=1}^M Omega_m^{1/2}`), zero cross-horizon mixing
  - `PreWhiten_Dial_ZCA`: separate 32x32 ZCA per raw horizon + separate 32x32 ZCA on raw horizon mean
  - `PCA_Eigen_Whiten`: 96x96 non-ZCA eigenbasis control (scrambles coordinate identity)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score
import torch

from scripts.run_laminar_paired_benchmarks import (
    C,
    CSMConfig,
    S,
    _build_schedule,
    _fast_val_checkpoint_train,
    _split_nominal_5block,
    _split_nominal_dev_4block,
    _train_fixed_epochs,
    compute_metrics,
    event_stats,
    load_stream,
)
from scripts.tnnls_revision.common_eval import (
    apply_horizon_whitening_operator,
    fit_horizon_whitening_operator,
)
from src.engine.trainer import build_ts_jepa_model
from src.scoring.circuit_breaker_panel import stabilize_whitened_subblocks
from src.scoring.laminar_lite_panel import PositionDriftLitePanel, build_position_drift_dials


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_json(obj: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@torch.no_grad()
def _extract_raw_residuals(model: torch.nn.Module, windows: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    dev = next(model.parameters()).device
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for i in range(0, len(windows), batch_size):
            b = torch.from_numpy(np.ascontiguousarray(windows[i : i + batch_size])).float().to(dev)
            _, _, _, zp, zt = model.forward(b[:, :C], b[:, C:])
            assert zt is not None
            chunks.append((zt - zp).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def _mean_entry_corr(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(len(a), -1).astype(np.float64)
    bf = b.reshape(len(b), -1).astype(np.float64)
    af = af - af.mean(axis=0, keepdims=True)
    bf = bf - bf.mean(axis=0, keepdims=True)
    denom = np.sqrt((af**2).sum(axis=0) * (bf**2).sum(axis=0)) + 1e-12
    return float(np.mean((af * bf).sum(axis=0) / denom))


def _eval_whitening_mode(
    panel: PositionDriftLitePanel,
    mode_label: str,
    op: dict[str, Any],
    q_raw: np.ndarray,
    va_raw: np.ndarray,
    te_raw: np.ndarray,
    len_va: int,
    len_test: int,
    y_test: np.ndarray,
) -> tuple[dict[str, dict[str, float]], np.ndarray]:
    n_cal_w = len(va_raw) // 2
    q_w = apply_horizon_whitening_operator(q_raw, op)
    va_w = apply_horizon_whitening_operator(va_raw, op)
    te_w = apply_horizon_whitening_operator(te_raw, op)
    va_ew, te_ew = stabilize_whitened_subblocks(va_w, te_w, va_raw, te_raw, n_cal_w, q_w, q_raw)

    if mode_label == "PreWhiten_Dial_ZCA":
        # Position dials from stabilized per-horizon ZCA; drift dials from separately ZCA-whitened raw horizon mean
        mu = op["mu"]
        w_bar = op["w_bar"]
        q_bar_r = (q_raw.astype(np.float64) - mu).mean(axis=1, keepdims=True)
        va_bar_r = (va_raw.astype(np.float64) - mu).mean(axis=1, keepdims=True)
        te_bar_r = (te_raw.astype(np.float64) - mu).mean(axis=1, keepdims=True)
        q_bar_w = (q_bar_r @ w_bar).astype(np.float32)
        va_bar_w = (va_bar_r @ w_bar).astype(np.float32)
        te_bar_w = (te_bar_r @ w_bar).astype(np.float32)
        va_bar_ew, te_bar_ew = stabilize_whitened_subblocks(
            va_bar_w, te_bar_w, va_bar_r.astype(np.float32), te_bar_r.astype(np.float32), n_cal_w, q_bar_w, q_bar_r.astype(np.float32)
        )
        m_cnt = va_ew.shape[1]
        dim = va_ew.shape[2]
        va_pd = np.concatenate([np.abs(va_ew).reshape(len(va_ew), m_cnt * dim), np.abs(va_bar_ew[:, 0, :])], axis=1)
        te_pd = np.concatenate([np.abs(te_ew).reshape(len(te_ew), m_cnt * dim), np.abs(te_bar_ew[:, 0, :])], axis=1)
        tau = np.percentile(va_pd[:n_cal_w], 98.5, axis=0, keepdims=True) + 1e-4
        bits: dict[str, np.ndarray | int] = {
            "bits_pd_va": (va_pd > tau).astype(np.float32),
            "bits_pd_te": (te_pd > tau).astype(np.float32),
            "num_horizons": int(m_cnt),
        }
        drift_cal = np.abs(va_bar_ew[:n_cal_w, 0, :])
    else:
        bits = panel.calibrate_and_trip_windows(va_ew, te_ew, n_cal_w)
        drift_cal = np.abs(va_ew[:n_cal_w].mean(axis=1))

    out: dict[str, dict[str, float]] = {}
    for mp in ("trailing", "smear"):
        dec = panel.evaluate_stream_decisions(bits, len_va, len_test, mp)
        tl = y_test[dec.valid_mask]
        has_both = bool(np.any(tl == 1) and np.any(tl == 0))
        mt = compute_metrics(tl, dec.test_pred_balanced, scores=dec.test_switch_counts, valid_mask=np.ones_like(tl, dtype=bool), use_pa=False)
        er, dl = event_stats(tl, dec.test_pred_balanced)
        pr = float(average_precision_score(tl, dec.test_switch_counts)) if has_both else 0.0
        out[mp] = {
            "f1": float(mt.get("f1", 0.0)),
            "pr_auc": pr,
            "event_recall": float(er) if er == er else float("nan"),
            "delay": float(dl) if dl == dl else float("nan"),
            "hold_fpr": float(dec.hold_pred_balanced.mean()) if len(dec.hold_pred_balanced) > 0 else 0.0,
            "test_fpr": float(dec.test_pred_balanced[tl == 0].mean()) if np.any(tl == 0) else 0.0,
        }
    return out, drift_cal


def main() -> None:
    out_csv = ROOT / "reports" / "whitening_structure_audit.csv"
    out_json = ROOT / "reports" / "r3_whitening_structure_summary.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panel = PositionDriftLitePanel(context_size=C, suspect_size=S)
    cfg = CSMConfig(model_type="laminar_jepa_lite", context_size=C, suspect_size=S, latent_dim=32)

    modes = [
        ("Full_MD_ZCA", "full_zca"),
        ("Shared_Kronecker_ZCA", "shared_kronecker_zca"),
        ("BlockDiag_PerHorizon_ZCA", "block_diag_zca"),
        ("PreWhiten_Dial_ZCA", "pre_dial_zca"),
        ("PCA_Eigen_Whiten", "pca_eigen"),
    ]

    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []

    # 1. Dev19 4-block (3 seeds) + 2. Untouched Archive 118 5-block (seed 42)
    dev_sched = [("dev19_4block", ds, ch, seeds) for ds, ch, seeds in _build_schedule("dev19_4block")]
    df_meta = pd.read_csv(ROOT / "reports" / "laminar_tier2_141stream_5block.csv")
    u_streams = (
        df_meta[df_meta["heldout"] == True][["dataset", "channel"]]
        .drop_duplicates()
        .to_dict(orient="records")
    )
    arch_sched = [("archive118_5block", str(r["dataset"]), str(r["channel"]), [42]) for r in u_streams]

    t_start = time.perf_counter()
    all_jobs = dev_sched + arch_sched
    for j_idx, (suite_name, ds, ch, seeds) in enumerate(all_jobs, start=1):
        train, test, y = load_stream(ds, ch)
        k_ch = int(train.shape[1])
        if suite_name == "dev19_4block":
            tr_w, q_w, va_w, te_w, len_va = _split_nominal_dev_4block(train, test)
            val_w = None
        else:
            tr_w, val_w, q_w, va_w, te_w, len_va = _split_nominal_5block(train, test)

        for sd in seeds:
            torch.manual_seed(sd)
            np.random.seed(sd)
            model = build_ts_jepa_model(cfg, input_dim=k_ch, device=device)
            if val_w is None:
                _train_fixed_epochs(model, tr_w, cfg, epochs=15, device=device)
            else:
                _fast_val_checkpoint_train(model, tr_w, val_w, cfg, device=device, epochs=12, patience=4)

            fit_raw = _extract_raw_residuals(model, tr_w)
            q_raw = _extract_raw_residuals(model, q_w)
            va_raw = _extract_raw_residuals(model, va_w)
            te_raw = _extract_raw_residuals(model, te_w)
            del model

            ops = {m_lbl: fit_horizon_whitening_operator(fit_raw, mode=m_key) for m_lbl, m_key in modes}

            # Structural diagnostics on Full_MD_ZCA vs Shared_Kronecker_ZCA & PCA_Eigen_Whiten
            w96 = ops["Full_MD_ZCA"]["w_mat"]
            m_cnt, dim = fit_raw.shape[1], fit_raw.shape[2]
            diag_blk_sq = sum(float(np.sum(w96[m * dim : (m + 1) * dim, m * dim : (m + 1) * dim] ** 2)) for m in range(m_cnt))
            tot_sq = float(np.sum(w96**2)) + 1e-12
            within_h_frac = diag_blk_sq / tot_sq

            n_cal_w = len(va_raw) // 2
            cal_raw = va_raw[:n_cal_w]
            cal_zca = apply_horizon_whitening_operator(cal_raw, ops["Full_MD_ZCA"])
            cal_pca = apply_horizon_whitening_operator(cal_raw, ops["PCA_Eigen_Whiten"])
            zca_corr = _mean_entry_corr(cal_zca, cal_raw)
            pca_corr = _mean_entry_corr(cal_pca, cal_raw)

            drift_map: dict[str, np.ndarray] = {}
            for m_lbl, _ in modes:
                res_mp, d_cal = _eval_whitening_mode(panel, m_lbl, ops[m_lbl], q_raw, va_raw, te_raw, len_va, len(test), y)
                drift_map[m_lbl] = d_cal
                for mp, mvals in res_mp.items():
                    rows.append({
                        "suite": suite_name,
                        "dataset": ds,
                        "channel": ch,
                        "stream": f"{ds}:{ch}",
                        "seed": sd,
                        "mode": m_lbl,
                        "mapping": mp,
                        **mvals,
                    })

            drift_corr = _mean_entry_corr(drift_map["Full_MD_ZCA"], drift_map["Shared_Kronecker_ZCA"])
            diag_rows.append({
                "suite": suite_name,
                "stream": f"{ds}:{ch}",
                "seed": sd,
                "within_horizon_energy_frac": within_h_frac,
                "cross_horizon_energy_frac": 1.0 - within_h_frac,
                "zca_raw_coord_corr": zca_corr,
                "pca_raw_coord_corr": pca_corr,
                "drift_dial_corr_full_vs_shared": drift_corr,
            })

        if j_idx % 20 == 0 or j_idx == len(all_jobs):
            print(f"[{j_idx:3d}/{len(all_jobs)}] {suite_name} {ds}:{ch} ({time.perf_counter()-t_start:.1f}s)", flush=True)

    df_out = pd.DataFrame(rows)
    df_diag = pd.DataFrame(diag_rows)
    _atomic_write_csv(df_out, out_csv)

    summary: dict[str, Any] = {"diagnostics": {}, "benchmarks": {}}
    for s_name, g_d in df_diag.groupby("suite"):
        s_avg_d = g_d.groupby("stream").mean(numeric_only=True)
        summary["diagnostics"][str(s_name)] = {
            col: {"mean": float(s_avg_d[col].mean()), "median": float(s_avg_d[col].median())}
            for col in ["within_horizon_energy_frac", "cross_horizon_energy_frac", "zca_raw_coord_corr", "pca_raw_coord_corr", "drift_dial_corr_full_vs_shared"]
        }

    for (s_name, mp), g_b in df_out.groupby(["suite", "mapping"]):
        s_avg_b = g_b.groupby(["mode", "stream"]).mean(numeric_only=True).reset_index()
        means = s_avg_b.groupby("mode")[["f1", "pr_auc", "event_recall", "delay", "hold_fpr", "test_fpr"]].mean().to_dict(orient="index")
        base = s_avg_b[s_avg_b["mode"] == "Full_MD_ZCA"].set_index("stream")
        paired: dict[str, Any] = {}
        for m_lbl, _ in modes[1:]:
            alt = s_avg_b[s_avg_b["mode"] == m_lbl].set_index("stream").loc[base.index]
            df1 = alt["f1"] - base["f1"]
            der = (alt["event_recall"] - base["event_recall"]).dropna()
            dfpr = alt["hold_fpr"] - base["hold_fpr"]
            paired[f"{m_lbl}_minus_Full_MD_ZCA"] = {
                "delta_f1": float(df1.mean()),
                "p_f1": float(wilcoxon(df1).pvalue) if not np.allclose(df1, 0) else 1.0,
                "delta_er": float(der.mean()),
                "p_er": float(wilcoxon(der).pvalue) if not np.allclose(der, 0) else 1.0,
                "delta_hold_fpr": float(dfpr.mean()),
                "p_hold_fpr": float(wilcoxon(dfpr).pvalue) if not np.allclose(dfpr, 0) else 1.0,
            }
        summary["benchmarks"][f"{s_name}_{mp}"] = {"means": means, "paired_vs_full_zca": paired}

    _atomic_write_json(summary, out_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
