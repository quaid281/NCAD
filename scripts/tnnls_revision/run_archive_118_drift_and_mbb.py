#!/usr/bin/env python3
"""Experiments 1B & 2C: 118-Stream Untouched Archive Failure Attribution & Block-Bootstrap Calibration.

Across all 118 untouched archive streams x 3 seeds `{42, 123, 456}` under the disjoint 5-block
protocol (`X_fit -> X_sel -> X_q -> X_cal -> X_hold`):
  1. Failure-Mode Attribution (Reviewer Weakness 2):
     - Exact Sokal `tau_int`, `N_eff = N_cal / tau_int`, and quorum `D_KS(cal, hold)` for ALL 118
       untouched streams (eliminating every `--` cell in `tab:tail_audit`).
     - Separates **Calibration Failure** (evaluated via block-permuted exchangeable calibration/hold
       splits of length `l = ceil(2*tau_int)`) from **Concept Drift** (chronological `X_cal -> X_hold`).
     - Decomposes internal drift drivers between `X_cal` and `X_hold`:
       (a) Latent context shift `||mu_z(hold) - mu_z(cal)||_2`,
       (b) Stabilized residual covariance shift `||Cov(hold) - Cov(cal)||_F / ||Cov(cal)||_F`,
       (c) Marginal per-dial exceedance ratio `mean(b_hold) / max(mean(b_cal), 1e-4)`,
       (d) Quorum variance-inflation factor `Var(N_hold) / max(Var(N_cal), 1e-4)`.
  2. Block-Bootstrap & Block-Subsampled Calibration Comparison (Reviewer Weakness 3):
     - Evaluates `Deployed` (`1.12 * q_0.995`), `MBB-Median`, `MBB-Mean`, `MBB-UCB90`
       (Moving-Block Bootstrap with `B = 400` resamples of block length `ceil(2*tau_int)`),
       and `Block-Subsampled` (thinned every `round(tau_int)` steps) on identical residuals.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.circuit_breaker_19stream.common import DEVELOPMENT_STREAMS
from scripts.run_laminar_paired_benchmarks import (
    C,
    CSMConfig,
    S,
    _build_schedule,
    _fast_val_checkpoint_train,
    _split_nominal_5block,
    compute_metrics,
    event_stats,
    load_stream,
)
from scripts.tnnls_revision.common_eval import (
    atomic_write_csv,
    atomic_write_json,
    effective_sample_size,
    mbb_quorum_thresholds,
)
from src.engine.builder import build_ts_jepa_model
from src.scoring.circuit_breaker_panel import stabilize_whitened_subblocks
from src.scoring.laminar_lite_panel import PositionDriftLitePanel


@torch.no_grad()
def _latent_mean_shift(model: torch.nn.Module, cal_w: np.ndarray, hold_w: np.ndarray) -> float:
    model.eval()
    enc = getattr(model, "context_encoder", model)
    dev = next(enc.parameters()).device

    def _mean_z(ws: np.ndarray) -> np.ndarray:
        sub = ws[:: max(1, len(ws) // 512), :C]
        with torch.inference_mode():
            zt = enc(torch.from_numpy(np.ascontiguousarray(sub)).float().to(dev))
            return zt.detach().cpu().numpy().mean(axis=0)

    if len(cal_w) == 0 or len(hold_w) == 0:
        return 0.0
    return float(np.linalg.norm(_mean_z(hold_w) - _mean_z(cal_w)))


def _stationary_block_perm_fpr(
    va_cnt: np.ndarray, half: int, block_len: int, safety: float = 1.12, floor: float = 3.5, seed: int = 42
) -> float:
    """Measure held-out FPR when `X_val` blocks are randomly interleaved (removing concept drift)."""
    rng = np.random.default_rng(seed)
    n = len(va_cnt)
    b_len = max(16, min(block_len, max(16, n // 6)))
    blocks = [va_cnt[i : i + b_len] for i in range(0, n, b_len)]
    fprs: list[float] = []
    for _ in range(25):
        perm = rng.permutation(len(blocks))
        shuf = np.concatenate([blocks[p] for p in perm])[:n]
        c_part = shuf[:half]
        h_part = shuf[half + S :]
        if len(c_part) == 0 or len(h_part) == 0:
            continue
        k_th = max(safety * float(np.percentile(c_part, 99.5)), floor)
        fprs.append(float((h_part > k_th).mean()))
    return float(np.mean(fprs)) if fprs else 0.0


def main() -> None:
    attr_csv = ROOT / "reports" / "archive_118_failure_attribution.csv"
    mbb_csv = ROOT / "reports" / "archive_118_mbb_calibration.csv"
    sum_json = ROOT / "reports" / "archive_118_drift_and_mbb_summary.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panel = PositionDriftLitePanel(context_size=C, suspect_size=S)

    schedule = [
        it for it in _build_schedule("tier2_141_5block")
        if (it[0].lower(), str(it[1])) not in DEVELOPMENT_STREAMS
    ]

    attr_rows: list[dict[str, Any]] = pd.read_csv(attr_csv).to_dict("records") if attr_csv.exists() else []
    mbb_rows: list[dict[str, Any]] = pd.read_csv(mbb_csv).to_dict("records") if mbb_csv.exists() else []
    done = {(str(r["dataset"]), str(r["channel"]), int(r["seed"])) for r in attr_rows}

    for idx, (ds, ch, seeds) in enumerate(schedule, start=1):
        ch_s = str(ch)
        need = [sd for sd in seeds if (ds, ch_s, sd) not in done]
        if not need:
            continue
        train, test, y = load_stream(ds, ch)
        tr_w, sel_w, q_w, va_w, te_w, len_va = _split_nominal_5block(train, test)
        k_ch = int(train.shape[1])
        n_cal_w = len(va_w) // 2

        for sd in need:
            torch.manual_seed(sd)
            np.random.seed(sd)
            cfg = CSMConfig(model_type="laminar_jepa_lite", context_size=C, suspect_size=S, latent_dim=32)
            m_l = build_ts_jepa_model(cfg, input_dim=k_ch, device=device)
            _fast_val_checkpoint_train(m_l, tr_w, sel_w, cfg, device, epochs=12, patience=4)
            m_l.fit_resolvent_whitening(tr_w, context_len=C, batch_size=512)

            va_r, va_w_res = m_l.extract_raw_and_whitened_residuals(va_w, context_len=C, batch_size=512)
            te_r, te_w_res = m_l.extract_raw_and_whitened_residuals(te_w, context_len=C, batch_size=512)
            q_r, q_w_res = m_l.extract_raw_and_whitened_residuals(q_w, context_len=C, batch_size=512)
            z_shift = _latent_mean_shift(m_l, va_w[:n_cal_w], va_w[n_cal_w + S :])
            del m_l

            va_ew, te_ew = stabilize_whitened_subblocks(va_w_res, te_w_res, va_r, te_r, n_cal_w, q_w_res, q_r)
            bits = panel.calibrate_and_trip_windows(va_ew, te_ew, n_cal_w)
            dec = panel.evaluate_stream_decisions(bits, len_va, len(test), "trailing")
            tl = y[dec.valid_mask]

            cal_cnt = dec.cal_switch_counts
            hold_cnt = dec.hold_switch_counts
            test_cnt = dec.test_switch_counts
            tau, n_eff = effective_sample_size(cal_cnt)
            ks_val = float(stats.ks_2samp(cal_cnt, hold_cnt).statistic) if len(hold_cnt) > 0 else 0.0

            # Residual covariance & dial-trip decomposition between cal and hold
            cal_flat = va_ew[:n_cal_w].reshape(n_cal_w, -1)
            hold_ew = va_ew[n_cal_w + S :]
            hold_flat = hold_ew.reshape(len(hold_ew), -1) if len(hold_ew) > 2 else cal_flat
            cov_c = np.cov(cal_flat, rowvar=False)
            cov_h = np.cov(hold_flat, rowvar=False)
            cov_rel_shift = float(np.linalg.norm(cov_h - cov_c) / max(np.linalg.norm(cov_c), 1e-6))

            b_va = np.asarray(bits["bits_pd_va"], dtype=np.float32)
            b_cal = b_va[:n_cal_w]
            b_hold = b_va[n_cal_w + S :]
            dial_rate_ratio = float(b_hold.mean() / max(float(b_cal.mean()), 1e-4)) if len(b_hold) > 0 else 1.0
            var_ratio = float(np.var(hold_cnt) / max(float(np.var(cal_cnt)), 1e-3)) if len(hold_cnt) > 0 else 1.0

            va_all_cnt = b_va.sum(axis=1)
            ths = mbb_quorum_thresholds(cal_cnt, floor=3.5, q=0.995, safety=1.12, n_boot=400, seed=sd)
            perm_fpr = _stationary_block_perm_fpr(va_all_cnt, len(cal_cnt), int(ths["block_len"]), seed=sd)

            attr_rows.append(
                {
                    "dataset": ds,
                    "channel": ch_s,
                    "k_channels": k_ch,
                    "t_nom": int(len(train)),
                    "seed": sd,
                    "n_cal": int(len(cal_cnt)),
                    "tau_int": tau,
                    "n_eff": n_eff,
                    "ks_drift": ks_val,
                    "hold_fpr_chron": float(dec.hold_pred_balanced.mean()) if len(hold_cnt) > 0 else 0.0,
                    "hold_fpr_block_perm": perm_fpr,
                    "test_fpr": float(dec.test_pred_balanced[tl == 0].mean()) if np.any(tl == 0) else 0.0,
                    "latent_mean_shift": z_shift,
                    "resid_cov_rel_shift": cov_rel_shift,
                    "dial_trip_rate_ratio": dial_rate_ratio,
                    "quorum_var_ratio": var_ratio,
                }
            )

            for rname in ("deployed", "mbb_median", "mbb_mean", "mbb_p90", "block_subsampled"):
                k_th = float(ths[rname])
                hp = (hold_cnt > k_th).astype(int)
                tp = (test_cnt > k_th).astype(int)
                mt = compute_metrics(tl, tp, scores=test_cnt, valid_mask=np.ones_like(tl, dtype=bool), use_pa=False)
                er, dl = event_stats(tl, tp)
                mbb_rows.append(
                    {
                        "dataset": ds,
                        "channel": ch_s,
                        "seed": sd,
                        "rule": rname,
                        "threshold": k_th,
                        "hold_fpr": float(hp.mean()) if len(hp) > 0 else 0.0,
                        "test_fpr": float(tp[tl == 0].mean()) if np.any(tl == 0) else 0.0,
                        "f1": float(mt.get("f1", 0.0)),
                        "event_recall": float(er) if er == er else float("nan"),
                        "delay": float(dl) if dl == dl else float("nan"),
                    }
                )

            atomic_write_csv(pd.DataFrame(attr_rows), attr_csv)
            atomic_write_csv(pd.DataFrame(mbb_rows), mbb_csv)
        if idx % 10 == 0 or idx == len(schedule):
            print(f"[Exp1B+2C {idx:3d}/{len(schedule)}] {ds}:{ch_s} completed", flush=True)

    df_a = pd.DataFrame(attr_rows)
    df_m = pd.DataFrame(mbb_rows)
    sa = df_a.groupby(["dataset", "channel"], as_index=False).mean(numeric_only=True)
    sm = df_m.groupby(["rule", "dataset", "channel"], as_index=False).mean(numeric_only=True)

    mbb_sum = []
    for rule, g in sm.groupby("rule"):
        mbb_sum.append(
            {
                "rule": rule,
                "f1_mean": float(g["f1"].mean()),
                "er_mean": float(g["event_recall"].mean()),
                "delay_mean": float(g["delay"].mean()),
                "hold_fpr_mean": float(g["hold_fpr"].mean()),
                "hold_fpr_med": float(g["hold_fpr"].median()),
                "hold_fpr_p90": float(np.percentile(g["hold_fpr"], 90.0)),
                "hold_fpr_p95": float(np.percentile(g["hold_fpr"], 95.0)),
                "pct_le_5pct": float(100.0 * (g["hold_fpr"] <= 0.05).mean()),
            }
        )

    hi = sa[sa["hold_fpr_chron"] > 0.05]
    lo = sa[sa["hold_fpr_chron"] <= 0.05]
    decomp = {
        "all_118_chron_fpr_mean": float(sa["hold_fpr_chron"].mean()),
        "all_118_block_perm_fpr_mean": float(sa["hold_fpr_block_perm"].mean()),
        "drift_share_of_excess_fpr": float(
            max(0.0, sa["hold_fpr_chron"].mean() - sa["hold_fpr_block_perm"].mean()) / max(sa["hold_fpr_chron"].mean(), 1e-6)
        ),
        "high_fpr_streams_n": int(len(hi)),
        "high_fpr_medians": hi[["latent_mean_shift", "resid_cov_rel_shift", "dial_trip_rate_ratio", "quorum_var_ratio", "ks_drift", "n_eff"]].median().round(4).to_dict(),
        "low_fpr_medians": lo[["latent_mean_shift", "resid_cov_rel_shift", "dial_trip_rate_ratio", "quorum_var_ratio", "ks_drift", "n_eff"]].median().round(4).to_dict(),
        "per_dataset_false_alarms": (
            sa.groupby("dataset", as_index=False)
            .agg(
                n=("hold_fpr_chron", "size"),
                fpr_mean=("hold_fpr_chron", "mean"),
                fpr_med=("hold_fpr_chron", "median"),
                fpr_p95=("hold_fpr_chron", lambda s: float(np.percentile(s, 95.0))),
                pct_le_5=("hold_fpr_chron", lambda s: float(100.0 * (s <= 0.05).mean())),
                ks_med=("ks_drift", "median"),
                neff_med=("n_eff", "median"),
            )
            .round(4)
            .to_dict("records")
        ),
    }
    atomic_write_json({"mbb_calibration": mbb_sum, "failure_decomposition": decomp}, sum_json)
    print("Exp 1B+2C summary saved to", sum_json, flush=True)


if __name__ == "__main__":
    main()
