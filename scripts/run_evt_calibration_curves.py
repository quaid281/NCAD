#!/usr/bin/env python3
"""EVT/SPOT Calibration Curves, Declustering, and Multi-Window Drift.

Addresses remaining requirements of Reviewer Major Concern 5 that the failure
decomposition table does not cover:

A) Calibration error as a function of requested risk level q:
   achieved held-out FPR vs. q in {1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4}
   for both SPOT (GPD tail model) and the plain empirical quantile.
B) Declustered tail estimation: runs-declustering of exceedances above the
   initial threshold u (gap tolerance r=5) vs. raw SPOT vs. empirical quantile,
   reporting achieved held-out FPR at q=1e-3.
C) Chronological calibration drift: the calibration interval is split into 3
   consecutive blocks; a threshold fit on block k is evaluated on all later
   blocks and the held-out evaluation partition, separating temporal drift
   from estimation error.

Models: TS-JEPA, TimesNet, TranAD (same training protocol as
run_evt_failure_decomposition.py: 12 epochs, C=256, S=64).

Outputs:
  reports/evt_calibration_curve_sweep.csv
  reports/evt_declustering_comparison.csv
  reports/evt_multiwindow_drift.csv
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CSMConfig
from src.data.data_loader import DataLoader
from src.engine.trainer import build_ts_jepa_model
from src.models.baselines import TimesNet, TranAD
from src.scoring.event_fusion import aggregate_window_scores
from src.scoring.evt_calibrator import fit_gpd

from scripts.run_evt_failure_decomposition import load_partitioned_telemetry

C_LEN, S_LEN = 256, 64
W_LEN = C_LEN + S_LEN
EPOCHS = 12
Q_LEVELS = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]


def gpd_threshold(x: np.ndarray, q: float, u: float, n_u_override: int | None = None,
                  excesses_override: np.ndarray | None = None) -> float:
    """POT/GPD threshold at risk level q given exceedances above u."""
    exc = excesses_override if excesses_override is not None else x[x > u] - u
    n_u = n_u_override if n_u_override is not None else len(exc)
    n = len(x)
    if len(exc) < 5 or n_u < 1:
        return float(np.quantile(x, 1.0 - q))
    fit = fit_gpd(exc)
    sigma, xi = fit.sigma, fit.gamma
    if abs(xi) < 1e-6:
        return float(u - sigma * math.log(q * n / n_u))
    return float(u + (sigma / xi) * ((q * n / n_u) ** (-xi) - 1.0))


def runs_decluster(exc_idx: np.ndarray, r: int = 5) -> List[np.ndarray]:
    """Cluster exceedance indices allowing inter-exceedance gaps <= r."""
    if len(exc_idx) == 0:
        return []
    clusters = [[exc_idx[0]]]
    for idx in exc_idx[1:]:
        if idx - clusters[-1][-1] <= r:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
    return [np.array(c) for c in clusters]


def train_and_score(dataset: str, channel: str, device: torch.device) -> Dict[str, Dict[str, np.ndarray]]:
    """Train the three models; return {model: {'cal': pt_scores, 'eval': pt_scores}}."""
    tr_raw, cal_raw, eval_raw = load_partitioned_telemetry(dataset, channel)
    k = tr_raw.shape[1]
    tr_win = DataLoader.create_windows(tr_raw, W_LEN, step=5, copy=False)
    cal_win = DataLoader.create_windows(cal_raw, W_LEN, step=1, copy=False)
    eval_win = DataLoader.create_windows(eval_raw, W_LEN, step=1, copy=False)

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for m_name in ["TS-JEPA", "TimesNet", "TranAD"]:
        print(f"  Training {m_name}...", flush=True)
        torch.manual_seed(42)
        np.random.seed(42)

        if m_name == "TS-JEPA":
            cfg = CSMConfig(model_type="ts_jepa", context_size=C_LEN, suspect_size=S_LEN,
                            latent_dim=32, filters=48, tcn_layers=3)
            m = build_ts_jepa_model(cfg, input_dim=k, device=device)
            opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
            for ep in range(EPOCHS):
                m.train()
                perm = np.random.permutation(len(tr_win))
                for b in range(0, len(perm), 32):
                    batch = torch.from_numpy(tr_win[perm[b:b + 32]]).float().to(device)
                    loss, _ = m.compute_objective(batch[:, :C_LEN], batch[:, C_LEN:], cfg)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    m.update_target_encoder()

            def score_fn(w_arr):
                m.eval()
                sc = []
                with torch.no_grad():
                    for i in range(0, len(w_arr), 256):
                        chunk = torch.from_numpy(w_arr[i:i + 256]).float().to(device)
                        sc.append(m.compute_predictive_discrepancy(chunk[:, :C_LEN], chunk[:, C_LEN:]).cpu().numpy())
                return np.concatenate(sc)

        elif m_name == "TimesNet":
            m = TimesNet(c_in=k, d_model=48, d_ff=96, e_layers=2).to(device)
            opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
            for ep in range(EPOCHS):
                m.train()
                perm = np.random.permutation(len(tr_win))
                for b in range(0, len(perm), 32):
                    batch = torch.from_numpy(tr_win[perm[b:b + 32]]).float().to(device)
                    loss = nn.functional.mse_loss(m(batch), batch)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

            def score_fn(w_arr):
                m.eval()
                sc = []
                with torch.no_grad():
                    for i in range(0, len(w_arr), 256):
                        chunk = torch.from_numpy(w_arr[i:i + 256]).float().to(device)
                        rec = m(chunk)
                        sc.append(torch.mean((rec[:, C_LEN:] - chunk[:, C_LEN:]) ** 2, dim=(-2, -1)).cpu().numpy())
                return np.concatenate(sc)

        else:  # TranAD
            m = TranAD(c_in=k, d_model=48, n_heads=4, e_layers=2, d_layers=2, d_ff=96).to(device)
            opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
            for ep in range(EPOCHS):
                m.train()
                perm = np.random.permutation(len(tr_win))
                for b in range(0, len(perm), 32):
                    batch = torch.from_numpy(tr_win[perm[b:b + 32]]).float().to(device)
                    rec1, rec2 = m(batch)
                    l1, l2 = m.adversarial_loss(rec1, rec2, batch, epoch=ep + 1)
                    loss = l1 + l2
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

            def score_fn(w_arr):
                m.eval()
                sc = []
                with torch.no_grad():
                    for i in range(0, len(w_arr), 256):
                        chunk = torch.from_numpy(w_arr[i:i + 256]).float().to(device)
                        _, rec2 = m(chunk)
                        sc.append(torch.mean((rec2[:, C_LEN:] - chunk[:, C_LEN:]) ** 2, dim=(-2, -1)).cpu().numpy())
                return np.concatenate(sc)

        cal_w = score_fn(cal_win)
        eval_w = score_fn(eval_win)
        pt_cal, mask_c = aggregate_window_scores(cal_w, n_points=len(cal_raw), context_size=C_LEN,
                                               suspect_size=S_LEN, step=1, mapping_method="trailing")
        pt_eval, mask_e = aggregate_window_scores(eval_w, n_points=len(eval_raw), context_size=C_LEN,
                                                suspect_size=S_LEN, step=1, mapping_method="trailing")
        out[m_name] = {
            "cal": pt_cal[pt_cal > 0],
            "eval": pt_eval[pt_eval > 0],
        }
    return out


def analyze(scores: Dict[str, Dict[str, np.ndarray]], dataset: str, channel: str,
            sweep_rows, decl_rows, drift_rows) -> None:
    for m_name, d in scores.items():
        cal, ev = d["cal"], d["eval"]
        if len(cal) < 50 or len(ev) < 50:
            continue
        u = float(np.percentile(cal, 98.0))

        # A) Risk-level sweep --------------------------------------------------
        for q in Q_LEVELS:
            for method in ["spot", "empirical"]:
                if method == "spot":
                    tau = gpd_threshold(cal, q, u)
                else:
                    tau = float(np.quantile(cal, 1.0 - q))
                fpr = float(np.mean(ev > tau))
                sweep_rows.append({
                    "dataset": dataset, "channel": channel, "model": m_name,
                    "method": method, "risk_q": q, "threshold": tau,
                    "achieved_fpr": fpr, "overshoot": fpr / q,
                    "log10_ratio": math.log10(max(fpr, 1e-9) / q),
                })

        # B) Declustered tail estimation at q=1e-3 ------------------------------
        q0 = 1e-3
        exc_idx = np.where(cal > u)[0]
        clusters = runs_decluster(exc_idx, r=5)
        if clusters:
            cluster_max = np.array([cal[c].max() for c in clusters])
            exc_decl = cluster_max - u
        else:
            exc_decl = np.array([])
        variants = {
            "spot_raw": gpd_threshold(cal, q0, u),
            "spot_declustered": gpd_threshold(cal, q0, u, n_u_override=max(len(clusters), 1),
                                            excesses_override=exc_decl),
            "empirical_quantile": float(np.quantile(cal, 1.0 - q0)),
        }
        for vname, tau in variants.items():
            fpr = float(np.mean(ev > tau))
            decl_rows.append({
                "dataset": dataset, "channel": channel, "model": m_name,
                "variant": vname, "n_clusters": len(clusters),
                "raw_nu": len(exc_idx), "threshold": tau,
                "achieved_fpr": fpr, "overshoot": fpr / q0,
            })

        # C) Multi-window chronological drift -----------------------------------
        n_blocks = 3
        block_len = len(cal) // n_blocks
        blocks = [cal[i * block_len:(i + 1) * block_len] for i in range(n_blocks)]
        for k, cal_blk in enumerate(blocks):
            if len(cal_blk) < 30:
                continue
            u_k = float(np.percentile(cal_blk, 98.0))
            tau_k = gpd_threshold(cal_blk, q0, u_k)
            eval_sets = {f"cal_block_{j}": blocks[j] for j in range(n_blocks)}
            eval_sets["heldout_eval"] = ev
            for ev_name, ev_blk in eval_sets.items():
                fpr = float(np.mean(ev_blk > tau_k))
                drift_rows.append({
                    "dataset": dataset, "channel": channel, "model": m_name,
                    "cal_block": k, "eval_block": ev_name,
                    "threshold": tau_k, "achieved_fpr": fpr, "overshoot": fpr / q0,
                })


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"EVT calibration curves on {device}")

    streams = [
        ("SMAP", "P-3"),
        ("MSL", "M-1"),
        ("SMD", "machine-1-2"),
        ("Daphnet", "S01R01E1"),
        ("GECCO", "water_quality"),
    ]

    sweep_rows, decl_rows, drift_rows = [], [], []
    for ds, ch in streams:
        print(f"\n=== {ds}:{ch} ===", flush=True)
        try:
            scores = train_and_score(ds, ch, device)
            analyze(scores, ds, ch, sweep_rows, decl_rows, drift_rows)
        except Exception as e:
            print(f"FAILED {ds}:{ch}: {e}")
        pd.DataFrame(sweep_rows).to_csv(ROOT / "reports" / "evt_calibration_curve_sweep.csv", index=False)
        pd.DataFrame(decl_rows).to_csv(ROOT / "reports" / "evt_declustering_comparison.csv", index=False)
        pd.DataFrame(drift_rows).to_csv(ROOT / "reports" / "evt_multiwindow_drift.csv", index=False)

    print("\n=== Calibration curve summary (mean overshoot by q) ===")
    sw = pd.DataFrame(sweep_rows)
    print(sw.groupby(["model", "method", "risk_q"])["achieved_fpr"].mean().to_string())
    print("\n=== Declustering summary ===")
    print(pd.DataFrame(decl_rows).groupby(["model", "variant"])["achieved_fpr"].mean().to_string())
    print("\n=== Drift summary (mean FPR by cal_block -> eval_block) ===")
    print(pd.DataFrame(drift_rows).groupby(["cal_block", "eval_block"])["achieved_fpr"].mean().to_string())


if __name__ == "__main__":
    main()
