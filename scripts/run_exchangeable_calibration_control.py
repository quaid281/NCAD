#!/usr/bin/env python3
"""Run Exchangeable / Shuffle Calibration Control.

Directly addresses Reviewer Concern A.2 (Drift vs. Tail Scarcity):
"Run a shuffle or block-permutation control. Draw calibration and evaluation
scores exchangeably from the pooled nominal blocks, which removes drift.
If overshoot persists, it comes from scarcity or autocorrelation.
If it vanishes, it comes from drift. Sweep q (10^-3, 10^-2, 5x10^-2)
and calibration length."

Protocol:
- For each dumped score file in reports/score_dumps/:
  Load nominal splits: cal (N_cal) and hold (N_hold).
  Pool them: X_nominal = [cal, hold]
- Three partitioning regimes:
  1. 'chronological': original temporal split (cal -> hold, subject to drift)
  2. 'point_shuffle': exchangeable uniform permutation of all points (removes drift + autocorrelation)
  3. 'block_shuffle': exchangeable permutation of L=64 blocks (removes inter-block drift, preserves intra-block correlation)
- Sweeps:
  - Design risk q in {0.001, 0.01, 0.05}
  - Calibration length: 1.0 * N_cal, 0.5 * N_cal
  - Models: TS-JEPA, OpEntropy, Reynolds, PotentialFlow, TimesNet, TranAD, TCN-ObsPred-LN
  - Mappings: trailing (causal), smear (buffered)

Output:
  reports/exchangeable_calibration_control.csv
  reports/exchangeable_calibration_summary.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scoring.evt_calibrator import fit_gpd

SCORE_DUMPS_DIR = ROOT / "reports" / "score_dumps"
OUT_CSV = ROOT / "reports" / "exchangeable_calibration_control.csv"
OUT_SUMMARY = ROOT / "reports" / "exchangeable_calibration_summary.md"

Q_LEVELS = [0.001, 0.01, 0.05]
CAL_FRACTIONS = [1.0, 0.5]
BLOCK_LEN = 64
U_QUANTILE = 0.98
RNG_SEED = 42


def calibrate_spot(x_cal: np.ndarray, q: float, u_q: float = 0.98) -> float:
    """Compute SPOT threshold on calibration array."""
    if len(x_cal) < 10:
        return float(np.quantile(x_cal, 1.0 - q))
    u = float(np.quantile(x_cal, u_q))
    exc = x_cal[x_cal > u] - u
    if len(exc) < 5:
        return float(np.quantile(x_cal, 1.0 - q))
    fit = fit_gpd(exc)
    sigma, xi = fit.sigma, fit.gamma
    n = len(x_cal)
    n_u = len(exc)
    r = (n / n_u) * q
    if abs(xi) > 1e-6:
        th = u + (sigma / xi) * (r ** (-xi) - 1.0)
    else:
        th = u - sigma * np.log(r)
    return float(th)


def partition_blocks(arr: np.ndarray, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Permute an array by contiguous blocks of length block_len."""
    n = len(arr)
    num_blocks = int(np.ceil(n / block_len))
    blocks = [arr[i * block_len : min((i + 1) * block_len, n)] for i in range(num_blocks)]
    perm = rng.permutation(num_blocks)
    shuffled_blocks = [blocks[p] for p in perm]
    return np.concatenate(shuffled_blocks)


def run_experiment() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    models = ["TS-JEPA", "OpEntropy", "Reynolds", "PotentialFlow", "TimesNet", "TranAD", "TCN-ObsPred-LN"]
    mappings = ["trailing", "smear"]
    
    files = list(SCORE_DUMPS_DIR.glob("*.npz"))
    print(f"Total dump files in {SCORE_DUMPS_DIR}: {len(files)}")
    
    rows = []
    
    for f in files:
        parts = f.stem.split("__")
        if len(parts) != 5:
            continue
        dataset, channel, seed_str, model, mapping = parts
        if model not in models or mapping not in mappings:
            continue
        
        try:
            data = np.load(f)
            cal = data["cal"]
            hold = data["hold"]
        except Exception:
            continue
        
        if len(cal) < 100 or len(hold) < 100:
            continue
            
        n_cal = len(cal)
        n_hold = len(hold)
        pooled = np.concatenate([cal, hold])
        total_len = len(pooled)
        
        for frac in CAL_FRACTIONS:
            cur_n_cal = int(round(n_cal * frac))
            if cur_n_cal < 50:
                continue
                
            for q in Q_LEVELS:
                # 1. Chronological (Real split)
                cal_chrono = cal[:cur_n_cal]
                th_chrono = calibrate_spot(cal_chrono, q=q, u_q=U_QUANTILE)
                fpr_chrono = float((hold > th_chrono).mean())
                
                # 2. Point-wise Shuffle (Exchangeable, removes drift + correlation)
                idx_point = rng.permutation(total_len)
                cal_point = pooled[idx_point[:cur_n_cal]]
                hold_point = pooled[idx_point[cur_n_cal:]]
                th_point = calibrate_spot(cal_point, q=q, u_q=U_QUANTILE)
                fpr_point = float((hold_point > th_point).mean())
                
                # 3. Block Permutation (L=64, removes drift across blocks, keeps local autocorrelation)
                shuffled_pooled = partition_blocks(pooled, BLOCK_LEN, rng)
                cal_block = shuffled_pooled[:cur_n_cal]
                hold_block = shuffled_pooled[cur_n_cal:]
                th_block = calibrate_spot(cal_block, q=q, u_q=U_QUANTILE)
                fpr_block = float((hold_block > th_block).mean())
                
                rows.append({
                    "dataset": dataset,
                    "channel": channel,
                    "seed": seed_str,
                    "model": model,
                    "mapping": mapping,
                    "cal_fraction": frac,
                    "n_cal": cur_n_cal,
                    "n_hold": n_hold,
                    "design_q": q,
                    "fpr_chrono": fpr_chrono,
                    "fpr_point_shuffle": fpr_point,
                    "fpr_block_shuffle": fpr_block,
                    "overshoot_chrono": fpr_chrono / q,
                    "overshoot_point": fpr_point / q,
                    "overshoot_block": fpr_block / q,
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} evaluations to {OUT_CSV}")
    return df


def generate_summary(df: pd.DataFrame) -> None:
    lines = []
    lines.append("# Exchangeable / Shuffle Calibration Control Summary")
    lines.append("")
    lines.append("Addresses Reviewer Concern A.2: Disentangling Inter-Block Distribution Drift from Finite-Sample Scarcity.")
    lines.append("")
    lines.append("## Overview")
    lines.append("- **Chronological:** Threshold calibrated on the nominal training tail, evaluated on the subsequent nominal block (subject to temporal drift).")
    lines.append("- **Point Shuffle:** Calibration and evaluation sets drawn exchangeably by uniform random permutation of pooled nominal blocks (removes both temporal drift and serial autocorrelation).")
    lines.append("- **Block Shuffle ($L=64$):** Calibration and evaluation sets drawn exchangeably by permuting 64-step contiguous blocks (eliminates macro drift between blocks while preserving intra-block autocorrelation).")
    lines.append("")
    
    # 1. Primary Design Point: q=1e-3, cal_fraction=1.0
    lines.append("## 1. Primary Design Point ($q = 10^{-3}$, Full Calibration Split)")
    lines.append("")
    lines.append("### Strictly Causal Mapping (Trailing)")
    lines.append("| Model | Chrono FPR | Point-Shuffle FPR | Block-Shuffle FPR | Chrono Overshoot | Point Overshoot | Block Overshoot |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    sub = df[(df.cal_fraction == 1.0) & (df.design_q == 0.001) & (df.mapping == "trailing")]
    for mdl, g in sub.groupby("model"):
        c_mean = g.fpr_chrono.mean()
        p_mean = g.fpr_point_shuffle.mean()
        b_mean = g.fpr_block_shuffle.mean()
        lines.append(f"| {mdl} | {c_mean:.4f} | {p_mean:.4f} | {b_mean:.4f} | {c_mean/0.001:.2f}x | {p_mean/0.001:.2f}x | {b_mean/0.001:.2f}x |")
    
    lines.append("")
    lines.append("### Buffered Mapping (Smear)")
    lines.append("| Model | Chrono FPR | Point-Shuffle FPR | Block-Shuffle FPR | Chrono Overshoot | Point Overshoot | Block Overshoot |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    sub_smear = df[(df.cal_fraction == 1.0) & (df.design_q == 0.001) & (df.mapping == "smear")]
    for mdl, g in sub_smear.groupby("model"):
        c_mean = g.fpr_chrono.mean()
        p_mean = g.fpr_point_shuffle.mean()
        b_mean = g.fpr_block_shuffle.mean()
        lines.append(f"| {mdl} | {c_mean:.4f} | {p_mean:.4f} | {b_mean:.4f} | {c_mean/0.001:.2f}x | {p_mean/0.001:.2f}x | {b_mean/0.001:.2f}x |")
        
    lines.append("")
    lines.append("## 2. Sweep across Design Risk $q \\in \\{10^{-3}, 10^{-2}, 5 \\times 10^{-2}\\}$ (Trailing, TS-JEPA)")
    lines.append("| Design $q$ | Chrono FPR | Point-Shuffle FPR | Block-Shuffle FPR | Chrono Overshoot | Point Overshoot | Block Overshoot |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    sub_q = df[(df.cal_fraction == 1.0) & (df.model == "TS-JEPA") & (df.mapping == "trailing")]
    for q, g in sub_q.groupby("design_q"):
        c_mean = g.fpr_chrono.mean()
        p_mean = g.fpr_point_shuffle.mean()
        b_mean = g.fpr_block_shuffle.mean()
        lines.append(f"| {q:.3f} | {c_mean:.4f} | {p_mean:.4f} | {b_mean:.4f} | {c_mean/q:.2f}x | {p_mean/q:.2f}x | {b_mean/q:.2f}x |")

    lines.append("")
    lines.append("## 3. Calibration Horizon Sensitivity (Full vs. 50% Calibration Split, TS-JEPA, Trailing, $q=10^{-3}$)")
    lines.append("| Calibration Split | Chrono FPR | Point-Shuffle FPR | Block-Shuffle FPR | Chrono Overshoot | Point Overshoot | Block Overshoot |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    sub_h = df[(df.design_q == 0.001) & (df.model == "TS-JEPA") & (df.mapping == "trailing")]
    for frac, g in sub_h.groupby("cal_fraction"):
        lbl = "100% N_cal" if frac == 1.0 else "50% N_cal"
        c_mean = g.fpr_chrono.mean()
        p_mean = g.fpr_point_shuffle.mean()
        b_mean = g.fpr_block_shuffle.mean()
        lines.append(f"| {lbl} | {c_mean:.4f} | {p_mean:.4f} | {b_mean:.4f} | {c_mean/0.001:.2f}x | {p_mean/0.001:.2f}x | {b_mean/0.001:.2f}x |")
        
    summary_text = "\n".join(lines)
    OUT_SUMMARY.write_text(summary_text, encoding="utf-8")
    print(f"Saved summary report to {OUT_SUMMARY}")
    print(summary_text)


if __name__ == "__main__":
    df = run_experiment()
    generate_summary(df)
