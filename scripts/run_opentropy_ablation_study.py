#!/usr/bin/env python3
"""Ablation Study Runner: Operator-Entropy JEPA Innovations.

Systematically ablates the core architectural and mathematical components:
  1. Historical-OpEntropy (alpha_K=0.5, alpha_Z=0.0, beta_Lyap=0.0, one-sided clamp scoring)
  2. OpEntropy-alpha0.02 (Bilateral scoring + alpha_Z=0.02)
  3. OpEntropy-alpha0.005 (Bilateral scoring + optimal alpha_Z=0.005)
  4. OpEntropy-Lyapunov (Bilateral scoring + alpha_Z=0.005 + beta_Lyapunov=0.02)
  5. TS-JEPA (Unconstrained 2-layer MLP predictor control)

Evaluates held-out calibration on 8 benchmark telemetry streams:
  * Held-out nominal FPR (SPOT u=0.98, q=1e-3, 64-step gap)
  * Point-F1, Event-Recall, PR-AUC, AUROC
  * Both strictly causal ('trailing') and buffered ('smear') mappings
Output: reports/opentropy_ablation_study.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CSMConfig
from src.engine.trainer import build_ts_jepa_model
from src.scoring.event_fusion import aggregate_window_scores, compute_metrics, _extract_segments
from src.scoring.evt_calibrator import EVTCalibrator, compute_excess_kurtosis

C, S = 256, 64
W = C + S

ABLATION_MODELS = [
    "Historical-OpEntropy",
    "OpEntropy-alpha0.02",
    "OpEntropy-alpha0.005",
    "OpEntropy-Lyapunov",
    "OpEntropy-Boosted",
    "OpEntropy-Cayley",
    "OpEntropy-Cayley-Lyapunov",
    "TS-JEPA",
]

DEFAULT_STREAMS = [
    ("SMAP", "P-3"),
    ("MSL", "M-1"),
    ("SMD", "machine-1-2"),
    ("SMAP", "A-1"),
    ("MSL", "C-1"),
    ("SMD", "machine-1-3"),
    ("Daphnet", "S01R01E1"),
    ("GECCO", "water_quality"),
]


def load_stream(ds: str, ch: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = ROOT / "mTSBench_data" / ds
    if ch == "default":
        tr = d / f"{ds}_train.csv"
        tr = tr if tr.exists() else d / f"{ds}.csv"
        te = d / f"{ds}_test.csv"
    else:
        tr = d / f"{ds}_{ch}_train.csv"
        tr = tr if tr.exists() else d / f"{ch}_train.csv"
        te = d / f"{ds}_{ch}_test.csv"
        te = te if te.exists() else d / f"{ch}_test.csv"
    a, b = pd.read_csv(tr), pd.read_csv(te)
    cols = [c for c in a.columns if c not in ["timestamp", "is_anomaly", "label"]]
    lc = [c for c in b.columns if c in ["is_anomaly", "label", "labels"]]
    y = b[lc[0]].to_numpy().astype(int) if lc else np.zeros(len(b), dtype=int)
    return (
        np.nan_to_num(a[cols].to_numpy(), nan=0.0).astype(np.float32),
        np.nan_to_num(b[cols].to_numpy(), nan=0.0).astype(np.float32),
        y,
    )


def event_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    segments = _extract_segments(y_true)
    if not segments:
        return 0.0, 0.0
    detected = 0
    delays = []
    for start, end in segments:
        seg_pred = y_pred[start:end]
        if seg_pred.any():
            detected += 1
            delay = int(np.argmax(seg_pred))
            delays.append(delay)
    rec = detected / len(segments)
    mean_delay = float(np.mean(delays)) if delays else float("nan")
    return rec, mean_delay


def block_bootstrap_ci(data: np.ndarray, block_size: int = 64, n_boot: int = 500) -> Tuple[float, float]:
    n = len(data)
    if n < block_size * 2:
        return float(data.mean()), float(data.mean())
    n_blocks = n // block_size
    means = []
    for _ in range(n_boot):
        b_idx = np.random.randint(0, n_blocks, size=n_blocks)
        sample = np.concatenate([data[i * block_size : (i + 1) * block_size] for i in b_idx])
        means.append(sample.mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compute_thresholds(cal: np.ndarray) -> Tuple[Dict[str, float], float]:
    cal_clean = np.nan_to_num(cal, nan=0.0)
    kurt = float(compute_excess_kurtosis(cal_clean))
    evt = EVTCalibrator(risk_level=1e-3, init_percentile=98.0, adaptive_kurtosis=False)
    evt.fit(cal_clean)
    t_spot = float(evt.threshold_)
    t_p995 = float(np.percentile(cal_clean, 99.5))
    return {"SPOT": t_spot, "p99.5": t_p995}, kurt


def train_ablation_model(
    name: str, tr_w: np.ndarray, k: int, device: torch.device, epochs: int
):
    n_win = len(tr_w)
    bs = min(64, n_win)
    idx = np.arange(n_win)

    if name == "Historical-OpEntropy":
        cfg = CSMConfig(
            model_type="operator_entropy_jepa",
            feature_dim=k,
            latent_dim=32,
            use_koopman_matrix=True,
            alpha_entropy=0.5,
            alpha_rep_entropy=0.0,
            beta_lyapunov=0.0,
        )
        is_historical = True
    elif name == "OpEntropy-alpha0.02":
        cfg = CSMConfig(
            model_type="operator_entropy_jepa",
            feature_dim=k,
            latent_dim=32,
            use_koopman_matrix=True,
            alpha_entropy=0.5,
            alpha_rep_entropy=0.02,
            beta_lyapunov=0.0,
        )
        is_historical = False
    elif name == "OpEntropy-alpha0.005":
        cfg = CSMConfig(
            model_type="operator_entropy_jepa",
            feature_dim=k,
            latent_dim=32,
            use_koopman_matrix=True,
            alpha_entropy=0.5,
            alpha_rep_entropy=0.005,
            beta_lyapunov=0.0,
        )
        is_historical = False
    elif name == "OpEntropy-Lyapunov":
        cfg = CSMConfig(
            model_type="operator_entropy_jepa",
            feature_dim=k,
            latent_dim=32,
            use_koopman_matrix=True,
            use_cayley_operator=False,
            alpha_entropy=0.5,
            alpha_rep_entropy=0.005,
            beta_lyapunov=0.02,
        )
        is_historical = False
    elif name == "OpEntropy-Boosted":
        cfg = CSMConfig(
            model_type="operator_entropy_jepa",
            feature_dim=k,
            latent_dim=32,
            use_koopman_matrix=True,
            use_cayley_operator=False,
            alpha_entropy=0.5,
            alpha_rep_entropy=0.0,
            beta_lyapunov=0.02,
            use_radial_heat=False,
            use_cohn_elkies=False,
        )
        is_historical = False
    elif name == "OpEntropy-Cayley":
        cfg = CSMConfig(
            model_type="operator_entropy_jepa",
            feature_dim=k,
            latent_dim=32,
            use_koopman_matrix=False,
            use_cayley_operator=True,
            alpha_entropy=0.0,
            alpha_rep_entropy=0.005,
            beta_lyapunov=0.0,
        )
        is_historical = False
    elif name == "OpEntropy-Cayley-Lyapunov":
        cfg = CSMConfig(
            model_type="operator_entropy_jepa",
            feature_dim=k,
            latent_dim=32,
            use_koopman_matrix=False,
            use_cayley_operator=True,
            alpha_entropy=0.0,
            alpha_rep_entropy=0.005,
            beta_lyapunov=0.02,
        )
        is_historical = False
    elif name == "TS-JEPA":
        cfg = CSMConfig(
            model_type="ts_jepa",
            feature_dim=k,
            latent_dim=32,
        )
        is_historical = False
    else:
        raise ValueError(f"Unknown ablation arm: {name}")

    m = build_ts_jepa_model(cfg, k, device).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-5)

    for ep in range(epochs):
        m.train()
        np.random.shuffle(idx)
        for s in range(0, n_win, bs):
            b = torch.from_numpy(tr_w[idx[s : s + bs]]).to(device)
            loss, _ = m.compute_objective(b[:, :C], b[:, C:], config=cfg)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 2.0)
            opt.step()
            if hasattr(m, "update_target_encoder"):
                m.update_target_encoder()

    m.eval()

    def score(wins: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for s in range(0, len(wins), 128):
                ch = torch.from_numpy(wins[s : s + 128]).to(device)
                ctx, tgt = ch[:, :C], ch[:, C:]
                if is_historical:
                    # Historical one-sided discrepancy: clamp(tgt - pred, min=0)
                    z_ctx = m._apply_filters(m.context_encoder(ctx))
                    z_pred = m._predict(z_ctx)
                    z_tgt = m._apply_filters(m.target_encoder(tgt))
                    diff = z_tgt - z_pred
                    disc = m.saliency_gate(diff)
                    e_tgt = torch.linalg.norm(z_tgt, dim=-1)
                    e_pred = torch.linalg.norm(z_pred, dim=-1)
                    leak = torch.clamp(e_tgt - e_pred, min=0.0)
                    score_val = disc + 0.2 * leak
                    out.append(score_val.cpu().numpy())
                else:
                    out.append(m.compute_predictive_discrepancy(ctx, tgt).cpu().numpy())
        return np.concatenate(out)

    return score


def run_ablation(
    streams: List[Tuple[str, str]],
    seeds: List[int],
    device: torch.device,
    epochs: int,
    val_frac: float,
    out_csv: Path,
    models: Optional[List[str]] = None,
):
    if out_csv.exists():
        try:
            existing_df = pd.read_csv(out_csv)
            out_rows = existing_df.to_dict("records")
            print(f"Loaded {len(out_rows)} existing records from {out_csv}", flush=True)
        except Exception:
            out_rows = []
    else:
        out_rows = []
    active_models = models if models is not None else ABLATION_MODELS

    for s_idx, (ds, ch) in enumerate(streams, 1):
        print(f"\n== [{s_idx}/{len(streams)}] Evaluating Stream: {ds}:{ch}", flush=True)
        try:
            train, test, y = load_stream(ds, ch)
        except Exception as e:
            print(f"   [Error loading {ds}:{ch}: {e}]", flush=True)
            continue

        n_tot = len(train)
        n_val = int(n_tot * val_frac)
        tr, va = train[: n_tot - n_val], train[n_tot - n_val :]

        # Z-score standardize using strictly the training split statistics
        mu = tr.mean(axis=0, keepdims=True)
        sigma = tr.std(axis=0, keepdims=True)
        sigma = np.where(sigma < 1e-6, 1.0, sigma)
        tr = (tr - mu) / sigma
        va = (va - mu) / sigma
        test = (test - mu) / sigma

        k = train.shape[-1]
        print(f"   dim={k}, train={len(tr)}, val={len(va)}, test={len(test)}, anom_rate={y.mean():.4f}", flush=True)

        tr_w = np.array([tr[i : i + W] for i in range(len(tr) - W + 1)], dtype=np.float32)
        va_w = np.array([va[i : i + W] for i in range(len(va) - W + 1)], dtype=np.float32)
        te_w = np.array([test[i : i + W] for i in range(len(test) - W + 1)], dtype=np.float32)

        if len(va_w) < 50:
            print("   [Skip] Held-out validation block too short.", flush=True)
            continue

        for seed in seeds:
            for m_name in active_models:
                torch.manual_seed(seed)
                np.random.seed(seed)

                print(f"   --> Arm: {m_name} (seed={seed})...", end=" ", flush=True)
                score_fn = train_ablation_model(m_name, tr_w, k, device, epochs)
                vws, tws = score_fn(va_w), score_fn(te_w)

                for mapping in ("smear", "trailing"):
                    vp, vm = aggregate_window_scores(vws, len(va), C, S, step=1, reducer="mean", mapping_method=mapping)
                    tp, tm = aggregate_window_scores(tws, len(test), C, S, step=1, reducer="mean", mapping_method=mapping)

                    v = vp[vm]
                    half = len(v) // 2
                    cal = v[:half]
                    eval_nom = v[half + S :]

                    if len(cal) < 30 or len(eval_nom) < 30:
                        continue

                    th_dict, kurt = compute_thresholds(cal)
                    te_sc = tp[tm]
                    te_y = y[tm]

                    pr_auc = float(average_precision_score(te_y, te_sc)) if len(np.unique(te_y)) > 1 else float("nan")
                    auroc = float(roc_auc_score(te_y, te_sc)) if len(np.unique(te_y)) > 1 else float("nan")

                    for calib_name, t_val in th_dict.items():
                        ho_pred = (eval_nom > t_val).astype(int)
                        ho_fpr = float(ho_pred.mean())
                        ci_lo, ci_hi = block_bootstrap_ci(ho_pred)

                        te_pred = (te_sc > t_val).astype(int)
                        point_m = compute_metrics(te_y, te_pred)
                        ev_rec, ev_delay = event_stats(te_y, te_pred)

                        row = {
                            "dataset": ds,
                            "channel": ch,
                            "seed": seed,
                            "model": m_name,
                            "mapping": mapping,
                            "calibrator": calib_name,
                            "threshold": t_val,
                            "excess_kurtosis": kurt,
                            "heldout_fpr": ho_fpr,
                            "heldout_fpr_ci_low": ci_lo,
                            "heldout_fpr_ci_high": ci_hi,
                            "pr_auc": pr_auc,
                            "auroc": auroc,
                            "point_f1": point_m.get("f1", 0.0),
                            "point_precision": point_m.get("precision", 0.0),
                            "point_recall": point_m.get("recall", 0.0),
                            "event_recall": ev_rec,
                            "event_delay": ev_delay,
                        }
                        out_rows.append(row)

                print("Done.", flush=True)

        df_out = pd.DataFrame(out_rows).drop_duplicates(
            subset=["dataset", "channel", "seed", "model", "mapping", "calibrator"],
            keep="last",
        )
        df_out.to_csv(out_csv, index=False)
        print(f"   [Checkpointed {len(df_out)} unique results to {out_csv}]", flush=True)

    print(f"\nAll ablation evaluations completed. Output saved to: {out_csv}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Operator-Entropy JEPA Ablation Study Runner.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per model")
    parser.add_argument("--val-frac", type=float, default=0.35, help="Fraction of training sequence for held-out validation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123], help="Random seeds")
    parser.add_argument("--models", type=str, nargs="+", default=None, help="Subset of ablation models to run")
    parser.add_argument("--out", type=str, default="reports/opentropy_ablation_study.csv", help="Output CSV path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")

    args = parser.parse_args()
    dev = torch.device(args.device)
    out_path = ROOT / args.out
    active_models = args.models if args.models is not None else ABLATION_MODELS

    print("Starting Operator-Entropy Ablation Study:")
    print(f"  Models: {active_models}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Streams: {len(DEFAULT_STREAMS)}")
    print(f"  Device: {dev}")
    print(f"  Output CSV: {out_path}")

    run_ablation(DEFAULT_STREAMS, args.seeds, dev, args.epochs, args.val_frac, out_path, models=active_models)


if __name__ == "__main__":
    main()
