#!/usr/bin/env python3
"""Exhaustive Tier-2 Held-Out Calibration Benchmark (All 141 Qualifying Multivariate Streams).

Evaluates 100% of streams across the 12 multivariate telemetry datasets in mTSBench_data/
that satisfy the objective held-out calibration requirement (N_cal >= 100 nominal calibration
windows after the 64-step isolation gap and >= 1 test anomaly):
  - SMAP (47/47), Daphnet (22/22), MSL (18/18), SMD (18/18), Exathlon (17/17),
    OPPORTUNITY (13/13), GECCO (1/1), Genesis (1/1), PSM (1/1), creditcard (1/1),
    metro (1/1), swan (1/1).

Includes the 224-Bit Multi-Perspective Circuit-Breaker Panel (`Speculative-224Bit-Breaker`
and `Speculative-224Bit-Strict`), which eliminates premature 1D scalar score collapse and
SPOT scalar thresholding by monitoring 224 binary perspective switches + 32 fast right-edge
onset relays directly on the latent coordinates.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CSMConfig
from src.data.data_loader import DataLoader
from src.engine.trainer import build_encoder, build_ts_jepa_model
from src.models.jepa.alien_jepas import SpeculativeJEPAModel
from src.scoring.circuit_breaker_panel import (
    MultiPerspectiveCircuitBreakerPanel,
    extract_multihorizon_4subblocks,
    stabilize_whitened_subblocks,
)
from scripts.run_multiseed_heldout_benchmark import (
    C,
    S,
    W,
    aggregate_window_scores,
    batches,
    block_bootstrap_ci,
    compute_metrics,
    event_stats,
    load_stream,
    thresholds,
)


def atomic_save_csv(df: pd.DataFrame, out_path: Path) -> None:
    """Write DataFrame atomically via PID-scoped temp file (`[INV-800LOC]` SOP 3)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(f".tmp.{os.getpid()}.csv")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)


@torch.no_grad()
def evaluate_jepa_validation(
    m: nn.Module, val_w: np.ndarray, cfg: CSMConfig, device: torch.device, is_speculative: bool
) -> tuple[float, float]:
    """Compute held-out nominal (val_loss, val_cos_err) without touching hold or test splits."""
    m.eval()
    total_loss, total_cos, total_n = 0.0, 0.0, 0
    for i in range(0, len(val_w), 64):
        b = torch.from_numpy(np.ascontiguousarray(val_w[i : i + 64])).float().to(device)
        ctx, tgt = b[:, :C], b[:, C:]
        loss, _ = m.compute_objective(ctx, tgt, cfg)
        if is_speculative:
            _, z_targets, z_drafts = m.forward(ctx, tgt)
            cos_sim = F.cosine_similarity(z_drafts, z_targets, dim=-1).mean()
        else:
            _, z_tgt, z_pred = m.forward(ctx, tgt)[:3]
            cos_sim = F.cosine_similarity(z_pred, z_tgt, dim=-1).mean()
        n = len(b)
        total_loss += float(loss.item()) * n
        total_cos += float((1.0 - cos_sim).item()) * n
        total_n += n
    denom = max(total_n, 1)
    return total_loss / denom, total_cos / denom


def train_jepa_with_val_checkpointing(
    m: nn.Module,
    model_label: str,
    tr_w: np.ndarray,
    val_w: np.ndarray,
    cfg: CSMConfig,
    device: torch.device,
    epochs: int,
    patience: int,
    is_speculative: bool,
    verbose_epochs: bool = True,
) -> dict:
    """Train JEPA with CosineAnnealingLR, per-epoch val tracking, early stopping, and best_state restore."""
    opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=1e-5)

    best_score = float("inf")
    best_ep = 1
    best_val_loss = float("inf")
    best_val_cos = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    no_improve = 0
    val_cos_hist = []

    for ep in range(1, epochs + 1):
        m.train()
        tr_loss_sum, tr_n = 0.0, 0
        for b in batches(tr_w, 16):
            b = b.to(device)
            loss, _ = m.compute_objective(b[:, :C], b[:, C:], cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
            opt.step()
            m.update_target_encoder(None)
            tr_loss_sum += float(loss.item()) * len(b)
            tr_n += len(b)
        sched.step()

        tr_loss = tr_loss_sum / max(tr_n, 1)
        val_loss, val_cos = evaluate_jepa_validation(m, val_w, cfg, device, is_speculative=is_speculative)
        val_cos_hist.append(val_cos)

        sel_metric = val_cos + 0.25 * val_loss
        improved = (ep == 1) or (sel_metric < best_score - 1e-4)
        if improved:
            best_score = sel_metric
            best_ep = ep
            best_val_loss = val_loss
            best_val_cos = val_cos
            best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            no_improve = 0
            mark = "[BEST *]"
        else:
            no_improve += 1
            mark = f"[patience {no_improve}/{patience}]" if patience > 0 else ""

        if verbose_epochs:
            lr_now = sched.get_last_lr()[0]
            print(
                f"      [{model_label}] ep {ep:02d}/{epochs:02d} (lr={lr_now:.1e}): "
                f"train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_cos_err={val_cos:.4f} {mark}",
                flush=True,
            )

        if patience > 0 and ep >= 6 and no_improve >= patience:
            if verbose_epochs:
                print(
                    f"      [{model_label}] early stop at ep {ep:02d} -> restoring best checkpoint from ep {best_ep:02d} "
                    f"(val_loss={best_val_loss:.4f}, val_cos_err={best_val_cos:.4f})",
                    flush=True,
                )
            break
    else:
        if verbose_epochs:
            print(
                f"      [{model_label}] finished {epochs:02d} eps -> restoring best checkpoint from ep {best_ep:02d} "
                f"(val_loss={best_val_loss:.4f}, val_cos_err={best_val_cos:.4f})",
                flush=True,
            )

    m.load_state_dict(best_state)
    return {
        "best_ep": best_ep,
        "best_val_loss": best_val_loss,
        "best_val_cos": best_val_cos,
        "init_val_cos": val_cos_hist[0] if val_cos_hist else float("nan"),
    }


def evaluate_standard_calibrators(ds, ch, seed, name, mapping, cal, hold, ts, tl, out_rows):
    ths, kurt = thresholds(cal)
    pr = float(average_precision_score(tl, ts)) if tl.sum() > 0 else float("nan")
    for cname, th in ths.items():
        ind = (hold > th).astype(float)
        lo, hi = block_bootstrap_ci(ind)
        pred = (ts > th).astype(int)
        mt = compute_metrics(tl, pred, scores=ts, valid_mask=np.ones_like(tl, dtype=bool), use_pa=False)
        er, dl = event_stats(tl, pred)
        out_rows.append(
            dict(
                dataset=ds,
                channel=ch,
                seed=seed,
                model=name,
                mapping=mapping,
                calibrator=cname,
                threshold=th,
                cal_kurtosis=kurt,
                n_cal=len(cal),
                n_hold=len(hold),
                hold_fpr=float(ind.mean()),
                hold_fpr_lo=lo,
                hold_fpr_hi=hi,
                test_fpr=float(mt["fp"] / max((tl == 0).sum(), 1)),
                f1=mt.get("f1", 0.0),
                precision=mt.get("precision", 0.0),
                recall=mt.get("recall", 0.0),
                pr_auc=pr,
                event_recall=er,
                mean_delay=dl,
            )
        )


def append_circuit_breaker_rows(
    ds: str,
    ch: str,
    seed: int,
    name: str,
    mapping: str,
    cal_cnt: np.ndarray,
    hold_pred: np.ndarray,
    test_pred: np.ndarray,
    test_cnt: np.ndarray,
    k_gate: float,
    tl: np.ndarray,
    out_rows: list[dict],
) -> None:
    """Record Zero-1D-Score / Zero-SPOT Circuit-Breaker decisions in the standard benchmark schema."""
    ind = hold_pred.astype(float)
    lo, hi = block_bootstrap_ci(ind)
    pr = float(average_precision_score(tl, test_cnt)) if tl.sum() > 0 else float("nan")
    mt = compute_metrics(tl, test_pred, scores=test_cnt, valid_mask=np.ones_like(tl, dtype=bool), use_pa=False)
    er, dl = event_stats(tl, test_pred)
    for cname in ("SPOT", "CircuitBreaker"):
        out_rows.append(
            dict(
                dataset=ds,
                channel=ch,
                seed=seed,
                model=name,
                mapping=mapping,
                calibrator=cname,
                threshold=float(k_gate),
                cal_kurtosis=0.0,
                n_cal=len(cal_cnt),
                n_hold=len(hold_pred),
                hold_fpr=float(ind.mean()),
                hold_fpr_lo=lo,
                hold_fpr_hi=hi,
                test_fpr=float(mt["fp"] / max((tl == 0).sum(), 1)),
                f1=mt.get("f1", 0.0),
                precision=mt.get("precision", 0.0),
                recall=mt.get("recall", 0.0),
                pr_auc=pr,
                event_recall=er,
                mean_delay=dl,
            )
        )


def score_ts_jepa_windows(m: nn.Module, w: np.ndarray, device: torch.device) -> np.ndarray:
    m.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(w), 128):
            ch = torch.from_numpy(np.ascontiguousarray(w[i : i + 128])).float().to(device)
            out.append(m.compute_predictive_discrepancy(ch[:, :C], ch[:, C:]).cpu().numpy())
    return np.concatenate(out)


def score_speculative_windows(m: SpeculativeJEPAModel, w: np.ndarray, device: torch.device, use_bank: bool) -> np.ndarray:
    m.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(w), 128):
            ch = torch.from_numpy(np.ascontiguousarray(w[i : i + 128])).float().to(device)
            s = m.compute_predictive_discrepancy(ch[:, :C], ch[:, C:], use_alien_bank=use_bank)
            out.append(s.cpu().numpy())
    return np.concatenate(out)


ALL_BENCHMARK_MODELS = (
    "TS-JEPA",
    "Speculative-JEPA",
    "Speculative-Alien-JEPA",
    "Speculative-224Bit-Strict",
    "Speculative-224Bit-Breaker",
)


def print_summary(df: pd.DataFrame) -> None:
    for mapping in ("trailing", "smear"):
        sub = df[(df["mapping"] == mapping) & (df["calibrator"] == "SPOT")]
        if sub.empty:
            continue
        stream_means = (
            sub.groupby(["dataset", "channel", "model"])[
                ["pr_auc", "f1", "precision", "recall", "event_recall", "hold_fpr", "test_fpr", "mean_delay"]
            ]
            .mean()
            .reset_index()
        )
        n_streams = stream_means[["dataset", "channel"]].drop_duplicates().shape[0]
        agg = stream_means.groupby("model")[
            ["pr_auc", "f1", "precision", "recall", "event_recall", "hold_fpr", "test_fpr", "mean_delay"]
        ].agg(["mean", "median"])
        print(f"\n=========================================================================================================")
        print(f"TIER-2 EXHAUSTIVE SUMMARY | MAPPING: {mapping.upper()} | N={n_streams} STREAMS (1D+SPOT vs 224-Bit Breaker)")
        print(f"=========================================================================================================")
        for mname in ALL_BENCHMARK_MODELS:
            if mname in agg.index:
                r = agg.loc[mname]
                print(
                    f"  {mname:<26} | PR-AUC: {r[('pr_auc', 'mean')]:.4f} (med {r[('pr_auc', 'median')]:.4f}) "
                    f"| F1: {r[('f1', 'mean')]:.4f} (P={r[('precision', 'mean')]:.4f}, R={r[('recall', 'mean')]:.4f}) "
                    f"| EvRec: {r[('event_recall', 'mean')]:.4f} "
                    f"| HoldFPR: {r[('hold_fpr', 'mean')]:.4f} | Delay: {r[('mean_delay', 'mean')]:.1f}"
                )
        piv_pr = stream_means.pivot(index=["dataset", "channel"], columns="model", values="pr_auc")
        piv_f1 = stream_means.pivot(index=["dataset", "channel"], columns="model", values="f1")
        for cand in ("Speculative-Alien-JEPA", "Speculative-224Bit-Strict", "Speculative-224Bit-Breaker"):
            for ref in ("TS-JEPA", "Speculative-Alien-JEPA"):
                if cand == ref or cand not in piv_f1.columns or ref not in piv_f1.columns:
                    continue
                df1 = (piv_f1[cand] - piv_f1[ref]).dropna()
                dpr = (piv_pr[cand] - piv_pr[ref]).dropna()
                nz_f1 = df1[np.abs(df1) > 1e-6]
                nz_pr = dpr[np.abs(dpr) > 1e-6]
                if len(nz_f1) >= 5 and len(nz_pr) >= 5:
                    wf1, lf1 = int((df1 > 1e-4).sum()), int((df1 < -1e-4).sum())
                    wpr, lpr = int((dpr > 1e-4).sum()), int((dpr < -1e-4).sum())
                    pf1 = wilcoxon(nz_f1).pvalue
                    ppr = wilcoxon(nz_pr).pvalue
                    print(
                        f"  -> {cand} vs {ref}: "
                        f"dF1={df1.mean():+.4f} (W/L={wf1}/{lf1}, p={pf1:.4e}) | "
                        f"dPR-AUC={dpr.mean():+.4f} (W/L={wpr}/{lpr}, p={ppr:.4e})"
                    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Tier-2 exhaustive multivariate benchmark (141 streams).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456], help="Random seeds per stream")
    ap.add_argument("--epochs", type=int, default=12, help="Maximum training epochs per model")
    ap.add_argument("--patience", type=int, default=4, help="Early-stopping patience on validation error (0=disable)")
    ap.add_argument("--val-frac", type=float, default=0.35, help="Held-out nominal fraction (cal + eval)")
    ap.add_argument("--streams", default=str(ROOT / "reports" / "streams_141_tier2.txt"), help="Stream manifest file")
    ap.add_argument(
        "--out",
        default=str(ROOT / "reports" / "heldout_tier2_141stream_circuit_breaker.csv"),
        help="Output CSV path (defaults to new circuit-breaker comparison CSV)",
    )
    ap.add_argument(
        "--baseline-csv",
        default=str(ROOT / "reports" / "heldout_tier2_141stream.csv"),
        help="Existing Tier-2 CSV to seed baseline TS-JEPA / Speculative-JEPA rows from",
    )
    ap.add_argument("--fresh", action="store_true", help="Start completely fresh (ignore any existing output CSV)")
    ap.add_argument("--quiet-epochs", action="store_true", help="Only print summary per seed instead of every epoch")
    a = ap.parse_args()

    out_path = Path(a.out)
    baseline_path = Path(a.baseline_csv)
    streams_path = Path(a.streams)
    streams = [tuple(ln.strip().split(":", 1)) for ln in streams_path.read_text().splitlines() if ln.strip()]

    existing_rows = []
    if not a.fresh and out_path.exists():
        existing_rows.extend(pd.read_csv(out_path).to_dict("records"))
    elif not a.fresh and baseline_path.exists() and out_path != baseline_path:
        base_df = pd.read_csv(baseline_path)
        existing_rows.extend(base_df.to_dict("records"))
        atomic_save_csv(base_df, out_path)
        print(
            f"Seeded {len(existing_rows)} baseline rows from {baseline_path.name} into new comparison CSV {out_path.name}.",
            flush=True,
        )

    done_keys = {(r["dataset"], str(r["channel"]), int(r["seed"]), r["model"]) for r in existing_rows}
    if existing_rows:
        print(
            f"Resuming with {len(existing_rows)} rows in {out_path.name}.",
            flush=True,
        )
    else:
        print(
            f"Starting fresh Tier-2 benchmark: {len(streams)} streams x {len(a.seeds)} seeds "
            f"(max_epochs={a.epochs}, patience={a.patience}).",
            flush=True,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panel = MultiPerspectiveCircuitBreakerPanel(context_size=C, suspect_size=S)
    rows = list(existing_rows)

    for idx, (ds, ch) in enumerate(streams, 1):
        needed_seeds = [
            sd
            for sd in a.seeds
            if not all((ds, ch, sd, m) in done_keys for m in ALL_BENCHMARK_MODELS)
        ]
        if not needed_seeds:
            print(f"[{idx:03d}/{len(streams):03d}] == {ds}:{ch} (already completed in {out_path.name}, skipping)", flush=True)
            continue

        try:
            train, test, y = load_stream(ds, ch)
            n_val = int(len(train) * a.val_frac)
            tr, va = train[:-n_val], train[-n_val:]
            mu, sd_arr = tr.mean(0), tr.std(0)
            sc = np.where(sd_arr < 1e-8, 1.0, np.maximum(sd_arr, 0.01))
            tr_s, va_s, te_s = (tr - mu) / sc, (va - mu) / sc, (test - mu) / sc
            tr_w = DataLoader.create_windows(tr_s, W, step=16, copy=False)
            va_w = DataLoader.create_windows(va_s, W, step=1, copy=False)
            te_w = DataLoader.create_windows(te_s, W, step=1, copy=False)
            cal_block_s = va_s[: len(va_s) // 2]
            val_mon_w = DataLoader.create_windows(
                cal_block_s, W, step=max(1, min(8, (len(cal_block_s) - W) // 32)), copy=False
            )
            if len(val_mon_w) == 0:
                val_mon_w = va_w[: max(1, len(va_w) // 2) : 8]

            k = train.shape[1]
            print(
                f"[{idx:03d}/{len(streams):03d}] == {ds}:{ch} (K={k}) train={len(tr)} val={len(va)} test={len(test)} anom={y.mean():.4f}",
                flush=True,
            )

            for seed in needed_seeds:
                # 1. Baseline TS-JEPA with validation checkpointing (if not already seeded)
                if (ds, ch, seed, "TS-JEPA") not in done_keys:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    cfg_ts = CSMConfig(model_type="ts_jepa", context_size=C, suspect_size=S, latent_dim=32)
                    m_ts = build_ts_jepa_model(cfg_ts, input_dim=k, device=device)
                    train_jepa_with_val_checkpointing(
                        m_ts,
                        f"TS-JEPA s={seed}",
                        tr_w,
                        val_mon_w,
                        cfg_ts,
                        device,
                        a.epochs,
                        a.patience,
                        is_speculative=False,
                        verbose_epochs=not a.quiet_epochs,
                    )
                    vws, tws = score_ts_jepa_windows(m_ts, va_w, device), score_ts_jepa_windows(m_ts, te_w, device)
                    for mapping in ("smear", "trailing"):
                        vp, vm = aggregate_window_scores(vws, len(va), C, S, step=1, reducer="mean", mapping_method=mapping)
                        tp, tm = aggregate_window_scores(tws, len(test), C, S, step=1, reducer="mean", mapping_method=mapping)
                        v = vp[vm]
                        half = len(v) // 2
                        cal, hold = v[:half], v[half + S :]
                        if len(cal) >= 100 and len(hold) >= 100:
                            evaluate_standard_calibrators(ds, ch, seed, "TS-JEPA", mapping, cal, hold, tp[tm], y[tm], rows)
                    done_keys.add((ds, ch, seed, "TS-JEPA"))

                # 2. Speculative-JEPA, Speculative-Alien-JEPA, and 224-Bit Circuit-Breaker Panel
                spec_models = (
                    "Speculative-JEPA",
                    "Speculative-Alien-JEPA",
                    "Speculative-224Bit-Strict",
                    "Speculative-224Bit-Breaker",
                )
                if any((ds, ch, seed, m) not in done_keys for m in spec_models):
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    cfg_sp = CSMConfig(model_type="ts_jepa", context_size=C, suspect_size=S, latent_dim=32)
                    encoder = build_encoder(cfg_sp, k, device)
                    m_sp = SpeculativeJEPAModel(
                        context_encoder=encoder,
                        latent_dim=cfg_sp.latent_dim,
                        predictor_hidden_dim=64,
                        ema_decay=0.996,
                        dropout=cfg_sp.dropout,
                    ).to(device)
                    train_jepa_with_val_checkpointing(
                        m_sp,
                        f"Spec-JEPA s={seed}",
                        tr_w,
                        val_mon_w,
                        cfg_sp,
                        device,
                        a.epochs,
                        a.patience,
                        is_speculative=True,
                        verbose_epochs=not a.quiet_epochs,
                    )
                    m_sp.fit_alien_bank(tr_w, context_len=C)

                    for mname, use_bank in [("Speculative-JEPA", False), ("Speculative-Alien-JEPA", True)]:
                        if (ds, ch, seed, mname) in done_keys:
                            continue
                        vws = score_speculative_windows(m_sp, va_w, device, use_bank=use_bank)
                        tws = score_speculative_windows(m_sp, te_w, device, use_bank=use_bank)
                        for mapping in ("smear", "trailing"):
                            vp, vm = aggregate_window_scores(vws, len(va), C, S, step=1, reducer="mean", mapping_method=mapping)
                            tp, tm = aggregate_window_scores(tws, len(test), C, S, step=1, reducer="mean", mapping_method=mapping)
                            v = vp[vm]
                            half = len(v) // 2
                            cal, hold = v[:half], v[half + S :]
                            if len(cal) >= 100 and len(hold) >= 100:
                                evaluate_standard_calibrators(ds, ch, seed, mname, mapping, cal, hold, tp[tm], y[tm], rows)
                        done_keys.add((ds, ch, seed, mname))

                    # Evaluate 224-Bit Multi-Perspective Circuit-Breaker Panel (Zero 1D Score, Zero SPOT)
                    if (
                        (ds, ch, seed, "Speculative-224Bit-Strict") not in done_keys
                        or (ds, ch, seed, "Speculative-224Bit-Breaker") not in done_keys
                    ):
                        va_raw4, va_white4 = extract_multihorizon_4subblocks(m_sp, va_w, C, device)
                        te_raw4, te_white4 = extract_multihorizon_4subblocks(m_sp, te_w, C, device)
                        n_cal_w = len(va_raw4) // 2
                        va_ew, te_ew = stabilize_whitened_subblocks(va_white4, te_white4, va_raw4, te_raw4, n_cal_w)
                        panel_bits = panel.calibrate_and_trip_windows(va_ew, te_ew, n_cal_w)

                        for mapping in ("smear", "trailing"):
                            dec = panel.evaluate_stream_decisions(panel_bits, len(va), len(test), mapping)
                            tl = y[dec.valid_mask]
                            if len(dec.cal_switch_counts) >= 100 and len(dec.hold_switch_counts) >= 100:
                                if (ds, ch, seed, "Speculative-224Bit-Strict") not in done_keys:
                                    append_circuit_breaker_rows(
                                        ds,
                                        ch,
                                        seed,
                                        "Speculative-224Bit-Strict",
                                        mapping,
                                        dec.cal_switch_counts,
                                        dec.hold_pred_strict,
                                        dec.test_pred_strict,
                                        dec.test_switch_counts,
                                        dec.k_gate_strict,
                                        tl,
                                        rows,
                                    )
                                if (ds, ch, seed, "Speculative-224Bit-Breaker") not in done_keys:
                                    append_circuit_breaker_rows(
                                        ds,
                                        ch,
                                        seed,
                                        "Speculative-224Bit-Breaker",
                                        mapping,
                                        dec.cal_switch_counts,
                                        dec.hold_pred_balanced,
                                        dec.test_pred_balanced,
                                        dec.test_switch_counts,
                                        dec.k_gate_balanced,
                                        tl,
                                        rows,
                                    )
                        done_keys.add((ds, ch, seed, "Speculative-224Bit-Strict"))
                        done_keys.add((ds, ch, seed, "Speculative-224Bit-Breaker"))

                print(f"   seed={seed} completed (saved to {out_path.name})", flush=True)
                atomic_save_csv(pd.DataFrame(rows), out_path)

        except Exception as exc:
            print(f"   FAILED {ds}:{ch}: {type(exc).__name__}: {exc}", flush=True)

    df_final = pd.DataFrame(rows)
    atomic_save_csv(df_final, out_path)
    print(f"\nSaved {len(df_final)} rows to {out_path}", flush=True)
    print_summary(df_final)


if __name__ == "__main__":
    main()
