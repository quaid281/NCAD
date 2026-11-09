#!/usr/bin/env python3
"""Evaluate the 224-bit circuit-breaker architecture across prediction horizons S in {1, 4, 16, 32, 64}.

Addresses Reviewer Concern 3:
- Evaluates SolitonReynoldsJEPAModel with:
  (1) 1D Scalar SPOT scoring,
  (2) 224-Bit Multi-Perspective Circuit Breaker, and
  (3) Laminar-JEPA (Surgical Weight-Space Quenching + 224-Bit Breaker)
  at prediction horizons S in {1, 4, 16, 32, 64} (context C = 256) across the 5 horizon-sweep
  streams x 3 seeds {42, 123, 456}.
- Reports causal and buffered Point-F1, Precision, Recall, Event Recall, FPR, PR-AUC,
  structural reporting delay (S - 1), and empirical detection delay (Delta t).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.circuit_breaker_19stream.common import VAL_FRAC, fit_spot
from scripts.experiment_wsp_soliton import extract_raw4_white4_from_latents
from scripts.run_multiseed_heldout_benchmark import (
    batches,
    compute_metrics,
    event_stats,
    load_stream,
)
from src.config import CSMConfig
from src.data.data_loader import DataLoader
from src.engine.builder import build_ts_jepa_model
from src.experimental.phase_embedding_synthesis import _forward_predictors_with_phase
from src.models.jepa.alien_jepas import _encode_with_multidepth_taps
from src.scoring.circuit_breaker_panel import (
    MultiPerspectiveCircuitBreakerPanel,
    aggregate_vector_stream,
    stabilize_whitened_subblocks,
)

STREAMS_5 = [
    ("daphnet", "S01R01E1"),
    ("smap", "P-3"),
    ("smap", "A-1"),
    ("msl", "M-1"),
    ("smd", "machine-1-2"),
]
HORIZONS = [1, 4, 16, 32, 64]
SEEDS = [42, 123, 456]
C = 256
EPOCHS = 10
TRAIN_BS = 64


@torch.no_grad()
def fast_extract_latents(
    model: torch.nn.Module,
    windows: np.ndarray,
    context_len: int = 256,
    batch_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract (z_ctx [N, D], z_sub_tgt [N, M, D]) with large batch size on GPU."""
    model.eval()
    dev = next(model.parameters()).device
    M = int(model.num_subblocks)
    z_ctx_list, z_sub_tgt_list = [], []
    for i in range(0, len(windows), batch_size):
        ch = torch.from_numpy(np.ascontiguousarray(windows[i : i + batch_size])).float().to(dev)
        ctx_win, tgt_win = ch[:, :context_len], ch[:, context_len:]
        z_ctx = _encode_with_multidepth_taps(model.context_encoder, ctx_win, model.context_tap_head)
        s_len = tgt_win.shape[1]
        t_list = [
            _encode_with_multidepth_taps(
                model.target_encoder,
                tgt_win[:, : (s_len if m == M - 1 else max(((m + 1) * s_len) // M, 2)), :],
                model.target_tap_head,
            )
            for m in range(M)
        ]
        z_ctx_list.append(z_ctx)
        z_sub_tgt_list.append(torch.stack(t_list, dim=1))
    return torch.cat(z_ctx_list, dim=0), torch.cat(z_sub_tgt_list, dim=0)


@torch.no_grad()
def fast_latent_surgical_quench(
    model: torch.nn.Module,
    z_ctx_tr: torch.Tensor,
    z_tgt_tr: torch.Tensor,
    z_ctx_cal: torch.Tensor,
    z_tgt_cal: torch.Tensor,
    k_rogue: int = 4,
    phase_align_step: float = 0.35,
    residual_mean_blend: float = 0.75,
    off_diagonal_retain: float = 0.15,
    clean_passband_rms: float = 0.50,
) -> None:
    """Zero-gradient surgical weight-space quenching directly from pre-extracted latents."""
    model.eval()
    dev = next(model.parameters()).device
    M, D = int(model.num_subblocks), int(model.latent_dim)

    tr_raw4, _ = extract_raw4_white4_from_latents(model, z_ctx_tr, z_tgt_tr)
    cal_raw4, cal_white4 = extract_raw4_white4_from_latents(model, z_ctx_cal, z_tgt_cal)

    tr_mean = tr_raw4.mean(axis=0)  # [M, D]
    cal_mean = cal_raw4.mean(axis=0)  # [M, D]
    tr_std = tr_raw4.std(axis=0) + 1e-4
    cal_std = cal_raw4.std(axis=0) + 1e-4

    shift_z = np.mean(np.abs(cal_mean - tr_mean) / tr_std, axis=0)  # [D]
    var_infl = np.mean((cal_std / tr_std) ** 2, axis=0)  # [D]
    half = len(cal_raw4) // 2
    intra_drift = np.mean(np.abs(cal_raw4[half:].mean(axis=0) - cal_raw4[:half].mean(axis=0)) / cal_std, axis=0)
    rogue_score = shift_z + np.maximum(0.0, var_infl - 1.0) + intra_drift
    med_r = float(np.median(rogue_score))
    mad_r = float(1.4826 * np.median(np.abs(rogue_score - med_r))) + 1e-4
    rogue_z = (rogue_score - med_r) / mad_r
    rogue_sub = [int(c) for c in np.argsort(rogue_score)[::-1][:k_rogue]]

    # 1. Predictor MLP output row damping & phase offset alignment on rogue coordinates
    for c in rogue_sub:
        damp = float(np.clip(1.0 / (1.0 + 0.5 * max(0.0, float(rogue_z[c]))), 0.35, 0.85))
        model.predictor.net[4].weight.data[c, :].mul_(damp)
        model.predictor.net[4].bias.data[c].mul_(damp)
        model.predictor.net[6].weight.data[c].mul_(damp)
        model.sub_predictor.net[4].weight.data[c, :].mul_(damp)
        model.sub_predictor.net[4].bias.data[c].mul_(damp)
        model.sub_predictor.net[6].weight.data[c].mul_(damp)
        phase_corr = torch.from_numpy((cal_mean[:, c] - tr_mean[:, c]) * phase_align_step).to(dev, dtype=torch.float32)
        model.subblock_embeddings.data[:, c].add_(phase_corr)

    # 2. Residual mean alignment on rogue coordinates
    cal_raw4_post = (
        (z_tgt_cal - _forward_predictors_with_phase(model, z_ctx_cal, model.subblock_embeddings))
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    cal_mean_post = cal_raw4_post.reshape(len(cal_raw4_post), M * D).mean(axis=0)
    res_mean_new = model.residual_mean.detach().cpu().numpy().copy()
    for c in rogue_sub:
        for m_idx in range(M):
            flat_idx = m_idx * D + c
            res_mean_new[flat_idx] = (
                (1.0 - residual_mean_blend) * res_mean_new[flat_idx] + residual_mean_blend * cal_mean_post[flat_idx]
            )
    model.residual_mean.copy_(torch.from_numpy(res_mean_new).to(dev, dtype=model.residual_mean.dtype))

    # 3. Precision matrix off-diagonal decoupling and passband guard scaling
    prec = model.precision_matrix.detach().cpu().numpy().copy()
    for c in rogue_sub:
        for m1 in range(M):
            idx1 = m1 * D + c
            for m2 in range(M):
                for c2 in range(D):
                    idx2 = m2 * D + c2
                    if c2 != c:
                        prec[idx1, idx2] *= off_diagonal_retain
                        prec[idx2, idx1] *= off_diagonal_retain

    evals, evecs = np.linalg.eigh(0.5 * (prec + prec.T))
    evals = np.maximum(evals, 1e-6)
    prec_spsd = (evecs * evals[None, :]) @ evecs.T
    omega_sqrt = (evecs * np.sqrt(evals)[None, :]) @ evecs.T
    diff_cal = cal_raw4_post.reshape(len(cal_raw4_post), M * D) - res_mean_new[None, :]
    w_cal = diff_cal @ omega_sqrt
    rms_cal = np.sqrt(np.mean(w_cal ** 2, axis=0)) + 1e-5

    target_rms = np.full(M * D, clean_passband_rms, dtype=np.float64)
    for c in rogue_sub:
        g_factor = float(np.clip(1.32 + 0.04 * max(0.0, float(rogue_z[c]) - 0.5), 1.30, 1.44))
        for m_idx in range(M):
            target_rms[m_idx * D + c] = clean_passband_rms * 1.50 * g_factor

    scale_vec = target_rms / rms_cal
    prec_programmed = (prec_spsd * scale_vec[:, None]) * scale_vec[None, :]
    model.precision_matrix.copy_(torch.from_numpy(prec_programmed).to(dev, dtype=model.precision_matrix.dtype))


def eval_1d_spot_from_latents(
    va_white4: np.ndarray,
    te_white4: np.ndarray,
    len_va: int,
    len_test: int,
    y_test: np.ndarray,
    s_len: int,
) -> dict[str, dict[str, float]]:
    """Evaluate 1D scalar energy score with standard SPOT (u=0.98, q=1e-3)."""
    va_1d = np.mean(va_white4 ** 2, axis=(1, 2))
    te_1d = np.mean(te_white4 ** 2, axis=(1, 2))
    n_cal_w = len(va_1d) // 2
    out: dict[str, dict[str, float]] = {}
    for mapping in ("trailing", "smear"):
        va_tl_2d, va_mask = aggregate_vector_stream(va_1d[:, None], len_va, C, s_len, mapping)
        te_tl_2d, te_mask = aggregate_vector_stream(te_1d[:, None], len_test, C, s_len, mapping)
        va_tl, te_tl = va_tl_2d[:, 0], te_tl_2d[:, 0]
        cal_end = C + n_cal_w if mapping == "trailing" else C + n_cal_w + s_len - 1
        cal_mask = np.zeros(len_va, dtype=bool)
        cal_mask[C : min(len_va, cal_end)] = True
        cal_scores = va_tl[va_mask & cal_mask]
        thr = fit_spot(cal_scores)
        preds = (te_tl[te_mask] > thr).astype(np.int64)
        tl = y_test[te_mask]
        mt = compute_metrics(
            tl,
            preds,
            scores=te_tl[te_mask],
            valid_mask=np.ones_like(tl, dtype=bool),
            use_pa=False,
        )
        er, delay = event_stats(tl, preds)
        tfpr = float(preds[tl == 0].mean()) if np.any(tl == 0) else 0.0
        out[mapping] = {
            "f1": float(mt["f1"]),
            "precision": float(mt["precision"]),
            "recall": float(mt["recall"]),
            "event_recall": float(er),
            "fpr": tfpr,
            "pr_auc": float(mt["pr_auc"]),
            "roc_auc": float(mt["roc_auc"]),
            "delay": float(delay),
        }
    return out


def eval_224_from_latents(
    model: torch.nn.Module,
    z_ctx_va: torch.Tensor,
    z_tgt_va: torch.Tensor,
    z_ctx_te: torch.Tensor,
    z_tgt_te: torch.Tensor,
    len_va: int,
    len_test: int,
    y_test: np.ndarray,
    s_len: int,
    apply_eq3_blend: bool = False,
) -> tuple[dict[str, dict[str, float]], np.ndarray, np.ndarray]:
    """Evaluate 224-Bit Circuit-Breaker Panel at arbitrary horizon s_len."""
    panel = MultiPerspectiveCircuitBreakerPanel(context_size=C, suspect_size=s_len)
    va_raw4, va_white4 = extract_raw4_white4_from_latents(model, z_ctx_va, z_tgt_va)
    te_raw4, te_white4 = extract_raw4_white4_from_latents(model, z_ctx_te, z_tgt_te)
    n_cal_w = len(va_raw4) // 2
    if apply_eq3_blend:
        va_ew, te_ew = stabilize_whitened_subblocks(va_white4, te_white4, va_raw4, te_raw4, n_cal_w)
    else:
        va_ew = np.clip(np.nan_to_num(va_white4.astype(np.float64), nan=0.0), -100.0, 100.0)
        te_ew = np.clip(np.nan_to_num(te_white4.astype(np.float64), nan=0.0), -100.0, 100.0)

    bits = panel.calibrate_and_trip_windows(va_ew, te_ew, n_cal_w)
    out: dict[str, dict[str, float]] = {}
    for mapping in ("trailing", "smear"):
        dec = panel.evaluate_stream_decisions(bits, len_va, len_test, mapping)
        tl = y_test[dec.valid_mask]
        mt = compute_metrics(
            tl,
            dec.test_pred_balanced,
            scores=dec.test_switch_counts,
            valid_mask=np.ones_like(tl, dtype=bool),
            use_pa=False,
        )
        er, delay = event_stats(tl, dec.test_pred_balanced)
        tfpr = float(dec.test_pred_balanced[tl == 0].mean()) if np.any(tl == 0) else 0.0
        out[mapping] = {
            "f1": float(mt["f1"]),
            "precision": float(mt["precision"]),
            "recall": float(mt["recall"]),
            "event_recall": float(er),
            "fpr": tfpr,
            "pr_auc": float(mt["pr_auc"]),
            "roc_auc": float(mt["roc_auc"]),
            "delay": float(delay),
        }
    return out, va_white4, te_white4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports" / "horizon_224bit_sweep.csv"))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, object]] = []
    t0 = time.time()

    for ds, ch in STREAMS_5:
        train, test, y_test = load_stream(ds, ch)
        n_val = int(len(train) * VAL_FRAC)
        tr, va = train[:-n_val], train[-n_val:]
        mu, sd = tr.mean(0), tr.std(0)
        sc = np.where(sd < 1e-8, 1.0, np.maximum(sd, 0.01))
        tr_norm = (tr - mu) / sc
        va_norm = (va - mu) / sc
        te_norm = (test - mu) / sc

        for s_len in HORIZONS:
            w_len = C + s_len
            tr_w = DataLoader.create_windows(tr_norm, w_len, step=16, copy=False)
            va_w = DataLoader.create_windows(va_norm, w_len, step=1, copy=False)
            te_w = DataLoader.create_windows(te_norm, w_len, step=1, copy=False)
            n_cal_w = len(va_w) // 2

            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)
                cfg = CSMConfig(model_type="soliton_reynolds_jepa", context_size=C, suspect_size=s_len, latent_dim=32)
                model = build_ts_jepa_model(cfg, input_dim=train.shape[1], device=device)
                opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                for _ in range(EPOCHS):
                    model.train()
                    for b in batches(tr_w, TRAIN_BS):
                        b = b.to(device)
                        loss, _ = model.compute_objective(b[:, :C], b[:, C:], cfg)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                        model.update_target_encoder(None)
                model.fit_alien_bank(tr_w, context_len=C, batch_size=512)

                z_ctx_tr, z_tgt_tr = fast_extract_latents(model, tr_w, C, batch_size=512)
                z_ctx_va, z_tgt_va = fast_extract_latents(model, va_w, C, batch_size=512)
                z_ctx_te, z_tgt_te = fast_extract_latents(model, te_w, C, batch_size=512)

                # 2. Base 224-Bit Circuit Breaker
                res_224, va_w4, te_w4 = eval_224_from_latents(
                    model, z_ctx_va, z_tgt_va, z_ctx_te, z_tgt_te, len(va), len(test), y_test, s_len, apply_eq3_blend=False
                )
                # 1. 1D Scalar SPOT
                res_1d = eval_1d_spot_from_latents(va_w4, te_w4, len(va), len(test), y_test, s_len)
                # 3. Laminar-JEPA (Surgical Weight-Space Quenching + 224-Bit Breaker)
                fast_latent_surgical_quench(model, z_ctx_tr, z_tgt_tr, z_ctx_va[:n_cal_w], z_tgt_va[:n_cal_w])
                res_wsp, _, _ = eval_224_from_latents(
                    model, z_ctx_va, z_tgt_va, z_ctx_te, z_tgt_te, len(va), len(test), y_test, s_len, apply_eq3_blend=True
                )

                for variant_name, res_dict in [
                    ("1D-Scalar-SPOT", res_1d),
                    ("Reynolds-224Bit", res_224),
                    ("Laminar-JEPA-224Bit", res_wsp),
                ]:
                    for mapping in ("trailing", "smear"):
                        m = res_dict[mapping]
                        rows.append(
                            {
                                "dataset": ds,
                                "stream": f"{ds}:{ch}",
                                "horizon_S": s_len,
                                "structural_delay": s_len - 1 if mapping == "trailing" else 0,
                                "seed": seed,
                                "variant": variant_name,
                                "mapping": mapping,
                                **m,
                            }
                        )
            print(f"[{time.time() - t0:5.1f}s] {ds}:{ch} S={s_len:2d} done", flush=True)

    df = pd.DataFrame(rows)
    tmp_path = out_path.with_suffix(".tmp.csv")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(out_path)

    print("\n=== HORIZON SWEEP SUMMARY (5 streams x 3 seeds, stream-averaged) ===")
    stream_avg = (
        df.groupby(["variant", "mapping", "horizon_S", "stream"])[
            ["f1", "precision", "recall", "event_recall", "fpr", "pr_auc", "delay"]
        ]
        .mean()
        .reset_index()
    )
    summary = (
        stream_avg.groupby(["variant", "mapping", "horizon_S"])[
            ["f1", "precision", "recall", "event_recall", "fpr", "pr_auc", "delay"]
        ]
        .mean()
        .round(4)
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
