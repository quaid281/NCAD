#!/usr/bin/env python3
"""Factorial and Strength Sweep of the Operator-Entropy Regularizer (Table II(c)).

Crosses the spectral-entropy deficiency term (Delta H) and the Bregman energy
barrier (Phi_energy) and sweeps lambda_ent in {0.05, 0.10, 0.50, 1.00, 2.00}
on 6 streams at 3 seeds (42, 123, 456) under the held-out calibration protocol
(val_frac=0.35, SPOT u=0.98, q=1e-3, 64-step isolation gap, buffered 'smear'
and strictly causal 'trailing' attribution).

Output: reports/opentropy_factorial_strength_ablation.csv
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CSMConfig
from src.data.data_loader import DataLoader
from src.engine.trainer import build_ts_jepa_model
from src.models.jepa.operator_entropy_jepa import compute_empirical_covariance
from src.scoring.event_fusion import _extract_segments, aggregate_window_scores, compute_metrics
from src.scoring.evt_calibrator import EVTCalibrator

C, S = 256, 64
W = C + S

STREAMS: List[Tuple[str, str]] = [
    ("SMAP", "P-3"),
    ("SMAP", "A-1"),
    ("MSL", "M-1"),
    ("MSL", "C-1"),
    ("Daphnet", "S01R01E1"),
    ("Genesis", "default"),
]

SEEDS = [42, 123, 456]

# (arm_name, alpha_entropy, entropy_weight, beta_norm)
ARMS = [
    ("Neither (Unconstrained K)", 0.0, 0.0, 0.0),
    ("Entropy-Only", 0.5, 1.0, 0.0),
    ("Barrier-Only", 0.5, 0.0, 1.0),
    ("Both (lambda=0.05)", 0.05, 1.0, 1.0),
    ("Both (lambda=0.10)", 0.10, 1.0, 1.0),
    ("Both (lambda=0.50, OpEntropy)", 0.50, 1.0, 1.0),
    ("Both (lambda=1.00)", 1.00, 1.0, 1.0),
    ("Both (lambda=2.00)", 2.00, 1.0, 1.0),
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
        np.nan_to_num(a[cols].to_numpy(np.float32), nan=0.0),
        np.nan_to_num(b[cols].to_numpy(np.float32), nan=0.0),
        y,
    )


def event_stats(y: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    segs = _extract_segments(y)
    if not segs:
        return float("nan"), float("nan")
    delays = []
    for gs, ge in segs:
        hit = np.where(pred[gs:ge] > 0)[0]
        if len(hit):
            delays.append(hit[0])
    return len(delays) / len(segs), (float(np.mean(delays)) if delays else float("nan"))


def operator_regularization(
    K: torch.Tensor,
    entropy_weight: float,
    beta_norm: float,
    gamma_norm: float = 1.0,
    eps: float = 1e-6,
    eps_log: float = 1e-12,
) -> torch.Tensor:
    Kf = K.float()
    D = Kf.shape[-1]
    gram = torch.matmul(Kf, Kf.transpose(-2, -1))
    gram = 0.5 * (gram + gram.transpose(-2, -1))
    eigvals = torch.linalg.eigvalsh(gram)
    s_sq = eigvals + (torch.clamp(eigvals, min=0.0) - eigvals).detach()

    total_energy = torch.sum(s_sq, dim=-1, keepdim=True)
    s_sq_smooth = s_sq + eps_log
    p = s_sq_smooth / torch.sum(s_sq_smooth, dim=-1, keepdim=True)
    spectral_entropy = -torch.sum(p * torch.log(p), dim=-1)
    entropy_deficiency = torch.clamp(math.log(max(D, 2)) - spectral_entropy, min=0.0)

    target_energy = gamma_norm * D
    energy_ratio = total_energy.squeeze(-1) / max(target_energy, eps)
    energy_barrier = torch.clamp(
        energy_ratio - 1.0 - torch.log(energy_ratio + eps) + math.log(1.0 + eps),
        min=0.0,
    )
    return torch.mean(entropy_weight * entropy_deficiency + beta_norm * energy_barrier)


def rankme_from_cov(cov: torch.Tensor, eps: float = 1e-12) -> float:
    eig = torch.linalg.eigvalsh(0.5 * (cov.double() + cov.double().T))
    eig = torch.clamp(eig, min=0.0)
    s = eig.sum()
    if s <= eps:
        return 1.0
    p = (eig + eps) / (s + eig.numel() * eps)
    h = -torch.sum(p * torch.log(p)).item()
    return float(math.exp(h))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_csv = ROOT / "reports" / "opentropy_factorial_strength_ablation.csv"
    epochs = 5
    val_frac = 0.35
    rows = []

    print(f"Running Operator-Entropy Factorial & Strength Sweep on {device}...")
    for ds, ch in STREAMS:
        train, test, y = load_stream(ds, ch)
        n_val = int(len(train) * val_frac)
        tr, va = train[:-n_val], train[-n_val:]
        mu = tr.mean(axis=0, keepdims=True)
        sd = tr.std(axis=0, keepdims=True)
        sc = np.where(sd < 1e-6, 1.0, sd)
        tr_s, va_s, te_s = (tr - mu) / sc, (va - mu) / sc, (test - mu) / sc
        tr_w = DataLoader.create_windows(tr_s, W, step=4, copy=False)
        va_w = DataLoader.create_windows(va_s, W, step=1, copy=False)
        te_w = DataLoader.create_windows(te_s, W, step=1, copy=False)
        k = train.shape[1]
        bs = min(64, len(tr_w))
        print(f"\n== {ds}:{ch} (K={k}, tr_win={len(tr_w)}, va_win={len(va_w)}, te_win={len(te_w)}) ==", flush=True)

        for seed in SEEDS:
            for arm_name, alpha_ent, ent_w, beta_n in ARMS:
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                cfg = CSMConfig(
                    model_type="operator_entropy_jepa",
                    context_size=C,
                    suspect_size=S,
                    latent_dim=32,
                    alpha_entropy=alpha_ent,
                    alpha_rep_entropy=0.0,
                    beta_lyapunov=0.02,
                )
                m = build_ts_jepa_model(cfg, input_dim=k, device=device)
                opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-5)

                batch_losses = []
                grad_norms = []
                for ep in range(epochs):
                    m.train()
                    perm = np.random.permutation(len(tr_w))
                    for i in range(0, len(perm), bs):
                        b = torch.from_numpy(np.ascontiguousarray(tr_w[perm[i : i + bs]])).float().to(device)
                        x_ctx, x_tgt = b[:, :C], b[:, C:]
                        # Base objective with alpha_entropy=0.0 so we can inject exact factorial (ent_w, beta_n)
                        base_loss, _ = m.compute_objective(x_ctx, x_tgt, config=cfg, alpha_entropy=0.0)
                        if alpha_ent > 0.0 and (ent_w > 0.0 or beta_n > 0.0):
                            reg_k = operator_regularization(m.K, entropy_weight=ent_w, beta_norm=beta_n)
                            loss = base_loss + alpha_ent * reg_k
                        else:
                            loss = base_loss

                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        gn = float(nn.utils.clip_grad_norm_(m.parameters(), 2.0).item())
                        grad_norms.append(gn)
                        batch_losses.append(float(loss.detach().item()))
                        opt.step()
                        m.update_target_encoder()

                m.eval()
                with torch.no_grad():
                    svals = torch.linalg.svdvals(m.K.detach().float())
                    s_sq = svals ** 2
                    p_k = (s_sq + 1e-12) / (s_sq.sum() + 32 * 1e-12)
                    op_rankme = float(torch.exp(-torch.sum(p_k * torch.log(p_k))).item())
                    op_sigma_max = float(svals[0].item())
                    op_sigma_min = float(svals[-1].item())
                    op_cond_num = op_sigma_max / max(op_sigma_min, 1e-8)
                    op_frob_ratio = float((s_sq.sum() / 32.0).item())

                    z_ctx_list, z_tgt_list = [], []
                    for i in range(0, min(len(va_w), 512), 128):
                        ch_b = torch.from_numpy(np.ascontiguousarray(va_w[i : i + 128])).float().to(device)
                        z_ctx_list.append(m._apply_filters(m.context_encoder(ch_b[:, :C])).cpu())
                        z_tgt_list.append(m._apply_filters(m.target_encoder(ch_b[:, C:])).cpu())
                    zc_all = torch.cat(z_ctx_list, dim=0)
                    zt_all = torch.cat(z_tgt_list, dim=0)
                    cov_zc = compute_empirical_covariance(zc_all)
                    cov_zt = compute_empirical_covariance(zt_all)
                    rep_ctx_rankme = rankme_from_cov(cov_zc)
                    rep_tgt_rankme = rankme_from_cov(cov_zt)
                    rep_tgt_var = float(torch.trace(cov_zt).item())

                loss_arr = np.array(batch_losses, dtype=np.float64)
                loss_cv = float(loss_arr.std() / max(abs(loss_arr.mean()), 1e-8))
                mean_grad_norm = float(np.mean(grad_norms))

                def score_fn(w_arr: np.ndarray) -> np.ndarray:
                    out = []
                    with torch.no_grad():
                        for i in range(0, len(w_arr), 128):
                            ch_b = torch.from_numpy(np.ascontiguousarray(w_arr[i : i + 128])).float().to(device)
                            out.append(m.compute_predictive_discrepancy(ch_b[:, :C], ch_b[:, C:]).cpu().numpy())
                    return np.concatenate(out)

                vws = score_fn(va_w)
                tws = score_fn(te_w)

                for mapping in ("trailing", "smear"):
                    vp, vm = aggregate_window_scores(vws, len(va), C, S, step=1, reducer="mean", mapping_method=mapping)
                    tp, tm = aggregate_window_scores(tws, len(test), C, S, step=1, reducer="mean", mapping_method=mapping)
                    v = vp[vm]
                    half = len(v) // 2
                    cal, hold = v[:half], v[half + S :]
                    evt = EVTCalibrator(risk_level=1e-3, init_percentile=98.0, adaptive_kurtosis=False)
                    evt.fit(cal)
                    th = float(evt.threshold_)

                    ts, tl = tp[tm], y[tm]
                    hold_fpr = float((hold > th).mean())
                    pred = (ts > th).astype(int)
                    mt = compute_metrics(tl, pred, scores=ts, valid_mask=np.ones_like(tl, dtype=bool), use_pa=False)
                    pr_auc = float(average_precision_score(tl, ts)) if tl.sum() > 0 else float("nan")
                    er, dl = event_stats(tl, pred)

                    rows.append({
                        "dataset": ds,
                        "channel": ch,
                        "seed": seed,
                        "arm": arm_name,
                        "alpha_entropy": alpha_ent,
                        "entropy_weight": ent_w,
                        "beta_norm": beta_n,
                        "mapping": mapping,
                        "op_rankme": op_rankme,
                        "op_sigma_max": op_sigma_max,
                        "op_sigma_min": op_sigma_min,
                        "op_cond_num": op_cond_num,
                        "op_frob_ratio": op_frob_ratio,
                        "rep_ctx_rankme": rep_ctx_rankme,
                        "rep_tgt_rankme": rep_tgt_rankme,
                        "rep_tgt_var": rep_tgt_var,
                        "mean_grad_norm": mean_grad_norm,
                        "loss_cv": loss_cv,
                        "heldout_fpr": hold_fpr,
                        "point_f1": mt.get("f1", 0.0),
                        "point_precision": mt.get("precision", 0.0),
                        "point_recall": mt.get("recall", 0.0),
                        "pr_auc": pr_auc,
                        "event_recall": er,
                        "event_delay": dl,
                    })
            print(f"   seed={seed} done", flush=True)
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("\n=== Summary Table II(c) ===")
    for arm, g in df.groupby("arm", sort=False):
        tr = g[g.mapping == "trailing"]
        sm = g[g.mapping == "smear"]
        print(
            f"{arm:30s} | K_rank={tr.op_rankme.mean():.2f} kappa={tr.op_cond_num.mean():.3f} "
            f"Z={tr.rep_ctx_rankme.mean():.2f}/{tr.rep_tgt_rankme.mean():.2f} CV={tr.loss_cv.mean():.3f} | "
            f"Buf: FPR={sm.heldout_fpr.mean():.3f} F1={sm.point_f1.mean():.3f} ER={sm.event_recall.mean():.3f} | "
            f"Caus: FPR={tr.heldout_fpr.mean():.3f} F1={tr.point_f1.mean():.3f} ER={tr.event_recall.mean():.3f}"
        )


if __name__ == "__main__":
    main()
