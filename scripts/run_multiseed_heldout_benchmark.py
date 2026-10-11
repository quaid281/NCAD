#!/usr/bin/env python3
"""Multi-seed benchmark with strictly held-out threshold calibration.

Addresses the fresh TNNLS review (Major 2, 3, 4, 5):
  * Train on the first (1 - val_frac) of the nominal training sequence.
  * Split the remaining held-out nominal block chronologically into a
    calibration half (fits every threshold) and an evaluation half
    (measures empirical FPR on nominal data the threshold never saw),
    separated by a gap of S=64 points.
  * Same protocol for every model; no thresholds are ever fit on residuals of
    windows the model was trained on.
  * Controls: Operator-Entropy with lambda_ent=0 (same K-matrix architecture),
    Reynolds-Stress with the stress loss off, and Reynolds-Stress scored with
    the mean-only term (isolates the stress term in the score).
  * Calibrators compared on identical score streams: SPOT (u=0.98, q=1e-3),
    SPOT with the tanh kurtosis multiplier (beta=1/6), 99.5th percentile, max.
  * Held-out FPR reported with a moving-block bootstrap 95% interval.
  * Both buffered ("smear", 64-step lookahead) and strictly causal
    ("trailing") score-to-timestamp mappings.

Output: reports/multiseed_heldout_benchmark.csv
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CSMConfig
from src.data.data_loader import DataLoader
from src.engine.trainer import build_ts_jepa_model
from src.models.baselines import TimesNet, TranAD
from src.models.encoders.tcn_encoder import HybridTCNEncoder
from src.scoring.event_fusion import (
    _extract_segments,
    aggregate_window_scores,
    compute_metrics,
)
from src.scoring.evt_calibrator import EVTCalibrator, compute_excess_kurtosis

C, S = 256, 64
W = C + S
STREAMS = [
    ("SMAP", "P-3"), ("MSL", "M-1"), ("SMD", "machine-1-2"), ("SMAP", "A-1"),
    ("MSL", "C-1"), ("SMD", "machine-1-3"), ("Daphnet", "S01R01E1"),
    ("GECCO", "water_quality"),
]
MODELS = [
    "TS-JEPA", "OpEntropy", "OpEntropy-lambda0", "Reynolds", "Reynolds-meanscore",
    "Reynolds-alpha0", "PotentialFlow", "TimesNet", "TranAD",
]


class TCNObsForecaster(nn.Module):
    """Matched-backbone control: forecast the suspect window in observation space."""

    def __init__(self, input_dim, context_len=C, suspect_len=S, latent_dim=32):
        super().__init__()
        self.suspect_len = suspect_len
        self.encoder = HybridTCNEncoder(input_dim=input_dim, latent_dim=latent_dim,
                                        filters=48, tcn_layers=3)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, suspect_len * input_dim),
        )
        self.input_dim = input_dim

    def forward(self, ctx):
        z = self.encoder(ctx)
        return self.decoder(z).view(-1, self.suspect_len, self.input_dim)


class TCNObsReconstructor(nn.Module):
    """Matched-backbone control: reconstruct the full window in observation space."""

    def __init__(self, input_dim, total_len=W, latent_dim=32):
        super().__init__()
        self.total_len = total_len
        self.encoder = HybridTCNEncoder(input_dim=input_dim, latent_dim=latent_dim,
                                        filters=48, tcn_layers=3)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, total_len * input_dim),
        )
        self.input_dim = input_dim

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z).view(-1, self.total_len, self.input_dim)


def load_stream(ds, ch):
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
    return a[cols].to_numpy(np.float32), b[cols].to_numpy(np.float32), y


def batches(arr, bs):
    perm = np.random.permutation(len(arr))
    for i in range(0, len(perm), bs):
        yield torch.from_numpy(np.ascontiguousarray(arr[perm[i:i + bs]])).float()


def train_model(name, tr_win, k, device, epochs):
    """Return a window-score function for the trained model."""
    # Training-free nonparametric references (reviewer-requested simple baselines).
    if name == "PCA-Recon":
        # PCA-subspace reconstruction on the suspect slice; k_pc=min(8, S*k)
        # components fit on standardized training windows only.
        X = tr_win[:, C:, :].reshape(len(tr_win), -1)
        mu = X.mean(axis=0)
        npc = min(8, X.shape[1] - 1)
        V = np.linalg.svd(X - mu, full_matrices=False)[2][:npc]
        proj = V.T @ V

        def score(w):
            Xw = w[:, C:, :].reshape(len(w), -1) - mu
            resid = Xw - Xw @ proj
            return (resid ** 2).mean(axis=1)
        return score

    if name == "ZScoreMax":
        # Per-channel z-score of the suspect slice against training-window
        # target statistics; window score = worst channel/timestep deviation.
        mu = tr_win[:, C:, :].mean(axis=(0, 1))
        sd = tr_win[:, C:, :].std(axis=(0, 1)) + 1e-8

        def score(w):
            z = np.abs((w[:, C:, :] - mu) / sd)
            return z.reshape(len(w), -1).max(axis=1)
        return score

    if name == "PCA-LinPred-LN":
        # Dimensionality/boundedness-matched observation-space control
        # (Round-4 reviewer): 32-d PCA embeddings of context and suspect
        # slices, a ridge linear forecast ctx->tgt, LayerNorm on both sides,
        # then the same saliency-gated norm (alpha=0.5, tau=0.5) used by the
        # JEPA scorers. Isolates the learned-encoder contribution from the
        # "bounded low-dim gated score" confound.
        d_lat = 32
        Xc = tr_win[:, :C, :].reshape(len(tr_win), -1)
        Xt = tr_win[:, C:, :].reshape(len(tr_win), -1)
        muc, mut = Xc.mean(0), Xt.mean(0)
        Vc = np.linalg.svd(Xc - muc, full_matrices=False)[2][:d_lat]
        Vt = np.linalg.svd(Xt - mut, full_matrices=False)[2][:d_lat]
        Cc = (Xc - muc) @ Vc.T
        Ct = (Xt - mut) @ Vt.T
        lam = 1e-3 * np.trace(Cc.T @ Cc) / Cc.shape[0]
        W = np.linalg.solve(Cc.T @ Cc + lam * np.eye(d_lat), Cc.T @ Ct)

        def _ln(v):
            return (v - v.mean(-1, keepdims=True)) / (v.std(-1, keepdims=True) + 1e-6)

        def _gate(e):
            a, tau = 0.5, 0.5
            w = np.exp(np.abs(e) / tau)
            w /= w.sum(-1, keepdims=True)
            return (1 - a) * np.linalg.norm(e, axis=-1) + a * np.sqrt(
                d_lat * (w * e ** 2).sum(-1))

        def score(w):
            cw = _ln((w[:, :C, :].reshape(len(w), -1) - muc) @ Vc.T)
            tw = _ln((w[:, C:, :].reshape(len(w), -1) - mut) @ Vt.T)
            return _gate(tw - cw @ W)
        return score

    if name == "Obs-MAD":
        # Robust median/MAD-normalized observation-space score (Round-4
        # reviewer): per-channel |x - med| / MAD over the suspect slice,
        # pooled by L2. Tests whether robust normalization alone reproduces
        # the latent score's tail conditioning.
        med = np.median(tr_win[:, C:, :], axis=(0, 1))
        mad = np.median(np.abs(tr_win[:, C:, :] - med), axis=(0, 1)) + 1e-6

        def score(w):
            z = np.abs(w[:, C:, :] - med) / mad
            return np.sqrt((z ** 2).sum(axis=(1, 2)) / z.shape[1])
        return score

    if name == "IsoForest":
        # Isolation Forest on flattened suspect windows (Round-4 baseline).
        from sklearn.ensemble import IsolationForest
        Xf = tr_win[:, C:, :].reshape(len(tr_win), -1)
        sub = Xf[np.random.RandomState(0).choice(len(Xf),
                  min(8192, len(Xf)), replace=False)]
        iso = IsolationForest(n_estimators=100, max_samples=min(1024, len(sub)),
                              contamination="auto", random_state=0).fit(sub)

        def score(w):
            Xw = w[:, C:, :].reshape(len(w), -1)
            return -iso.decision_function(Xw)
        return score

    if name == "KNN-Dist":
        # Distance to the 5th nearest training window (classic kNN-AD).
        Xf = tr_win[:, C:, :].reshape(len(tr_win), -1)
        ref = Xf[np.random.RandomState(0).choice(len(Xf),
                 min(4096, len(Xf)), replace=False)]

        def score(w):
            Xw = w[:, C:, :].reshape(len(w), -1)
            out = np.empty(len(Xw))
            for i in range(0, len(Xw), 512):
                dd = np.sqrt(((Xw[i:i + 512, None, :] - ref[None]) ** 2)
                             .sum(-1))
                out[i:i + 512] = np.sort(dd, axis=1)[:, 4]
            return out
        return score


    if name in ("TCN-ObsRecon", "TCN-ObsPred"):
        if name == "TCN-ObsRecon":
            m = TCNObsReconstructor(input_dim=k).to(device)
        else:
            m = TCNObsForecaster(input_dim=k).to(device)
        opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        for ep in range(epochs):
            m.train()
            for b in batches(tr_win, 16):
                b = b.to(device)
                if name == "TCN-ObsRecon":
                    loss = nn.functional.mse_loss(m(b), b)
                else:
                    loss = nn.functional.mse_loss(m(b[:, :C]), b[:, C:])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        def score(w):
            m.eval()
            out = []
            with torch.no_grad():
                for i in range(0, len(w), 128):
                    ch = torch.from_numpy(np.ascontiguousarray(w[i:i + 128])).float().to(device)
                    if name == "TCN-ObsRecon":
                        rec = m(ch)[:, C:]
                    else:
                        rec = m(ch[:, :C])
                    out.append(((rec - ch[:, C:]) ** 2).mean(dim=(-2, -1)).cpu().numpy())
            return np.concatenate(out)
        return score

    if name in ("TimesNet", "TranAD"):
        if name == "TimesNet":
            m = TimesNet(c_in=k, d_model=48, d_ff=96, e_layers=2).to(device)
        else:
            m = TranAD(c_in=k, d_model=48, n_heads=4, e_layers=2, d_layers=2, d_ff=96).to(device)
        opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        for ep in range(epochs):
            m.train()
            for b in batches(tr_win, 16):
                b = b.to(device)
                if name == "TimesNet":
                    loss = nn.functional.mse_loss(m(b), b)
                else:
                    r1, r2 = m(b)
                    l1, l2 = m.adversarial_loss(r1, r2, b, epoch=ep + 1)
                    loss = l1 + l2
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        def score(w):
            m.eval()
            out = []
            with torch.no_grad():
                for i in range(0, len(w), 128):
                    ch = torch.from_numpy(np.ascontiguousarray(w[i:i + 128])).float().to(device)
                    out.append(m.compute_anomaly_scores(ch)[:, C:].mean(dim=-1).cpu().numpy())
            return np.concatenate(out)
        return score

    mt = {"TS-JEPA": "ts_jepa", "OpEntropy": "operator_entropy_jepa",
          "OpEntropy-lambda0": "operator_entropy_jepa", "Reynolds": "reynolds_stress_jepa",
          "Reynolds-meanscore": "reynolds_stress_jepa", "Reynolds-alpha0": "reynolds_stress_jepa",
          "Reynolds-tempcov": "reynolds_stress_jepa", "Reynolds-batchcov": "reynolds_stress_jepa",
          "JEPA-CtxRecon": "ts_jepa", "JEPA-NoEMA": "ts_jepa", "TS-JEPA-NoGate": "ts_jepa",
          "PotentialFlow": "potential_flow_jepa", "PotentialFlow-noregime": "potential_flow_jepa"}[name]
    extra = {"use_regimes": False} if name == "PotentialFlow-noregime" else {}
    cfg = CSMConfig(model_type=mt, context_size=C, suspect_size=S, latent_dim=32, **extra)
    m = build_ts_jepa_model(cfg, input_dim=k, device=device)
    if name == "Reynolds-alpha0":
        m.alpha_stress = 0.0
    if name == "Reynolds-tempcov":
        m.cov_mode = "temporal"
    if name == "Reynolds-batchcov":
        m.cov_mode = "batch"
    opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    kw = {"alpha_entropy": 0.0} if name == "OpEntropy-lambda0" else {}
    ctx_recon = name == "JEPA-CtxRecon"
    for ep in range(epochs):
        m.train()
        for b in batches(tr_win, 16):
            b = b.to(device)
            # JEPA-CtxRecon: latent-reconstruction control. Identical JEPA machinery
            # (EMA target encoder, predictor, VICReg loss) but the target is the EMA
            # encoding of the *context* window rather than the future window --
            # isolates prediction vs. latent compression/reconstruction.
            tgt = b[:, :C] if ctx_recon else b[:, C:]
            loss, _ = m.compute_objective(b[:, :C], tgt, cfg, **kw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            # JEPA-NoEMA: online-target control (hard copy each step) isolates the
            # EMA target encoder's contribution to residual conditioning.
            m.update_target_encoder(1.0 if name == "JEPA-NoEMA" else None)

    mean_only = name in ("Reynolds-meanscore", "Reynolds-alpha0")
    nogate = name == "TS-JEPA-NoGate"  # equalized-gating control: plain L2 discrepancy

    def score(w):
        m.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(w), 128):
                ch = torch.from_numpy(np.ascontiguousarray(w[i:i + 128])).float().to(device)
                if nogate:
                    d = m.target_encoder(ch[:, C:]) - m.predictor(m.context_encoder(ch[:, :C]))
                    out.append(d.norm(dim=-1).cpu().numpy())
                elif mean_only:
                    zc = m._extract_context(ch[:, :C])
                    d = m.target_encoder(ch[:, C:]) - m.predictor(zc)
                    out.append(m.saliency_gate(d).cpu().numpy())
                elif ctx_recon:
                    out.append(m.compute_predictive_discrepancy(ch[:, :C], ch[:, :C]).cpu().numpy())
                else:
                    out.append(m.compute_predictive_discrepancy(ch[:, :C], ch[:, C:]).cpu().numpy())
        return np.concatenate(out)
    return score


def block_bootstrap_ci(ind, block=W, n_boot=400, rng=None):
    """Moving-block bootstrap 95% CI for the mean of a 0/1 exceedance indicator."""
    rng = rng or np.random.default_rng(0)
    n = len(ind)
    if n < 2:
        return float("nan"), float("nan")
    block = max(16, min(block, n // 8))
    n_blocks = int(math.ceil(n / block))
    starts = np.arange(0, n - block + 1)
    means = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.choice(starts, n_blocks)
        means[i] = np.concatenate([ind[j:j + block] for j in s])[:n].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def event_stats(y, pred):
    segs = _extract_segments(y)
    if not segs:
        return float("nan"), float("nan")
    delays = []
    for gs, ge in segs:
        hit = np.where(pred[gs:ge] > 0)[0]
        if len(hit):
            delays.append(hit[0])
    return len(delays) / len(segs), (float(np.mean(delays)) if delays else float("nan"))


def thresholds(cal):
    base = EVTCalibrator(risk_level=1e-3, init_percentile=98.0, adaptive_kurtosis=False)
    base.fit(cal)
    t0 = float(base.threshold_)
    kurt = float(compute_excess_kurtosis(cal))
    return {
        "SPOT": t0,
        "SPOT+kurt(1/6)": t0 * (1.0 + math.tanh(kurt) / 6.0),
        "p99.5": float(np.percentile(cal, 99.5)),
        "max-val": float(np.max(cal)),
    }, kurt


def run(ds, ch, seeds, device, epochs, val_frac, out_rows, dump_dir=None):
    train, test, y = load_stream(ds, ch)
    n_val = int(len(train) * val_frac)
    tr, va = train[:-n_val], train[-n_val:]
    mu, sd = tr.mean(0), tr.std(0)
    sc = np.where(sd < 1e-8, 1.0, np.maximum(sd, 0.01))
    tr_s, va_s, te_s = (tr - mu) / sc, (va - mu) / sc, (test - mu) / sc
    tr_w = DataLoader.create_windows(tr_s, W, step=16, copy=False)
    va_w = DataLoader.create_windows(va_s, W, step=1, copy=False)
    te_w = DataLoader.create_windows(te_s, W, step=1, copy=False)
    k = train.shape[1]
    print(f"== {ds}:{ch} train={len(tr)} val={len(va)} test={len(test)} anomaly_rate={y.mean():.4f}", flush=True)
    if len(va_w) < 50:
        print("   skip: held-out block too short", flush=True)
        return
    for seed in seeds:
        for name in MODELS:
            dump_paths = None
            if dump_dir is not None:
                dump_paths = {
                    mp: dump_dir / f"{ds}__{ch}__{seed}__{name}__{mp}.npz"
                    for mp in ("smear", "trailing")
                }
                if all(p.exists() for p in dump_paths.values()):
                    print(f"   seed={seed} {name} cached, skipping", flush=True)
                    continue
            torch.manual_seed(seed)
            np.random.seed(seed)
            score = train_model(name, tr_w, k, device, epochs)
            vws, tws = score(va_w), score(te_w)
            for mapping in ("smear", "trailing"):
                vp, vm = aggregate_window_scores(vws, len(va), C, S, step=1, reducer="mean", mapping_method=mapping)
                tp, tm = aggregate_window_scores(tws, len(test), C, S, step=1, reducer="mean", mapping_method=mapping)
                v = vp[vm]
                half = len(v) // 2
                # Gap of S points between halves. Short streams (SMAP P-3 has ~2k nominal
                # steps) cannot afford a full-window gap, so residual overlap-correlation is
                # handled by the block bootstrap instead.
                cal, hold = v[:half], v[half + S:]
                if len(cal) < 100 or len(hold) < 100:
                    continue
                ths, kurt = thresholds(cal)
                ts, tl = tp[tm], y[tm]
                if dump_paths is not None:
                    np.savez_compressed(
                        dump_paths[mapping],
                        cal=cal, hold=hold, test_scores=ts, test_labels=tl,
                    )
                pr = float(average_precision_score(tl, ts)) if tl.sum() > 0 else float("nan")
                for cname, th in ths.items():
                    ind = (hold > th).astype(float)
                    lo, hi = block_bootstrap_ci(ind)
                    pred = (ts > th).astype(int)
                    mt = compute_metrics(tl, pred, scores=ts, valid_mask=np.ones_like(tl, dtype=bool), use_pa=False)
                    er, dl = event_stats(tl, pred)
                    out_rows.append(dict(
                        dataset=ds, channel=ch, seed=seed, model=name, mapping=mapping, calibrator=cname,
                        threshold=th, cal_kurtosis=kurt, n_cal=len(cal), n_hold=len(hold),
                        hold_fpr=float(ind.mean()), hold_fpr_lo=lo, hold_fpr_hi=hi,
                        test_fpr=float(mt["fp"] / max((tl == 0).sum(), 1)),
                        f1=mt.get("f1", 0.0), precision=mt.get("precision", 0.0),
                        recall=mt.get("recall", 0.0), pr_auc=pr, event_recall=er, mean_delay=dl,
                    ))
            print(f"   seed={seed} {name} done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--val-frac", type=float, default=0.3)
    ap.add_argument("--streams", nargs="*", default=None, help="DS:channel entries")
    ap.add_argument("--models", nargs="*", default=None, help="subset of MODELS (or PotentialFlow-noregime)")
    ap.add_argument("--out", default=str(ROOT / "reports" / "multiseed_heldout_benchmark.csv"))
    ap.add_argument("--dump-scores", default=None,
                    help="Directory for raw per-timestep score arrays (npz per stream/seed/model/mapping)")
    a = ap.parse_args()
    global MODELS
    if a.models:
        MODELS = a.models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.streams and len(a.streams) == 1 and Path(a.streams[0]).exists():
        raw_streams = [line.strip() for line in Path(a.streams[0]).read_text().splitlines() if line.strip()]
        streams = [tuple(s.split(":")) for s in raw_streams]
    elif a.streams:
        streams = [tuple(s.split(":")) for s in a.streams]
    else:
        streams = STREAMS
    dump_dir = Path(a.dump_scores) if a.dump_scores else None
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds, ch in streams:
        try:
            run(ds, ch, a.seeds, device, a.epochs, a.val_frac, rows, dump_dir=dump_dir)
        except Exception as e:  # keep going; log the failure in the output
            print(f"   FAILED {ds}:{ch}: {type(e).__name__}: {e}", flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
    print("saved", a.out)


if __name__ == "__main__":
    main()
