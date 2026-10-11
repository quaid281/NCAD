#!/usr/bin/env python3
"""PotentialFlow inference-path test (Round-4, MC 3.6).

Trains PotentialFlow-JEPA on a few streams, then scores the held-out test
block two ways:
  (a) mean-path: query the flow field at z = 0.5*z_tgt, t = 0.5
      (current deployment path == z_noise = 0);
  (b) noise-matched: query at z = 0.5*(z_noise + z_tgt) with z_noise ~ N(0,I),
      matching the training interpolation, averaged over 16 draws.
If the curl-free explanation is right, (b) should not help; if the poor
performance is train/test mismatch, (b) improves recall/FPR.

Output: reports/pf_noise_path_test.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_multiseed_heldout_benchmark import (  # noqa: E402
    C, S, W, load_stream, batches, event_stats,
    aggregate_window_scores, compute_metrics)
from src.config import CSMConfig  # noqa: E402
from src.engine.trainer import build_ts_jepa_model  # noqa: E402
from src.scoring.evt_calibrator import EVTCalibrator  # noqa: E402

STREAMS = [("Daphnet", "S01R01E1"), ("SMAP", "P-3"), ("swan", "sf")]
SEED = 42
EPOCHS = 10
K_NOISE = 16
OUT = ROOT / "reports" / "pf_noise_path_test.csv"


def score_path(m, w, device, noisy, k=K_NOISE):
    """PF discrepancy at z_eval = 0.5*(e + z_tgt), e=0 or e~N(0,I)."""
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(w), 128):
            ch = torch.from_numpy(np.ascontiguousarray(w[i:i + 128])).float().to(device)
            z_ctx = m.context_encoder(ch[:, :C])
            z_tgt = m.target_encoder(ch[:, C:])
            t_eval = torch.full((len(ch),), 0.5, device=device, dtype=ch.dtype)
            if m.grassmannian_codebook is not None:
                p_regime, _, _ = m.grassmannian_codebook(z_ctx, hard=m.eval_hard_routing)
            else:
                p_regime = torch.zeros_like(z_ctx)
            if not noisy:
                v = m.flow_predictor(0.5 * z_tgt, t_eval, z_ctx,
                                     p_regime=p_regime, create_graph=False)
                out.append(m.saliency_gate(v - z_tgt).cpu().numpy())
            else:
                acc = torch.zeros(len(ch), device=device)
                for _ in range(k):
                    zn = torch.randn_like(z_tgt)
                    z_eval = 0.5 * (zn + z_tgt)
                    v = m.flow_predictor(z_eval, t_eval, z_ctx,
                                         p_regime=p_regime, create_graph=False)
                    acc += m.saliency_gate(v - z_tgt)
                out.append((acc / k).cpu().numpy())
    return np.concatenate(out)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for ds, ch_name in STREAMS:
        train, test, y = load_stream(ds, ch_name)
        n_val = int(len(train) * 0.30)
        tr, va = train[:-n_val], train[-n_val:]
        mu, sd = tr.mean(0), tr.std(0)
        sc = np.where(sd < 1e-8, 1.0, np.maximum(sd, 0.01))
        tr_s, va_s, te_s = (tr - mu) / sc, (va - mu) / sc, (test - mu) / sc
        from src.data.data_loader import DataLoader
        tr_w = DataLoader.create_windows(tr_s, W, step=16, copy=False)
        va_w = DataLoader.create_windows(va_s, W, step=1, copy=False)
        te_w = DataLoader.create_windows(te_s, W, step=1, copy=False)
        k = train.shape[1]
        torch.manual_seed(SEED); np.random.seed(SEED)
        cfg = CSMConfig(model_type="potential_flow_jepa", context_size=C,
                        suspect_size=S, latent_dim=32)
        m = build_ts_jepa_model(cfg, input_dim=k, device=device)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        for ep in range(EPOCHS):
            m.train()
            for b in batches(tr_w, 16):
                b = b.to(device)
                loss, _ = m.compute_objective(b[:, :C], b[:, C:], cfg)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            m.update_target_encoder(None)
            print(f"  {ds}:{ch_name} epoch {ep+1}/{EPOCHS}", flush=True)
        for noisy in (False, True):
            vws = score_path(m, va_w, device, noisy)
            tws = score_path(m, te_w, device, noisy)
            for mapping in ("smear", "trailing"):
                vp, vm = aggregate_window_scores(vws, len(va), C, S, step=1,
                                               reducer="mean", mapping_method=mapping)
                tp, tm = aggregate_window_scores(tws, len(test), C, S, step=1,
                                               reducer="mean", mapping_method=mapping)
                v = vp[vm]
                half = len(v) // 2
                cal = v[:half]
                evt = EVTCalibrator(risk_level=1e-3, init_percentile=98.0,
                                    adaptive_kurtosis=False)
                evt.fit(cal)
                th = float(evt.threshold_)
                ts, tl = tp[tm], y[tm]
                pred = (ts > th).astype(int)
                er, dl = event_stats(tl, pred)
                nom = max(int((tl == 0).sum()), 1)
                rows.append(dict(
                    dataset=ds, channel=ch_name, mapping=mapping,
                    noise_matched=noisy, threshold=th,
                    realized_fpr=float(((pred == 1) & (tl == 0)).sum() / nom),
                    event_recall=er, mean_delay=dl,
                    point_f1=float(compute_metrics(
                        tl, pred, scores=ts,
                        valid_mask=np.ones_like(tl, dtype=bool),
                        use_pa=False).get("f1", 0.0))))
                print(f"   {ds}:{ch_name} {mapping} noisy={noisy} "
                      f"FPR={rows[-1]['realized_fpr']:.3f} "
                      f"ER={er:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
