#!/usr/bin/env python3
"""Expanded multi-horizon scaling sweep for the 224-bit circuit breaker (thin wrapper).

Runs exactly the protocol of ``run_224bit_horizon_sweep.py`` (same model, training, latent
extraction, 1D scalar SPOT, 224-bit panel and Laminar-JEPA quench; S in {1, 4, 16, 32, 64},
C = 256) on a user-supplied stream list, and checkpoints the CSV after every
(stream, horizon) so partial results survive.  Already-finished (stream, S, seed) triples in
the output file are skipped, so the job can be resumed.

Each row carries ``split`` = ``development`` if (dataset, channel) is in
``scripts.circuit_breaker_19stream.common.DEVELOPMENT_STREAMS`` else ``heldout``.

Usage::

    python scripts/reviewer_analyses/horizon_sweep_expanded.py \
        --streams reports/streams_horizon_expanded.txt --seeds 42 \
        --out reports/horizon_224bit_sweep_expanded.csv
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

from scripts.circuit_breaker_19stream.common import DEVELOPMENT_STREAMS, VAL_FRAC  # noqa: E402
from scripts.reviewer_analyses.run_224bit_horizon_sweep import (  # noqa: E402
    C,
    EPOCHS,
    HORIZONS,
    TRAIN_BS,
    eval_1d_spot_from_latents,
    eval_224_from_latents,
    fast_extract_latents,
    fast_latent_surgical_quench,
)
from scripts.run_multiseed_heldout_benchmark import batches, load_stream  # noqa: E402
from src.config import CSMConfig  # noqa: E402
from src.data.data_loader import DataLoader  # noqa: E402
from src.engine.builder import build_ts_jepa_model  # noqa: E402

DEV = {(d.lower(), c) for d, c in DEVELOPMENT_STREAMS}


def read_streams(path: Path) -> list[tuple[str, str]]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ds, ch = line.split(":", 1)
            out.append((ds, ch))
    return out


def run_one(ds, ch, s_len, seed, data, device) -> list[dict[str, object]]:
    """One (stream, horizon, seed) cell; mirrors ``run_224bit_horizon_sweep.main``."""
    train, test, y_test, tr_norm, va_norm, te_norm, len_va = data
    w_len = C + s_len
    tr_w = DataLoader.create_windows(tr_norm, w_len, step=16, copy=False)
    va_w = DataLoader.create_windows(va_norm, w_len, step=1, copy=False)
    te_w = DataLoader.create_windows(te_norm, w_len, step=1, copy=False)
    n_cal_w = len(va_w) // 2
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
    len_test = len(test)
    res_224, va_w4, te_w4 = eval_224_from_latents(
        model, z_ctx_va, z_tgt_va, z_ctx_te, z_tgt_te, len_va, len_test, y_test, s_len, apply_eq3_blend=False
    )
    res_1d = eval_1d_spot_from_latents(va_w4, te_w4, len_va, len_test, y_test, s_len)
    fast_latent_surgical_quench(model, z_ctx_tr, z_tgt_tr, z_ctx_va[:n_cal_w], z_tgt_va[:n_cal_w])
    res_wsp, _, _ = eval_224_from_latents(
        model, z_ctx_va, z_tgt_va, z_ctx_te, z_tgt_te, len_va, len_test, y_test, s_len, apply_eq3_blend=True
    )
    del model, opt, z_ctx_tr, z_tgt_tr, z_ctx_va, z_tgt_va, z_ctx_te, z_tgt_te
    torch.cuda.empty_cache()
    rows = []
    split = "development" if (ds.lower(), ch) in DEV else "heldout"
    for variant, res in [("1D-Scalar-SPOT", res_1d), ("Reynolds-224Bit", res_224), ("Laminar-JEPA-224Bit", res_wsp)]:
        for mapping in ("trailing", "smear"):
            rows.append({
                "dataset": ds.lower(), "stream": f"{ds.lower()}:{ch}", "split": split, "K": train.shape[1],
                "horizon_S": s_len, "structural_delay": s_len - 1 if mapping == "trailing" else 0,
                "seed": seed, "variant": variant, "mapping": mapping, **res[mapping],
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    ap.add_argument("--out", default=str(ROOT / "reports" / "horizon_224bit_sweep_expanded.csv"))
    args = ap.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows: list[dict[str, object]] = []
    done: set[tuple[str, int, int]] = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        rows = prev.to_dict("records")
        done = set(zip(prev["stream"], prev["horizon_S"].astype(int), prev["seed"].astype(int)))
    t0 = time.time()
    for ds, ch in read_streams(Path(args.streams)):
        name = f"{ds.lower()}:{ch}"
        todo = [(s, sd) for sd in args.seeds for s in args.horizons if (name, s, sd) not in done]
        if not todo:
            continue
        train, test, y_test = load_stream(ds, ch)
        n_val = int(len(train) * VAL_FRAC)
        tr, va = train[:-n_val], train[-n_val:]
        mu, sd_ = tr.mean(0), tr.std(0)
        sc = np.where(sd_ < 1e-8, 1.0, np.maximum(sd_, 0.01))
        data = (train, test, y_test, (tr - mu) / sc, (va - mu) / sc, (test - mu) / sc, len(va))
        for s_len, seed in todo:
            t_c = time.time()
            rows.extend(run_one(ds, ch, s_len, seed, data, device))
            df = pd.DataFrame(rows)
            tmp = out_path.with_suffix(".tmp.csv")
            df.to_csv(tmp, index=False)
            tmp.replace(out_path)
            print(f"[{time.time() - t0:7.1f}s] {name} S={s_len:2d} seed={seed} done in {time.time() - t_c:5.1f}s "
                  f"(ntr={len(train)}, nte={len(test)}, K={train.shape[1]})", flush=True)


if __name__ == "__main__":
    main()
