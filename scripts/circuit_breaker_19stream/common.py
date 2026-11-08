"""Shared helpers for the 19-stream circuit-breaker studies: paths, stream list, cache I/O, SPOT fit."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CSMConfig  # noqa: E402
from src.data.data_loader import DataLoader  # noqa: E402
from src.engine.trainer import build_ts_jepa_model  # noqa: E402
from src.models._jepa_utils import whiten_residuals  # noqa: E402
from src.scoring.evt_calibrator import EVTCalibrator  # noqa: E402
from scripts.run_multiseed_heldout_benchmark import batches, load_stream  # noqa: E402

C, S = 256, 64
W = C + S
SEEDS = (42, 123, 456)
CACHE_EPOCHS = 15
VAL_FRAC = 0.35

# Override with NCAD_PANEL_CACHE to reuse an existing cache directory.
CACHE_DIR = Path(os.environ.get("NCAD_PANEL_CACHE", ROOT / "reports" / "cache_19stream_panel"))

# Streams inspected (with test labels) while the panel constants were selected.
DEVELOPMENT_STREAMS = {
    ("daphnet", c) for c in ("S01R01E1", "S02R01E0", "S02R02E0", "S03R01E0", "S03R01E1", "S03R02E0", "S09R01E2")
} | {("genesis", "default"), ("metro", "traffic-volume"), ("swan", "sf"), ("opportunity", "S1-ADL3")} | {
    ("msl", c) for c in ("C-1", "D-14", "D-15", "M-1", "P-10")
} | {("smap", c) for c in ("A-1", "A-2", "A-7", "D-3", "P-3", "E-1", "T-2")}


def load_streams_19() -> list[tuple[str, str]]:
    """Return the 19 non-GHL primary streams as (dataset, channel) pairs."""
    text = (ROOT / "reports" / "streams_19_nonghl.txt").read_text()
    return [tuple(line.strip().split(":")) for line in text.splitlines() if line.strip()]


def fit_spot(cal_1d: np.ndarray) -> float:
    """Standard SPOT threshold (q=1e-3, u=p98) with a p99.5 fallback."""
    try:
        th = float(EVTCalibrator(risk_level=1e-3, init_percentile=98.0, adaptive_kurtosis=False).fit(cal_1d).threshold_)
    except Exception:  # noqa: BLE001 - GPD MLE can fail on degenerate tails; use the documented fallback
        th = float("nan")
    return th if np.isfinite(th) and th > 0 else float(np.percentile(cal_1d, 99.5))


def cache_path(ds: str, ch: str, seed: int) -> Path:
    return CACHE_DIR / f"{ds}_{ch.replace('/', '_')}_seed{seed}.npz"


@torch.no_grad()
def extract_32dial_tensors(model, windows: np.ndarray, batch_size: int = 256) -> dict[str, np.ndarray]:
    """Whitened signed multi-horizon residuals (N, M, D) and the 1D scalar score (N,)."""
    model.eval()
    device = next(model.parameters()).device
    m_sub, dim = model.num_subblocks, model.latent_dim
    ew_list, s_base_list = [], []
    for i in range(0, len(windows), batch_size):
        b = torch.from_numpy(np.ascontiguousarray(windows[i : i + batch_size])).float().to(device)
        ctx, tgt = b[:, :C], b[:, C:]
        _, _, _, z_sub_pred, z_sub_tgt = model.forward(ctx, tgt)
        diffs = z_sub_tgt - z_sub_pred
        bsz = diffs.shape[0]
        s_base = model.compute_predictive_discrepancy(ctx, tgt)
        e_eff = whiten_residuals(diffs.reshape(bsz, m_sub * dim) - model.residual_mean, model.precision_matrix)
        s_base_list.append(s_base.cpu().numpy().astype(np.float32))
        ew_list.append(e_eff.reshape(bsz, m_sub, dim).cpu().numpy().astype(np.float32))
    return {"s_base": np.concatenate(s_base_list), "ew_signed": np.concatenate(ew_list)}


def train_and_cache(ds: str, ch: str, seed: int, device: torch.device, epochs: int = CACHE_EPOCHS) -> dict:
    """Train Soliton-Reynolds JEPA (AdamW 1e-3, batch 16, constant LR, EMA 0.996) and cache residuals."""
    path = cache_path(ds, ch, seed)
    if path.exists():
        data = np.load(path)
        return {k: data[k] for k in data.files}

    train, test, y = load_stream(ds, ch)
    n_val = int(len(train) * VAL_FRAC)
    tr, va = train[:-n_val], train[-n_val:]
    mu, sd = tr.mean(0), tr.std(0)
    sc = np.where(sd < 1e-8, 1.0, np.maximum(sd, 0.01))
    tr_w = DataLoader.create_windows((tr - mu) / sc, W, step=16, copy=False)
    va_w = DataLoader.create_windows((va - mu) / sc, W, step=1, copy=False)
    te_w = DataLoader.create_windows((test - mu) / sc, W, step=1, copy=False)

    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = CSMConfig(model_type="soliton_reynolds_jepa", context_size=C, suspect_size=S, latent_dim=32)
    model = build_ts_jepa_model(cfg, input_dim=train.shape[1], device=device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for b in batches(tr_w, 16):
            b = b.to(device)
            loss, _ = model.compute_objective(b[:, :C], b[:, C:], cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.update_target_encoder(None)
    model.fit_alien_bank(tr_w, context_len=C)

    va_d, te_d = extract_32dial_tensors(model, va_w), extract_32dial_tensors(model, te_w)
    payload = {
        "va_s_base": va_d["s_base"],
        "va_ew_signed": va_d["ew_signed"],
        "te_s_base": te_d["s_base"],
        "te_ew_signed": te_d["ew_signed"],
        "len_va": np.array(len(va), dtype=np.int64),
        "len_test": np.array(len(test), dtype=np.int64),
        "y": y.astype(np.int64),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return payload


def load_cached(ds: str, ch: str, seed: int) -> dict:
    """Load cached residuals, clipped and NaN-cleaned exactly as in the reported runs."""
    data = np.load(cache_path(ds, ch, seed))
    clean = lambda a: np.clip(np.nan_to_num(a.astype(np.float64), nan=0.0), -100.0, 100.0)  # noqa: E731
    return {
        "len_va": int(data["len_va"]),
        "len_test": int(data["len_test"]),
        "y": data["y"],
        "va_ew": clean(data["va_ew_signed"]),
        "te_ew": clean(data["te_ew_signed"]),
        "va_s_base": data["va_s_base"].astype(np.float64),
        "te_s_base": data["te_s_base"].astype(np.float64),
    }
