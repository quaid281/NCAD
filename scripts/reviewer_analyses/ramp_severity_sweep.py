#!/usr/bin/env python3
"""Controlled slow-ramp severity sweep (operating boundary for gradual process drift).

Model and scoring are identical to ``run_224bit_horizon_sweep.py`` at S = 64, C = 256:
SolitonReynoldsJEPAModel trained for 10 epochs on the nominal training split (first 65% of
the training file), latent extraction with ``fast_extract_latents``, calibration on the first
half of the nominal held-out block (last 35% of the training file), and three detectors:

  * ``1D-Scalar-SPOT``       mean squared whitened residual + standard SPOT (u = p98, q = 1e-3)
  * ``Reynolds-224Bit``      224-bit multi-perspective circuit-breaker panel (balanced gate)
  * ``Laminar-JEPA-224Bit``  ``fast_latent_surgical_quench`` + Eq.-3 blend + 224-bit panel

One model per stream (seed 42).  The quenched model is a deep copy, so the two 224-bit
variants and 1D SPOT see the same encoder latents.

Fault-free test segment
-----------------------
``seg`` = the longest contiguous run of label-0 points in the test file, truncated to its
first ``MAX_SEG`` (16384) points.  It contains no labeled anomalies by construction.

Injection model (raw units, applied before the training z-normalisation)
-----------------------------------------------------------------------
For an event with onset ``t0``, ramp duration ``L``, final amplitude ``a`` and channel set
``R``::

    x_c(t) += a * sigma_c * (t - t0) / L        for t0 <= t < t0 + L
    x_c(t) += a * sigma_c                       for t0 + L <= t < t0 + L + P,   P = 256
    x_c(t) unchanged otherwise                  (offset removed after the plateau)

for every ``c`` in ``R``, where ``sigma_c`` is the standard deviation of channel ``c`` on the
training split.  After normalisation with the training scale ``max(sigma_c, 0.01)`` the
injected offset is ``a`` channel-std units for every channel with ``sigma_c >= 0.01``.
Slope = a / L std units per step.

Grid: a in {0.5, 1, 2, 4} x L in {64, 256, 1024, 4096}; two variants:
  * ``multi``  : |R| = ceil(0.3 K) channels (capped at the number of non-constant channels)
  * ``single`` : |R| = 1 channel
Channel pool = channels with training std > 1e-8.  For each stream and variant, 3 channel
subsets are drawn once (``np.random.default_rng([42, stream_idx, variant_idx])``) and reused
across all 16 grid cells so that cells are comparable.

Event placement (per stream x variant x subset x cell)
------------------------------------------------------
Labels = [t0, t0 + L + P).  After every event an ignore zone [t0 + L + P, t0 + L + P + C + S)
is excluded from both detection and false-alarm accounting, because windows whose context
still contains the injected plateau or the removal step are neither nominal nor part of the
labeled ramp.  Events are non-overlapping and separated by that ignore zone.  The first
onset is at least C + S points into the segment.  ``N_EVENTS`` = 5 events per subset are
required; each injected copy of the segment holds ``n = min(n_fit, remaining)`` events, where
``n_fit = floor((len(seg) - (C+S) - (L+P)) / (L + P + C + S)) + 1`` (the final ignore zone
may be truncated by the segment end).  The slack is split into n + 1 random gaps (sorted uniform
integers, ``np.random.default_rng([42, stream_idx, variant_idx, subset, ai, li, copy])``).
Additional copies are drawn until 5 events per subset are placed (15 events per cell per
variant per stream).  Cells for which one event does not fit are recorded as infeasible.

Metrics (per copy, per method, per mapping: causal ``trailing`` and buffered ``smear``)
--------------------------------------------------------------------------------------
  * detection: an event is detected if any alarm falls inside [t0, t0 + L + P)
  * delay: first alarm index minus t0 (includes the structural S - 1 for trailing mapping)
  * false alarms: alarm rate over valid points outside events and ignore zones (``fa_rate``)
    and the number of alarm episodes (rising edges) there (``fa_episodes``)
The clean (uninjected) segment is also scored to give each method's baseline alarm rate.

Outputs: ``reports/ramp_severity_sweep.csv`` (one row per copy x method x mapping),
``reports/ramp_severity_sweep_summary.csv`` and ``.md`` (pivot tables).
"""

from __future__ import annotations

import argparse
import copy as copylib
import math
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

from scripts.circuit_breaker_19stream.common import VAL_FRAC, fit_spot  # noqa: E402
from scripts.experiment_wsp_soliton import extract_raw4_white4_from_latents  # noqa: E402
from scripts.reviewer_analyses.run_224bit_horizon_sweep import (  # noqa: E402
    EPOCHS,
    TRAIN_BS,
    fast_extract_latents,
    fast_latent_surgical_quench,
)
from scripts.run_multiseed_heldout_benchmark import batches, load_stream  # noqa: E402
from src.config import CSMConfig  # noqa: E402
from src.data.data_loader import DataLoader  # noqa: E402
from src.engine.builder import build_ts_jepa_model  # noqa: E402
from src.scoring.circuit_breaker_panel import (  # noqa: E402
    MultiPerspectiveCircuitBreakerPanel,
    aggregate_vector_stream,
    stabilize_whitened_subblocks,
)

C, S = 256, 64
W = C + S
P = 256
SEED = 42
MAX_SEG = 16384
N_EVENTS = 5
N_SUBSETS = 3
AMPS = [0.5, 1.0, 2.0, 4.0]
LENS = [64, 256, 1024, 4096]
VARIANTS = ["multi", "single"]
METHODS = ["1D-Scalar-SPOT", "Reynolds-224Bit", "Laminar-JEPA-224Bit"]
MAPPINGS = ["trailing", "smear"]
DEFAULT_STREAMS = [
    ("daphnet", "S01R01E1"),
    ("daphnet", "S02R02E0"),
    ("daphnet", "S03R01E0"),
    ("genesis", "default"),
    ("smap", "A-1"),
    ("smap", "A-7"),
    ("smap", "D-3"),
    ("smap", "E-1"),
    ("smap", "P-3"),
    ("msl", "D-14"),
]


def longest_nominal_run(y: np.ndarray) -> tuple[int, int]:
    """Return (start, length) of the longest contiguous run of y == 0."""
    best_s, best_n, cur_s, cur_n = 0, 0, 0, 0
    for i, v in enumerate(y):
        if v == 0:
            if cur_n == 0:
                cur_s = i
            cur_n += 1
            if cur_n > best_n:
                best_s, best_n = cur_s, cur_n
        else:
            cur_n = 0
    return best_s, best_n


def place_events(n_seg: int, L: int, n_want: int, rng: np.random.Generator) -> list[int]:
    """Random non-overlapping onsets with C+S lead and C+S ignore guard between events."""
    ev = L + P
    block = ev + W
    # last event needs only its labeled span inside the segment (guard may be truncated)
    n_fit = max(0, (n_seg - W - ev) // block + 1) if n_seg - W >= ev else 0
    n = min(n_fit, n_want)
    if n == 0:
        return []
    slack = n_seg - W - ev - (n - 1) * block
    cuts = np.sort(rng.integers(0, slack + 1, size=n))
    return [int(W + cuts[k] + k * block) for k in range(n)]


def inject(seg_norm: np.ndarray, onsets: list[int], L: int, amp: float, chans: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Apply the linear ramp + plateau in normalised units (gain_c = sigma_c / scale_c)."""
    out = seg_norm.copy()
    t = np.arange(L + P, dtype=np.float32)
    prof = np.where(t < L, t / L, 1.0).astype(np.float32) * amp
    for t0 in onsets:
        out[t0 : t0 + L + P, chans] += prof[:, None] * gain[None, chans]
    return out


class Scorer:
    """Holds validation-side calibration for the two models and scores a test latent set."""

    def __init__(self, model, model_q, z_ctx_va, z_tgt_va, len_va: int):
        self.model, self.model_q, self.len_va = model, model_q, len_va
        self.va_raw, self.va_white = extract_raw4_white4_from_latents(model, z_ctx_va, z_tgt_va)
        self.vq_raw, self.vq_white = extract_raw4_white4_from_latents(model_q, z_ctx_va, z_tgt_va)
        self.n_cal_w = len(self.va_raw) // 2
        self.panel = MultiPerspectiveCircuitBreakerPanel(context_size=C, suspect_size=S)
        va_1d = np.mean(self.va_white**2, axis=(1, 2))
        self.va_1d = va_1d
        self.spot_thr = {}
        for mapping in MAPPINGS:
            va_tl_2d, va_mask = aggregate_vector_stream(va_1d[:, None], len_va, C, S, mapping)
            cal_end = C + self.n_cal_w if mapping == "trailing" else C + self.n_cal_w + S - 1
            cal_mask = np.zeros(len_va, dtype=bool)
            cal_mask[C : min(len_va, cal_end)] = True
            self.spot_thr[mapping] = fit_spot(va_tl_2d[:, 0][va_mask & cal_mask])

    def _panel_preds(self, va_ew, te_ew, len_te: int) -> dict[str, np.ndarray]:
        bits = self.panel.calibrate_and_trip_windows(va_ew, te_ew, self.n_cal_w)
        out = {}
        for mapping in MAPPINGS:
            dec = self.panel.evaluate_stream_decisions(bits, self.len_va, len_te, mapping)
            full = np.zeros(len_te, dtype=np.int8)
            full[dec.valid_mask] = dec.test_pred_balanced
            out[mapping] = (full, dec.valid_mask)
        return out

    def score(self, z_ctx_te, z_tgt_te, len_te: int) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
        te_raw, te_white = extract_raw4_white4_from_latents(self.model, z_ctx_te, z_tgt_te)
        tq_raw, tq_white = extract_raw4_white4_from_latents(self.model_q, z_ctx_te, z_tgt_te)
        res: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        # 1D scalar SPOT (unquenched model, as in the horizon sweep)
        te_1d = np.mean(te_white**2, axis=(1, 2))
        for mapping in MAPPINGS:
            te_tl_2d, te_mask = aggregate_vector_stream(te_1d[:, None], len_te, C, S, mapping)
            full = np.zeros(len_te, dtype=np.int8)
            full[te_mask] = (te_tl_2d[:, 0][te_mask] > self.spot_thr[mapping]).astype(np.int8)
            res[("1D-Scalar-SPOT", mapping)] = (full, te_mask)
        va_ew = np.clip(np.nan_to_num(self.va_white.astype(np.float64), nan=0.0), -100.0, 100.0)
        te_ew = np.clip(np.nan_to_num(te_white.astype(np.float64), nan=0.0), -100.0, 100.0)
        for mapping, v in self._panel_preds(va_ew, te_ew, len_te).items():
            res[("Reynolds-224Bit", mapping)] = v
        vq_ew, tq_ew = stabilize_whitened_subblocks(self.vq_white, tq_white, self.vq_raw, tq_raw, self.n_cal_w)
        for mapping, v in self._panel_preds(vq_ew, tq_ew, len_te).items():
            res[("Laminar-JEPA-224Bit", mapping)] = v
        return res


def copy_metrics(pred: np.ndarray, valid: np.ndarray, onsets: list[int], L: int) -> dict[str, object]:
    """Event detection, delays and out-of-event false alarms for one injected copy."""
    n = len(pred)
    outside = valid.copy()
    delays = []
    for t0 in onsets:
        e1 = min(n, t0 + L + P)
        outside[t0 : min(n, e1 + W)] = False
        hit = np.where((pred[t0:e1] > 0) & valid[t0:e1])[0]
        delays.append(int(hit[0]) if len(hit) else -1)
    p_out = pred.astype(bool) & outside
    rising = p_out & ~np.concatenate([[False], p_out[:-1]])
    n_out = int(outside.sum())
    return {
        "n_events": len(onsets),
        "n_detected": int(sum(d >= 0 for d in delays)),
        "delays": ";".join(str(d) for d in delays),
        "n_out_points": n_out,
        "fa_points": int(p_out.sum()),
        "fa_rate": float(p_out.sum() / n_out) if n_out else float("nan"),
        "fa_episodes": int(rising.sum()),
    }


def train_stream(ds: str, ch: str, device: torch.device):
    train, test, y_test = load_stream(ds, ch)
    n_val = int(len(train) * VAL_FRAC)
    tr, va = train[:-n_val], train[-n_val:]
    mu, sd = tr.mean(0), tr.std(0)
    sc = np.where(sd < 1e-8, 1.0, np.maximum(sd, 0.01))
    tr_norm, va_norm = (tr - mu) / sc, (va - mu) / sc
    tr_w = DataLoader.create_windows(tr_norm, W, step=16, copy=False)
    va_w = DataLoader.create_windows(va_norm, W, step=1, copy=False)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cfg = CSMConfig(model_type="soliton_reynolds_jepa", context_size=C, suspect_size=S, latent_dim=32)
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
    model.eval()
    z_ctx_tr, z_tgt_tr = fast_extract_latents(model, tr_w, C, batch_size=512)
    z_ctx_va, z_tgt_va = fast_extract_latents(model, va_w, C, batch_size=512)
    n_cal_w = len(va_w) // 2
    model_q = copylib.deepcopy(model)
    fast_latent_surgical_quench(model_q, z_ctx_tr, z_tgt_tr, z_ctx_va[:n_cal_w], z_tgt_va[:n_cal_w])
    del z_ctx_tr, z_tgt_tr
    s0, n0 = longest_nominal_run(y_test)
    seg_norm = ((test[s0 : s0 + min(n0, MAX_SEG)] - mu) / sc).astype(np.float32)
    gain = (np.where(sd < 1e-8, 0.0, sd) / sc).astype(np.float32)
    pool = np.where(sd > 1e-8)[0]
    scorer = Scorer(model, model_q, z_ctx_va, z_tgt_va, len(va))
    return model, scorer, seg_norm, gain, pool, train.shape[1]


@torch.no_grad()
def splice_latents(model, seg_inj, z_ctx_clean, z_tgt_clean, onsets, L):
    """Re-extract latents only for windows overlapping an injected span."""
    n_win = len(z_ctx_clean)
    z_ctx, z_tgt = z_ctx_clean.clone(), z_tgt_clean.clone()
    for t0 in onsets:
        lo = max(0, t0 - W + 1)
        hi = min(n_win - 1, t0 + L + P - 1)
        if hi < lo:
            continue
        wins = DataLoader.create_windows(seg_inj[lo : hi + W], W, step=1, copy=False)
        zc, zt = fast_extract_latents(model, wins, C, batch_size=1024)
        z_ctx[lo : hi + 1], z_tgt[lo : hi + 1] = zc, zt
    return z_ctx, z_tgt


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Per (variant, mapping, method, a, L): stream-mean detection prob, pooled median delay, FA rate."""
    inj = df[df["kind"] == "injected"].copy()
    keys = ["variant", "mapping", "method", "amp", "L"]
    per_stream = inj.groupby(keys + ["stream"]).agg(
        n_events=("n_events", "sum"), n_detected=("n_detected", "sum"), fa_points=("fa_points", "sum"),
        n_out=("n_out_points", "sum"), fa_episodes=("fa_episodes", "sum")
    ).reset_index()
    per_stream["det_prob"] = per_stream["n_detected"] / per_stream["n_events"]
    per_stream["fa_rate"] = per_stream["fa_points"] / per_stream["n_out"].clip(lower=1)
    per_stream["fa_ep_per_10k"] = 1e4 * per_stream["fa_episodes"] / per_stream["n_out"].clip(lower=1)
    rows = []
    for k, g in per_stream.groupby(keys):
        sub = inj.set_index(keys).loc[k]
        sub = sub if isinstance(sub, pd.DataFrame) else sub.to_frame().T
        d = [int(x) for s in sub["delays"] for x in str(s).split(";") if x != "" and int(x) >= 0]
        rows.append(dict(zip(keys, k), n_streams=len(g), n_events=int(g["n_events"].sum()),
                         det_prob=g["det_prob"].mean(), median_delay=float(np.median(d)) if d else float("nan"),
                         fa_rate=g["fa_rate"].mean(), fa_ep_per_10k=g["fa_ep_per_10k"].mean()))
    summ = pd.DataFrame(rows)
    clean = df[df["kind"] == "clean"].groupby(["mapping", "method"])["fa_rate"].mean().rename("clean_fa_rate")
    md = ["# Ramp severity sweep summary\n", "Detection probability is the stream mean of per-stream event recall; "
          "delay is the pooled median over detected events (steps from onset t0); FA rate is the stream-mean alarm "
          "rate outside events and ignore zones.\n", "## Baseline alarm rate on the clean segment\n",
          clean.reset_index().round(4).to_markdown(index=False), ""]
    for variant in VARIANTS:
        for mapping in MAPPINGS:
            for metric in ["det_prob", "median_delay", "fa_rate"]:
                sub = summ[(summ["variant"] == variant) & (summ["mapping"] == mapping)]
                if sub.empty:
                    continue
                pv = sub.pivot_table(index=["method", "amp"], columns="L", values=metric).round(3)
                md += [f"## {metric} | variant={variant} | mapping={mapping}\n", pv.to_markdown(), ""]
    ns = summ.pivot_table(index="amp", columns="L", values="n_streams", aggfunc="max")
    md += ["## Streams contributing per cell (feasibility)\n", ns.to_markdown(), ""]
    return summ, "\n".join(md)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports" / "ramp_severity_sweep.csv"))
    ap.add_argument("--max-streams", type=int, default=len(DEFAULT_STREAMS))
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    done_streams: set[str] = set()
    rows: list[dict[str, object]] = []
    if out_path.exists():
        prev = pd.read_csv(out_path)
        rows = prev.to_dict("records")
        done_streams = set(prev["stream"].unique())
    t_start = time.time()

    for s_idx, (ds, ch) in enumerate(DEFAULT_STREAMS[: args.max_streams]):
        name = f"{ds}:{ch}"
        if args.summarize_only or name in done_streams:
            continue
        t_s = time.time()
        model, scorer, seg, gain, pool, K = train_stream(ds, ch, device)
        seg_w = DataLoader.create_windows(seg, W, step=1, copy=False)
        z_ctx_c, z_tgt_c = fast_extract_latents(model, seg_w, C, batch_size=1024)
        n_seg = len(seg)
        stream_rows: list[dict[str, object]] = []
        base = {"dataset": ds, "stream": name, "K": K, "seg_len": n_seg, "seed": SEED}
        for (method, mapping), (pred, valid) in scorer.score(z_ctx_c, z_tgt_c, n_seg).items():
            m = copy_metrics(pred, valid, [], 0)
            stream_rows.append({**base, "kind": "clean", "variant": "none", "method": method, "mapping": mapping, **m})
        for v_idx, variant in enumerate(VARIANTS):
            r = min(len(pool), math.ceil(0.3 * K)) if variant == "multi" else 1
            rng_sub = np.random.default_rng([SEED, s_idx, v_idx])
            subsets = [np.sort(rng_sub.choice(pool, size=r, replace=False)) for _ in range(N_SUBSETS)]
            for sub_i, chans in enumerate(subsets):
                for ai, amp in enumerate(AMPS):
                    for li, L in enumerate(LENS):
                        placed, cp = 0, 0
                        while placed < N_EVENTS:
                            rng = np.random.default_rng([SEED, s_idx, v_idx, sub_i, ai, li, cp])
                            onsets = place_events(n_seg, L, N_EVENTS - placed, rng)
                            if not onsets:
                                break
                            seg_inj = inject(seg, onsets, L, amp, chans, gain)
                            zc, zt = splice_latents(model, seg_inj, z_ctx_c, z_tgt_c, onsets, L)
                            for (method, mapping), (pred, valid) in scorer.score(zc, zt, n_seg).items():
                                m = copy_metrics(pred, valid, onsets, L)
                                stream_rows.append({**base, "kind": "injected", "variant": variant, "n_chan": r,
                                                    "subset": sub_i, "chans": "|".join(map(str, chans)),
                                                    "copy": cp, "amp": amp, "L": L, "slope": amp / L,
                                                    "onsets": "|".join(map(str, onsets)), "method": method,
                                                    "mapping": mapping, **m})
                            placed += len(onsets)
                            cp += 1
                        if placed == 0:
                            stream_rows.append({**base, "kind": "infeasible", "variant": variant, "subset": sub_i,
                                                "amp": amp, "L": L})
        rows.extend(stream_rows)
        df = pd.DataFrame(rows)
        tmp = out_path.with_suffix(".tmp.csv")
        df.to_csv(tmp, index=False)
        tmp.replace(out_path)
        del model, scorer
        torch.cuda.empty_cache()
        print(f"[{time.time() - t_start:7.1f}s] {name} done in {time.time() - t_s:6.1f}s (seg={n_seg}, K={K})", flush=True)

    df = pd.read_csv(out_path)
    summ, md = summarize(df)
    summ.to_csv(out_path.with_name(out_path.stem + "_summary.csv"), index=False)
    out_path.with_name(out_path.stem + "_summary.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
