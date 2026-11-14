#!/usr/bin/env python3
"""Operating-point and false-alarm-tail audit on the 118 untouched archive streams (five-block protocol).

`run` retrains Multi-Horizon JEPA (1D) and Laminar-JEPA-Lite per stream x seed on identical partitions
and records, under causal (trailing) attribution:

1. Deployed rules: Lite balanced quorum, 1D SPOT, and an effective-sample-size (ESS) supported Lite quorum
   whose 99.5% count quantile is replaced by its distribution-free 90% upper confidence bound computed with
   `N_eff = N_cal / tau_int` (Sokal) effective samples (`Lite-UCB`).
2. Healthy-only matched calibration alarm rates r in {0.5, 1, 2, 5}% for Lite counts, 1D scores, and the
   Fisher / raw top-k / max-z aggregators on the same frozen Lite residuals.
3. Post-hoc diagnostics at matched test FPR f in {1, 2, 5, 10}% (uses test labels; diagnostic only).
4. Per-stream tail diagnostics: N_cal, tau_int, N_eff, expected effective exceedances of the 99.5% quorum
   quantile, KS(cal, hold) drift of the deployed statistic, and calibration excess kurtosis.

`analyze` aggregates seeds within streams and prints every number cited in the revision.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CSMConfig
from src.engine.trainer import build_encoder, build_ts_jepa_model
from src.models.jepa.alien_jepas import SpeculativeJEPAModel
from src.scoring.circuit_breaker_panel import stabilize_whitened_subblocks
from src.scoring.laminar_lite_panel import PositionDriftLitePanel
from scripts.circuit_breaker_19stream.common import DEVELOPMENT_STREAMS
from scripts.reviewer_analyses.calibration_quantile_reliability import tau_sokal
from scripts.reviewer_analyses.run_matched_budget_archive import _rate_threshold, _series, lite_bank_scores
from scripts.run_laminar_paired_benchmarks import (
    _build_schedule, _fast_val_checkpoint_train, _score_1d_windows, _split_nominal_5block,
)
from scripts.run_multiseed_heldout_benchmark import C, S, event_stats, load_stream, thresholds
from scripts.run_tier2_exhaustive_benchmark import atomic_save_csv

CAL_RATES = (0.005, 0.01, 0.02, 0.05)
TEST_FPRS = (0.01, 0.02, 0.05, 0.10)
Q_QUORUM, SAFETY, UCB_CONF = 0.995, 1.12, 0.90


def _f1(y: np.ndarray, pred: np.ndarray) -> float:
    tp = float(np.sum((pred == 1) & (y == 1)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def _posthoc(test: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """ER / Point-F1 when the threshold is set post hoc to a target test FPR (diagnostic only)."""
    out: dict[str, float] = {}
    if not (np.any(y == 1) and np.any(y == 0)):
        return out
    nom = test[y == 0]
    for f in TEST_FPRS:
        pred = (test > float(np.quantile(nom, 1.0 - f, method="higher"))).astype(int)
        er, _ = event_stats(y, pred)
        out[f"er_tf{int(f * 100):02d}"], out[f"f1_tf{int(f * 100):02d}"] = float(er), _f1(y, pred)
    return out


def _diag(cal: np.ndarray, hold: np.ndarray) -> dict[str, float]:
    tau = tau_sokal(cal) if len(cal) > 8 else 1.0
    n_eff = len(cal) / tau
    return {
        "tau_int": float(tau), "n_eff": float(n_eff), "exp_exceed_995": float(n_eff * (1 - Q_QUORUM)),
        "ks_cal_hold": float(stats.ks_2samp(cal, hold).statistic) if len(cal) and len(hold) else float("nan"),
        "cal_kurtosis": float(stats.kurtosis(cal)) if np.std(cal) > 1e-12 else 0.0,
    }


def _row(meta: dict[str, Any], model: str, rule: str, th: float, cal: np.ndarray, hold: np.ndarray,
         test: np.ndarray, y: np.ndarray, posthoc: bool = False) -> dict[str, Any]:
    pt = (test > th).astype(int)
    er, dl = event_stats(y, pt)
    row = {
        **meta, "model": model, "rule": rule, "threshold": float(th), "n_cal": int(len(cal)),
        "cal_rate": float(np.mean(cal > th)) if len(cal) else float("nan"),
        "hold_fpr": float(np.mean(hold > th)) if len(hold) else 0.0,
        "test_fpr": float(pt[y == 0].mean()) if np.any(y == 0) else 0.0,
        "f1": _f1(y, pt), "event_recall": float(er), "mean_delay": float(dl),
        **_diag(cal, hold),
    }
    if posthoc:
        row.update(_posthoc(test, y))
    return row


def ucb_quorum(cal_cnt: np.ndarray, floor: float, q: float = Q_QUORUM, conf: float = UCB_CONF) -> tuple[float, bool]:
    """Distribution-free upper confidence bound of the q-quantile with `N_eff` effective samples.

    The bound is the order statistic of rank `r = F_Binom(N_eff, q)^{-1}(conf) + 1`, mapped to level
    `r / N_eff` on the full calibration sample. When `r > N_eff` the quantile is unsupported and the
    calibration maximum is used. The safety factor and floor are applied exactly as in the deployed rule.
    """
    n_eff = max(1, int(np.floor(len(cal_cnt) / tau_sokal(cal_cnt)))) if len(cal_cnt) > 8 else 1
    r = int(stats.binom.ppf(conf, n_eff, q)) + 1
    supported = r <= n_eff
    base = float(np.quantile(cal_cnt, min(1.0, r / n_eff), method="higher")) if supported else float(np.max(cal_cnt))
    return max(base * SAFETY, floor), supported


def run(args: argparse.Namespace) -> None:
    out_path = Path(args.out)
    rows: list[dict[str, Any]] = pd.read_csv(out_path).to_dict("records") if out_path.exists() else []
    done = {(str(r["dataset"]), str(r["channel"]), int(r["seed"])) for r in rows}
    schedule = [it for it in _build_schedule("tier2_141_5block")
                if (it[0].lower(), str(it[1])) not in DEVELOPMENT_STREAMS]
    if args.num_shards > 1:
        schedule = [it for i, it in enumerate(schedule) if i % args.num_shards == args.shard_id]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panel = PositionDriftLitePanel(context_size=C, suspect_size=S, use_hijack_guard=True)
    print(f"=== tail audit: {len(schedule)} untouched streams -> {out_path} ===", flush=True)

    for idx, (ds, ch, seeds) in enumerate(schedule, start=1):
        need = [sd for sd in seeds if (ds, str(ch), sd) not in done]
        if not need:
            continue
        train, test, y = load_stream(ds, ch)
        tr_w, sel_w, q_w, va_w, te_w, len_va = _split_nominal_5block(train, test)
        len_te, k_ch = len(test), int(train.shape[1])
        t0 = time.perf_counter()
        for seed in need:
            meta = {"dataset": ds, "channel": str(ch), "k_channels": k_ch, "t_nom": int(len(train)), "seed": int(seed)}
            torch.manual_seed(seed); np.random.seed(seed)
            cfg_sp = CSMConfig(model_type="ts_jepa", context_size=C, suspect_size=S, latent_dim=32)
            m_sp = SpeculativeJEPAModel(context_encoder=build_encoder(cfg_sp, k_ch, device), latent_dim=32,
                                        predictor_hidden_dim=64, ema_decay=0.996, dropout=cfg_sp.dropout).to(device)
            _fast_val_checkpoint_train(m_sp, tr_w, sel_w, cfg_sp, device, epochs=12, patience=4)
            m_sp.fit_alien_bank(tr_w, context_len=C, batch_size=512)
            ws_1d = (_score_1d_windows(m_sp, va_w, device, use_bank=True), _score_1d_windows(m_sp, te_w, device, use_bank=True))
            del m_sp

            torch.manual_seed(seed); np.random.seed(seed)
            cfg_l = CSMConfig(model_type="laminar_jepa_lite", context_size=C, suspect_size=S, latent_dim=32)
            m_l = build_ts_jepa_model(cfg_l, input_dim=k_ch, device=device)
            _fast_val_checkpoint_train(m_l, tr_w, sel_w, cfg_l, device, epochs=12, patience=4)
            m_l.fit_resolvent_whitening(tr_w, context_len=C, batch_size=512)
            va_r3, va_w3 = m_l.extract_raw_and_whitened_residuals(va_w, context_len=C, batch_size=512)
            te_r3, te_w3 = m_l.extract_raw_and_whitened_residuals(te_w, context_len=C, batch_size=512)
            q_r3, q_w3 = m_l.extract_raw_and_whitened_residuals(q_w, context_len=C, batch_size=512)
            del m_l
            n_cal_w = len(va_w) // 2
            va_ew, te_ew = stabilize_whitened_subblocks(va_w3, te_w3, va_r3, te_r3, n_cal_w, adapt_white4=q_w3, adapt_raw4=q_r3)
            dec = panel.evaluate_stream_decisions(panel.calibrate_and_trip_windows(va_ew, te_ew, n_cal_w), len_va, len_te, "trailing")
            tl = y[dec.valid_mask]
            cal, hold, tst = dec.cal_switch_counts, dec.hold_switch_counts, dec.test_switch_counts
            series = {"Lite": (cal, hold, tst, tl)}

            rows.append(_row(meta, "Lite", "deployed", float(dec.k_gate_balanced), cal, hold, tst, tl, posthoc=True))
            k_ucb, ok = ucb_quorum(cal, floor=3.5)
            rows.append({**_row(meta, "Lite", "ucb", k_ucb, cal, hold, tst, tl), "ucb_supported": bool(ok)})

            v1, t1, tm1 = _series(ws_1d, len_va, len_te, "trailing")
            h1 = len(v1) // 2
            series["MH1D"] = (v1[:h1], v1[h1 + S:], t1, y[tm1])
            rows.append(_row(meta, "MH1D", "deployed", float(thresholds(v1[:h1])[0]["SPOT"]),
                             v1[:h1], v1[h1 + S:], t1, y[tm1], posthoc=True))
            for name, s_va, s_te in lite_bank_scores(va_ew, te_ew):
                v, t, tm = _series((s_va, s_te), len_va, len_te, "trailing")
                hh = len(v) // 2
                series[name] = (v[:hh], v[hh + S:], t, y[tm])
                rows.append(_row(meta, name, "posthoc_only", float("inf"), v[:hh], v[hh + S:], t, y[tm], posthoc=True))
            for name, (c_, h_, t_, y_) in series.items():
                for r in CAL_RATES:
                    rows.append(_row(meta, name, f"cal{r:.3f}", _rate_threshold(c_, r), c_, h_, t_, y_))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            done.add((ds, str(ch), seed))
            atomic_save_csv(pd.DataFrame(rows), out_path)
        print(f"[{idx:3d}/{len(schedule):3d}] {ds}:{ch} K={k_ch} T={len(train)} ({time.perf_counter() - t0:.1f}s)", flush=True)


def _stream_mean(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.groupby(["dataset", "channel"], as_index=False)[cols].mean(numeric_only=True)


def _paired(a: pd.DataFrame, b: pd.DataFrame, col: str) -> tuple[float, float, float, float]:
    m = a.merge(b, on=["dataset", "channel"], suffixes=("_a", "_b")).dropna(subset=[f"{col}_a", f"{col}_b"])
    d = m[f"{col}_a"] - m[f"{col}_b"]
    p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return float(m[f"{col}_a"].mean()), float(m[f"{col}_b"].mean()), float(d.mean()), p


def analyze(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.out)
    cols = ["hold_fpr", "test_fpr", "f1", "event_recall", "mean_delay", "n_cal", "tau_int", "n_eff",
            "exp_exceed_995", "ks_cal_hold", "cal_kurtosis", "cal_rate"]
    ph = [c for c in df.columns if c.startswith(("er_tf", "f1_tf"))]
    get = lambda m, r: _stream_mean(df[(df.model == m) & (df.rule == r)], cols + ph)  # noqa: E731
    lite, mh, ucb = get("Lite", "deployed"), get("MH1D", "deployed"), get("Lite", "ucb")
    print(f"streams: Lite={len(lite)} MH1D={len(mh)} datasets={lite.dataset.nunique()}")

    print("\n[1] Lite causal FPR_hold tail")
    f = lite.hold_fpr
    print(f"  mean={f.mean():.4f} med={f.median():.4f} p90={f.quantile(.9):.4f} p95={f.quantile(.95):.4f}")
    for thr in (0.01, 0.02, 0.05, 0.10, 0.20, 0.30):
        print(f"  >{thr:.0%}: Lite {int((f > thr).sum())}  MH1D {int((mh.hold_fpr > thr).sum())}")
    top = lite.sort_values("hold_fpr", ascending=False).head(args.top)
    print("\n[2] highest-FPR Lite streams (seed-averaged)")
    print(top[["dataset", "channel", "hold_fpr", "test_fpr", "n_cal", "tau_int", "n_eff", "exp_exceed_995",
               "ks_cal_hold", "cal_kurtosis"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    hi, lo = lite[lite.hold_fpr > 0.10], lite[lite.hold_fpr <= 0.05]
    for c in ("n_cal", "n_eff", "tau_int", "ks_cal_hold"):
        print(f"  {c}: FPR>10% median {hi[c].median():.3f} | FPR<=5% median {lo[c].median():.3f} "
              f"(MWU p={stats.mannwhitneyu(hi[c], lo[c]).pvalue:.2g})")
    print(f"  Spearman(KS, FPR)={stats.spearmanr(lite.ks_cal_hold, lite.hold_fpr)[0]:.3f} "
          f"Spearman(N_eff, FPR)={stats.spearmanr(lite.n_eff, lite.hold_fpr)[0]:.3f}")

    print("\n[3] reliability condition: N_eff * (1 - 0.995) >= 1  (N_eff >= 200)")
    for name, mask in (("supported", lite.exp_exceed_995 >= 1), ("unsupported", lite.exp_exceed_995 < 1)):
        s = lite[mask]
        m = mh[mh.set_index(["dataset", "channel"]).index.isin(s.set_index(["dataset", "channel"]).index)]
        print(f"  {name}: n={len(s)} Lite FPR mean {s.hold_fpr.mean():.4f} med {s.hold_fpr.median():.4f} "
              f">5% {int((s.hold_fpr > .05).sum())} F1 {s.f1.mean():.3f} ER {s.event_recall.mean():.3f} | "
              f"MH1D FPR {m.hold_fpr.mean():.4f} F1 {m.f1.mean():.3f} ER {m.event_recall.mean():.3f}")

    print("\n[4] Lite-UCB (ESS-supported quorum) vs deployed Lite")
    for c in ("hold_fpr", "f1", "event_recall"):
        a, b, d, p = _paired(ucb, lite, c)
        print(f"  {c}: UCB {a:.4f} vs Lite {b:.4f} (d={d:+.4f}, p={p:.2g})")
    print(f"  UCB p95 FPR={ucb.hold_fpr.quantile(.95):.4f} >5%: {int((ucb.hold_fpr > .05).sum())} "
          f">10%: {int((ucb.hold_fpr > .10).sum())}")

    print("\n[5] selection rules (Lite when condition holds, else MH1D)")
    key = ["dataset", "channel"]
    mm = lite.merge(mh, on=key, suffixes=("_l", "_m"))
    for name, cond in (("N_cal>=500", mm.n_cal_l >= 500), ("N_eff>=200", mm.exp_exceed_995_l >= 1)):
        sel = {c: np.where(cond, mm[f"{c}_l"], mm[f"{c}_m"]) for c in ("hold_fpr", "f1", "event_recall")}
        print(f"  {name}: Lite on {int(cond.sum())} | FPR mean {np.mean(sel['hold_fpr']):.4f} p95 "
              f"{np.quantile(sel['hold_fpr'], .95):.4f} F1 {np.mean(sel['f1']):.3f} ER {np.mean(sel['event_recall']):.3f}")
    print(f"  Lite-only : FPR {mm.hold_fpr_l.mean():.4f} F1 {mm.f1_l.mean():.3f} ER {mm.event_recall_l.mean():.3f}")
    print(f"  MH1D-only : FPR {mm.hold_fpr_m.mean():.4f} F1 {mm.f1_m.mean():.3f} ER {mm.event_recall_m.mean():.3f}")

    print("\n[6] healthy-only matched calibration alarm rates (Lite vs aggregators)")
    for r in CAL_RATES:
        base = get("Lite", f"cal{r:.3f}")
        line = f"  r={r:.1%}: Lite F1 {base.f1.mean():.3f} ER {base.event_recall.mean():.3f} FPR {base.hold_fpr.mean():.4f}"
        for other in ("Fisher", "TopK", "MaxZ", "MH1D"):
            o = get(other, f"cal{r:.3f}")
            _, b, d, p = _paired(base, o, "event_recall")
            _, bf, df1, pf = _paired(base, o, "f1")
            line += f" | {other} ER {b:.3f} (d={d:+.3f},p={p:.2g}) F1 {bf:.3f} (p={pf:.2g}) FPR {o.hold_fpr.mean():.4f}"
        print(line)
    print("\n[7] post-hoc matched test FPR (diagnostic)")
    for fv in TEST_FPRS:
        c_er, c_f1 = f"er_tf{int(fv * 100):02d}", f"f1_tf{int(fv * 100):02d}"
        line = f"  f={fv:.0%}: Lite ER {lite[c_er].mean():.3f} F1 {lite[c_f1].mean():.3f}"
        for other in ("Fisher", "TopK", "MaxZ"):
            o = get(other, "posthoc_only")
            _, b, d, p = _paired(lite, o, c_er)
            line += f" | {other} ER {b:.3f} (d={d:+.3f},p={p:.2g}) F1 {o[c_f1].mean():.3f}"
        _, b, d, p = _paired(lite, mh, c_er)
        print(line + f" | MH1D ER {b:.3f} (d={d:+.3f},p={p:.2g}) F1 {mh[c_f1].mean():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("run", "analyze"))
    ap.add_argument("--out", default="reports/reviewer_tail_audit_118.csv")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    run(args) if args.mode == "run" else analyze(args)


if __name__ == "__main__":
    main()
