"""Single-factor architectural isolation study for Laminar-JEPA-Lite (19 dev streams x 3 seeds).

Isolates the five architectural simplifications between Laminar-JEPA and Laminar-JEPA-Lite
under the disjoint 4-block protocol (`dev19_4block`):
  1. Lite_Base          : 4 TCN blocks, 64D heads, parallel predictor, M=3, 128 P+D dials
  2. Depth_5Blocks      : 5 TCN blocks, 64D heads, parallel predictor, M=3, 128 P+D dials
  3. Pool_128D          : 4 TCN blocks, 128D heads, parallel predictor, M=3, 128 P+D dials
  4. Horizons_M4_160D   : 4 TCN blocks, 64D heads, parallel predictor, M=4, 160 P+D dials
  5. SeqPred_M4_160D    : 4 TCN blocks, 64D heads, sequential predictor, M=4, 160 P+D dials
  6. Panel_224D_on_M4   : Exact same M=4 parallel residuals (Arm 4) scored with 224-dial panel
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score

from scripts.run_laminar_paired_benchmarks import (
    C,
    CSMConfig,
    S,
    _build_schedule,
    _split_nominal_dev_4block,
    _train_fixed_epochs,
    compute_metrics,
    event_stats,
    load_stream,
)
from src.models.jepa.ts_jepa import LatentPredictor
from src.models.laminar_lite import LaminarJEPALiteModel, LaminarLiteEncoder
from src.scoring.circuit_breaker_panel import MultiPerspectiveCircuitBreakerPanel, stabilize_whitened_subblocks
from src.scoring.laminar_lite_panel import PositionDriftLitePanel


class SequentialLiteControlModel(LaminarJEPALiteModel):
    """Identical to LaminarJEPALiteModel except sequential rollout (sub_predictor + anchor predictor)."""

    def __init__(self, context_encoder: nn.Module, latent_dim: int = 32, predictor_hidden_dim: int = 64, num_horizons: int = 4):
        super().__init__(
            context_encoder=context_encoder,
            latent_dim=latent_dim,
            predictor_hidden_dim=predictor_hidden_dim,
            ema_decay=0.996,
            num_horizons=num_horizons,
        )
        del self.predictor
        self.sub_predictor = LatentPredictor(latent_dim=latent_dim, hidden_dim=predictor_hidden_dim, num_layers=2, dropout=0.0)
        self.anchor_predictor = LatentPredictor(latent_dim=latent_dim, hidden_dim=predictor_hidden_dim, num_layers=2, dropout=0.0)

    def predict_horizons(self, z_ctx: torch.Tensor) -> torch.Tensor:
        d = (self.latent_dim,)
        horizons: list[torch.Tensor] = []
        z_prev = z_ctx
        for m in range(self.num_horizons - 1):
            z_prev = F.layer_norm(self.sub_predictor(z_prev + self.horizon_embeddings[m].unsqueeze(0)), d)
            horizons.append(z_prev)
        horizons.append(F.layer_norm(self.anchor_predictor(z_ctx + self.horizon_embeddings[-1].unsqueeze(0)), d))
        return torch.stack(horizons, dim=1)


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _atomic_write_json(obj: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _eval_panel_arm(
    panel: PositionDriftLitePanel | MultiPerspectiveCircuitBreakerPanel,
    va_raw: np.ndarray,
    va_white: np.ndarray,
    te_raw: np.ndarray,
    te_white: np.ndarray,
    q_raw: np.ndarray,
    q_white: np.ndarray,
    len_va: int,
    len_test: int,
    y_test: np.ndarray,
) -> dict[str, dict[str, float]]:
    n_cal_w = len(va_raw) // 2
    va_ew, te_ew = stabilize_whitened_subblocks(va_white, te_white, va_raw, te_raw, n_cal_w, q_white, q_raw)
    bits = panel.calibrate_and_trip_windows(va_ew, te_ew, n_cal_w)
    out: dict[str, dict[str, float]] = {}
    for mapping in ("trailing", "smear"):
        dec = panel.evaluate_stream_decisions(bits, len_va, len_test, mapping)
        tl = y_test[dec.valid_mask]
        has_both = bool(np.any(tl == 1) and np.any(tl == 0))
        mt = compute_metrics(tl, dec.test_pred_balanced, scores=dec.test_switch_counts, valid_mask=np.ones_like(tl, dtype=bool), use_pa=False)
        er, dl = event_stats(tl, dec.test_pred_balanced)
        pr = float(average_precision_score(tl, dec.test_switch_counts)) if has_both else 0.0
        out[mapping] = {
            "f1": float(mt.get("f1", 0.0)),
            "pr_auc": pr,
            "event_recall": float(er) if er == er else float("nan"),
            "delay": float(dl) if dl == dl else float("nan"),
            "hold_fpr": float(dec.hold_pred_balanced.mean()) if len(dec.hold_pred_balanced) > 0 else 0.0,
            "test_fpr": float(dec.test_pred_balanced[tl == 0].mean()) if np.any(tl == 0) else 0.0,
        }
    return out


def _build_arm_model(arm: str, k_ch: int, device: torch.device) -> LaminarJEPALiteModel:
    if arm == "Lite_Base":
        enc = LaminarLiteEncoder(k_ch, latent_dim=32, filters=48, tcn_layers=4, proj_dim=64)
        return LaminarJEPALiteModel(enc, latent_dim=32, predictor_hidden_dim=64, num_horizons=3).to(device)
    if arm == "Depth_5Blocks":
        enc = LaminarLiteEncoder(k_ch, latent_dim=32, filters=48, tcn_layers=5, proj_dim=64)
        return LaminarJEPALiteModel(enc, latent_dim=32, predictor_hidden_dim=64, num_horizons=3).to(device)
    if arm == "Pool_128D":
        enc = LaminarLiteEncoder(k_ch, latent_dim=32, filters=48, tcn_layers=4, proj_dim=128)
        return LaminarJEPALiteModel(enc, latent_dim=32, predictor_hidden_dim=64, num_horizons=3).to(device)
    if arm == "Horizons_M4_160D":
        enc = LaminarLiteEncoder(k_ch, latent_dim=32, filters=48, tcn_layers=4, proj_dim=64)
        return LaminarJEPALiteModel(enc, latent_dim=32, predictor_hidden_dim=64, num_horizons=4).to(device)
    if arm == "SeqPred_M4_160D":
        enc = LaminarLiteEncoder(k_ch, latent_dim=32, filters=48, tcn_layers=4, proj_dim=64)
        return SequentialLiteControlModel(enc, latent_dim=32, predictor_hidden_dim=64, num_horizons=4).to(device)
    raise ValueError(f"Unknown trainable arm: {arm}")


def main() -> None:
    out_csv = ROOT / "reports" / "lite_factor_ablations_dev19.csv"
    out_json = ROOT / "reports" / "lite_factor_ablations_dev19_summary.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = _build_schedule("dev19_4block")

    panel_pd = PositionDriftLitePanel(context_size=C, suspect_size=S)
    panel_224 = MultiPerspectiveCircuitBreakerPanel(context_size=C, suspect_size=S)
    cfg_train = CSMConfig(model_type="laminar_jepa_lite", context_size=C, suspect_size=S, latent_dim=32)

    rows: list[dict[str, Any]] = []
    if out_csv.exists():
        rows = pd.read_csv(out_csv).to_dict(orient="records")
    done = {(str(r["stream"]), int(r["seed"]), str(r["arm"])) for r in rows}

    trainable_arms = ("Lite_Base", "Depth_5Blocks", "Pool_128D", "Horizons_M4_160D", "SeqPred_M4_160D")

    for idx, (ds, ch, seeds) in enumerate(schedule, start=1):
        s_key = f"{ds}:{ch}"
        needed_seeds = [sd for sd in seeds if not all((s_key, sd, a) in done for a in (*trainable_arms, "Panel_224D_on_M4"))]
        if not needed_seeds:
            continue
        train, test, y = load_stream(ds, ch)
        tr_w, q_w, va_w, te_w, len_va = _split_nominal_dev_4block(train, test)
        k_ch = int(train.shape[1])

        for sd in needed_seeds:
            for arm in trainable_arms:
                if (s_key, sd, arm) in done and not (arm == "Horizons_M4_160D" and (s_key, sd, "Panel_224D_on_M4") not in done):
                    continue
                torch.manual_seed(sd)
                np.random.seed(sd)
                model = _build_arm_model(arm, k_ch, device)
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                _train_fixed_epochs(model, tr_w, cfg_train, epochs=15, device=device)
                model.fit_resolvent_whitening(tr_w, context_len=C, batch_size=512)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                va_raw, va_white = model.extract_raw_and_whitened_residuals(va_w, context_len=C, batch_size=512)
                te_raw, te_white = model.extract_raw_and_whitened_residuals(te_w, context_len=C, batch_size=512)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                infer_ms = 1000.0 * (time.perf_counter() - t0) / max(len(va_w) + len(te_w), 1)
                q_raw, q_white = model.extract_raw_and_whitened_residuals(q_w, context_len=C, batch_size=512)

                if (s_key, sd, arm) not in done:
                    res = _eval_panel_arm(panel_pd, va_raw, va_white, te_raw, te_white, q_raw, q_white, len_va, len(test), y)
                    for mp, mvals in res.items():
                        rows.append({"stream": s_key, "k_channels": k_ch, "seed": sd, "arm": arm, "mapping": mp, "params": n_params, "infer_ms": infer_ms, **mvals})

                if arm == "Horizons_M4_160D" and (s_key, sd, "Panel_224D_on_M4") not in done:
                    res224 = _eval_panel_arm(panel_224, va_raw, va_white, te_raw, te_white, q_raw, q_white, len_va, len(test), y)
                    for mp, mvals in res224.items():
                        rows.append({"stream": s_key, "k_channels": k_ch, "seed": sd, "arm": "Panel_224D_on_M4", "mapping": mp, "params": n_params, "infer_ms": infer_ms, **mvals})
                del model

            _atomic_write_csv(pd.DataFrame(rows), out_csv)
        print(f"[{idx:2d}/{len(schedule)}] {s_key} completed", flush=True)

    df = pd.DataFrame(rows)
    s_avg = df.groupby(["arm", "mapping", "stream"]).mean(numeric_only=True).reset_index()
    summary: dict[str, Any] = {"arms": {}, "paired_tests": {}}
    for mp in ("trailing", "smear"):
        sub = s_avg[s_avg["mapping"] == mp]
        summary["arms"][mp] = sub.groupby("arm")[["f1", "pr_auc", "event_recall", "delay", "hold_fpr", "test_fpr", "params", "infer_ms"]].mean().to_dict(orient="index")
        pairs = [
            ("Depth_5_minus_Lite", "Depth_5Blocks", "Lite_Base"),
            ("Pool_128D_minus_Lite", "Pool_128D", "Lite_Base"),
            ("M4_160D_minus_Lite_128D", "Horizons_M4_160D", "Lite_Base"),
            ("SeqM4_minus_ParM4", "SeqPred_M4_160D", "Horizons_M4_160D"),
            ("Panel224_minus_PD160_on_M4", "Panel_224D_on_M4", "Horizons_M4_160D"),
        ]
        summary["paired_tests"][mp] = {}
        for pname, a1, a2 in pairs:
            g1 = sub[sub["arm"] == a1].set_index("stream")
            g2 = sub[sub["arm"] == a2].set_index("stream").loc[g1.index]
            summary["paired_tests"][mp][pname] = {
                "delta_f1": float((g1["f1"] - g2["f1"]).mean()),
                "p_f1": float(wilcoxon(g1["f1"], g2["f1"]).pvalue) if not np.allclose(g1["f1"], g2["f1"]) else 1.0,
                "delta_er": float((g1["event_recall"] - g2["event_recall"]).dropna().mean()),
                "delta_delay": float((g1["delay"] - g2["delay"]).dropna().mean()),
                "delta_hold_fpr": float((g1["hold_fpr"] - g2["hold_fpr"]).mean()),
            }
    _atomic_write_json(summary, out_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
