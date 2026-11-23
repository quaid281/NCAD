"""Single-factor and dial-family ablation on the *deployed* Laminar-JEPA-Lite (32D head).

Mirrors `run_lite_factor_ablations.py` (19 dev streams x 3 seeds, disjoint 4-block
protocol) but starts every arm from the deployed configuration (`proj_dim=None` -> 32D
head, ~120-129k trainable parameters) instead of the 64D-head factor base.

Arms (one factor flipped at a time from `Lite32_Base`):
  Lite32_Base         : 4 TCN blocks, 32D head, parallel predictor, M=3, 128 P+D dials
  Lite32_PosOnly96    : same residuals as Lite32_Base, 96 position dials only (drift dials muted)
  Lite32_Depth5       : 5 TCN blocks
  Lite32_Head64       : 64D head
  Lite32_M4_160       : M=4 parallel, 160 P+D dials
  Lite32_SeqM4_160    : M=4 sequential predictor, 160 P+D dials
  Lite32_Panel224_M4  : Lite32_M4_160 residuals scored by the 224-dial panel
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "scripts", ROOT / "scripts" / "reviewer_analyses"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from run_lite_factor_ablations import (  # noqa: E402
    SequentialLiteControlModel,
    _atomic_write_csv,
    _atomic_write_json,
    _eval_panel_arm,
)
from scripts.run_laminar_paired_benchmarks import (  # noqa: E402
    C,
    CSMConfig,
    S,
    _build_schedule,
    _split_nominal_dev_4block,
    _train_fixed_epochs,
    load_stream,
)
from src.models.laminar_lite import LaminarJEPALiteModel, LaminarLiteEncoder  # noqa: E402
from src.scoring.circuit_breaker_panel import MultiPerspectiveCircuitBreakerPanel  # noqa: E402
from src.scoring.laminar_lite_panel import PositionDriftLitePanel  # noqa: E402


class PositionOnlyLitePanel(PositionDriftLitePanel):
    """Same tripwires and quorum rule as the P+D panel, with the D drift dials muted."""

    def calibrate_and_trip_windows(self, va_ew, te_ew, n_cal_w, scale_homogeneous_floor=None):
        out = super().calibrate_and_trip_windows(va_ew, te_ew, n_cal_w, scale_homogeneous_floor)
        m, d = int(out["num_horizons"]), va_ew.shape[2]
        for key in ("bits_pd_va", "bits_pd_te"):
            out[key][:, m * d :] = 0.0
        return out


TRAINABLE = ("Lite32_Base", "Lite32_Depth5", "Lite32_Head64", "Lite32_M4_160", "Lite32_SeqM4_160")


def _build(arm: str, k_ch: int, device: torch.device) -> LaminarJEPALiteModel:
    blocks = 5 if arm == "Lite32_Depth5" else 4
    proj = 64 if arm == "Lite32_Head64" else None
    horizons = 4 if arm in ("Lite32_M4_160", "Lite32_SeqM4_160") else 3
    enc = LaminarLiteEncoder(k_ch, latent_dim=32, filters=48, tcn_layers=blocks, proj_dim=proj)
    cls = SequentialLiteControlModel if arm == "Lite32_SeqM4_160" else LaminarJEPALiteModel
    if cls is SequentialLiteControlModel:
        return cls(enc, latent_dim=32, predictor_hidden_dim=64, num_horizons=horizons).to(device)
    return cls(enc, latent_dim=32, predictor_hidden_dim=64, ema_decay=0.996, num_horizons=horizons).to(device)


def main() -> None:
    out_dir = ROOT / "reports" / "latent_diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "lite32_deployed_ablations_dev19.csv"
    out_json = out_dir / "lite32_deployed_ablations_dev19_summary.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = _build_schedule("dev19_4block")
    panels = {
        "pd": PositionDriftLitePanel(context_size=C, suspect_size=S),
        "pos": PositionOnlyLitePanel(context_size=C, suspect_size=S),
        "p224": MultiPerspectiveCircuitBreakerPanel(context_size=C, suspect_size=S),
    }
    cfg = CSMConfig(model_type="laminar_jepa_lite", context_size=C, suspect_size=S, latent_dim=32)

    rows: list[dict[str, Any]] = pd.read_csv(out_csv).to_dict(orient="records") if out_csv.exists() else []
    done = {(str(r["stream"]), int(r["seed"]), str(r["arm"])) for r in rows}

    for idx, (ds, ch, seeds) in enumerate(schedule, start=1):
        key = f"{ds}:{ch}"
        train, test, y = load_stream(ds, ch)
        tr_w, q_w, va_w, te_w, len_va = _split_nominal_dev_4block(train, test)
        k_ch = int(train.shape[1])
        for sd in seeds:
            for arm in TRAINABLE:
                if (key, sd, arm) in done:
                    continue
                torch.manual_seed(sd)
                np.random.seed(sd)
                model = _build(arm, k_ch, device)
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                _train_fixed_epochs(model, tr_w, cfg, epochs=15, device=device)
                model.fit_resolvent_whitening(tr_w, context_len=C, batch_size=512)
                with torch.no_grad():
                    va = model.extract_raw_and_whitened_residuals(va_w, context_len=C, batch_size=512)
                    te = model.extract_raw_and_whitened_residuals(te_w, context_len=C, batch_size=512)
                    qq = model.extract_raw_and_whitened_residuals(q_w, context_len=C, batch_size=512)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                evals = [(arm, panels["pd"])]
                if arm == "Lite32_Base":
                    evals.append(("Lite32_PosOnly96", panels["pos"]))
                if arm == "Lite32_M4_160":
                    evals.append(("Lite32_Panel224_M4", panels["p224"]))
                for name, panel in evals:
                    res = _eval_panel_arm(panel, va[0], va[1], te[0], te[1], qq[0], qq[1], len_va, len(test), y)
                    for mp, vals in res.items():
                        rows.append({"stream": key, "k_channels": k_ch, "seed": sd, "arm": name,
                                     "mapping": mp, "params": n_params, **vals})
                    done.add((key, sd, name))
            _atomic_write_csv(pd.DataFrame(rows), out_csv)
        print(f"[{idx:2d}/{len(schedule)}] {key} completed", flush=True)

    df = pd.DataFrame(rows)
    s_avg = df.groupby(["arm", "mapping", "stream"]).mean(numeric_only=True).reset_index()
    cols = ["f1", "pr_auc", "event_recall", "delay", "hold_fpr", "test_fpr", "params"]
    summary: dict[str, Any] = {"arms": {}, "paired": {}}
    pairs = [
        ("Drift_vs_PosOnly", "Lite32_Base", "Lite32_PosOnly96"),
        ("Depth5_vs_Base", "Lite32_Depth5", "Lite32_Base"),
        ("Head64_vs_Base", "Lite32_Head64", "Lite32_Base"),
        ("M4_vs_Base", "Lite32_M4_160", "Lite32_Base"),
        ("Seq_vs_ParM4", "Lite32_SeqM4_160", "Lite32_M4_160"),
        ("P224_vs_PD160", "Lite32_Panel224_M4", "Lite32_M4_160"),
    ]
    for mp in ("trailing", "smear"):
        sub = s_avg[s_avg["mapping"] == mp]
        summary["arms"][mp] = sub.groupby("arm")[cols].mean().to_dict(orient="index")
        summary["paired"][mp] = {}
        for name, a, b in pairs:
            g1 = sub[sub["arm"] == a].set_index("stream")
            g2 = sub[sub["arm"] == b].set_index("stream").loc[g1.index]
            entry: dict[str, float] = {}
            for met in ("f1", "event_recall", "delay", "hold_fpr"):
                d = (g1[met] - g2[met]).dropna()
                entry[f"delta_{met}"] = float(d.mean())
                entry[f"p_{met}"] = float(wilcoxon(d).pvalue) if len(d) > 0 and not np.allclose(d, 0) else 1.0
            summary["paired"][mp][name] = entry
    _atomic_write_json(summary, out_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
