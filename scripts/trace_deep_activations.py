"""Deep Activation Trace for the Physics-Informed JEPA Triad.

Goes beyond the module-level trace in `trace_model_activations.py` by hooking
every leaf submodule via `src.engine.activation_tracer.LayerActivationTracer`,
capturing per-layer statistics (mean/std/norm/min/max, sparsity, stable rank,
batch variance, health flags) plus LayerAutopsy diagnostics:
- dimensional collapse / dead-neuron / NaN-Inf / explosion detection,
- sequential drift (cosine similarity between adjacent layer activations),
- nominal vs. anomalous perturbation divergence per layer.

Models:
1. OperatorEntropyJEPAModel (Koopman Spectral Mechanics)
2. ReynoldsStressJEPAModel (Hydrodynamic Transport & Turbulent Stress)
3. PotentialFlowJEPAModel (Conservative Potential Field Theory)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.engine.activation_tracer import LayerActivationTracer, LayerAutopsy
from src.models import (
    HybridTCNEncoder,
    OperatorEntropyJEPAModel,
    PotentialFlowJEPAModel,
    ReynoldsStressJEPAModel,
)

try:
    from scripts.trace_model_activations import generate_synthetic_telemetry
except ImportError:
    from trace_model_activations import generate_synthetic_telemetry


def build_operator_entropy(channels: int, device: torch.device) -> nn.Module:
    encoder = HybridTCNEncoder(input_dim=channels, latent_dim=32, filters=48, tcn_layers=3).to(device)
    model = OperatorEntropyJEPAModel(
        context_encoder=encoder,
        latent_dim=32,
        hidden_dim=64,
        predictor_layers=2,
        use_radial_heat=True,
    ).to(device)
    model.eval()
    return model


def build_reynolds_stress(channels: int, device: torch.device) -> nn.Module:
    encoder = HybridTCNEncoder(input_dim=channels, latent_dim=32, filters=48, tcn_layers=3).to(device)
    model = ReynoldsStressJEPAModel(
        context_encoder=encoder,
        latent_dim=32,
        stress_dim=16,
        use_shearing_pulses=True,
        use_admissible_cone=True,
    ).to(device)
    model.eval()
    return model


def build_potential_flow(channels: int, device: torch.device) -> nn.Module:
    encoder = HybridTCNEncoder(input_dim=channels, latent_dim=32, filters=48, tcn_layers=3).to(device)
    model = PotentialFlowJEPAModel(
        context_encoder=encoder,
        latent_dim=32,
        predictor_hidden_dim=64,
        predictor_layers=2,
        n_regimes=4,
        subspace_dim=8,
        use_regimes=True,
    ).to(device)
    model.eval()
    return model


def _forward_model(
    model: nn.Module,
    model_key: str,
    ctx: torch.Tensor,
    tgt: torch.Tensor,
    device: torch.device,
) -> Any:
    """Deterministic forward pass. PotentialFlow gets fixed t / zero noise so
    nominal and anomalous traces traverse the same interpolation point."""
    if model_key == "potential_flow":
        B = ctx.size(0)
        t_batch = torch.linspace(0.1, 0.9, B, device=device, dtype=ctx.dtype)
        z_zero = torch.zeros(B, model.latent_dim, device=device, dtype=ctx.dtype)
        return model(ctx.to(device), tgt.to(device), t=t_batch, z_noise=z_zero)
    return model(ctx.to(device), tgt.to(device))



def deep_trace_model(
    model_key: str,
    display_name: str,
    model: nn.Module,
    ctx_nom: torch.Tensor,
    tgt_nom: torch.Tensor,
    ctx_anom: torch.Tensor,
    tgt_anom: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    # Fit Mahalanobis scoring on nominal telemetry (matches shallow trace protocol)
    model.fit_mahalanobis_covariance(ctx_nom, tgt_nom, batch_size=4)

    with LayerActivationTracer(model, leaf_only=True, capture_tensors=True, cpu_offload=True) as nom_tracer:
        with torch.no_grad():
            _forward_model(model, model_key, ctx_nom, tgt_nom, device)

    with LayerActivationTracer(model, leaf_only=True, capture_tensors=True, cpu_offload=True) as anom_tracer:
        with torch.no_grad():
            _forward_model(model, model_key, ctx_anom, tgt_anom, device)

    report = LayerAutopsy.inspect(nominal_tracer=nom_tracer, anomaly_tracer=anom_tracer, model_name=display_name)

    disc_kw = {"use_mahalanobis": True}
    if model_key == "potential_flow":
        disc_kw["include_curvature"] = True
    disc_nom = model.compute_predictive_discrepancy(ctx_nom.to(device), tgt_nom.to(device), **disc_kw)
    disc_anom = model.compute_predictive_discrepancy(ctx_anom.to(device), tgt_anom.to(device), **disc_kw)

    return {
        "model_name": display_name,
        "report": report,
        "anom_records": anom_tracer.records,
        "metrics": {
            "discrepancy_nominal": disc_nom.mean().item(),
            "discrepancy_anomalous": disc_anom.mean().item(),
            "discrepancy_ratio": (disc_anom.mean() / (disc_nom.mean() + 1e-6)).item(),
        },
    }


def _record_row(r, divergence: Optional[float] = None) -> Dict[str, Any]:
    return {
        "layer_name": r.name,
        "module_type": r.module_type,
        "input_shapes": str(r.input_shapes),
        "output_shape": str(r.output_shape),
        "mean": r.mean,
        "std": r.std,
        "l2_norm": r.norm,
        "min_val": r.min_val,
        "max_val": r.max_val,
        "sparsity": r.sparsity,
        "stable_rank": r.stable_rank,
        "batch_variance": r.batch_variance,
        "health_flag": r.health_flag,
        "anomaly_divergence": divergence,
    }


def print_report(res: Dict[str, Any], top_k: int = 8) -> None:
    report = res["report"]
    print(report.summary())
    print(f"  Predictive discrepancy: {res['metrics']['discrepancy_nominal']:.4f} -> "
          f"{res['metrics']['discrepancy_anomalous']:.4f} "
          f"({res['metrics']['discrepancy_ratio']:.2f}x)")

    if report.anomaly_divergence:
        top = sorted(report.anomaly_divergence.items(), key=lambda x: x[1], reverse=True)[:top_k]
        print(f"\n  Top-{top_k} anomaly-divergent layers:")
        for name, div in top:
            print(f"    {name:<55} {div:>8.4f}")

    ranks = [(r.name, r.stable_rank) for r in report.records if r.stable_rank > 0]
    if ranks:
        low = sorted(ranks, key=lambda x: x[1])[:top_k]
        print(f"\n  Lowest stable-rank layers (dimensional compression):")
        for name, rk in low:
            print(f"    {name:<55} {rk:>8.3f}")

    flagged = [(r.name, r.health_flag) for r in report.records if r.health_flag != "HEALTHY"]
    if flagged:
        print(f"\n  Flagged layers:")
        for name, flag in flagged:
            print(f"    {name:<55} {flag}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"Executing DEEP Activation Trace on device: {device}")

    ctx_nom, tgt_nom, ctx_anom, tgt_anom = generate_synthetic_telemetry(
        batch_size=4, context_len=256, target_len=64, channels=9, seed=42
    )
    channels = ctx_nom.size(-1)

    builders = [
        ("operator_entropy", "OperatorEntropyJEPA", build_operator_entropy),
        ("reynolds_stress", "ReynoldsStressJEPA", build_reynolds_stress),
        ("potential_flow", "PotentialFlowJEPA", build_potential_flow),
    ]

    out_dir = Path("reports/deep_trace")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {}

    for model_key, display_name, builder in builders:
        print("\n" + "=" * 100)
        print(f"DEEP TRACE: {display_name}")
        print("=" * 100)

        torch.manual_seed(42)
        model = builder(channels, device)
        res = deep_trace_model(
            model_key, display_name, model,
            ctx_nom, tgt_nom, ctx_anom, tgt_anom, device,
        )
        print_report(res)

        report = res["report"]
        df = report.to_dataframe()
        df.to_csv(out_dir / f"{model_key}_layers.csv", index=False)

        # Nominal vs. anomalous per-layer comparison CSV
        anom_map = res["anom_records"]
        rows = []
        for r in report.records:
            base_key = r.name.split("#")[0]
            anom_r = anom_map.get(r.name) or anom_map.get(base_key)
            rows.append(
                _record_row(r, divergence=report.anomaly_divergence.get(r.name))
                | {
                    "anom_mean": anom_r.mean if anom_r else None,
                    "anom_std": anom_r.std if anom_r else None,
                    "anom_l2_norm": anom_r.norm if anom_r else None,
                    "anom_sparsity": anom_r.sparsity if anom_r else None,
                    "anom_stable_rank": anom_r.stable_rank if anom_r else None,
                }
            )
        pd.DataFrame(rows).to_csv(out_dir / f"{model_key}_nom_vs_anom.csv", index=False)

        summary[model_key] = {
            "model_name": display_name,
            "total_layers": len(report.records),
            "global_health": report.global_health,
            "collapsed_layers": report.collapsed_layers,
            "dead_layers": report.dead_layers,
            "unstable_layers": report.unstable_layers,
            "metrics": res["metrics"],
            "top_anomaly_divergence": sorted(
                report.anomaly_divergence.items(), key=lambda x: x[1], reverse=True
            )[:15],
            "sequential_drift": report.sequential_drift,
            "layer_table_markdown": report.markdown_table(max_rows=60),
        }

    out_file = out_dir / "deep_trace_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDeep trace artifacts saved to: {out_file.as_posix()}")


if __name__ == "__main__":
    main()
