"""Script to perform deep activation autopsies across NCAD-CS model families.

Executes non-invasive layer-to-layer activation tracing, dimensional collapse
detection, numerical stability checks, and perturbation divergence profiling
across JEPA architectures, SOTA baselines, and sequence encoders.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from src.config import CSMConfig
from src.engine.activation_tracer import LayerActivationTracer, LayerAutopsy, autopsy_model
from src.engine.trainer import build_encoder, build_ts_jepa_model
from src.models.baselines import AnomalyTransformer, DCdetector, TimesNet, TranAD
from src.models.losses.anomaly_injector import AnomalyInjectionConfig, ContextualAnomalyInjector


def generate_synthetic_data(
    batch_size: int = 8,
    context_size: int = 64,
    suspect_size: int = 16,
    input_dim: int = 4,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate reproducible nominal and anomalous temporal sequence windows."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Base harmonic / quasi-periodic normal dynamics with distinct sample parameters
    batch_series = []
    t = torch.linspace(0, 4 * np.pi, context_size + suspect_size)
    for b in range(batch_size):
        signals = []
        for c in range(input_dim):
            freq = 1.0 + 0.4 * c + 0.15 * b
            phase = c * (np.pi / 4) + b * 0.3
            sig = torch.sin(freq * t + phase) + 0.15 * torch.randn(len(t))
            signals.append(sig)
        batch_series.append(torch.stack(signals, dim=-1))
    full_series = torch.stack(batch_series, dim=0)

    nom_context = full_series[:, :context_size, :]
    nom_suspect = full_series[:, context_size:, :]

    # Inject contextual anomalies using ContextualAnomalyInjector
    injector = ContextualAnomalyInjector(
        AnomalyInjectionConfig(injection_ratio=1.0),
        seed=seed,
    )
    full_np = full_series.numpy()
    anom_full_np, _ = injector.inject_batch(full_np, context_size=context_size)
    anom_full = torch.from_numpy(anom_full_np).float()
    anom_context = anom_full[:, :context_size, :]
    anom_suspect = anom_full[:, context_size:, :]

    return nom_context, nom_suspect, anom_context, anom_suspect


def run_autopsies(output_dir: Path) -> List[Dict[str, Any]]:
    """Run autopsies across all model families and compile comparative diagnostics."""
    device = torch.device("cpu")
    input_dim = 4
    context_size = 64
    suspect_size = 16
    batch_size = 8

    nom_c, nom_s, anom_c, anom_s = generate_synthetic_data(
        batch_size=batch_size,
        context_size=context_size,
        suspect_size=suspect_size,
        input_dim=input_dim,
    )
    nom_full = torch.cat([nom_c, nom_s], dim=1)
    anom_full = torch.cat([anom_c, anom_s], dim=1)

    results: List[Dict[str, Any]] = []

    # 1. Representative JEPA Variants
    jepa_variants = [
        "ts_jepa",
        "patch_ts_jepa",
        "flow_jepa",
        "causal_ssm_flow_jepa",
        "potential_flow_jepa",
        "latent_world_jepa",
        "recurrent_koopman_jepa",
        "harmonic_spring_jepa",
        "cosine_cycle_jepa",
    ]

    print("\n--- Running Deep Autopsies: JEPA Architectures ---")
    for variant in jepa_variants:
        print(f"-> Autopsying JEPA: {variant}...")
        config = CSMConfig(
            model_type=variant,
            context_size=context_size,
            suspect_size=suspect_size,
            patch_size=8,
            latent_dim=16,
            filters=16,
            tcn_layers=2,
            rank=2,
            n_regimes=2,
            subspace_dim=4,
        )
        model = build_ts_jepa_model(config, input_dim=input_dim, device=device)
        report = autopsy_model(
            model=model,
            sample_input=(nom_c, nom_s),
            anomalous_input=(anom_c, anom_s),
            model_name=variant,
        )
        ranks = [r.stable_rank for r in report.records if r.stable_rank > 0]
        avg_rank = float(np.mean(ranks)) if ranks else 0.0
        peak_div = max(report.anomaly_divergence.values()) if report.anomaly_divergence else 0.0
        peak_div_layer = (
            max(report.anomaly_divergence.items(), key=lambda x: x[1])[0]
            if report.anomaly_divergence
            else "N/A"
        )
        drift_sims = list(report.sequential_drift.values())
        avg_drift = float(np.mean(drift_sims)) if drift_sims else 0.0

        res = {
            "family": "JEPA",
            "model_name": variant,
            "total_layers": len(report.records),
            "global_health": report.global_health,
            "collapsed_layers": len(report.collapsed_layers),
            "dead_layers": len(report.dead_layers),
            "unstable_layers": len(report.unstable_layers),
            "avg_stable_rank": round(avg_rank, 2),
            "avg_layer_similarity": round(avg_drift, 3),
            "peak_anomaly_divergence": round(peak_div, 3),
            "peak_divergence_layer": peak_div_layer,
        }
        results.append(res)

    # 2. Modern Baselines
    print("\n--- Running Deep Autopsies: SOTA Baselines ---")
    baselines = [
        (
            "AnomalyTransformer",
            AnomalyTransformer(c_in=input_dim, d_model=32, n_heads=2, e_layers=2, d_ff=64),
        ),
        (
            "DCdetector",
            DCdetector(c_in=input_dim, patch_size1=8, patch_size2=16, d_model=32, n_heads=2, e_layers=2),
        ),
        (
            "TimesNet",
            TimesNet(c_in=input_dim, d_model=32, d_ff=32, e_layers=2, top_k=2),
        ),
        (
            "TranAD",
            TranAD(c_in=input_dim, d_model=32, n_heads=2, e_layers=2, d_layers=2, d_ff=64),
        ),
    ]

    for name, b_model in baselines:
        print(f"-> Autopsying Baseline: {name}...")
        report = autopsy_model(
            model=b_model,
            sample_input=nom_full,
            anomalous_input=anom_full,
            model_name=name,
        )
        ranks = [r.stable_rank for r in report.records if r.stable_rank > 0]
        avg_rank = float(np.mean(ranks)) if ranks else 0.0
        peak_div = max(report.anomaly_divergence.values()) if report.anomaly_divergence else 0.0
        peak_div_layer = (
            max(report.anomaly_divergence.items(), key=lambda x: x[1])[0]
            if report.anomaly_divergence
            else "N/A"
        )
        drift_sims = list(report.sequential_drift.values())
        avg_drift = float(np.mean(drift_sims)) if drift_sims else 0.0

        res = {
            "family": "Baseline",
            "model_name": name,
            "total_layers": len(report.records),
            "global_health": report.global_health,
            "collapsed_layers": len(report.collapsed_layers),
            "dead_layers": len(report.dead_layers),
            "unstable_layers": len(report.unstable_layers),
            "avg_stable_rank": round(avg_rank, 2),
            "avg_layer_similarity": round(avg_drift, 3),
            "peak_anomaly_divergence": round(peak_div, 3),
            "peak_divergence_layer": peak_div_layer,
        }
        results.append(res)

    # 3. Sequence Encoders
    print("\n--- Running Deep Autopsies: Sequence Encoders ---")
    encoders = [
        ("HybridTCNEncoder", build_encoder(CSMConfig(encoder_architecture="hybrid_tcn", latent_dim=16, filters=16, tcn_layers=2), input_dim, device)),
        ("MultiScaleTCNEncoder", build_encoder(CSMConfig(encoder_architecture="multi_scale_tcn", latent_dim=16, filters=16, tcn_layers=2), input_dim, device)),
        ("RelationalGATEncoder", build_encoder(CSMConfig(encoder_architecture="relational_gat", latent_dim=16, filters=16, tcn_layers=2), input_dim, device)),
        ("SelectiveSSMContextEncoder", build_encoder(CSMConfig(encoder_architecture="selective_ssm", latent_dim=16, filters=16, tcn_layers=2), input_dim, device)),
    ]

    for name, enc_model in encoders:
        print(f"-> Autopsying Encoder: {name}...")
        report = autopsy_model(
            model=enc_model,
            sample_input=nom_full,
            anomalous_input=anom_full,
            model_name=name,
        )
        ranks = [r.stable_rank for r in report.records if r.stable_rank > 0]
        avg_rank = float(np.mean(ranks)) if ranks else 0.0
        peak_div = max(report.anomaly_divergence.values()) if report.anomaly_divergence else 0.0
        peak_div_layer = (
            max(report.anomaly_divergence.items(), key=lambda x: x[1])[0]
            if report.anomaly_divergence
            else "N/A"
        )
        drift_sims = list(report.sequential_drift.values())
        avg_drift = float(np.mean(drift_sims)) if drift_sims else 0.0

        res = {
            "family": "Encoder",
            "model_name": name,
            "total_layers": len(report.records),
            "global_health": report.global_health,
            "collapsed_layers": len(report.collapsed_layers),
            "dead_layers": len(report.dead_layers),
            "unstable_layers": len(report.unstable_layers),
            "avg_stable_rank": round(avg_rank, 2),
            "avg_layer_similarity": round(avg_drift, 3),
            "peak_anomaly_divergence": round(peak_div, 3),
            "peak_divergence_layer": peak_div_layer,
        }
        results.append(res)

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = output_dir / "autopsy_cross_model_summary.csv"
    json_path = output_dir / "autopsy_cross_model_summary.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved cross-model autopsy summary to {csv_path}")
    print("\n" + df.to_string(index=False))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deep activation autopsy across models.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/autopsy",
        help="Directory to save autopsy findings.",
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    run_autopsies(out_dir)


if __name__ == "__main__":
    main()
