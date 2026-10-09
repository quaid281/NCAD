#!/usr/bin/env python3
"""Benchmark Computational Profile, Latency, Memory, and Parameter Complexity.

Resolves Reviewer Major Concern 4.2:
Measures on RTX 5090:
- Trainable Parameter Count
- Model Size (MB)
- Forward Inference Latency (ms per window, and ms per 1k timesteps)
- Inference Throughput (windows/sec and timesteps/sec)
- Training Step Throughput (windows/sec)
- Peak GPU Memory Footprint (VRAM in MB)

Evaluates:
1. TS-JEPA (Unconstrained Control)
2. Operator-Entropy JEPA
3. Reynolds-Stress JEPA
4. Potential-Flow JEPA
5. Helmholtz JEPA
6. TimesNet
7. TranAD

Outputs:
- reports/computational_profile_benchmark.csv
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CSMConfig
from src.engine.trainer import build_ts_jepa_model
from src.models.baselines import TimesNet, TranAD


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_model_profile(
    model_name: str,
    input_dim: int = 25,
    context_size: int = 256,
    suspect_size: int = 64,
    device: torch.device = torch.device("cuda"),
    batch_size: int = 32,
    n_warmup: int = 20,
    n_benchmark: int = 100,
) -> dict:
    window_size = context_size + suspect_size
    model_key = "flow_jepa" if model_name == "Helmholtz_JEPA" else model_name.lower().replace("-", "_")
    cfg = CSMConfig(
        model_type=model_key,
        context_size=context_size,
        suspect_size=suspect_size,
        latent_dim=32,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Instantiate model
    if model_name == "TimesNet":
        model = TimesNet(c_in=input_dim, d_model=48, d_ff=96, e_layers=2).to(device)
    elif model_name == "TranAD":
        model = TranAD(c_in=input_dim).to(device)
    else:
        model = build_ts_jepa_model(cfg, input_dim=input_dim, device=device)

    num_params = count_parameters(model)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    # Dummy inputs
    x_batch = torch.randn(batch_size, window_size, input_dim, device=device)
    ctx = x_batch[:, :context_size]
    tgt = x_batch[:, context_size:]

    # 1. Benchmark Inference Latency & Throughput
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            if isinstance(model, TimesNet):
                _ = model(x_batch)
            elif isinstance(model, TranAD):
                _ = model(x_batch)
            else:
                _ = model.compute_predictive_discrepancy(ctx, tgt)
        torch.cuda.synchronize()

        start_time = time.perf_counter()
        for _ in range(n_benchmark):
            if isinstance(model, TimesNet):
                _ = model(x_batch)
            elif isinstance(model, TranAD):
                _ = model(x_batch)
            else:
                _ = model.compute_predictive_discrepancy(ctx, tgt)
        torch.cuda.synchronize()
        infer_total_time = time.perf_counter() - start_time

    total_windows_inferred = batch_size * n_benchmark
    infer_latency_ms_per_window = (infer_total_time / total_windows_inferred) * 1000.0
    infer_throughput_win_sec = total_windows_inferred / infer_total_time
    infer_throughput_steps_sec = infer_throughput_win_sec  # in streaming step=1

    # 2. Benchmark Training Step Latency & Throughput
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    # Warmup
    for _ in range(n_warmup):
        optimizer.zero_grad(set_to_none=True)
        if isinstance(model, TimesNet):
            out = model(x_batch)
            loss = nn.functional.mse_loss(out, x_batch)
        elif isinstance(model, TranAD):
            out = model(x_batch)
            loss = nn.functional.mse_loss(out[0], x_batch)
        else:
            loss, _ = model.compute_objective(ctx, tgt, cfg)
        loss.backward()
        optimizer.step()
        if hasattr(model, "update_target_encoder"):
            model.update_target_encoder()
    torch.cuda.synchronize()

    start_train = time.perf_counter()
    for _ in range(n_benchmark):
        optimizer.zero_grad(set_to_none=True)
        if isinstance(model, TimesNet):
            out = model(x_batch)
            loss = nn.functional.mse_loss(out, x_batch)
        elif isinstance(model, TranAD):
            out = model(x_batch)
            loss = nn.functional.mse_loss(out[0], x_batch)
        else:
            loss, _ = model.compute_objective(ctx, tgt, cfg)
        loss.backward()
        optimizer.step()
        if hasattr(model, "update_target_encoder"):
            model.update_target_encoder()
    torch.cuda.synchronize()
    train_total_time = time.perf_counter() - start_train

    total_windows_trained = batch_size * n_benchmark
    train_throughput_win_sec = total_windows_trained / train_total_time
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return {
        "model": model_name,
        "parameters": num_params,
        "model_size_mb": round(model_size_mb, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "infer_latency_ms_per_window": round(infer_latency_ms_per_window, 4),
        "infer_throughput_win_sec": round(infer_throughput_win_sec, 1),
        "train_throughput_win_sec": round(train_throughput_win_sec, 1),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking computational profiles on device: {device}")

    models = [
        "TS-JEPA",
        "Operator-Entropy_JEPA",
        "Reynolds-Stress_JEPA",
        "Potential-Flow_JEPA",
        "Helmholtz_JEPA",
        "TimesNet",
        "TranAD",
    ]

    records = []
    for m in models:
        print(f"Profiling {m}...")
        res = benchmark_model_profile(m, input_dim=25, device=device)
        records.append(res)

    df = pd.DataFrame(records)
    out_csv = ROOT / "reports" / "computational_profile_benchmark.csv"
    df.to_csv(out_csv, index=False)
    print("\n" + "="*80)
    print("COMPUTATIONAL PROFILE BENCHMARK SUMMARY (RTX 5090)")
    print("="*80)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
