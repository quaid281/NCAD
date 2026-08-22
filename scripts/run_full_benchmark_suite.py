"""Comprehensive multi-dataset benchmark evaluation suite for NCAD-CS v5.

Evaluates NCAD-CS v5 (Spatial-Temporal Relational GAT + EVT/SPOT Tail Calibration)
across mTSBench benchmark datasets and exports summary reports, CSVs, and LaTeX tables.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.anomaly_injector import AnomalyInjectionConfig, ContextualAnomalyInjector
from src.models.tcn_encoder import HybridTCNEncoder, contrastive_loss
from src.models.relational_gat_encoder import RelationalGATEncoder
from src.models.selective_ssm_encoder import SelectiveSSMContextEncoder
from src.models.successor_memory import CounterfactualSuccessorMemory, SuccessorMemoryConfig
from src.data.data_loader import DataLoader
from src.models.train_model import encode_windows, split_train_validation
from src.utils.event_fusion import (
    adaptive_elbow_score_floor,
    aggregate_window_scores,
    calibrate_evt_threshold,
    compute_metrics,
    event_level_filter,
    fuse_evidence_scores,
    local_deviation_scores,
    moving_average,
    positive_robust_z,
    robust_stats,
)
from src.utils.evt_calibrator import EVTCalibrator


def evaluate_single_channel(
    train_path: Path,
    test_path: Path,
    chan_name: str,
    context_size: int = 256,
    suspect_size: int = 64,
    epochs: int = 10,
    risk_level: float = 1e-3,
    encoder_arch: str = "relational_gat",
    device: torch.device = torch.device("cpu"),
) -> Optional[Dict]:
    window_size = context_size + suspect_size
    batch_size = 32
    latent_dim = 32

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    numeric_cols = [c for c in train_df.columns if c not in ["timestamp", "is_anomaly"]]
    test_labels = test_df["is_anomaly"].to_numpy() if "is_anomaly" in test_df.columns else None
    if test_labels is None:
        return None

    train_mean = train_df[numeric_cols].mean().to_numpy()
    train_std = train_df[numeric_cols].std().to_numpy()
    scale = np.maximum(train_std, 0.01)
    train_scaled = (train_df[numeric_cols].to_numpy() - train_mean) / scale
    test_scaled = (test_df[numeric_cols].to_numpy() - train_mean) / scale

    train_windows = DataLoader.create_windows(train_scaled, window_size, step=10)
    test_windows = DataLoader.create_windows(test_scaled, window_size, step=1)

    if len(train_windows) < 10 or len(test_windows) < 10:
        return None

    training_data, val_data = split_train_validation(train_windows, val_split=0.1, seed=42)

    if encoder_arch == "relational_gat":
        model = RelationalGATEncoder(
            input_dim=len(numeric_cols),
            latent_dim=latent_dim,
            filters=48,
            tcn_layers=3,
            gat_layers=2,
            gat_heads=4,
            dropout=0.20,
        ).to(device)
    elif encoder_arch == "hybrid_tcn":
        model = HybridTCNEncoder(
            input_dim=len(numeric_cols),
            latent_dim=latent_dim,
            filters=48,
            tcn_layers=3,
            dropout=0.20,
        ).to(device)
    elif encoder_arch == "ssm":
        model = SelectiveSSMContextEncoder(
            input_dim=len(numeric_cols),
            latent_dim=latent_dim,
            hidden_dim=48,
            layers=2,
            dropout=0.20,
        ).to(device)
    else:
        raise ValueError(f"Unknown encoder: {encoder_arch}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    injector = ContextualAnomalyInjector(AnomalyInjectionConfig(injection_ratio=0.5, min_anomaly_len=16, max_anomaly_len=64), seed=42)

    best_val = float("inf")
    best_state = None
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        indices = np.random.permutation(len(training_data))
        for b in range(0, len(indices), batch_size):
            b_idx = indices[b : b + batch_size]
            clean = training_data[b_idx]
            mod, lbl = injector.inject_batch(clean, context_size)
            loss = contrastive_loss(
                model(torch.from_numpy(mod).float().to(device)),
                model(torch.from_numpy(clean[:, :context_size]).float().to(device)),
                torch.from_numpy(lbl).float().to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        v_loss = 0.0
        v_cnt = 0
        with torch.no_grad():
            for b in range(0, len(val_data), batch_size):
                clean = val_data[b : b + batch_size]
                mod, lbl = injector.inject_batch(clean, context_size)
                l = contrastive_loss(
                    model(torch.from_numpy(mod).float().to(device)),
                    model(torch.from_numpy(clean[:, :context_size]).float().to(device)),
                    torch.from_numpy(lbl).float().to(device),
                )
                v_loss += float(l.item()) * len(clean)
                v_cnt += len(clean)
        val_score = v_loss / max(v_cnt, 1)

        if val_score < best_val - 1e-4:
            best_val = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 4:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_ctx = encode_windows(model, train_windows[:, :context_size], batch_size=2048, device=device)
    train_suc = encode_windows(model, train_windows[:, context_size:], batch_size=2048, device=device)
    memory = CounterfactualSuccessorMemory(SuccessorMemoryConfig(n_neighbors=8, max_memory_windows=4000, seed=42))
    memory.fit(train_ctx, train_suc)

    train_local = local_deviation_scores(train_windows, context_size)
    succ_stats = robust_stats(memory.calibration_successor_scores)
    local_stats = robust_stats(train_local[memory.sample_indices])

    # Test scoring
    test_ctx = encode_windows(model, test_windows[:, :context_size], batch_size=2048, device=device)
    test_suc = encode_windows(model, test_windows[:, context_size:], batch_size=2048, device=device)
    query = memory.query(test_ctx, test_suc)
    test_local = local_deviation_scores(test_windows, context_size)

    succ_z = positive_robust_z(query.successor_scores, succ_stats, clip=20.0)
    loc_z = positive_robust_z(test_local, local_stats, clip=20.0)
    ctx_ratio = query.context_distances / max(float(memory.context_threshold), 1e-6)
    ctx_ratio = np.minimum(ctx_ratio, 3.0).astype(np.float32)

    win_scores = fuse_evidence_scores(successor_z=succ_z, local_z=loc_z, context_ratio=ctx_ratio)
    pt_scores, valid_mask = aggregate_window_scores(
        win_scores,
        n_points=len(test_df),
        context_size=context_size,
        suspect_size=suspect_size,
        step=1,
        reducer="mean",
        mapping_method="middle",
    )
    test_scores = moving_average(pt_scores, 12)
    valid_scores = test_scores[valid_mask]

    # 1. Training Calibration Distribution for EVT
    train_succ_z = positive_robust_z(memory.calibration_successor_scores, succ_stats, clip=20.0)
    train_loc_z = positive_robust_z(train_local[memory.sample_indices], local_stats, clip=20.0)
    train_ctx_ratio = memory.calibration_context_distances / max(float(memory.context_threshold), 1e-6)
    train_ctx_ratio = np.minimum(train_ctx_ratio, 3.0).astype(np.float32)

    train_win_scores = fuse_evidence_scores(successor_z=train_succ_z, local_z=train_loc_z, context_ratio=train_ctx_ratio)
    train_pt_scores, train_valid_mask = aggregate_window_scores(
        train_win_scores,
        n_points=(len(train_windows) - 1) * 10 + window_size,
        context_size=context_size,
        suspect_size=suspect_size,
        step=10,
        window_indices=memory.sample_indices,
        reducer="mean",
        mapping_method="middle",
    )
    train_smoothed = moving_average(train_pt_scores, 12)
    train_valid_scores = train_smoothed[train_valid_mask]

    # EVT Calibration
    evt_res = calibrate_evt_threshold(train_valid_scores, risk_level=risk_level, init_percentile=98.0)
    evt_th = evt_res.threshold

    # Legacy Elbow Floor
    floor_res = adaptive_elbow_score_floor(valid_scores)
    elbow_th = floor_res.threshold

    # Metrics
    preds_elbow = event_level_filter(test_scores, elbow_th, valid_mask, min_run=2, extreme_factor=1.75) * valid_mask.astype(np.float32)
    m_elbow_pa = compute_metrics(test_labels, preds_elbow, valid_mask=valid_mask, use_pa=True)
    m_elbow_std = compute_metrics(test_labels, preds_elbow, valid_mask=valid_mask, use_pa=False)

    preds_evt = event_level_filter(test_scores, evt_th, valid_mask, min_run=2, extreme_factor=1.75) * valid_mask.astype(np.float32)
    m_evt_pa = compute_metrics(test_labels, preds_evt, valid_mask=valid_mask, use_pa=True)
    m_evt_std = compute_metrics(test_labels, preds_evt, valid_mask=valid_mask, use_pa=False)

    # Oracle Best
    best_pa_f1 = 0.0
    best_std_f1 = 0.0
    candidates = np.percentile(valid_scores, np.linspace(0.0, 100.0, 150))
    for th in candidates:
        p = event_level_filter(test_scores, th, valid_mask, min_run=2, extreme_factor=1.75) * valid_mask.astype(np.float32)
        mpa = compute_metrics(test_labels, p, valid_mask=valid_mask, use_pa=True)
        mstd = compute_metrics(test_labels, p, valid_mask=valid_mask, use_pa=False)
        if mpa.get("f1", 0.0) > best_pa_f1:
            best_pa_f1 = mpa["f1"]
        if mstd.get("f1", 0.0) > best_std_f1:
            best_std_f1 = mstd["f1"]

    return {
        "channel": chan_name,
        "length": len(test_df),
        "anomaly_ratio": float(np.mean(test_labels)),
        "elbow_threshold": elbow_th,
        "elbow_std_f1": m_elbow_std.get("f1", 0.0),
        "elbow_pa_f1": m_elbow_pa.get("f1", 0.0),
        "evt_threshold": evt_th,
        "evt_std_f1": m_evt_std.get("f1", 0.0),
        "evt_pa_f1": m_evt_pa.get("f1", 0.0),
        "oracle_std_f1": best_std_f1,
        "oracle_pa_f1": best_pa_f1,
        "evt_method": evt_res.method,
        "gamma": evt_res.gpd_fit.gamma,
        "sigma": evt_res.gpd_fit.sigma,
    }


def run_benchmark_suite(
    datasets: Optional[List[str]] = None,
    max_channels_per_dataset: int = 10,
    epochs: int = 8,
    risk_level: float = 1e-3,
    encoder_arch: str = "relational_gat",
    device_str: str = "auto",
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() and device_str == "auto" else device_str if device_str != "auto" else "cpu")
    out_dir = ROOT / "results" / "benchmark_v5"
    out_dir.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        datasets = ["Daphnet", "OPPORTUNITY", "CalIt2", "Exathlon", "GECCO", "Genesis", "room-occupancy"]

    print(f"===============================================================")
    print(f"NCAD-CS v5 Multi-Dataset Benchmark Sweep on {device}")
    print(f"Encoder: {encoder_arch} | EVT Risk Level: {risk_level} | Epochs: {epochs}")
    print(f"===============================================================\n")

    all_summaries = []

    for dname in datasets:
        dataset_dir = ROOT / "mTSBench_data" / dname
        if not dataset_dir.exists():
            print(f"Dataset {dname} not found, skipping.")
            continue

        train_files = sorted(glob.glob(str(dataset_dir / f"{dname}*_train.csv")))
        if not train_files:
            train_files = sorted(glob.glob(str(dataset_dir / "*_train.csv")))
        if not train_files:
            continue

        channels = []
        for f in train_files:
            fname = Path(f).name
            if fname == f"{dname}_train.csv":
                chan = ""
            else:
                chan = fname.replace(f"{dname}_", "").replace("_train.csv", "")
            channels.append((f, chan))

        if max_channels_per_dataset > 0:
            channels = channels[:max_channels_per_dataset]

        print(f"\n--- Running Dataset: {dname} ({len(channels)} channels) ---")
        dataset_results = []

        for idx, (train_f, chan) in enumerate(channels):
            chan_display = chan if chan != "" else "default"
            print(f"  [{idx+1}/{len(channels)}] {chan_display} ... ", end="", flush=True)

            if chan == "":
                test_f = dataset_dir / f"{dname}_test.csv"
            else:
                test_f = dataset_dir / f"{dname}_{chan}_test.csv"
                if not test_f.exists():
                    test_f = Path(str(train_f).replace("_train.csv", "_test.csv"))

            if not test_f.exists():
                print("Missing test file, skipping.")
                continue

            try:
                res = evaluate_single_channel(
                    train_path=Path(train_f),
                    test_path=Path(test_f),
                    chan_name=chan_display,
                    epochs=epochs,
                    risk_level=risk_level,
                    encoder_arch=encoder_arch,
                    device=device,
                )
                if res is not None:
                    res["dataset"] = dname
                    dataset_results.append(res)
                    print(f"Elbow PA-F1: {res['elbow_pa_f1']:.4f} | EVT PA-F1: {res['evt_pa_f1']:.4f} | Oracle: {res['oracle_pa_f1']:.4f}")
                else:
                    print("Skipped (no valid labels/windows)")
            except Exception as e:
                print(f"Error: {e}")

        if dataset_results:
            df_dataset = pd.DataFrame(dataset_results)
            df_dataset.to_csv(out_dir / f"{dname}_evaluation.csv", index=False)
            all_summaries.extend(dataset_results)

    all_df = pd.DataFrame(all_summaries)
    if not all_df.empty:
        all_df.to_csv(out_dir / "all_datasets_summary.csv", index=False)

        # Aggregate by dataset
        agg = all_df.groupby("dataset").agg({
            "channel": "count",
            "elbow_pa_f1": "mean",
            "evt_pa_f1": "mean",
            "oracle_pa_f1": "mean",
            "elbow_std_f1": "mean",
            "evt_std_f1": "mean",
            "oracle_std_f1": "mean",
        }).rename(columns={"channel": "num_channels"}).reset_index()

        print("\n===============================================================")
        print("CONSOLIDATED MULTI-DATASET BENCHMARK SUMMARY (NCAD-CS v5)")
        print("===============================================================")
        print(agg[["dataset", "num_channels", "elbow_pa_f1", "evt_pa_f1", "oracle_pa_f1"]].to_string(index=False))
        print(f"\nTotal Channels Evaluated: {len(all_df)}")
        print(f"Overall Mean Legacy Elbow PA-F1: {all_df['elbow_pa_f1'].mean():.4f}")
        print(f"Overall Mean NCAD-CS v5 EVT PA-F1: {all_df['evt_pa_f1'].mean():.4f}")
        print(f"Overall Mean Oracle Ceiling PA-F1: {all_df['oracle_pa_f1'].mean():.4f}")
        print("===============================================================\n")

    return all_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NCAD-CS v5 Multi-Dataset Benchmark Suite.")
    parser.add_argument("--datasets", type=str, nargs="+", default=["Daphnet", "OPPORTUNITY", "CalIt2", "Exathlon", "GECCO", "Genesis", "room-occupancy"])
    parser.add_argument("--max-channels", type=int, default=8, help="Max channels to run per dataset.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--risk-level", type=float, default=1e-3)
    parser.add_argument("--encoder", type=str, default="relational_gat")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_benchmark_suite(
        datasets=args.datasets,
        max_channels_per_dataset=args.max_channels,
        epochs=args.epochs,
        risk_level=args.risk_level,
        encoder_arch=args.encoder,
        device_str=args.device,
    )
