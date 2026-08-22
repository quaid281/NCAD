"""Evaluation runner comparing EVT calibration vs Legacy Elbow Floor on mTSBench benchmarks."""

import argparse
import glob
import os
from pathlib import Path
import sys
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


def evaluate_dataset(
    dataset_name: str = "Daphnet",
    max_channels: int = 6,
    epochs: int = 15,
    risk_level: float = 1e-3,
    encoder_arch: str = "relational_gat",
    device_str: str = "auto",
):
    device = torch.device("cuda" if torch.cuda.is_available() and device_str == "auto" else device_str if device_str != "auto" else "cpu")
    dataset_dir = ROOT / "mTSBench_data" / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    train_files = sorted(glob.glob(str(dataset_dir / f"{dataset_name}*_train.csv")))
    if not train_files:
        raise FileNotFoundError(f"No train files found for {dataset_name} in {dataset_dir}")

    channels = []
    for f in train_files:
        fname = Path(f).name
        if fname == f"{dataset_name}_train.csv":
            chan = ""
        else:
            chan = fname.replace(f"{dataset_name}_", "").replace("_train.csv", "")
        channels.append(chan)

    if max_channels > 0:
        channels = channels[:max_channels]

    context_size = 256
    suspect_size = 64
    window_size = context_size + suspect_size
    batch_size = 32
    latent_dim = 32

    print(f"\n=======================================================")
    print(f"Evaluating {dataset_name} ({len(channels)} channels) on {device} | Encoder: {encoder_arch}")
    print(f"EVT Risk Level: {risk_level} | Window: {window_size} (Ctx:{context_size}, Susp:{suspect_size})")
    print(f"=======================================================\n")

    results = []

    for idx, chan in enumerate(channels):
        chan_display = chan if chan != "" else "default"
        print(f"[{idx+1}/{len(channels)}] Channel: {chan_display} ... ", end="", flush=True)

        if chan == "":
            train_path = dataset_dir / f"{dataset_name}_train.csv"
            test_path = dataset_dir / f"{dataset_name}_test.csv"
        else:
            train_path = dataset_dir / f"{dataset_name}_{chan}_train.csv"
            test_path = dataset_dir / f"{dataset_name}_{chan}_test.csv"

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        numeric_cols = [c for c in train_df.columns if c not in ["timestamp", "is_anomaly"]]
        test_labels = test_df["is_anomaly"].to_numpy() if "is_anomaly" in test_df.columns else None

        train_mean = train_df[numeric_cols].mean().to_numpy()
        train_std = train_df[numeric_cols].std().to_numpy()
        scale = np.maximum(train_std, 0.01)
        train_scaled = (train_df[numeric_cols].to_numpy() - train_mean) / scale
        test_scaled = (test_df[numeric_cols].to_numpy() - train_mean) / scale

        train_windows = DataLoader.create_windows(train_scaled, window_size, step=10)
        test_windows = DataLoader.create_windows(test_scaled, window_size, step=1)

        if len(train_windows) < 10 or len(test_windows) < 10:
            print("Skipped (Too few windows)")
            continue

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
            raise ValueError(f"Unknown encoder architecture: {encoder_arch}")

        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        injector = ContextualAnomalyInjector(
            AnomalyInjectionConfig(injection_ratio=0.5, min_anomaly_len=16, max_anomaly_len=64),
            seed=42,
        )

        best_val_loss = float("inf")
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
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for b in range(0, len(val_data), batch_size):
                    clean = val_data[b : b + batch_size]
                    mod, lbl = injector.inject_batch(clean, context_size)
                    v_loss = contrastive_loss(
                        model(torch.from_numpy(mod).float().to(device)),
                        model(torch.from_numpy(clean[:, :context_size]).float().to(device)),
                        torch.from_numpy(lbl).float().to(device),
                    )
                    val_loss += float(v_loss.item()) * len(clean)
                    val_count += len(clean)
            val_loss /= max(val_count, 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 4:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Build Counterfactual Successor Memory
        train_ctx_emb = encode_windows(model, train_windows[:, :context_size], device=device)
        train_suc_emb = encode_windows(model, train_windows[:, context_size:], device=device)
        memory = CounterfactualSuccessorMemory(SuccessorMemoryConfig(n_neighbors=8, max_memory_windows=4000, seed=42))
        memory.fit(train_ctx_emb, train_suc_emb)

        train_local = local_deviation_scores(train_windows, context_size)
        succ_stats = robust_stats(memory.calibration_successor_scores)
        local_stats = robust_stats(train_local[memory.sample_indices])

        # Query test set
        test_ctx_emb = encode_windows(model, test_windows[:, :context_size], batch_size=2048, device=device)
        test_suc_emb = encode_windows(model, test_windows[:, context_size:], batch_size=2048, device=device)
        query = memory.query(test_ctx_emb, test_suc_emb)
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
        # 1. Calibrate on Training Normal Scores
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

        # 2. EVT Threshold Calibration (on clean normal training distribution)
        evt_res = calibrate_evt_threshold(train_valid_scores, risk_level=risk_level, init_percentile=98.0)
        evt_th = evt_res.threshold

        # 3. Legacy Elbow Threshold (test floor)
        floor_res = adaptive_elbow_score_floor(valid_scores)
        elbow_th = floor_res.threshold

        # Compute Metrics
        preds_elbow = event_level_filter(test_scores, elbow_th, valid_mask, min_run=2, extreme_factor=1.75) * valid_mask.astype(np.float32)
        m_elbow_pa = compute_metrics(test_labels, preds_elbow, valid_mask=valid_mask, use_pa=True)

        preds_evt = event_level_filter(test_scores, evt_th, valid_mask, min_run=2, extreme_factor=1.75) * valid_mask.astype(np.float32)
        m_evt_pa = compute_metrics(test_labels, preds_evt, valid_mask=valid_mask, use_pa=True)

        # Oracle Best
        best_pa_f1 = 0.0
        candidates = np.percentile(valid_scores, np.linspace(0.0, 100.0, 150))
        for th in candidates:
            p = event_level_filter(test_scores, th, valid_mask, min_run=2, extreme_factor=1.75) * valid_mask.astype(np.float32)
            m = compute_metrics(test_labels, p, valid_mask=valid_mask, use_pa=True)
            if m.get("f1", 0.0) > best_pa_f1:
                best_pa_f1 = m["f1"]

        f1_elbow = m_elbow_pa.get("f1", 0.0)
        f1_evt = m_evt_pa.get("f1", 0.0)
        print(f"Legacy Elbow PA-F1: {f1_elbow:.4f} | EVT PA-F1: {f1_evt:.4f} | Oracle: {best_pa_f1:.4f}")

        results.append({
            "channel": chan_display,
            "length": len(test_df),
            "elbow_threshold": elbow_th,
            "elbow_pa_f1": f1_elbow,
            "evt_threshold": evt_th,
            "evt_pa_f1": f1_evt,
            "oracle_pa_f1": best_pa_f1,
            "evt_method": evt_res.method,
            "gamma": evt_res.gpd_fit.gamma,
            "sigma": evt_res.gpd_fit.sigma,
        })

    summary_df = pd.DataFrame(results)
    print("\n=======================================================")
    print("Benchmark Comparison Summary:")
    print(summary_df[["channel", "elbow_pa_f1", "evt_pa_f1", "oracle_pa_f1", "evt_method"]].to_string(index=False))
    print(f"\nAverage Legacy Elbow PA-F1: {summary_df['elbow_pa_f1'].mean():.4f}")
    print(f"Average New EVT PA-F1:      {summary_df['evt_pa_f1'].mean():.4f}")
    print(f"Average Oracle PA-F1:       {summary_df['oracle_pa_f1'].mean():.4f}")
    print("=======================================================\n")
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Daphnet", help="Dataset name (e.g. Daphnet, OPPORTUNITY, CalIt2, GECCO)")
    parser.add_argument("--max-channels", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--risk-level", type=float, default=1e-3)
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["relational_gat", "hybrid_tcn", "ssm"],
        default="relational_gat",
        help="Encoder architecture to benchmark.",
    )
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    evaluate_dataset(
        dataset_name=args.dataset,
        max_channels=args.max_channels,
        epochs=args.epochs,
        risk_level=args.risk_level,
        encoder_arch=args.encoder,
        device_str=args.device,
    )
