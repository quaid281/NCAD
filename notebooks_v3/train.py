"""NCAD-CS v3 training and inference pipeline.

This is a clean implementation of the method described in paper/NCAD_CS.sty:
feature engineering, contrastive TCN encoder training, Context Memory Bank
construction, confidence-weighted context substitution, smoothing, robust
thresholding, and point-level evaluation.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from models.anomaly_injector import AnomalyInjectionConfig, ContextualAnomalyInjector
from models.memory_bank import ContextMemoryBank, MemoryBankConfig
from models.tcn_encoder import HybridTCNEncoder, contrastive_loss
from utils.data_loader import ChannelData, DataLoader
from utils.features import FeatureConfig, NCADFeatureExtractor
from utils.scoring import (
    aggregate_suspect_scores,
    calculate_label_calibrated_threshold,
    calculate_threshold,
    compute_metrics,
    confidence_over_threshold,
    dynamic_weighted_smoothing,
    euclidean_distance,
    temporal_consistency_filter,
)
from utils.visualization import plot_channel_diagnostics


@dataclass
class NCADCSConfig:
    data_dir: Optional[str] = None
    output_dir: Optional[str] = None
    context_size: int = 284
    suspect_size: int = 16
    step: int = 1
    feature_dim: int = 64
    latent_dim: int = 16
    filters: int = 64
    tcn_layers: int = 4
    kernel_size: int = 5
    dropout: float = 0.20
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    margin: float = 1.0
    val_split: float = 0.10
    patience: int = 8
    injection_ratio: float = 0.70
    min_clusters: int = 2
    max_clusters: int = 50
    memory_percentile: float = 99.0
    boost_factor: float = 2.5
    min_substitution_confidence: float = 0.10
    threshold_percentile: float = 95.0
    threshold_alpha: float = 1.0
    threshold_tail_guard_multiplier: float = 5.0
    threshold_tail_guard_percentile: float = 95.0
    threshold_sparse_tail_guard_multiplier: float = 3.0
    threshold_sparse_tail_guard_percentile: float = 97.0
    threshold_sparse_tail_guard_max_base: float = 0.02
    max_threshold_calibration_windows: Optional[int] = 1024
    min_threshold_calibration_f1: float = 0.10
    use_original_on_degenerate_memory: bool = True
    degenerate_memory_threshold: float = 1e-8
    short_smoothing_window: int = 50
    long_smoothing_window: int = 200
    min_prediction_run: int = 3
    extreme_score_factor: float = 1.8
    max_train_windows: Optional[int] = None
    max_test_windows: Optional[int] = None
    max_memory_windows: Optional[int] = 5000
    save_plots: bool = True
    seed: int = 42
    device: str = "auto"

    @property
    def full_window_size(self) -> int:
        return self.context_size + self.suspect_size


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def derive_channel_seed(base_seed: int, channel_id: str) -> int:
    return int(base_seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / "results" / timestamp


def encode_windows(model: HybridTCNEncoder, windows: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size]).float().to(device)
            embeddings.append(model(batch).cpu().numpy())
    if not embeddings:
        return np.empty((0, model.latent_dim), dtype=np.float32)
    return np.concatenate(embeddings, axis=0).astype(np.float32)


def limit_windows(windows: np.ndarray, max_windows: Optional[int]) -> np.ndarray:
    if max_windows is None or len(windows) <= max_windows:
        return windows
    return windows[:max_windows]


def split_train_validation(windows: np.ndarray, val_split: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(windows))
    rng.shuffle(indices)

    if len(indices) < 10:
        return windows[indices], np.empty((0,) + windows.shape[1:], dtype=windows.dtype)

    n_val = max(1, int(len(indices) * val_split))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    return windows[train_indices], windows[val_indices]


def train_encoder(train_windows: np.ndarray, config: NCADCSConfig, input_dim: int, device: torch.device) -> tuple[HybridTCNEncoder, dict]:
    model = HybridTCNEncoder(
        input_dim=input_dim,
        latent_dim=config.latent_dim,
        filters=config.filters,
        tcn_layers=config.tcn_layers,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    injector = ContextualAnomalyInjector(
        AnomalyInjectionConfig(injection_ratio=config.injection_ratio), seed=config.seed
    )
    val_injector = ContextualAnomalyInjector(
        AnomalyInjectionConfig(injection_ratio=config.injection_ratio), seed=config.seed + 1
    )
    training_data, validation_data = split_train_validation(train_windows, config.val_split, config.seed)

    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_indices = np.random.permutation(len(training_data))
        total_loss = 0.0
        total_count = 0

        for batch_start in range(0, len(epoch_indices), config.batch_size):
            batch_indices = epoch_indices[batch_start : batch_start + config.batch_size]
            clean_batch = training_data[batch_indices]
            modified_batch, labels = injector.inject_batch(clean_batch, config.context_size)

            full_tensor = torch.from_numpy(modified_batch).float().to(device)
            context_tensor = torch.from_numpy(clean_batch[:, : config.context_size]).float().to(device)
            label_tensor = torch.from_numpy(labels).float().to(device)

            optimizer.zero_grad(set_to_none=True)
            z_full = model(full_tensor)
            z_context = model(context_tensor)
            loss = contrastive_loss(z_full, z_context, label_tensor, margin=config.margin)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item()) * len(clean_batch)
            total_count += len(clean_batch)

        train_loss = total_loss / max(total_count, 1)
        val_loss = evaluate_contrastive_loss(model, validation_data, val_injector, config, device)
        if not np.isfinite(val_loss):
            val_loss = train_loss

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"  epoch {epoch:03d}/{config.epochs}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"  early stopping after {epoch} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_loss"] = best_val_loss
    return model, history


def evaluate_contrastive_loss(
    model: HybridTCNEncoder,
    validation_data: np.ndarray,
    injector: ContextualAnomalyInjector,
    config: NCADCSConfig,
    device: torch.device,
) -> float:
    if len(validation_data) == 0:
        return float("nan")

    model.eval()
    losses = []
    counts = []
    with torch.no_grad():
        for batch_start in range(0, len(validation_data), config.batch_size):
            clean_batch = validation_data[batch_start : batch_start + config.batch_size]
            modified_batch, labels = injector.inject_batch(clean_batch, config.context_size)
            full_tensor = torch.from_numpy(modified_batch).float().to(device)
            context_tensor = torch.from_numpy(clean_batch[:, : config.context_size]).float().to(device)
            label_tensor = torch.from_numpy(labels).float().to(device)
            loss = contrastive_loss(model(full_tensor), model(context_tensor), label_tensor, margin=config.margin)
            losses.append(float(loss.item()) * len(clean_batch))
            counts.append(len(clean_batch))
    return sum(losses) / max(sum(counts), 1)


def build_memory_bank(
    model: HybridTCNEncoder,
    train_windows: np.ndarray,
    config: NCADCSConfig,
    device: torch.device,
) -> tuple[ContextMemoryBank, np.ndarray, dict]:
    context_windows = train_windows[:, : config.context_size]
    context_windows_for_bank = limit_windows(context_windows, config.max_memory_windows)
    context_embeddings = encode_windows(model, context_windows_for_bank, config.batch_size, device)

    memory_bank = ContextMemoryBank(
        MemoryBankConfig(
            min_clusters=config.min_clusters,
            max_clusters=config.max_clusters,
            percentile=config.memory_percentile,
            distance_metric="euclidean",
        )
    ).fit(context_embeddings, context_windows_for_bank)

    change_scores = np.linalg.norm(np.diff(context_embeddings, axis=0), axis=1) if len(context_embeddings) > 1 else np.array([0.0])
    transition_scores = np.mean(np.std(context_windows_for_bank, axis=1), axis=1)
    calibration = {
        "change_threshold": float(np.percentile(change_scores, 99.0)) if len(change_scores) else 0.0,
        "transition_threshold": float(np.percentile(transition_scores, 99.0)) if len(transition_scores) else 0.0,
        "memory_threshold": float(memory_bank.threshold),
        "n_clusters": int(memory_bank.n_clusters),
    }
    reference_embeddings = encode_windows(model, memory_bank.representative_windows, config.batch_size, device)
    return memory_bank, reference_embeddings, calibration


def score_with_context_substitution(
    model: HybridTCNEncoder,
    test_windows: np.ndarray,
    memory_bank: ContextMemoryBank,
    reference_embeddings: np.ndarray,
    calibration: dict,
    config: NCADCSConfig,
    device: torch.device,
) -> dict:
    full_embeddings = encode_windows(model, test_windows, config.batch_size, device)
    context_windows = test_windows[:, : config.context_size]
    context_embeddings = encode_windows(model, context_windows, config.batch_size, device)

    window_scores = np.zeros(len(test_windows), dtype=np.float32)
    original_scores = np.zeros(len(test_windows), dtype=np.float32)
    substituted_scores = np.zeros(len(test_windows), dtype=np.float32)
    substitutions = np.zeros(len(test_windows), dtype=bool)
    substitution_confidences = np.zeros(len(test_windows), dtype=np.float32)
    context_distances = np.zeros(len(test_windows), dtype=np.float32)
    change_scores = np.zeros(len(test_windows), dtype=np.float32)
    transition_scores = np.zeros(len(test_windows), dtype=np.float32)

    previous_context_embedding = None
    memory_degenerate = (
        config.use_original_on_degenerate_memory
        and calibration.get("memory_threshold", 0.0) <= config.degenerate_memory_threshold
    )
    for index in range(len(test_windows)):
        z_full = full_embeddings[index]
        z_context = context_embeddings[index]
        original_score = euclidean_distance(z_full, z_context)
        context_distance, prototype_index = memory_bank.query(z_context)
        change_score = 0.0 if previous_context_embedding is None else euclidean_distance(z_context, previous_context_embedding)
        transition_score = float(np.mean(np.std(context_windows[index], axis=0)))

        distance_confidence = confidence_over_threshold(context_distance, calibration["memory_threshold"])
        change_confidence = confidence_over_threshold(change_score, calibration["change_threshold"])
        transition_confidence = confidence_over_threshold(transition_score, calibration["transition_threshold"])
        substitution_confidence = max(distance_confidence, change_confidence, transition_confidence)

        should_substitute = False if memory_degenerate else (
            context_distance > calibration["memory_threshold"]
            and substitution_confidence >= config.min_substitution_confidence
        )

        substituted_score = 0.0
        if memory_degenerate:
            final_score = original_score
        elif should_substitute:
            substituted_score = euclidean_distance(z_full, reference_embeddings[prototype_index])
            final_score = (1.0 - substitution_confidence) * original_score
            final_score += substitution_confidence * (config.boost_factor * substituted_score)
        else:
            final_score = original_score

        window_scores[index] = final_score
        original_scores[index] = original_score
        substituted_scores[index] = substituted_score
        substitutions[index] = should_substitute
        substitution_confidences[index] = substitution_confidence
        context_distances[index] = context_distance
        change_scores[index] = change_score
        transition_scores[index] = transition_score
        previous_context_embedding = z_context

    return {
        "window_scores": window_scores,
        "original_scores": original_scores,
        "substituted_scores": substituted_scores,
        "substitutions": substitutions,
        "substitution_confidences": substitution_confidences,
        "context_distances": context_distances,
        "change_scores": change_scores,
        "transition_scores": transition_scores,
        "memory_degenerate": memory_degenerate,
    }


def calibrate_threshold_from_training_windows(
    model: HybridTCNEncoder,
    train_windows: np.ndarray,
    memory_bank: ContextMemoryBank,
    reference_embeddings: np.ndarray,
    calibration: dict,
    fallback_threshold: object,
    config: NCADCSConfig,
    device: torch.device,
) -> object:
    """Learn a channel-specific threshold from synthetic calibration labels."""

    if len(train_windows) < 4:
        return replace(fallback_threshold, threshold_method="adaptive_distribution_fallback")

    calibration_windows = limit_windows(train_windows, config.max_threshold_calibration_windows)
    injector = ContextualAnomalyInjector(
        AnomalyInjectionConfig(injection_ratio=config.injection_ratio), seed=config.seed + 10_000
    )
    synthetic_windows, labels = injector.inject_batch(calibration_windows, config.context_size)
    if len(np.unique(labels)) < 2:
        return replace(fallback_threshold, threshold_method="adaptive_distribution_fallback")

    score_details = score_with_context_substitution(
        model,
        synthetic_windows,
        memory_bank,
        reference_embeddings,
        calibration,
        config,
        device,
    )
    calibrated = calculate_label_calibrated_threshold(
        score_details["window_scores"],
        labels,
        fallback_threshold,
        alpha=config.threshold_alpha,
    )
    if calibrated.final_threshold < fallback_threshold.final_threshold:
        return replace(
            fallback_threshold,
            threshold_method="hybrid_dynamic_distribution_floor",
            calibration_threshold=calibrated.calibration_threshold,
            calibration_precision=calibrated.calibration_precision,
            calibration_recall=calibrated.calibration_recall,
            calibration_f1=calibrated.calibration_f1,
            calibration_positive_rate=calibrated.calibration_positive_rate,
        )
    if calibrated.calibration_f1 < config.min_threshold_calibration_f1:
        return replace(
            fallback_threshold,
            threshold_method="adaptive_distribution_low_calibration_f1",
            calibration_threshold=calibrated.calibration_threshold,
            calibration_precision=calibrated.calibration_precision,
            calibration_recall=calibrated.calibration_recall,
            calibration_f1=calibrated.calibration_f1,
            calibration_positive_rate=calibrated.calibration_positive_rate,
        )
    return replace(calibrated, threshold_method="hybrid_dynamic_self_supervised")


def run_channel(channel_data: ChannelData, run_dir: Path, config: NCADCSConfig, device: torch.device) -> dict:
    config = replace(config, seed=derive_channel_seed(config.seed, channel_data.channel_id))
    set_seed(config.seed)
    channel_start = time.time()
    channel_dir = run_dir / channel_data.channel_id
    channel_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{channel_data.channel_id}] extracting features")
    feature_extractor = NCADFeatureExtractor(FeatureConfig(max_features=config.feature_dim))
    train_features = feature_extractor.fit_transform(channel_data.train_normalized)
    test_features = feature_extractor.transform(channel_data.test_normalized)

    train_windows = DataLoader.create_windows(train_features, config.full_window_size, config.step)
    test_windows = DataLoader.create_windows(test_features, config.full_window_size, config.step)
    train_windows = limit_windows(train_windows, config.max_train_windows)
    test_windows = limit_windows(test_windows, config.max_test_windows)
    if len(train_windows) == 0 or len(test_windows) == 0:
        raise ValueError(f"Channel {channel_data.channel_id} does not have enough samples for window size {config.full_window_size}.")

    print(f"[{channel_data.channel_id}] windows: train={len(train_windows):,}, test={len(test_windows):,}, features={train_features.shape[1]}")
    print(f"[{channel_data.channel_id}] training contrastive TCN encoder")
    model, training_history = train_encoder(train_windows, config, train_features.shape[1], device)

    print(f"[{channel_data.channel_id}] building Context Memory Bank")
    memory_bank, reference_embeddings, calibration = build_memory_bank(model, train_windows, config, device)
    print(
        f"[{channel_data.channel_id}] memory bank: clusters={memory_bank.n_clusters}, "
        f"tau={memory_bank.threshold:.5f}"
    )

    print(f"[{channel_data.channel_id}] scoring test windows with contextual substitution")
    score_details = score_with_context_substitution(
        model, test_windows, memory_bank, reference_embeddings, calibration, config, device
    )
    point_scores, substitution_map, valid_mask = aggregate_suspect_scores(
        score_details["window_scores"],
        score_details["substitutions"],
        n_points=len(channel_data.test_raw),
        context_size=config.context_size,
        suspect_size=config.suspect_size,
        step=config.step,
    )
    smoothed_scores = dynamic_weighted_smoothing(
        point_scores,
        substitution_map,
        short_window=config.short_smoothing_window,
        long_window=config.long_smoothing_window,
    )
    threshold_info = calculate_threshold(
        smoothed_scores,
        substitution_map,
        valid_mask,
        percentile=config.threshold_percentile,
        alpha=config.threshold_alpha,
        tail_guard_multiplier=config.threshold_tail_guard_multiplier,
        tail_guard_percentile=config.threshold_tail_guard_percentile,
        sparse_tail_guard_multiplier=config.threshold_sparse_tail_guard_multiplier,
        sparse_tail_guard_percentile=config.threshold_sparse_tail_guard_percentile,
        sparse_tail_guard_max_base=config.threshold_sparse_tail_guard_max_base,
    )
    threshold_info = calibrate_threshold_from_training_windows(
        model,
        train_windows,
        memory_bank,
        reference_embeddings,
        calibration,
        threshold_info,
        config,
        device,
    )
    preliminary_flags = (smoothed_scores > threshold_info.final_threshold) & valid_mask
    predictions = temporal_consistency_filter(
        preliminary_flags,
        smoothed_scores,
        threshold_info.final_threshold,
        min_run=config.min_prediction_run,
        extreme_factor=config.extreme_score_factor,
    )
    predictions = predictions * valid_mask.astype(np.float32)
    metrics = compute_metrics(channel_data.labels, predictions, valid_mask=valid_mask)

    elapsed = time.time() - channel_start
    result = {
        "channel": channel_data.channel_id,
        "channel_seed": config.seed,
        "elapsed_seconds": elapsed,
        "training_history": training_history,
        "calibration": calibration,
        "threshold": asdict(threshold_info),
        "metrics": metrics,
        "substitution_rate_window": float(np.mean(score_details["substitutions"])) if len(test_windows) else 0.0,
        "substitution_rate_point": float(np.mean(substitution_map[valid_mask])) if np.any(valid_mask) else 0.0,
        "memory_degenerate": bool(score_details.get("memory_degenerate", False)),
    }

    save_channel_outputs(
        channel_dir,
        channel_data,
        config,
        model,
        feature_extractor,
        memory_bank,
        result,
        score_details,
        point_scores,
        smoothed_scores,
        substitution_map,
        valid_mask,
        predictions,
    )

    if config.save_plots:
        plot_channel_diagnostics(
            channel_data.channel_id,
            channel_data.test_raw,
            smoothed_scores,
            threshold_info.final_threshold,
            predictions,
            channel_data.labels,
            substitution_map,
            score_details["context_distances"],
            channel_dir / f"{channel_data.channel_id}_ncad_cs_diagnostics.png",
        )

    print(f"[{channel_data.channel_id}] done in {elapsed:.1f}s: {metrics if metrics else 'no labels'}")
    return result


def save_channel_outputs(
    channel_dir: Path,
    channel_data: ChannelData,
    config: NCADCSConfig,
    model: HybridTCNEncoder,
    feature_extractor: NCADFeatureExtractor,
    memory_bank: ContextMemoryBank,
    result: dict,
    score_details: dict,
    point_scores: np.ndarray,
    smoothed_scores: np.ndarray,
    substitution_map: np.ndarray,
    valid_mask: np.ndarray,
    predictions: np.ndarray,
) -> None:
    with (channel_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    with (channel_dir / "feature_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(feature_extractor.to_metadata(), file, indent=2)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "input_dim": model.input_dim,
            "latent_dim": model.latent_dim,
            "filters": model.filters,
        },
        channel_dir / "encoder.pt",
    )
    memory_bank.save(channel_dir / "memory_bank.npz")

    labels = channel_data.labels if channel_data.labels is not None else np.full(len(channel_data.test_raw), np.nan)
    predictions_df = pd.DataFrame(
        {
            "time_index": np.arange(len(channel_data.test_raw)),
            "telemetry": channel_data.test_raw,
            "point_score": point_scores,
            "smoothed_score": smoothed_scores,
            "prediction": predictions.astype(np.float32),
            "label": labels[: len(channel_data.test_raw)],
            "context_substituted": substitution_map.astype(np.float32),
            "valid_score": valid_mask.astype(np.float32),
        }
    )
    predictions_df.to_csv(channel_dir / "point_predictions.csv", index=False)

    window_df = pd.DataFrame(
        {
            "window_index": np.arange(len(score_details["window_scores"])),
            "window_score": score_details["window_scores"],
            "original_score": score_details["original_scores"],
            "substituted_score": score_details["substituted_scores"],
            "context_substituted": score_details["substitutions"].astype(np.float32),
            "substitution_confidence": score_details["substitution_confidences"],
            "context_distance": score_details["context_distances"],
            "change_score": score_details["change_scores"],
            "transition_score": score_details["transition_scores"],
        }
    )
    window_df.to_csv(channel_dir / "window_scores.csv", index=False)


def run_experiment(channels: list[str], config: NCADCSConfig) -> tuple[Path, pd.DataFrame]:
    set_seed(config.seed)
    device = resolve_device(config.device)
    run_dir = Path(config.output_dir).resolve() if config.output_dir else default_output_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")
    print(f"Results directory: {run_dir}")

    with (run_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=2)

    loader = DataLoader(config.data_dir)
    results = []
    for channel_id in channels:
        channel_data = loader.load_channel(channel_id)
        try:
            results.append(run_channel(channel_data, run_dir, config, device))
        except Exception as exc:
            print(f"[{channel_id}] failed: {exc}")
            results.append({"channel": channel_id, "error": str(exc)})

    summary_rows = []
    for result in results:
        row = {"channel": result.get("channel"), "error": result.get("error")}
        row.update(result.get("metrics", {}))
        row["channel_seed"] = result.get("channel_seed")
        row["substitution_rate_window"] = result.get("substitution_rate_window")
        row["substitution_rate_point"] = result.get("substitution_rate_point")
        row["memory_degenerate"] = result.get("memory_degenerate")
        row["elapsed_seconds"] = result.get("elapsed_seconds")
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(run_dir / "summary.csv", index=False)
    return run_dir, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NCAD-CS v3 on SMAP/MSL telemetry channels.")
    parser.add_argument("--channel", type=str, default=None, help="Run one channel, for example A-1.")
    parser.add_argument("--channels", type=str, nargs="+", default=None, help="Run a selected list of channels.")
    parser.add_argument("--all", action="store_true", help="Run every channel with train and test files.")
    parser.add_argument("--list-channels", action="store_true", help="List available channels and exit.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to the data directory. Defaults to ../data from this file.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for run outputs.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--max-train-windows", type=int, default=None, help="Optional cap for quick smoke runs.")
    parser.add_argument("--max-test-windows", type=int, default=None, help="Optional cap for quick smoke runs.")
    parser.add_argument("--max-memory-windows", type=int, default=5000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-plots", action="store_true", help="Skip diagnostic plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = DataLoader(args.data_dir)
    available_channels = loader.list_channels()

    if args.list_channels:
        print("Available channels:")
        for channel in available_channels:
            print(f"  {channel}")
        return

    if args.channel:
        channels = [args.channel]
    elif args.channels:
        channels = args.channels
    else:
        channels = available_channels if args.all else ["A-1"]

    missing = [channel for channel in channels if channel not in available_channels]
    if missing:
        raise ValueError(f"Unknown or incomplete channels: {missing}")

    config = NCADCSConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        feature_dim=args.feature_dim,
        max_train_windows=args.max_train_windows,
        max_test_windows=args.max_test_windows,
        max_memory_windows=args.max_memory_windows,
        save_plots=not args.no_plots,
        device=args.device,
    )
    run_dir, summary = run_experiment(channels, config)
    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nSaved run to: {run_dir}")


if __name__ == "__main__":
    main()
