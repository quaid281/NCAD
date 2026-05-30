"""NCAD-CS v4: Counterfactual Successor Memory training and evaluation routine."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from src.models.anomaly_injector import AnomalyInjectionConfig, ContextualAnomalyInjector
from src.models.tcn_encoder import HybridTCNEncoder, contrastive_loss
from src.data.data_loader import ChannelData, DataLoader
from src.features.features import FeatureConfig, NCADFeatureExtractor
from src.data.pipeline import NCADPipeline
from src.utils.logging_utils import setup_logging
from src.utils.plotting import plot_channel_diagnostics

from src.models.multi_scale_tcn_encoder import MultiScaleTCNEncoder
from src.models.successor_memory import CounterfactualSuccessorMemory, SuccessorMemoryConfig
from src.utils.event_fusion import (
    adaptive_elbow_score_floor,
    aggregate_window_scores,
    compute_metrics,
    event_level_filter,
    fuse_evidence_scores,
    local_deviation_scores,
    moving_average,
    percentile_score_floor,
    positive_robust_z,
    robust_dispersion_floor,
    robust_stats,
    dispersion_confidence,
    successor_manifold_uncertainty_scores,
)

logger = logging.getLogger("NCAD.train")

EncoderModel = HybridTCNEncoder | MultiScaleTCNEncoder


@dataclass
class CSMConfig:
    data_dir: Optional[str] = None
    output_dir: Optional[str] = None
    context_size: int = 284
    suspect_size: int = 16
    step: int = 1
    feature_dim: int = 64
    encoder_architecture: str = "hybrid_tcn"
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
    successor_neighbors: int = 8
    context_percentile: float = 99.0
    event_threshold_percentile: float = 99.0
    score_floor_percentile: Optional[float] = None
    degenerate_score_epsilon: float = 1e-6
    local_tail_size: int = 64
    manifold_uncertainty: bool = False
    manifold_dispersion_floor_percentile: float = 50.0
    manifold_min_confidence: float = 0.35
    smoothing_window: int = 12
    min_event_run: int = 2
    extreme_event_factor: float = 1.75
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


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_encoder(config: CSMConfig, input_dim: int, device: torch.device) -> EncoderModel:
    common_kwargs = {
        "input_dim": input_dim,
        "latent_dim": config.latent_dim,
        "filters": config.filters,
        "tcn_layers": config.tcn_layers,
        "kernel_size": config.kernel_size,
        "dropout": config.dropout,
    }
    if config.encoder_architecture == "hybrid_tcn":
        model = HybridTCNEncoder(**common_kwargs)
    elif config.encoder_architecture == "multi_scale_tcn":
        model = MultiScaleTCNEncoder(**common_kwargs)
    else:
        raise ValueError(f"Unknown encoder architecture: {config.encoder_architecture}")
    model.architecture = config.encoder_architecture
    return model.to(device)


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Resolve to project root / results / timestamp (src/models/train_model.py is 2 levels deep)
    return Path(__file__).resolve().parents[2] / "results" / timestamp


def limit_windows(windows: np.ndarray, max_windows: Optional[int]) -> np.ndarray:
    if max_windows is None or len(windows) <= max_windows:
        return windows
    return windows[:max_windows]


def encode_windows(model: EncoderModel, windows: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size]).float().to(device)
            embeddings.append(model(batch).cpu().numpy())
    if not embeddings:
        return np.empty((0, model.latent_dim), dtype=np.float32)
    return np.concatenate(embeddings, axis=0).astype(np.float32)


def split_train_validation(windows: np.ndarray, val_split: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(windows))
    rng.shuffle(indices)
    if len(indices) < 10:
        return windows[indices], np.empty((0,) + windows.shape[1:], dtype=windows.dtype)
    n_val = max(1, int(len(indices) * val_split))
    return windows[indices[n_val:]], windows[indices[:n_val]]


def train_encoder(train_windows: np.ndarray, config: CSMConfig, input_dim: int, device: torch.device) -> tuple[EncoderModel, dict]:
    model = build_encoder(config, input_dim, device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    injector = ContextualAnomalyInjector(AnomalyInjectionConfig(injection_ratio=config.injection_ratio), seed=config.seed)
    val_injector = ContextualAnomalyInjector(AnomalyInjectionConfig(injection_ratio=config.injection_ratio), seed=config.seed + 1)
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
            loss = contrastive_loss(model(full_tensor), model(context_tensor), label_tensor, margin=config.margin)
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
        logger.info(f"  epoch {epoch:03d}/{config.epochs}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"  early stopping after {epoch} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_loss"] = best_val_loss
    return model, history


def evaluate_contrastive_loss(
    model: EncoderModel,
    validation_data: np.ndarray,
    injector: ContextualAnomalyInjector,
    config: CSMConfig,
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


def build_successor_memory(
    model: EncoderModel,
    train_windows: np.ndarray,
    config: CSMConfig,
    device: torch.device,
) -> CounterfactualSuccessorMemory:
    context_windows = train_windows[:, : config.context_size]
    successor_windows = train_windows[:, config.context_size :]
    context_embeddings = encode_windows(model, context_windows, config.batch_size, device)
    memory = CounterfactualSuccessorMemory(
        SuccessorMemoryConfig(
            n_neighbors=config.successor_neighbors,
            max_memory_windows=config.max_memory_windows,
            context_percentile=config.context_percentile,
            seed=config.seed,
        )
    )
    return memory.fit(context_embeddings, successor_windows)


def score_windows(
    model: EncoderModel,
    windows: np.ndarray,
    memory: CounterfactualSuccessorMemory,
    successor_stats,
    local_stats,
    manifold_stats,
    dispersion_stats,
    dispersion_floor: float,
    config: CSMConfig,
    device: torch.device,
) -> dict:
    context_windows = windows[:, : config.context_size]
    observed_successors = windows[:, config.context_size :]
    context_embeddings = encode_windows(model, context_windows, config.batch_size, device)
    query = memory.query(context_embeddings, observed_successors)

    local_raw_scores = local_deviation_scores(windows, config.context_size, tail_size=config.local_tail_size)
    if successor_stats.percentile_99 <= config.degenerate_score_epsilon:
        successor_z = np.zeros_like(query.successor_scores, dtype=np.float32)
    else:
        successor_z = positive_robust_z(query.successor_scores, successor_stats)
    local_z = positive_robust_z(local_raw_scores, local_stats)
    if config.manifold_uncertainty:
        manifold_raw_scores = successor_manifold_uncertainty_scores(
            query.successor_scores,
            query.successor_dispersion,
            dispersion_floor,
        )
        if manifold_stats.percentile_99 <= config.degenerate_score_epsilon:
            manifold_z = np.zeros_like(manifold_raw_scores, dtype=np.float32)
        else:
            manifold_z = positive_robust_z(manifold_raw_scores, manifold_stats)
        uncertainty_confidence = dispersion_confidence(
            query.successor_dispersion,
            dispersion_stats,
            min_confidence=config.manifold_min_confidence,
        )
    else:
        manifold_raw_scores = np.zeros_like(query.successor_scores, dtype=np.float32)
        manifold_z = np.zeros_like(query.successor_scores, dtype=np.float32)
        uncertainty_confidence = np.ones_like(query.successor_scores, dtype=np.float32)
    if float(memory.context_threshold) <= config.degenerate_score_epsilon:
        context_ratio = np.ones_like(query.context_distances, dtype=np.float32)
    else:
        context_ratio = query.context_distances / float(memory.context_threshold)
    window_scores = fuse_evidence_scores(
        successor_z,
        local_z,
        context_ratio,
        manifold_z=manifold_z if config.manifold_uncertainty else None,
        uncertainty_confidence=uncertainty_confidence if config.manifold_uncertainty else None,
    )
    context_ood = context_ratio > 1.0

    return {
        "window_scores": window_scores,
        "successor_scores": query.successor_scores,
        "successor_median_scores": query.successor_median_scores,
        "successor_dispersion": query.successor_dispersion,
        "manifold_uncertainty_scores": manifold_raw_scores,
        "manifold_z": manifold_z,
        "uncertainty_confidence": uncertainty_confidence,
        "local_scores": local_raw_scores,
        "successor_z": successor_z,
        "local_z": local_z,
        "context_distances": query.context_distances,
        "context_ratio": context_ratio.astype(np.float32),
        "context_ood": context_ood.astype(bool),
    }


def calibrate_event_threshold(
    train_windows: np.ndarray,
    memory: CounterfactualSuccessorMemory,
    config: CSMConfig,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    if (
        memory.calibration_successor_scores is None
        or memory.calibration_successor_dispersion is None
        or memory.sample_indices is None
    ):
        raise RuntimeError("Successor memory is missing calibration scores.")

    train_local_scores = local_deviation_scores(train_windows, config.context_size, tail_size=config.local_tail_size)
    calibration_local_scores = train_local_scores[memory.sample_indices]
    successor_stats = robust_stats(memory.calibration_successor_scores)
    local_stats = robust_stats(calibration_local_scores)
    dispersion_floor = robust_dispersion_floor(
        memory.calibration_successor_dispersion,
        percentile=config.manifold_dispersion_floor_percentile,
        minimum=config.degenerate_score_epsilon,
    )
    calibration_manifold_scores = successor_manifold_uncertainty_scores(
        memory.calibration_successor_scores,
        memory.calibration_successor_dispersion,
        dispersion_floor,
    )
    manifold_stats = robust_stats(calibration_manifold_scores)
    dispersion_stats = robust_stats(memory.calibration_successor_dispersion)

    if successor_stats.percentile_99 <= config.degenerate_score_epsilon:
        successor_z = np.zeros_like(memory.calibration_successor_scores, dtype=np.float32)
    else:
        successor_z = positive_robust_z(memory.calibration_successor_scores, successor_stats)
    local_z = positive_robust_z(calibration_local_scores, local_stats)
    if config.manifold_uncertainty:
        if manifold_stats.percentile_99 <= config.degenerate_score_epsilon:
            manifold_z = np.zeros_like(calibration_manifold_scores, dtype=np.float32)
        else:
            manifold_z = positive_robust_z(calibration_manifold_scores, manifold_stats)
        uncertainty_confidence = dispersion_confidence(
            memory.calibration_successor_dispersion,
            dispersion_stats,
            min_confidence=config.manifold_min_confidence,
        )
    else:
        manifold_z = None
        uncertainty_confidence = None
    if float(memory.context_threshold) <= config.degenerate_score_epsilon:
        context_ratio = np.ones_like(memory.calibration_context_distances, dtype=np.float32)
    else:
        context_ratio = memory.calibration_context_distances / float(memory.context_threshold)
    calibration_window_scores = fuse_evidence_scores(
        successor_z,
        local_z,
        context_ratio,
        manifold_z=manifold_z,
        uncertainty_confidence=uncertainty_confidence,
    )

    n_points = (len(train_windows) - 1) * config.step + config.full_window_size
    calibration_point_scores, calibration_valid_mask = aggregate_window_scores(
        calibration_window_scores,
        n_points=n_points,
        context_size=config.context_size,
        suspect_size=config.suspect_size,
        step=config.step,
        window_indices=memory.sample_indices,
        reducer="mean",
    )
    calibration_smoothed_scores = moving_average(calibration_point_scores, config.smoothing_window)
    valid_scores = calibration_smoothed_scores[calibration_valid_mask]
    if len(valid_scores) == 0:
        valid_scores = np.array([0.0], dtype=np.float32)
    threshold = float(np.percentile(valid_scores, config.event_threshold_percentile))
    threshold = max(threshold, 1e-6)

    calibration = {
        "threshold_method": "counterfactual_successor_training_distribution",
        "event_threshold": threshold,
        "event_threshold_percentile": config.event_threshold_percentile,
        "successor_score_median": successor_stats.median,
        "successor_score_iqr": successor_stats.iqr,
        "successor_manifold_uncertainty_enabled": bool(config.manifold_uncertainty),
        "successor_manifold_score_median": manifold_stats.median,
        "successor_manifold_score_iqr": manifold_stats.iqr,
        "successor_dispersion_median": dispersion_stats.median,
        "successor_dispersion_iqr": dispersion_stats.iqr,
        "successor_dispersion_floor": float(dispersion_floor),
        "successor_dispersion_floor_percentile": config.manifold_dispersion_floor_percentile,
        "successor_uncertainty_min_confidence": config.manifold_min_confidence,
        "local_score_median": local_stats.median,
        "local_score_iqr": local_stats.iqr,
        "context_threshold": float(memory.context_threshold),
        "successor_degenerate": bool(successor_stats.percentile_99 <= config.degenerate_score_epsilon),
        "context_degenerate": bool(float(memory.context_threshold) <= config.degenerate_score_epsilon),
        "memory_size": int(len(memory.context_embeddings)) if memory.context_embeddings is not None else 0,
        "memory_sampled": bool(config.max_memory_windows is not None and len(train_windows) > config.max_memory_windows),
    }
    return calibration, calibration_window_scores, successor_stats, local_stats, manifold_stats, dispersion_stats, dispersion_floor


def run_channel(channel_data: ChannelData, run_dir: Path, config: CSMConfig, device: torch.device) -> dict:
    set_seed(config.seed)
    channel_start = time.time()
    channel_dir = run_dir / channel_data.channel_id
    channel_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{channel_data.channel_id}] extracting features")
    feature_extractor = NCADFeatureExtractor(FeatureConfig(max_features=config.feature_dim))
    train_features = feature_extractor.fit_transform(channel_data.train_normalized)
    test_features = feature_extractor.transform(channel_data.test_normalized)

    train_windows = DataLoader.create_windows(train_features, config.full_window_size, config.step)
    test_windows = DataLoader.create_windows(test_features, config.full_window_size, config.step)
    train_windows = limit_windows(train_windows, config.max_train_windows)
    test_windows = limit_windows(test_windows, config.max_test_windows)
    if len(train_windows) == 0 or len(test_windows) == 0:
        raise ValueError(f"Channel {channel_data.channel_id} does not have enough samples for window size {config.full_window_size}.")

    logger.info(f"[{channel_data.channel_id}] windows: train={len(train_windows):,}, test={len(test_windows):,}, features={train_features.shape[1]}")
    logger.info(f"[{channel_data.channel_id}] training {config.encoder_architecture} contrastive TCN encoder")
    model, training_history = train_encoder(train_windows, config, train_features.shape[1], device)

    logger.info(f"[{channel_data.channel_id}] building Counterfactual Successor Memory")
    memory = build_successor_memory(model, train_windows, config, device)
    (
        calibration,
        calibration_window_scores,
        successor_stats,
        local_stats,
        manifold_stats,
        dispersion_stats,
        dispersion_floor,
    ) = calibrate_event_threshold(train_windows, memory, config)
    logger.info(
        f"[{channel_data.channel_id}] memory={calibration['memory_size']:,}, "
        f"context_tau={calibration['context_threshold']:.5f}, event_tau={calibration['event_threshold']:.5f}"
    )

    logger.info(f"[{channel_data.channel_id}] scoring test windows with counterfactual successors")
    score_details = score_windows(
        model,
        test_windows,
        memory,
        successor_stats,
        local_stats,
        manifold_stats,
        dispersion_stats,
        dispersion_floor,
        config,
        device,
    )
    point_scores, valid_mask = aggregate_window_scores(
        score_details["window_scores"],
        n_points=len(channel_data.test_raw),
        context_size=config.context_size,
        suspect_size=config.suspect_size,
        step=config.step,
        reducer="mean",
    )
    smoothed_scores = moving_average(point_scores, config.smoothing_window)
    valid_smoothed_scores = smoothed_scores[valid_mask]
    if config.score_floor_percentile is None:
        score_floor = adaptive_elbow_score_floor(valid_smoothed_scores)
    else:
        score_floor = percentile_score_floor(valid_smoothed_scores, config.score_floor_percentile)
    score_floor_threshold = score_floor.threshold
    training_event_threshold = float(calibration["event_threshold"])
    calibration["training_event_threshold"] = training_event_threshold
    calibration["score_floor_threshold"] = score_floor_threshold
    calibration["score_floor_percentile"] = config.score_floor_percentile
    calibration["score_floor_method"] = score_floor.method
    calibration["score_floor_plateau_adjusted"] = score_floor.plateau_adjusted
    calibration["score_floor_details"] = score_floor.to_dict()
    if score_floor_threshold > training_event_threshold:
        calibration["event_threshold"] = score_floor_threshold
        calibration["threshold_method"] = f"counterfactual_successor_training_plus_{score_floor.method}"

    predictions = event_level_filter(
        smoothed_scores,
        calibration["event_threshold"],
        valid_mask,
        min_run=config.min_event_run,
        extreme_factor=config.extreme_event_factor,
    )
    predictions = predictions * valid_mask.astype(np.float32)
    metrics = compute_metrics(channel_data.labels, predictions, valid_mask=valid_mask)

    context_ood_point, _ = aggregate_window_scores(
        score_details["context_ood"].astype(np.float32),
        n_points=len(channel_data.test_raw),
        context_size=config.context_size,
        suspect_size=config.suspect_size,
        step=config.step,
        reducer="mean",
    )
    context_ood_map = context_ood_point > 0.0
    elapsed = time.time() - channel_start
    result = {
        "channel": channel_data.channel_id,
        "elapsed_seconds": elapsed,
        "training_history": training_history,
        "calibration": calibration,
        "threshold": {"final_threshold": calibration["event_threshold"], "threshold_method": calibration["threshold_method"]},
        "metrics": metrics,
        "context_ood_rate_window": float(np.mean(score_details["context_ood"])) if len(test_windows) else 0.0,
        "context_ood_rate_point": float(np.mean(context_ood_map[valid_mask])) if np.any(valid_mask) else 0.0,
        "score_mode": "counterfactual_successor_memory_uncertainty" if config.manifold_uncertainty else "counterfactual_successor_memory",
    }

    save_channel_outputs(
        channel_dir,
        channel_data,
        config,
        model,
        feature_extractor,
        memory,
        result,
        score_details,
        point_scores,
        smoothed_scores,
        context_ood_map,
        valid_mask,
        predictions,
        calibration_window_scores,
    )
    if config.save_plots:
        plot_channel_diagnostics(channel_dir, channel_data, smoothed_scores, calibration["event_threshold"], predictions, context_ood_map)

    logger.info(f"[{channel_data.channel_id}] done in {elapsed:.1f}s: {metrics if metrics else 'no labels'}")
    return result


def save_channel_outputs(
    channel_dir: Path,
    channel_data: ChannelData,
    config: CSMConfig,
    model: EncoderModel,
    feature_extractor: NCADFeatureExtractor,
    memory: CounterfactualSuccessorMemory,
    result: dict,
    score_details: dict,
    point_scores: np.ndarray,
    smoothed_scores: np.ndarray,
    context_ood_map: np.ndarray,
    valid_mask: np.ndarray,
    predictions: np.ndarray,
    calibration_window_scores: np.ndarray,
) -> None:
    with (channel_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    with (channel_dir / "feature_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(feature_extractor.to_metadata(), file, indent=2)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "encoder_architecture": getattr(model, "architecture", "unknown"),
            "input_dim": model.input_dim,
            "latent_dim": model.latent_dim,
            "filters": model.filters,
            "tcn_layers": getattr(model, "tcn_layers", config.tcn_layers),
            "kernel_sizes": list(getattr(model, "kernel_sizes", [config.kernel_size])),
            "branch_channels": getattr(model, "branch_channels", None),
        },
        channel_dir / "encoder.pt",
    )
    memory.save(channel_dir / "successor_memory.npz")

    labels = channel_data.labels if channel_data.labels is not None else np.full(len(channel_data.test_raw), np.nan)
    predictions_df = pd.DataFrame(
        {
            "time_index": np.arange(len(channel_data.test_raw)),
            "telemetry": channel_data.test_raw,
            "point_score": point_scores,
            "smoothed_score": smoothed_scores,
            "prediction": predictions.astype(np.float32),
            "label": labels[: len(channel_data.test_raw)],
            "context_ood": context_ood_map.astype(np.float32),
            "valid_score": valid_mask.astype(np.float32),
        }
    )
    predictions_df.to_csv(channel_dir / "point_predictions.csv", index=False)

    window_df = pd.DataFrame(
        {
            "window_index": np.arange(len(score_details["window_scores"])),
            "window_score": score_details["window_scores"],
            "successor_score": score_details["successor_scores"],
            "successor_median_score": score_details["successor_median_scores"],
            "successor_dispersion": score_details["successor_dispersion"],
            "manifold_uncertainty_score": score_details["manifold_uncertainty_scores"],
            "manifold_z": score_details["manifold_z"],
            "uncertainty_confidence": score_details["uncertainty_confidence"],
            "local_score": score_details["local_scores"],
            "successor_z": score_details["successor_z"],
            "local_z": score_details["local_z"],
            "context_distance": score_details["context_distances"],
            "context_ratio": score_details["context_ratio"],
            "context_ood": score_details["context_ood"].astype(np.float32),
        }
    )
    window_df.to_csv(channel_dir / "window_scores.csv", index=False)
    pd.DataFrame({"calibration_window_score": calibration_window_scores}).to_csv(
        channel_dir / "calibration_window_scores.csv", index=False
    )


def run_experiment(channels: list[str], config: CSMConfig) -> tuple[Path, pd.DataFrame]:
    set_seed(config.seed)
    device = resolve_device(config.device)
    run_dir = Path(config.output_dir).resolve() if config.output_dir else default_output_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using device: {device}")
    logger.info(f"Results directory: {run_dir}")
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=2)

    loader = DataLoader(config.data_dir)
    results = []
    for channel_id in channels:
        channel_data = loader.load_channel(channel_id)
        try:
            results.append(run_channel(channel_data, run_dir, config, device))
        except Exception as exc:
            logger.error(f"[{channel_id}] failed: {exc}", exc_info=True)
            results.append({"channel": channel_id, "error": str(exc)})

    summary_rows = []
    for result in results:
        row = {"channel": result.get("channel"), "error": result.get("error")}
        row.update(result.get("metrics", {}))
        row["context_ood_rate_window"] = result.get("context_ood_rate_window")
        row["context_ood_rate_point"] = result.get("context_ood_rate_point")
        row["score_mode"] = result.get("score_mode")
        row["elapsed_seconds"] = result.get("elapsed_seconds")
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(run_dir / "summary.csv", index=False)
    return run_dir, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NCAD-CS Counterfactual Successor Memory.")
    parser.add_argument("--channel", type=str, default=None, help="Run one channel, for example A-1.")
    parser.add_argument("--channels", type=str, nargs="+", default=None, help="Run selected channels.")
    parser.add_argument("--all", action="store_true", help="Run every channel with train and test files.")
    parser.add_argument("--list-channels", action="store_true", help="List available channels and exit.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to the data directory. Defaults to ../data.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for run outputs.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["hybrid_tcn", "multi_scale_tcn"],
        default="hybrid_tcn",
        help="Encoder architecture.",
    )
    parser.add_argument("--successor-neighbors", type=int, default=8)
    parser.add_argument("--event-threshold-percentile", type=float, default=99.0)
    parser.add_argument(
        "--score-floor-percentile",
        type=float,
        default=None,
        help="Legacy fixed percentile floor override. Omit to use the adaptive elbow floor.",
    )
    parser.add_argument(
        "--manifold-uncertainty",
        action="store_true",
        help="Enable the experimental successor manifold uncertainty scorer.",
    )
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-test-windows", type=int, default=None)
    parser.add_argument("--max-memory-windows", type=int, default=5000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Structured logger level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
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

    config = CSMConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        feature_dim=args.feature_dim,
        encoder_architecture=args.encoder,
        successor_neighbors=args.successor_neighbors,
        event_threshold_percentile=args.event_threshold_percentile,
        score_floor_percentile=args.score_floor_percentile,
        manifold_uncertainty=args.manifold_uncertainty,
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
