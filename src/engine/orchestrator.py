"""Channel execution, multi-channel orchestration, and output serialization for NCAD-CS."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.config import CSMConfig
from src.data.data_loader import ChannelData, DataLoader
from src.engine.evaluator import build_successor_memory, calibrate_event_threshold, score_windows
from src.engine.trainer import (
    EncoderModel,
    build_ts_jepa_model,
    limit_windows,
    resolve_device,
    set_seed,
    train_encoder,
    train_ts_jepa,
)
from src.features.features import FeatureConfig, NCADFeatureExtractor
from src.models.memory.successor_memory import CounterfactualSuccessorMemory
from src.scoring.event_fusion import (
    adaptive_elbow_score_floor,
    aggregate_window_scores,
    compute_metrics,
    event_level_filter,
    moving_average,
    percentile_score_floor,
    positive_robust_z,
)

logger = logging.getLogger("NCAD.engine.orchestrator")


def default_output_dir() -> Path:
    """Generate default timestamped output directory under project root / results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parents[2] / "results" / timestamp


def run_channel(
    channel_data: ChannelData,
    run_dir: Path,
    config: CSMConfig,
    device: torch.device,
) -> dict:
    """Execute complete training, memory construction, calibration, scoring, and output saving for one channel."""
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

    from src.models.registry import is_jepa_model

    is_jepa = is_jepa_model(config.model_type)

    if is_jepa:
        logger.info(f"[{channel_data.channel_id}] training {config.model_type} self-supervised physical dynamics model")
        model, training_history = train_ts_jepa(train_windows, config, train_features.shape[1], device)

        logger.info(f"[{channel_data.channel_id}] calibrating TS-JEPA EVT decision threshold on training discrepancies")
        from src.engine.evaluator import calibrate_jepa_threshold, compute_jepa_discrepancy

        calibration, train_z, disc_stats, evt_calibrator = calibrate_jepa_threshold(train_windows, model, config, device)
        logger.info(
            f"[{channel_data.channel_id}] TS-JEPA event_tau={calibration['event_threshold']:.5f} ({calibration['threshold_method']})"
        )

        logger.info(f"[{channel_data.channel_id}] scoring test windows via predictive discrepancy")
        test_disc = compute_jepa_discrepancy(model, test_windows, config, device)
        # Strict training scale transfer (zero test distribution leakage)
        test_z = positive_robust_z(test_disc, disc_stats)
        point_scores, valid_mask = aggregate_window_scores(
            test_z,
            n_points=len(channel_data.test_raw),
            context_size=config.context_size,
            suspect_size=config.suspect_size,
            step=config.step,
            reducer="mean",
            mapping_method=config.mapping_method,
        )
        smoothed_scores = moving_average(point_scores, config.smoothing_window)
        valid_smoothed_scores = smoothed_scores[valid_mask]
        score_details = {
            "window_scores": test_z,
            "raw_discrepancy": test_disc,
        }

    else:
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
            evt_calibrator,
        ) = calibrate_event_threshold(train_windows, memory, config)
        logger.info(
            f"[{channel_data.channel_id}] memory={calibration['memory_size']:,}, "
            f"context_tau={calibration['context_threshold']:.5f}, event_tau={calibration['event_threshold']:.5f} ({calibration['threshold_method']})"
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
            mapping_method=config.mapping_method,
        )
        smoothed_scores = moving_average(point_scores, config.smoothing_window)
        valid_smoothed_scores = smoothed_scores[valid_mask]

    if config.threshold_method == "evt" or is_jepa:
        training_event_threshold = float(calibration["event_threshold"])
        calibration["training_event_threshold"] = training_event_threshold
        calibration["score_floor_threshold"] = training_event_threshold
        calibration["score_floor_method"] = "evt"
        if evt_calibrator is not None:
            anomaly_probabilities = evt_calibrator.predict_anomaly_probability(smoothed_scores)
        else:
            anomaly_probabilities = np.zeros_like(smoothed_scores)
    elif config.score_floor_percentile is None:
        score_floor = adaptive_elbow_score_floor(valid_smoothed_scores)
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
        anomaly_probabilities = np.zeros_like(smoothed_scores)
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
        anomaly_probabilities = np.zeros_like(smoothed_scores)

    predictions = event_level_filter(
        smoothed_scores,
        calibration["event_threshold"],
        valid_mask,
        min_run=config.min_event_run,
        extreme_factor=config.extreme_event_factor,
    )
    predictions = predictions * valid_mask.astype(np.float32)
    metrics_pt = compute_metrics(channel_data.labels, predictions, valid_mask=valid_mask, use_pa=False)
    metrics_pa = compute_metrics(channel_data.labels, predictions, valid_mask=valid_mask, use_pa=True)

    if "context_ood" in score_details:
        context_ood_point, _ = aggregate_window_scores(
            score_details["context_ood"].astype(np.float32),
            n_points=len(channel_data.test_raw),
            context_size=config.context_size,
            suspect_size=config.suspect_size,
            step=config.step,
            reducer="mean",
            mapping_method=config.mapping_method,
        )
        context_ood_map = context_ood_point > 0.0
        context_ood_rate_win = float(np.mean(score_details["context_ood"])) if len(test_windows) else 0.0
        context_ood_rate_pt = float(np.mean(context_ood_map[valid_mask])) if np.any(valid_mask) else 0.0
    else:
        context_ood_map = np.zeros(len(channel_data.test_raw), dtype=bool)
        context_ood_rate_win = 0.0
        context_ood_rate_pt = 0.0

    elapsed = time.time() - channel_start
    result = {
        "channel": channel_data.channel_id,
        "elapsed_seconds": elapsed,
        "training_history": training_history,
        "calibration": calibration,
        "threshold": {"final_threshold": calibration["event_threshold"], "threshold_method": calibration["threshold_method"]},
        "metrics": metrics_pa if config.use_pa else metrics_pt,
        "point_metrics": metrics_pt,
        "pa_metrics": metrics_pa,
        "context_ood_rate_window": context_ood_rate_win,
        "context_ood_rate_point": context_ood_rate_pt,
        "score_mode": config.model_type if is_jepa else ("counterfactual_successor_memory_uncertainty" if config.manifold_uncertainty else "counterfactual_successor_memory"),
    }

    save_channel_outputs(
        channel_dir,
        channel_data,
        config,
        model,
        feature_extractor,
        memory if not is_jepa else None,
        result,
        score_details,
        point_scores,
        smoothed_scores,
        context_ood_map,
        valid_mask,
        predictions,
        train_z if is_jepa else calibration_window_scores,
        anomaly_probabilities=anomaly_probabilities,
    )
    if config.save_plots:
        from src.utils.plotting import plot_channel_diagnostics

        plot_channel_diagnostics(channel_dir, channel_data, smoothed_scores, calibration["event_threshold"], predictions, context_ood_map)

    logger.info(f"[{channel_data.channel_id}] done in {elapsed:.1f}s: Point-F1={metrics_pt.get('f1', 0.0):.4f}, PA-F1={metrics_pa.get('f1', 0.0):.4f}")
    return result


def save_channel_outputs(
    channel_dir: Path,
    channel_data: ChannelData,
    config: CSMConfig,
    model: EncoderModel,
    feature_extractor: NCADFeatureExtractor,
    memory: Optional[CounterfactualSuccessorMemory],
    result: dict,
    score_details: dict,
    point_scores: np.ndarray,
    smoothed_scores: np.ndarray,
    context_ood_map: np.ndarray,
    valid_mask: np.ndarray,
    predictions: np.ndarray,
    calibration_window_scores: np.ndarray,
    anomaly_probabilities: Optional[np.ndarray] = None,
) -> None:
    """Serialize channel metrics, model weights, memory index, and per-point/per-window predictions."""
    with (channel_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    with (channel_dir / "feature_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(feature_extractor.to_metadata(), file, indent=2)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "model_type": config.model_type,
            "encoder_architecture": getattr(model, "architecture", "unknown"),
        },
        channel_dir / "model.pt",
    )
    if memory is not None:
        memory.save(channel_dir / "successor_memory.npz")

    labels = channel_data.labels if channel_data.labels is not None else np.full(len(channel_data.test_raw), np.nan)
    prob_col = (
        anomaly_probabilities.astype(np.float32)
        if anomaly_probabilities is not None
        else np.zeros(len(channel_data.test_raw), dtype=np.float32)
    )
    predictions_df = pd.DataFrame(
        {
            "time_index": np.arange(len(channel_data.test_raw)),
            "telemetry": channel_data.test_raw if channel_data.test_raw.ndim == 1 else channel_data.test_raw[:, 0],
            "point_score": point_scores,
            "smoothed_score": smoothed_scores,
            "prediction": predictions.astype(np.float32),
            "anomaly_probability": prob_col,
            "label": labels[: len(channel_data.test_raw)],
            "context_ood": context_ood_map.astype(np.float32),
            "valid_score": valid_mask.astype(np.float32),
        }
    )
    predictions_df.to_csv(channel_dir / "point_predictions.csv", index=False)

    w_data = {
        "window_index": np.arange(len(score_details["window_scores"])),
        "window_score": score_details["window_scores"],
    }
    for k in [
        "raw_discrepancy",
        "successor_scores",
        "successor_median_scores",
        "successor_dispersion",
        "manifold_uncertainty_scores",
        "manifold_z",
        "uncertainty_confidence",
        "local_scores",
        "successor_z",
        "local_z",
        "context_distances",
        "context_ratio",
    ]:
        if k in score_details:
            w_data[k] = score_details[k]
    if "context_ood" in score_details:
        w_data["context_ood"] = score_details["context_ood"].astype(np.float32)

    window_df = pd.DataFrame(w_data)
    window_df.to_csv(channel_dir / "window_predictions.csv", index=False)
    pd.DataFrame({"calibration_window_score": calibration_window_scores}).to_csv(
        channel_dir / "calibration_window_scores.csv", index=False
    )


def run_experiment(channels: list[str], config: CSMConfig) -> tuple[Path, pd.DataFrame]:
    """Run full NCAD-CS benchmark evaluation across multiple telemetry channels."""
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
