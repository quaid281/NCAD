"""Evaluation, window scoring, and EVT/SPOT threshold calibration engine for NCAD-CS."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch

from src.config import CSMConfig
from src.engine.trainer import EncoderModel, encode_windows
from src.models.successor_memory import CounterfactualSuccessorMemory, SuccessorMemoryConfig
from src.utils.event_fusion import (
    aggregate_window_scores,
    dispersion_confidence,
    fuse_evidence_scores,
    local_deviation_scores,
    moving_average,
    positive_robust_z,
    robust_dispersion_floor,
    robust_stats,
    successor_manifold_uncertainty_scores,
)
from src.utils.evt_calibrator import EVTCalibrator

logger = logging.getLogger("NCAD.engine.evaluator")


def build_successor_memory(
    model: EncoderModel,
    train_windows: np.ndarray,
    config: CSMConfig,
    device: torch.device,
) -> CounterfactualSuccessorMemory:
    """Build and fit the Counterfactual Successor Memory on nominal training windows."""
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
    """Compute fused anomaly evidence scores for sliding windows."""
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
) -> tuple:
    """Calibrate decision threshold using EVT/SPOT tail modeling or empirical distribution."""
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
        mapping_method=config.mapping_method,
    )
    calibration_smoothed_scores = moving_average(calibration_point_scores, config.smoothing_window)
    valid_scores = calibration_smoothed_scores[calibration_valid_mask]
    if len(valid_scores) == 0:
        valid_scores = np.array([0.0], dtype=np.float32)

    evt_calibrator = None
    evt_info = None
    if config.threshold_method == "evt":
        evt_calibrator = EVTCalibrator(
            risk_level=config.evt_risk_level,
            init_percentile=config.evt_init_percentile,
        )
        evt_calibrator.fit(valid_scores)
        evt_res = evt_calibrator.compute_threshold(valid_scores, risk_level=config.evt_risk_level)
        threshold = float(evt_res.threshold)
        threshold_method_name = f"evt_gpd_{evt_res.method}"
        evt_info = evt_res.to_dict()
    else:
        threshold = float(np.percentile(valid_scores, config.event_threshold_percentile))
        threshold_method_name = "counterfactual_successor_training_distribution"
    threshold = max(threshold, 1e-6)

    calibration = {
        "threshold_method": threshold_method_name,
        "event_threshold": threshold,
        "event_threshold_percentile": config.event_threshold_percentile,
        "evt_details": evt_info,
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
    return (
        calibration,
        calibration_window_scores,
        successor_stats,
        local_stats,
        manifold_stats,
        dispersion_stats,
        dispersion_floor,
        evt_calibrator,
    )


def compute_jepa_discrepancy(
    model: torch.nn.Module,
    windows: np.ndarray,
    config: CSMConfig,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Compute predictive discrepancy scores for sliding windows using TS-JEPA."""
    model.eval()
    discrepancies = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = windows[i : i + batch_size]
            ctx = torch.from_numpy(batch[:, : config.context_size]).float().to(device)
            tgt = torch.from_numpy(batch[:, config.context_size :]).float().to(device)
            disc = model.compute_predictive_discrepancy(ctx, tgt, use_mahalanobis=config.use_mahalanobis)
            discrepancies.append(disc.cpu().numpy())
    if not discrepancies:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(discrepancies, axis=0).astype(np.float32)


def calibrate_jepa_threshold(
    train_windows: np.ndarray,
    model: torch.nn.Module,
    config: CSMConfig,
    device: torch.device,
) -> tuple:
    """Calibrate decision threshold on nominal training discrepancies using EVT tail estimation."""
    train_disc = compute_jepa_discrepancy(model, train_windows, config, device)
    disc_stats = robust_stats(train_disc)
    train_z = positive_robust_z(train_disc, disc_stats)

    n_points = (len(train_windows) - 1) * config.step + config.full_window_size
    train_pt_scores, train_valid_mask = aggregate_window_scores(
        train_z,
        n_points=n_points,
        context_size=config.context_size,
        suspect_size=config.suspect_size,
        step=config.step,
        reducer="mean",
        mapping_method=config.mapping_method,
    )
    train_smoothed = moving_average(train_pt_scores, config.smoothing_window)
    valid_scores = train_smoothed[train_valid_mask]
    if len(valid_scores) == 0:
        valid_scores = np.array([0.0], dtype=np.float32)

    evt_calibrator = None
    evt_info = None
    if config.threshold_method == "evt":
        evt_calibrator = EVTCalibrator(
            risk_level=config.evt_risk_level,
            init_percentile=config.evt_init_percentile,
        )
        evt_calibrator.fit(valid_scores)
        evt_res = evt_calibrator.compute_threshold(valid_scores, risk_level=config.evt_risk_level)
        threshold = float(evt_res.threshold)
        threshold_method_name = f"evt_gpd_{evt_res.method}"
        evt_info = evt_res.to_dict()
    else:
        threshold = float(np.percentile(valid_scores, config.event_threshold_percentile))
        threshold_method_name = "jepa_training_distribution"
    threshold = max(threshold, 1e-6)

    calibration = {
        "threshold_method": threshold_method_name,
        "event_threshold": threshold,
        "event_threshold_percentile": config.event_threshold_percentile,
        "evt_details": evt_info,
        "discrepancy_median": float(disc_stats.median),
        "discrepancy_iqr": float(disc_stats.iqr),
        "discrepancy_p99": float(disc_stats.percentile_99),
        "model_type": config.model_type,
        "use_mahalanobis": config.use_mahalanobis,
    }
    return calibration, train_z, disc_stats, evt_calibrator
