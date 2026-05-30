"""Scoring, smoothing, thresholding, and metrics for NCAD-CS."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


@dataclass
class ThresholdInfo:
    final_threshold: float
    base_threshold: float
    robust_threshold: float
    percentile_threshold: float
    score_tail_threshold: float
    percentile: float
    alpha: float
    tail_guard_applied: bool
    tail_guard_multiplier: float
    tail_guard_percentile: float
    sparse_tail_threshold: float
    sparse_tail_guard_applied: bool
    sparse_tail_guard_multiplier: float
    sparse_tail_guard_percentile: float
    sparse_tail_guard_max_base: float
    threshold_method: str
    calibration_threshold: float
    calibration_precision: float
    calibration_recall: float
    calibration_f1: float
    calibration_positive_rate: float


def euclidean_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(left) - np.asarray(right)))


def confidence_over_threshold(value: float, threshold: float) -> float:
    """Map a threshold exceedance to [0, 1] with smooth saturation."""

    if not np.isfinite(value) or not np.isfinite(threshold) or threshold <= 1e-12:
        return 0.0
    ratio = (value - threshold) / threshold
    if ratio <= 0:
        return 0.0
    return float(np.clip(ratio / (1.0 + ratio), 0.0, 0.95))


def aggregate_suspect_scores(
    window_scores: np.ndarray,
    substitutions: np.ndarray,
    n_points: int,
    context_size: int,
    suspect_size: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate window-level suspect scores into point-level score arrays."""

    score_sum = np.zeros(n_points, dtype=np.float64)
    counts = np.zeros(n_points, dtype=np.float64)
    substitution_sum = np.zeros(n_points, dtype=np.float64)

    for window_index, score in enumerate(window_scores):
        start = window_index * step + context_size
        end = min(start + suspect_size, n_points)
        if start >= n_points or end <= start:
            continue
        score_sum[start:end] += score
        counts[start:end] += 1.0
        if substitutions[window_index]:
            substitution_sum[start:end] += 1.0

    point_scores = np.divide(score_sum, counts, out=np.zeros_like(score_sum), where=counts > 0)
    substitution_map = np.divide(substitution_sum, counts, out=np.zeros_like(substitution_sum), where=counts > 0) > 0.0
    valid_mask = counts > 0
    return point_scores.astype(np.float32), substitution_map, valid_mask


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float32)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def dynamic_weighted_smoothing(
    point_scores: np.ndarray,
    substitution_map: np.ndarray,
    short_window: int = 50,
    long_window: int = 200,
) -> np.ndarray:
    """Blend responsive and stable smoothing using the substitution indicator map."""

    short_scores = moving_average(point_scores, short_window)
    long_scores = moving_average(point_scores, long_window)
    weights = moving_average(substitution_map.astype(np.float32), short_window)
    weights = np.clip(weights, 0.0, 1.0)
    return ((1.0 - weights) * short_scores + weights * long_scores).astype(np.float32)


def calculate_threshold(
    smoothed_scores: np.ndarray,
    substitution_map: np.ndarray,
    valid_mask: np.ndarray,
    percentile: float = 95.0,
    alpha: float = 1.0,
    tail_guard_multiplier: float = 5.0,
    tail_guard_percentile: float = 95.0,
    sparse_tail_guard_multiplier: float = 3.0,
    sparse_tail_guard_percentile: float = 97.0,
    sparse_tail_guard_max_base: float = 0.02,
) -> ThresholdInfo:
    """Calculate the paper's robust normal-context threshold."""

    finite_valid_mask = valid_mask & np.isfinite(smoothed_scores)
    baseline_mask = finite_valid_mask & ~substitution_map
    baseline = smoothed_scores[baseline_mask]
    if len(baseline) < 20:
        baseline = smoothed_scores[finite_valid_mask]
    if len(baseline) == 0:
        baseline = np.array([0.0], dtype=np.float32)

    q25, q75 = np.percentile(baseline, [25, 75])
    iqr = q75 - q25
    robust_threshold = float(np.median(baseline) + 3.0 * iqr)
    percentile_threshold = float(np.percentile(baseline, percentile))
    if np.isfinite(percentile_threshold) and percentile_threshold > 0:
        base_threshold = percentile_threshold
    elif np.isfinite(robust_threshold) and robust_threshold > 0:
        base_threshold = robust_threshold
    else:
        base_threshold = max(float(np.max(baseline)), 1e-6)

    valid_scores = smoothed_scores[finite_valid_mask]
    score_tail_threshold = 0.0
    tail_guard_applied = False
    sparse_tail_threshold = 0.0
    sparse_tail_guard_applied = False
    if len(valid_scores) >= 20:
        score_tail_threshold = float(np.percentile(valid_scores, tail_guard_percentile))
        sparse_tail_threshold = float(np.percentile(valid_scores, sparse_tail_guard_percentile))

    final_threshold = float(alpha * base_threshold)

    return ThresholdInfo(
        final_threshold=final_threshold,
        base_threshold=base_threshold,
        robust_threshold=robust_threshold,
        percentile_threshold=percentile_threshold,
        score_tail_threshold=score_tail_threshold,
        percentile=percentile,
        alpha=alpha,
        tail_guard_applied=tail_guard_applied,
        tail_guard_multiplier=tail_guard_multiplier,
        tail_guard_percentile=tail_guard_percentile,
        sparse_tail_threshold=sparse_tail_threshold,
        sparse_tail_guard_applied=sparse_tail_guard_applied,
        sparse_tail_guard_multiplier=sparse_tail_guard_multiplier,
        sparse_tail_guard_percentile=sparse_tail_guard_percentile,
        sparse_tail_guard_max_base=sparse_tail_guard_max_base,
        threshold_method="adaptive_distribution",
        calibration_threshold=0.0,
        calibration_precision=0.0,
        calibration_recall=0.0,
        calibration_f1=0.0,
        calibration_positive_rate=0.0,
    )


def calculate_label_calibrated_threshold(
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
    fallback: ThresholdInfo,
    alpha: float = 1.0,
    max_candidates: int = 1024,
) -> ThresholdInfo:
    """Choose a threshold from self-supervised calibration labels.

    The threshold is selected per channel from synthetic clean/contaminated
    calibration scores. No test labels are used. If the synthetic labels are not
    usable, the distribution-only fallback threshold is returned unchanged.
    """

    scores = np.asarray(calibration_scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(calibration_labels, dtype=bool).reshape(-1)
    finite_mask = np.isfinite(scores)
    scores = scores[finite_mask]
    labels = labels[finite_mask]
    if len(scores) < 4 or not np.any(labels) or np.all(labels):
        return replace(fallback, threshold_method="adaptive_distribution_fallback")

    unique_scores = np.unique(scores)
    if len(unique_scores) == 1:
        return replace(fallback, threshold_method="adaptive_distribution_fallback")

    if len(unique_scores) > max_candidates:
        quantiles = np.linspace(0.1, 99.9, max_candidates)
        candidates = np.unique(np.percentile(unique_scores, quantiles))
    else:
        candidates = (unique_scores[:-1] + unique_scores[1:]) * 0.5

    best_threshold = float(fallback.base_threshold)
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = -1.0
    for threshold in candidates:
        predictions = scores > threshold
        tp = float(np.sum(labels & predictions))
        fp = float(np.sum(~labels & predictions))
        fn = float(np.sum(labels & ~predictions))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if (f1, precision, recall) > (best_f1, best_precision, best_recall):
            best_threshold = float(threshold)
            best_precision = float(precision)
            best_recall = float(recall)
            best_f1 = float(f1)

    return replace(
        fallback,
        final_threshold=float(alpha * best_threshold),
        base_threshold=float(best_threshold),
        threshold_method="self_supervised_calibration",
        calibration_threshold=float(best_threshold),
        calibration_precision=best_precision,
        calibration_recall=best_recall,
        calibration_f1=max(best_f1, 0.0),
        calibration_positive_rate=float(np.mean(labels)),
    )


def temporal_consistency_filter(
    preliminary_flags: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    min_run: int = 3,
    extreme_factor: float = 1.8,
) -> np.ndarray:
    """Remove isolated detections unless they are extreme high-score points."""

    flags = preliminary_flags.astype(bool)
    filtered = np.zeros_like(flags, dtype=bool)
    index = 0
    while index < len(flags):
        if not flags[index]:
            index += 1
            continue
        start = index
        while index < len(flags) and flags[index]:
            index += 1
        end = index
        run_scores = scores[start:end]
        if (end - start) >= min_run or np.any(run_scores > extreme_factor * threshold):
            filtered[start:end] = True
    return filtered.astype(np.float32)


def compute_metrics(labels: Optional[np.ndarray], predictions: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> dict:
    if labels is None:
        return {}
    labels = labels[: len(predictions)].astype(np.float32)
    predictions = predictions[: len(labels)].astype(np.float32)
    if valid_mask is not None:
        mask = valid_mask[: len(labels)].astype(bool)
        labels = labels[mask]
        predictions = predictions[mask]
    if len(labels) == 0:
        return {}

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0.0, 1.0]).ravel()
    return {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
