"""Scoring and event-fusion utilities for the v4 successor-memory prototype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.scoring.evt_calibrator import EVTCalibrator, EVTThresholdResult


@dataclass
class RobustStats:
    median: float
    iqr: float
    percentile_95: float
    percentile_99: float


@dataclass
class AdaptiveScoreFloor:
    threshold: float
    method: str
    reason: str
    plateau_adjusted: bool
    plateau_fraction: float
    lower_elbow_threshold: float
    upper_elbow_threshold: float
    otsu_threshold: float
    selected_candidate: str
    selected_percentile: float
    effective_count: int

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "method": self.method,
            "reason": self.reason,
            "plateau_adjusted": self.plateau_adjusted,
            "plateau_fraction": self.plateau_fraction,
            "lower_elbow_threshold": self.lower_elbow_threshold,
            "upper_elbow_threshold": self.upper_elbow_threshold,
            "otsu_threshold": self.otsu_threshold,
            "selected_candidate": self.selected_candidate,
            "selected_percentile": self.selected_percentile,
            "effective_count": self.effective_count,
        }


def robust_stats(values: np.ndarray) -> RobustStats:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        values = np.array([0.0], dtype=np.float64)
    q25, q75 = np.percentile(values, [25.0, 75.0])
    return RobustStats(
        median=float(np.median(values)),
        iqr=float(max(q75 - q25, 1e-6)),
        percentile_95=float(np.percentile(values, 95.0)),
        percentile_99=float(np.percentile(values, 99.0)),
    )


def adaptive_elbow_score_floor(values: np.ndarray) -> AdaptiveScoreFloor:
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    scores = scores[np.isfinite(scores)]
    if len(scores) < 20:
        return _empty_score_floor("too_few_scores", int(len(scores)))

    max_score = float(np.max(scores))
    plateau_epsilon = max(1e-6, 1e-5 * max(abs(max_score), 1.0))
    plateau_mask = scores >= max_score - plateau_epsilon
    plateau_fraction = float(np.mean(plateau_mask))
    working_scores = scores
    plateau_adjusted = False
    non_plateau_scores = scores[~plateau_mask]
    if plateau_fraction >= 0.005 and len(non_plateau_scores) >= 20:
        working_scores = non_plateau_scores
        plateau_adjusted = True

    if len(working_scores) < 20:
        return _empty_score_floor("too_few_non_plateau_scores", int(len(working_scores)), plateau_adjusted, plateau_fraction)
    if float(np.percentile(working_scores, 90.0)) <= 1e-8:
        return _empty_score_floor("near_zero_score_bulk", int(len(working_scores)), plateau_adjusted, plateau_fraction)

    sorted_scores = np.sort(working_scores)
    score_span = float(sorted_scores[-1] - sorted_scores[0])
    if score_span <= max(1e-6, 1e-6 * max(abs(float(sorted_scores[-1])), 1.0)):
        return _empty_score_floor("flat_score_distribution", int(len(working_scores)), plateau_adjusted, plateau_fraction)

    lower_elbow = _score_curve_elbow(sorted_scores, direction="lower")
    upper_elbow = _score_curve_elbow(sorted_scores, direction="upper")
    otsu_threshold = _otsu_score_threshold(working_scores)
    selected_candidate, threshold = _select_elbow_candidate(
        lower_elbow,
        upper_elbow,
        otsu_threshold,
        plateau_fraction,
    )
    selected_percentile = float(np.mean(working_scores <= threshold) * 100.0)

    return AdaptiveScoreFloor(
        threshold=float(threshold),
        method="adaptive_elbow",
        reason="ok",
        plateau_adjusted=plateau_adjusted,
        plateau_fraction=plateau_fraction,
        lower_elbow_threshold=float(lower_elbow),
        upper_elbow_threshold=float(upper_elbow),
        otsu_threshold=float(otsu_threshold),
        selected_candidate=selected_candidate,
        selected_percentile=selected_percentile,
        effective_count=int(len(working_scores)),
    )


def percentile_score_floor(values: np.ndarray, percentile: float) -> AdaptiveScoreFloor:
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return _empty_score_floor("no_scores", 0)

    threshold = float(np.percentile(scores, percentile))
    max_score = float(np.max(scores))
    plateau_epsilon = max(1e-6, 1e-5 * max(abs(max_score), 1.0))
    plateau_fraction = float(np.mean(scores >= max_score - plateau_epsilon))
    plateau_adjusted = False
    if threshold >= max_score - plateau_epsilon:
        non_plateau_scores = scores[scores < max_score - plateau_epsilon]
        if len(non_plateau_scores) >= 20:
            threshold = float(np.percentile(non_plateau_scores, percentile))
            scores = non_plateau_scores
            plateau_adjusted = True

    return AdaptiveScoreFloor(
        threshold=threshold,
        method="percentile_floor",
        reason="ok",
        plateau_adjusted=plateau_adjusted,
        plateau_fraction=plateau_fraction,
        lower_elbow_threshold=0.0,
        upper_elbow_threshold=0.0,
        otsu_threshold=0.0,
        selected_candidate=f"p{percentile:g}",
        selected_percentile=float(percentile),
        effective_count=int(len(scores)),
    )


def calibrate_evt_threshold(
    values: np.ndarray,
    risk_level: float = 1e-3,
    init_percentile: float = 98.0,
) -> EVTThresholdResult:
    calibrator = EVTCalibrator(risk_level=risk_level, init_percentile=init_percentile)
    calibrator.fit(values)
    return calibrator.compute_threshold(values, risk_level=risk_level)


def _empty_score_floor(
    reason: str,
    effective_count: int,
    plateau_adjusted: bool = False,
    plateau_fraction: float = 0.0,
) -> AdaptiveScoreFloor:
    return AdaptiveScoreFloor(
        threshold=0.0,
        method="adaptive_elbow",
        reason=reason,
        plateau_adjusted=plateau_adjusted,
        plateau_fraction=plateau_fraction,
        lower_elbow_threshold=0.0,
        upper_elbow_threshold=0.0,
        otsu_threshold=0.0,
        selected_candidate="none",
        selected_percentile=0.0,
        effective_count=effective_count,
    )


def _score_curve_elbow(sorted_scores: np.ndarray, direction: str) -> float:
    lower_index = int(np.floor((len(sorted_scores) - 1) * 0.05))
    upper_index = int(np.ceil((len(sorted_scores) - 1) * 0.995))
    upper_index = max(lower_index + 2, min(upper_index, len(sorted_scores) - 1))
    segment = sorted_scores[lower_index : upper_index + 1]
    span = float(segment[-1] - segment[0])
    if span <= 0.0:
        return float(segment[-1])

    x_axis = np.linspace(0.0, 1.0, len(segment))
    y_axis = (segment - segment[0]) / span
    if direction == "upper":
        elbow_index = int(np.argmax(y_axis - x_axis))
    else:
        elbow_index = int(np.argmax(x_axis - y_axis))
    return float(segment[elbow_index])


def _otsu_score_threshold(scores: np.ndarray) -> float:
    bin_count = min(256, max(32, int(np.sqrt(len(scores)))))
    histogram, bin_edges = np.histogram(scores, bins=bin_count)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weight_low = np.cumsum(histogram)
    weight_high = np.cumsum(histogram[::-1])[::-1]
    mean_low = np.cumsum(histogram * centers) / np.maximum(weight_low, 1)
    mean_high = (np.cumsum((histogram * centers)[::-1]) / np.maximum(weight_high[::-1], 1))[::-1]
    between_class_variance = weight_low[:-1] * weight_high[1:] * (mean_low[:-1] - mean_high[1:]) ** 2
    if len(between_class_variance) == 0:
        return float(np.median(scores))
    return float(centers[int(np.argmax(between_class_variance))])


def _select_elbow_candidate(
    lower_elbow: float,
    upper_elbow: float,
    otsu_threshold: float,
    plateau_fraction: float,
) -> tuple[str, float]:
    if otsu_threshold >= lower_elbow and otsu_threshold >= upper_elbow:
        return "otsu", otsu_threshold
    if lower_elbow >= otsu_threshold and lower_elbow >= upper_elbow:
        if upper_elbow > otsu_threshold:
            return "upper_elbow", upper_elbow
        return "lower_elbow", lower_elbow
    if plateau_fraction >= 0.05 and upper_elbow > otsu_threshold:
        return "upper_elbow", upper_elbow
    if otsu_threshold > lower_elbow:
        return "otsu", otsu_threshold
    return "lower_elbow", lower_elbow


def positive_robust_z(values: np.ndarray, stats: RobustStats, clip: float = 20.0) -> np.ndarray:
    scores = (np.asarray(values, dtype=np.float64) - stats.median) / max(stats.iqr, 1e-6)
    return np.clip(scores, 0.0, clip).astype(np.float32)


def local_deviation_scores(windows: np.ndarray, context_size: int, tail_size: int = 64, mad_floor: float = 0.20) -> np.ndarray:
    """Measure abrupt suspect deviations across ALL features relative to the recent context."""

    n_features = windows.shape[2]
    all_scores = []
    for f in range(n_features):
        raw_values = np.asarray(windows[:, :, f], dtype=np.float64)
        context_tail = raw_values[:, max(0, context_size - tail_size) : context_size]
        suspects = raw_values[:, context_size:]
        medians = np.median(context_tail, axis=1)
        mad = np.median(np.abs(context_tail - medians[:, None]), axis=1)
        scale = np.maximum(1.4826 * mad, mad_floor)
        point_z = np.max(np.abs((suspects - medians[:, None]) / scale[:, None]), axis=1)
        mean_shift = np.abs(np.mean(suspects, axis=1) - medians) / scale
        all_scores.append(np.maximum(point_z, mean_shift))
    return np.max(all_scores, axis=0).astype(np.float32)


def robust_dispersion_floor(values: np.ndarray, percentile: float = 50.0, minimum: float = 1e-6) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float(minimum)
    return float(max(np.percentile(np.maximum(values, 0.0), percentile), minimum))


def reconstruction_deviation_scores(
    observed_successors: np.ndarray,
    expected_successors: np.ndarray,
) -> np.ndarray:
    """Per-window RMSE between observed and KNN-expected successors."""
    residuals = np.asarray(observed_successors, dtype=np.float64) - np.asarray(expected_successors, dtype=np.float64)
    return np.sqrt(np.mean(residuals ** 2, axis=tuple(range(1, residuals.ndim)))).astype(np.float32)


def successor_manifold_uncertainty_scores(
    successor_scores: np.ndarray,
    successor_dispersion: np.ndarray,
    dispersion_floor: float,
) -> np.ndarray:
    residuals = np.maximum(np.asarray(successor_scores, dtype=np.float64), 0.0)
    dispersion = np.maximum(np.asarray(successor_dispersion, dtype=np.float64), 0.0)
    scale = dispersion + max(float(dispersion_floor), 1e-6)
    return (residuals / scale).astype(np.float32)


def dispersion_confidence(
    successor_dispersion: np.ndarray,
    dispersion_stats: RobustStats,
    min_confidence: float = 0.35,
) -> np.ndarray:
    dispersion_z = positive_robust_z(successor_dispersion, dispersion_stats, clip=10.0)
    confidence = 1.0 / (1.0 + 0.35 * dispersion_z)
    return np.clip(confidence, min_confidence, 1.0).astype(np.float32)


def fuse_evidence_scores(
    successor_z: np.ndarray,
    local_z: np.ndarray,
    context_ratio: np.ndarray,
    manifold_z: Optional[np.ndarray] = None,
    reconstruction_z: Optional[np.ndarray] = None,
    sindy_z: Optional[np.ndarray] = None,
    uncertainty_confidence: Optional[np.ndarray] = None,
    successor_weight: float = 1.0,
    local_weight: float = 0.80,
    context_weight: float = 0.35,
    reconstruction_weight: float = 0.60,
    sindy_weight: float = 0.50,
    normalize_components: bool = True,
) -> np.ndarray:
    successor_z = np.asarray(successor_z, dtype=np.float32)
    local_z = np.asarray(local_z, dtype=np.float32)
    if normalize_components:
        succ_scale = max(float(np.percentile(successor_z, 95.0)), 1e-4)
        local_scale = max(float(np.percentile(local_z, 95.0)), 1e-4)
        successor_z = successor_z / succ_scale
        local_z = local_z / local_scale
    if uncertainty_confidence is None:
        successor_evidence = successor_z
    else:
        successor_evidence = successor_z * np.asarray(uncertainty_confidence, dtype=np.float32)
    if manifold_z is not None:
        m_z = np.asarray(manifold_z, dtype=np.float32)
        if normalize_components:
            m_scale = max(float(np.percentile(m_z, 95.0)), 1e-4)
            m_z = m_z / m_scale
        successor_evidence = np.maximum(successor_evidence, m_z)
    context_excess = np.maximum(np.asarray(context_ratio, dtype=np.float32) - 1.0, 0.0)
    context_gain = 1.0 + context_weight * np.clip(context_excess, 0.0, 2.0)
    contextual_successor = successor_weight * successor_evidence * context_gain
    local_component = local_weight * local_z
    fused = np.maximum(contextual_successor, local_component)
    if reconstruction_z is not None:
        rec_z = np.asarray(reconstruction_z, dtype=np.float32)
        if normalize_components:
            rec_scale = max(float(np.percentile(rec_z, 95.0)), 1e-4)
            rec_z = rec_z / rec_scale
        reconstruction_component = reconstruction_weight * rec_z
        fused = np.maximum(fused, reconstruction_component)
    if sindy_z is not None:
        sin_z = np.asarray(sindy_z, dtype=np.float32)
        if normalize_components:
            sin_scale = max(float(np.percentile(sin_z, 95.0)), 1e-4)
            sin_z = sin_z / sin_scale
        sindy_component = sindy_weight * sin_z
        fused = np.maximum(fused, sindy_component)
    return fused.astype(np.float32)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def aggregate_window_scores(
    window_scores: np.ndarray,
    n_points: int,
    context_size: int,
    suspect_size: int,
    step: int,
    window_indices: Optional[np.ndarray] = None,
    reducer: str = "mean",
    mapping_method: str = "smear",
) -> tuple[np.ndarray, np.ndarray]:
    window_scores = np.asarray(window_scores, dtype=np.float64).reshape(-1)
    if window_indices is None:
        window_indices = np.arange(len(window_scores), dtype=np.int64)
    else:
        window_indices = np.asarray(window_indices, dtype=np.int64)

    if mapping_method == "smear":
        if reducer == "max":
            point_scores = np.full(n_points, -np.inf, dtype=np.float64)
            counts = np.zeros(n_points, dtype=np.float64)
            for window_index, score in zip(window_indices, window_scores):
                start = int(window_index) * step + context_size
                end = min(start + suspect_size, n_points)
                if start >= n_points or end <= start:
                    continue
                point_scores[start:end] = np.maximum(point_scores[start:end], score)
                counts[start:end] += 1.0
            point_scores[~np.isfinite(point_scores)] = 0.0
            return point_scores.astype(np.float32), counts > 0

        score_sum = np.zeros(n_points, dtype=np.float64)
        counts = np.zeros(n_points, dtype=np.float64)
        for window_index, score in zip(window_indices, window_scores):
            start = int(window_index) * step + context_size
            end = min(start + suspect_size, n_points)
            if start >= n_points or end <= start:
                continue
            score_sum[start:end] += score
            counts[start:end] += 1.0
        point_scores = np.divide(score_sum, counts, out=np.zeros_like(score_sum), where=counts > 0)
        return point_scores.astype(np.float32), counts > 0

    score_sum = np.zeros(n_points, dtype=np.float64)
    counts = np.zeros(n_points, dtype=np.float64)
    for window_index, score in zip(window_indices, window_scores):
        if mapping_method in ["last", "trailing"]:
            idx = int(window_index) * step + context_size + suspect_size - 1
            if idx < n_points:
                score_sum[idx] += score
                counts[idx] += 1.0
        elif mapping_method in ["first", "leading"]:
            idx = int(window_index) * step + context_size
            if idx < n_points:
                score_sum[idx] += score
                counts[idx] += 1.0
        elif mapping_method in ["middle", "center"]:
            idx = int(window_index) * step + context_size + suspect_size // 2
            if idx < n_points:
                score_sum[idx] += score
                counts[idx] += 1.0
        elif mapping_method == "suspect_trailing":
            idx = int(window_index) * step + context_size + suspect_size - 1
            if idx < n_points:
                score_sum[idx] += score
                counts[idx] += 1.0

    point_scores = np.divide(score_sum, counts, out=np.zeros_like(score_sum), where=counts > 0)
    return point_scores.astype(np.float32), counts > 0



def event_level_filter(
    scores: np.ndarray,
    threshold: float,
    valid_mask: np.ndarray,
    min_run: int = 2,
    extreme_factor: float = 1.75,
    min_area_factor: float = 0.75,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    flags = (scores > threshold) & valid_mask.astype(bool)
    predictions = np.zeros_like(flags, dtype=bool)
    index = 0
    safe_threshold = max(float(threshold), 1e-6)
    while index < len(flags):
        if not flags[index]:
            index += 1
            continue
        start = index
        while index < len(flags) and flags[index]:
            index += 1
        end = index
        event_scores = scores[start:end]
        peak = float(np.max(event_scores)) if len(event_scores) else 0.0
        area = float(np.sum(np.maximum(event_scores - safe_threshold, 0.0)))
        keep = (end - start) >= min_run or peak >= extreme_factor * safe_threshold
        keep = keep or area >= min_area_factor * safe_threshold * max(min_run, 1)
        if keep:
            predictions[start:end] = True
    return predictions.astype(np.float32)


def point_adjustment(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Adjust predictions such that if any point in a ground-truth anomaly segment
    is correctly predicted, the entire segment is marked as predicted anomaly (True Positive).
    """
    adjusted = np.asarray(predictions).copy()
    labels = np.asarray(labels)
    in_anomaly = False
    start = 0
    n = len(labels)
    for i in range(n):
        if labels[i] == 1.0:
            if not in_anomaly:
                in_anomaly = True
                start = i
        else:
            if in_anomaly:
                in_anomaly = False
                if np.any(predictions[start:i] == 1.0):
                    adjusted[start:i] = 1.0
    if in_anomaly:
        if np.any(predictions[start:n] == 1.0):
            adjusted[start:n] = 1.0
    return adjusted


def compute_metrics(
    labels: Optional[np.ndarray],
    predictions: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    use_pa: bool = False,
) -> dict:
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

    if use_pa:
        predictions = point_adjustment(labels, predictions)

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


def confidence_over_threshold(value: float, threshold: float) -> float:
    """Map a threshold exceedance to [0, 1] with smooth saturation."""
    if not np.isfinite(value) or not np.isfinite(threshold) or threshold <= 1e-12:
        return 0.0
    ratio = (value - threshold) / threshold
    if ratio <= 0:
        return 0.0
    return float(np.clip(ratio / (1.0 + ratio), 0.0, 0.95))


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

