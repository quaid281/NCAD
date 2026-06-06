"""Counterfactual successor memory for NCAD-CS v4.

The memory stores normal context embeddings together with the normal suspect
segment that followed each context. At inference time, nearest normal contexts
provide plausible normal successors for counterfactual scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class SuccessorMemoryConfig:
    n_neighbors: int = 8
    max_memory_windows: Optional[int] = 5000
    context_percentile: float = 99.0
    distance_metric: str = "euclidean"
    seed: int = 42


@dataclass
class SuccessorQueryResult:
    context_distances: np.ndarray
    neighbor_indices: np.ndarray
    successor_scores: np.ndarray
    successor_median_scores: np.ndarray
    successor_dispersion: np.ndarray
    expected_successors: np.ndarray


class CounterfactualSuccessorMemory:
    """KNN memory mapping normal contexts to plausible normal successors."""

    def __init__(self, config: Optional[SuccessorMemoryConfig] = None):
        self.config = config or SuccessorMemoryConfig()
        self.context_embeddings: Optional[np.ndarray] = None
        self.successor_windows: Optional[np.ndarray] = None
        self.sample_indices: Optional[np.ndarray] = None
        self.neighbor_model: Optional[NearestNeighbors] = None
        self.context_threshold: float = 0.0
        self.calibration_context_distances: Optional[np.ndarray] = None
        self.calibration_successor_scores: Optional[np.ndarray] = None
        self.calibration_successor_median_scores: Optional[np.ndarray] = None
        self.calibration_successor_dispersion: Optional[np.ndarray] = None
        self.calibration_expected_successors: Optional[np.ndarray] = None

    def fit(self, context_embeddings: np.ndarray, successor_windows: np.ndarray) -> "CounterfactualSuccessorMemory":
        context_embeddings = np.asarray(context_embeddings, dtype=np.float32)
        successor_windows = np.asarray(successor_windows, dtype=np.float32)
        if len(context_embeddings) == 0:
            raise ValueError("Cannot fit successor memory from zero embeddings.")
        if len(context_embeddings) != len(successor_windows):
            raise ValueError("Context embeddings and successor windows must have matching lengths.")

        selected_indices = np.arange(len(context_embeddings), dtype=np.int64)
        max_windows = self.config.max_memory_windows
        if max_windows is not None and len(selected_indices) > max_windows:
            rng = np.random.default_rng(self.config.seed)
            selected_indices = np.sort(rng.choice(selected_indices, size=max_windows, replace=False))

        self.sample_indices = selected_indices
        self.context_embeddings = context_embeddings[selected_indices].astype(np.float32)
        self.successor_windows = successor_windows[selected_indices].astype(np.float32)

        n_neighbors = max(1, min(self.config.n_neighbors, len(self.context_embeddings)))
        self.neighbor_model = NearestNeighbors(n_neighbors=n_neighbors, metric=self.config.distance_metric)
        self.neighbor_model.fit(self.context_embeddings)
        self._calibrate_leave_one_out()
        return self

    def query(self, context_embeddings: np.ndarray, observed_successors: np.ndarray) -> SuccessorQueryResult:
        if self.neighbor_model is None or self.context_embeddings is None or self.successor_windows is None:
            raise RuntimeError("Successor memory has not been fitted.")
        context_embeddings = np.asarray(context_embeddings, dtype=np.float32)
        observed_successors = np.asarray(observed_successors, dtype=np.float32)
        n_neighbors = max(1, min(self.config.n_neighbors, len(self.context_embeddings)))
        distances, neighbor_indices = self.neighbor_model.kneighbors(context_embeddings, n_neighbors=n_neighbors)
        return self._score_against_neighbors(observed_successors, distances, neighbor_indices)

    def save(self, path: str | Path) -> None:
        if self.context_embeddings is None or self.successor_windows is None:
            raise RuntimeError("Cannot save an unfitted successor memory.")
        np.savez_compressed(
            path,
            context_embeddings=self.context_embeddings,
            successor_windows=self.successor_windows,
            sample_indices=self.sample_indices,
            context_threshold=np.array([self.context_threshold], dtype=np.float32),
            calibration_context_distances=self.calibration_context_distances,
            calibration_successor_scores=self.calibration_successor_scores,
            calibration_successor_median_scores=self.calibration_successor_median_scores,
            calibration_successor_dispersion=self.calibration_successor_dispersion,
            calibration_expected_successors=self.calibration_expected_successors,
            n_neighbors=np.array([self.config.n_neighbors], dtype=np.int64),
            distance_metric=np.array([self.config.distance_metric]),
        )

    def _calibrate_leave_one_out(self) -> None:
        if self.neighbor_model is None or self.context_embeddings is None or self.successor_windows is None:
            raise RuntimeError("Successor memory has not been fitted.")
        if len(self.context_embeddings) == 1:
            self.calibration_context_distances = np.zeros(1, dtype=np.float32)
            self.calibration_successor_scores = np.zeros(1, dtype=np.float32)
            self.calibration_successor_median_scores = np.zeros(1, dtype=np.float32)
            self.calibration_successor_dispersion = np.zeros(1, dtype=np.float32)
            self.calibration_expected_successors = np.zeros_like(self.successor_windows)
            self.context_threshold = 0.0
            return

        n_neighbors = min(self.config.n_neighbors + 1, len(self.context_embeddings))
        distances, indices = self.neighbor_model.kneighbors(self.context_embeddings, n_neighbors=n_neighbors)
        clean_distances = []
        clean_indices = []
        for row_index, row_indices in enumerate(indices):
            keep = row_indices[row_indices != row_index]
            if len(keep) == 0:
                keep = row_indices[:1]
            keep = keep[: self.config.n_neighbors]
            clean_indices.append(keep)
            distance_lookup = {int(index): float(distance) for index, distance in zip(row_indices, distances[row_index])}
            clean_distances.append([distance_lookup.get(int(index), 0.0) for index in keep])

        max_len = max(len(row) for row in clean_indices)
        padded_indices = np.zeros((len(clean_indices), max_len), dtype=np.int64)
        padded_distances = np.zeros((len(clean_distances), max_len), dtype=np.float32)
        for row_index, row in enumerate(clean_indices):
            padded_indices[row_index, : len(row)] = row
            padded_indices[row_index, len(row) :] = row[-1]
            padded_distances[row_index, : len(row)] = clean_distances[row_index]
            padded_distances[row_index, len(row) :] = clean_distances[row_index][-1]

        result = self._score_against_neighbors(self.successor_windows, padded_distances, padded_indices)
        self.calibration_context_distances = result.context_distances.astype(np.float32)
        self.calibration_successor_scores = result.successor_scores.astype(np.float32)
        self.calibration_successor_median_scores = result.successor_median_scores.astype(np.float32)
        self.calibration_successor_dispersion = result.successor_dispersion.astype(np.float32)
        self.calibration_expected_successors = result.expected_successors.astype(np.float32)
        self.context_threshold = float(np.percentile(self.calibration_context_distances, self.config.context_percentile))

    def _score_against_neighbors(
        self,
        observed_successors: np.ndarray,
        distances: np.ndarray,
        neighbor_indices: np.ndarray,
    ) -> SuccessorQueryResult:
        if self.successor_windows is None:
            raise RuntimeError("Successor memory has not been fitted.")
        observed_successors = np.asarray(observed_successors, dtype=np.float32)
        neighbor_successors = self.successor_windows[neighbor_indices]
        # Generic over successor shape (works for raw (T, F) or latent (D,)).
        # neighbor_successors: (B, K, *S);  observed: (B, *S)
        feature_axes_neighbors = tuple(range(2, neighbor_successors.ndim))
        feature_axes_observed = tuple(range(1, observed_successors.ndim))
        expected_successors = np.median(neighbor_successors, axis=1).astype(np.float32)

        residuals = np.sqrt(
            np.mean(
                (neighbor_successors - observed_successors[:, None, ...]) ** 2,
                axis=feature_axes_neighbors,
            )
        )
        successor_scores = np.min(residuals, axis=1).astype(np.float32)
        successor_median_scores = np.sqrt(
            np.mean((expected_successors - observed_successors) ** 2, axis=feature_axes_observed)
        ).astype(np.float32)
        successor_dispersion = np.mean(
            np.std(neighbor_successors, axis=1), axis=feature_axes_observed
        ).astype(np.float32)
        context_distances = distances[:, 0].astype(np.float32)

        return SuccessorQueryResult(
            context_distances=context_distances,
            neighbor_indices=neighbor_indices.astype(np.int64),
            successor_scores=successor_scores,
            successor_median_scores=successor_median_scores,
            successor_dispersion=successor_dispersion,
            expected_successors=expected_successors,
        )
