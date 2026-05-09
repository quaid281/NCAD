"""Context Memory Bank for NCAD-CS contextual substitution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


@dataclass
class MemoryBankConfig:
    min_clusters: int = 2
    max_clusters: int = 50
    percentile: float = 99.0
    distance_metric: str = "euclidean"
    sample_size: int = 5000


class ContextMemoryBank:
    """Stores normal latent prototypes and raw reference context windows."""

    def __init__(self, config: Optional[MemoryBankConfig] = None):
        self.config = config or MemoryBankConfig()
        self.centroids: Optional[np.ndarray] = None
        self.representative_windows: Optional[np.ndarray] = None
        self.representative_indices: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None
        self.n_clusters: int = 0
        self.training_distances: Optional[np.ndarray] = None

    def fit(self, embeddings: np.ndarray, context_windows: np.ndarray) -> "ContextMemoryBank":
        embeddings = np.asarray(embeddings, dtype=np.float64)
        context_windows = np.asarray(context_windows, dtype=np.float32)
        if len(embeddings) == 0:
            raise ValueError("Cannot build memory bank from zero embeddings.")
        if len(embeddings) != len(context_windows):
            raise ValueError("Embeddings and context windows must have the same length.")

        self.n_clusters = self._choose_cluster_count(embeddings)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        self.centroids = kmeans.cluster_centers_.astype(np.float32)

        representative_windows = []
        representative_indices = []
        for cluster_index in range(self.n_clusters):
            member_indices = np.flatnonzero(labels == cluster_index)
            if len(member_indices) == 0:
                nearest_global = int(np.argmin(self._distance_matrix(embeddings, self.centroids[cluster_index : cluster_index + 1])[:, 0]))
            else:
                distances = self._distance_matrix(embeddings[member_indices], self.centroids[cluster_index : cluster_index + 1])[:, 0]
                nearest_global = int(member_indices[int(np.argmin(distances))])
            representative_indices.append(nearest_global)
            representative_windows.append(context_windows[nearest_global])

        self.representative_indices = np.array(representative_indices, dtype=np.int64)
        self.representative_windows = np.stack(representative_windows).astype(np.float32)

        nearest_distances = self._nearest_distances(embeddings)
        self.training_distances = nearest_distances.astype(np.float32)
        self.threshold = float(np.percentile(nearest_distances, self.config.percentile))
        return self

    def query(self, embedding: np.ndarray) -> tuple[float, int]:
        if self.centroids is None:
            raise RuntimeError("Memory bank has not been fitted.")
        distances = self._distance_matrix(np.asarray(embedding, dtype=np.float64).reshape(1, -1), self.centroids)[0]
        index = int(np.argmin(distances))
        return float(distances[index]), index

    def save(self, path: str | Path) -> None:
        if self.centroids is None or self.representative_windows is None or self.threshold is None:
            raise RuntimeError("Cannot save an unfitted memory bank.")
        np.savez_compressed(
            path,
            centroids=self.centroids,
            representative_windows=self.representative_windows,
            representative_indices=self.representative_indices,
            threshold=np.array([self.threshold], dtype=np.float32),
            n_clusters=np.array([self.n_clusters], dtype=np.int64),
            training_distances=self.training_distances,
            distance_metric=np.array([self.config.distance_metric]),
            percentile=np.array([self.config.percentile], dtype=np.float32),
        )

    def _choose_cluster_count(self, embeddings: np.ndarray) -> int:
        n_samples = len(embeddings)
        if n_samples < 2:
            return 1

        unique_embeddings = np.unique(np.round(embeddings, decimals=8), axis=0)
        unique_count = len(unique_embeddings)
        if unique_count < 2:
            return 1

        max_clusters = min(self.config.max_clusters, n_samples, unique_count)
        min_clusters = min(max(2, self.config.min_clusters), max_clusters)
        if max_clusters <= 2:
            return max_clusters

        sample_embeddings = embeddings
        if n_samples > self.config.sample_size:
            rng = np.random.default_rng(42)
            sample_indices = rng.choice(n_samples, size=self.config.sample_size, replace=False)
            sample_embeddings = embeddings[sample_indices]

        sample_unique_count = len(np.unique(np.round(sample_embeddings, decimals=8), axis=0))
        max_clusters = min(max_clusters, sample_unique_count)
        if max_clusters < 2:
            return 1
        min_clusters = min(min_clusters, max_clusters)

        best_score = -np.inf
        best_k = min_clusters
        candidate_counts = np.unique(np.linspace(min_clusters, max_clusters, num=min(10, max_clusters - min_clusters + 1), dtype=int))
        for n_clusters in candidate_counts:
            if n_clusters >= len(sample_embeddings):
                continue
            labels = KMeans(n_clusters=int(n_clusters), random_state=42, n_init=10).fit_predict(sample_embeddings)
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(sample_embeddings, labels, metric=self.config.distance_metric)
            if score > best_score:
                best_score = float(score)
                best_k = int(n_clusters)
        return max(1, min(best_k, n_samples))

    def _nearest_distances(self, embeddings: np.ndarray) -> np.ndarray:
        if self.centroids is None:
            raise RuntimeError("Memory bank has not been fitted.")
        distances = self._distance_matrix(embeddings, self.centroids)
        return np.min(distances, axis=1)

    def _distance_matrix(self, x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
        if self.config.distance_metric == "cosine":
            return cosine_distances(x_values, y_values)
        if self.config.distance_metric == "euclidean":
            return euclidean_distances(x_values, y_values)
        raise ValueError(f"Unsupported distance metric: {self.config.distance_metric}")
