"""Modern Hopfield-style associative context retrieval.

This is an isolated alternative to the K-Means Context Memory Bank. It keeps the
NCAD-CS idea of replacing a contaminated context, but retrieves the replacement
through differentiable associative attention over normal context states instead
of discrete clustering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class HopfieldContextConfig:
    inverse_temperature: float = 12.0
    update_steps: int = 1
    percentile: float = 99.0
    max_memory_items: Optional[int] = 5000
    chunk_size: int = 512
    normalize: bool = True
    seed: int = 42


@dataclass
class HopfieldRetrieval:
    distance: float
    index: int
    confidence: float
    retrieved_embedding: np.ndarray
    attention_weights: np.ndarray


class HopfieldContextMemory:
    """Associative memory for normal context embeddings.

    The fitted object stores normal context embeddings, but it does not cluster
    them. Query retrieval is a continuous Hopfield update:

    attention(q, K) = softmax(beta q K^T)
    retrieved(q) = attention(q, K) V

    The residual between the observed context and its retrieved normal state is
    used as the contamination distance.
    """

    def __init__(self, config: Optional[HopfieldContextConfig] = None):
        self.config = config or HopfieldContextConfig()
        self.keys: Optional[np.ndarray] = None
        self.values: Optional[np.ndarray] = None
        self.reference_windows: Optional[np.ndarray] = None
        self.memory_indices: Optional[np.ndarray] = None
        self.training_distances: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None

    def fit(self, context_embeddings: np.ndarray, reference_windows: Optional[np.ndarray] = None) -> "HopfieldContextMemory":
        embeddings = self._as_2d(context_embeddings).astype(np.float32)
        if len(embeddings) == 0:
            raise ValueError("Cannot fit Hopfield context memory from zero embeddings.")

        selected_indices = self._select_memory_indices(len(embeddings))
        selected_embeddings = embeddings[selected_indices]
        self.values = selected_embeddings
        self.keys = self._normalize(selected_embeddings) if self.config.normalize else selected_embeddings.copy()
        self.memory_indices = selected_indices.astype(np.int64)

        if reference_windows is not None:
            windows = np.asarray(reference_windows, dtype=np.float32)
            if len(windows) != len(embeddings):
                raise ValueError("reference_windows must have the same length as context_embeddings.")
            self.reference_windows = windows[selected_indices]

        retrieved, _ = self._retrieve_many(selected_embeddings, exclude_self=len(selected_embeddings) > 1)
        distances = np.linalg.norm(selected_embeddings - retrieved, axis=1)
        self.training_distances = distances.astype(np.float32)
        self.threshold = float(np.percentile(distances, self.config.percentile))
        return self

    def query(self, embedding: np.ndarray) -> tuple[float, int]:
        retrieval = self.retrieve(embedding)
        return retrieval.distance, retrieval.index

    def retrieve(self, embedding: np.ndarray) -> HopfieldRetrieval:
        if self.keys is None or self.values is None or self.threshold is None:
            raise RuntimeError("Hopfield context memory has not been fitted.")

        query = self._as_2d(embedding).astype(np.float32)
        if len(query) != 1:
            raise ValueError("retrieve expects a single embedding.")

        retrieved, weights = self._retrieve_many(query)
        distance = float(np.linalg.norm(query[0] - retrieved[0]))
        index = int(np.argmax(weights[0]))
        confidence = self._confidence_over_threshold(distance, self.threshold)
        return HopfieldRetrieval(
            distance=distance,
            index=index,
            confidence=confidence,
            retrieved_embedding=retrieved[0].astype(np.float32),
            attention_weights=weights[0].astype(np.float32),
        )

    def retrieve_reference_window(self, embedding: np.ndarray) -> Optional[np.ndarray]:
        if self.reference_windows is None:
            return None
        retrieval = self.retrieve(embedding)
        flat_windows = self.reference_windows.reshape(len(self.reference_windows), -1)
        weighted = retrieval.attention_weights @ flat_windows
        return weighted.reshape(self.reference_windows.shape[1:]).astype(np.float32)

    def _retrieve_many(self, queries: np.ndarray, exclude_self: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if self.keys is None or self.values is None:
            raise RuntimeError("Hopfield context memory has not been fitted.")

        queries = self._as_2d(queries).astype(np.float32)
        updated = queries.copy()
        weights = np.zeros((len(queries), len(self.keys)), dtype=np.float32)

        for _ in range(max(1, self.config.update_steps)):
            retrieved_chunks = []
            weight_chunks = []
            for start in range(0, len(updated), self.config.chunk_size):
                end = min(start + self.config.chunk_size, len(updated))
                chunk = updated[start:end]
                chunk_keys = self._normalize(chunk) if self.config.normalize else chunk
                logits = self.config.inverse_temperature * (chunk_keys @ self.keys.T)
                if exclude_self and len(updated) == len(self.keys):
                    rows = np.arange(end - start)
                    cols = np.arange(start, end)
                    logits[rows, cols] = -np.inf
                chunk_weights = self._softmax(logits)
                retrieved_chunks.append(chunk_weights @ self.values)
                weight_chunks.append(chunk_weights)
            updated = np.concatenate(retrieved_chunks, axis=0).astype(np.float32)
            weights = np.concatenate(weight_chunks, axis=0).astype(np.float32)
        return updated, weights

    def _select_memory_indices(self, n_items: int) -> np.ndarray:
        if self.config.max_memory_items is None or n_items <= self.config.max_memory_items:
            return np.arange(n_items)
        rng = np.random.default_rng(self.config.seed)
        return np.sort(rng.choice(n_items, size=self.config.max_memory_items, replace=False))

    @staticmethod
    def _as_2d(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim == 1:
            return array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError("Expected embeddings with shape (n, dim) or (dim,).")
        return array

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.clip(norms, 1e-8, None)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        row_max = np.max(logits, axis=1, keepdims=True)
        stable = logits - np.where(np.isfinite(row_max), row_max, 0.0)
        exp_values = np.exp(stable)
        exp_values[~np.isfinite(exp_values)] = 0.0
        denominators = np.sum(exp_values, axis=1, keepdims=True)
        fallback = np.full_like(exp_values, 1.0 / exp_values.shape[1])
        return np.divide(exp_values, denominators, out=fallback, where=denominators > 0)

    @staticmethod
    def _confidence_over_threshold(value: float, threshold: float) -> float:
        if threshold <= 1e-8:
            return 1.0 if value > threshold else 0.0
        return float(np.clip((value - threshold) / threshold, 0.0, 1.0))
