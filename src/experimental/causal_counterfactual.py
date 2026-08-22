"""Counterfactual context substitution via a lightweight structural model.

This module reframes substitution as a causal intervention. A normal-only
structural equation learns how context embeddings generate full-window
embeddings. At inference, we can ask: what score would this window receive under
do(context = healthy_context)?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CounterfactualConfig:
    ridge_alpha: float = 1e-3
    residual_percentile: float = 99.0
    shift_percentile: float = 99.0
    score_boost: float = 2.5
    min_intervention_confidence: float = 0.10


@dataclass
class CounterfactualContextResult:
    observed_residual: float
    counterfactual_residual: float
    context_shift: float
    intervention_confidence: float
    final_score: float
    expected_full_embedding: np.ndarray
    counterfactual_full_embedding: np.ndarray


class CounterfactualContextSubstitutor:
    """Normal-context structural equation for causal substitution.

    The learned map is intentionally simple and transparent:

    z_full = f(z_context) + epsilon

    Replacing z_context with a healthy intervention lets us score the same
    observed full-window embedding against the counterfactual normal context.
    """

    def __init__(self, config: Optional[CounterfactualConfig] = None):
        self.config = config or CounterfactualConfig()
        self.coefficients: Optional[np.ndarray] = None
        self.residual_threshold: Optional[float] = None
        self.context_shift_threshold: Optional[float] = None
        self.training_residuals: Optional[np.ndarray] = None

    def fit(self, context_embeddings: np.ndarray, full_embeddings: np.ndarray) -> "CounterfactualContextSubstitutor":
        contexts = self._as_2d(context_embeddings).astype(np.float64)
        full = self._as_2d(full_embeddings).astype(np.float64)
        if len(contexts) == 0:
            raise ValueError("Cannot fit counterfactual substitutor from zero embeddings.")
        if len(contexts) != len(full):
            raise ValueError("context_embeddings and full_embeddings must have the same length.")

        design = self._with_intercept(contexts)
        regularizer = self.config.ridge_alpha * np.eye(design.shape[1], dtype=np.float64)
        regularizer[0, 0] = 0.0
        lhs = design.T @ design + regularizer
        rhs = design.T @ full
        try:
            self.coefficients = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            self.coefficients = np.linalg.pinv(lhs) @ rhs

        expected = design @ self.coefficients
        residuals = np.linalg.norm(full - expected, axis=1)
        self.training_residuals = residuals.astype(np.float32)
        self.residual_threshold = float(np.percentile(residuals, self.config.residual_percentile))

        if len(contexts) > 1:
            shifts = np.linalg.norm(np.diff(contexts, axis=0), axis=1)
            self.context_shift_threshold = float(np.percentile(shifts, self.config.shift_percentile))
        else:
            self.context_shift_threshold = 0.0
        return self

    def predict_full(self, context_embedding: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("Counterfactual substitutor has not been fitted.")
        context = self._as_2d(context_embedding).astype(np.float64)
        predicted = self._with_intercept(context) @ self.coefficients
        return predicted.astype(np.float32)

    def score(
        self,
        full_embedding: np.ndarray,
        observed_context_embedding: np.ndarray,
        intervened_context_embedding: Optional[np.ndarray] = None,
    ) -> CounterfactualContextResult:
        if self.residual_threshold is None or self.context_shift_threshold is None:
            raise RuntimeError("Counterfactual substitutor has not been fitted.")

        full = np.asarray(full_embedding, dtype=np.float32).reshape(-1)
        observed_context = np.asarray(observed_context_embedding, dtype=np.float32).reshape(-1)
        healthy_context = observed_context if intervened_context_embedding is None else np.asarray(intervened_context_embedding, dtype=np.float32).reshape(-1)

        expected_observed = self.predict_full(observed_context)[0]
        expected_counterfactual = self.predict_full(healthy_context)[0]
        observed_residual = float(np.linalg.norm(full - expected_observed))
        counterfactual_residual = float(np.linalg.norm(full - expected_counterfactual))
        context_shift = float(np.linalg.norm(observed_context - healthy_context))

        residual_confidence = self._confidence_over_threshold(observed_residual, self.residual_threshold)
        shift_confidence = self._confidence_over_threshold(context_shift, self.context_shift_threshold)
        intervention_confidence = max(residual_confidence, shift_confidence)
        if intervention_confidence < self.config.min_intervention_confidence:
            intervention_confidence = 0.0

        final_score = (1.0 - intervention_confidence) * observed_residual
        final_score += intervention_confidence * (self.config.score_boost * counterfactual_residual)

        return CounterfactualContextResult(
            observed_residual=observed_residual,
            counterfactual_residual=counterfactual_residual,
            context_shift=context_shift,
            intervention_confidence=float(intervention_confidence),
            final_score=float(final_score),
            expected_full_embedding=expected_observed.astype(np.float32),
            counterfactual_full_embedding=expected_counterfactual.astype(np.float32),
        )

    def batch_score(
        self,
        full_embeddings: np.ndarray,
        observed_context_embeddings: np.ndarray,
        intervened_context_embeddings: Optional[np.ndarray] = None,
    ) -> list[CounterfactualContextResult]:
        full = self._as_2d(full_embeddings)
        observed = self._as_2d(observed_context_embeddings)
        if len(full) != len(observed):
            raise ValueError("full_embeddings and observed_context_embeddings must have the same length.")
        if intervened_context_embeddings is None:
            intervened = [None] * len(full)
        else:
            intervened = list(self._as_2d(intervened_context_embeddings))
            if len(intervened) != len(full):
                raise ValueError("intervened_context_embeddings must match the batch length.")
        return [self.score(full_item, observed_item, intervened_item) for full_item, observed_item, intervened_item in zip(full, observed, intervened)]

    @staticmethod
    def _with_intercept(values: np.ndarray) -> np.ndarray:
        return np.concatenate([np.ones((len(values), 1), dtype=values.dtype), values], axis=1)

    @staticmethod
    def _as_2d(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim == 1:
            return array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError("Expected embeddings with shape (n, dim) or (dim,).")
        return array

    @staticmethod
    def _confidence_over_threshold(value: float, threshold: float) -> float:
        if threshold <= 1e-8:
            return 1.0 if value > threshold else 0.0
        return float(np.clip((value - threshold) / threshold, 0.0, 1.0))