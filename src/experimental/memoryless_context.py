"""Prototype-free contamination detection and context denoising.

This module answers the memory-bank question directly: the contaminate-vs-normal
idea can be implemented without storing a bank of reference contexts. A compact
neural head learns two functions from synthetic contamination pairs:

1. classify whether a context embedding is contaminated;
2. project a contaminated context embedding back to the normal context manifold.

At inference, only learned weights and scalar calibration thresholds are used.
No per-sample memory bank, prototypes, or nearest-neighbor lookup are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MemorylessContextConfig:
    hidden_dim: int = 64
    dropout: float = 0.10
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    denoise_weight: float = 1.0
    threshold_percentile: float = 99.0
    score_boost: float = 2.5
    min_contamination_confidence: float = 0.10
    synthetic_noise_scale: float = 0.75
    synthetic_shift_scale: float = 1.50
    seed: int = 42
    device: str = "auto"


@dataclass
class MemorylessContextResult:
    original_score: float
    denoised_score: float
    reconstruction_error: float
    contamination_probability: float
    contamination_confidence: float
    final_score: float
    denoised_context_embedding: np.ndarray


class ContaminationDenoisingHead(nn.Module):
    """Small MLP with a contamination logit and a denoised embedding head."""

    def __init__(self, latent_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.delta_head = nn.Linear(hidden_dim, latent_dim)
        self.logit_head = nn.Linear(hidden_dim, 1)

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(embeddings)
        denoised = embeddings + self.delta_head(features)
        logits = self.logit_head(features).squeeze(-1)
        return denoised, logits


class MemorylessContextSubstitutor:
    """Contaminate-vs-normal substitution without a memory bank."""

    def __init__(self, config: Optional[MemorylessContextConfig] = None):
        self.config = config or MemorylessContextConfig()
        self.model: Optional[ContaminationDenoisingHead] = None
        self.latent_dim: Optional[int] = None
        self.reconstruction_threshold: Optional[float] = None
        self.probability_threshold: Optional[float] = None
        self.training_history: dict[str, list[float]] = {"loss": [], "classification_loss": [], "denoise_loss": []}

    def fit(
        self,
        clean_context_embeddings: np.ndarray,
        contaminated_context_embeddings: Optional[np.ndarray] = None,
        target_clean_embeddings: Optional[np.ndarray] = None,
    ) -> "MemorylessContextSubstitutor":
        clean = self._as_2d(clean_context_embeddings).astype(np.float32)
        if len(clean) == 0:
            raise ValueError("Cannot fit memoryless substitutor from zero embeddings.")

        if contaminated_context_embeddings is None:
            contaminated, contaminated_targets = self._synthesize_contaminated_pairs(clean)
        else:
            contaminated = self._as_2d(contaminated_context_embeddings).astype(np.float32)
            if target_clean_embeddings is None:
                if len(contaminated) != len(clean):
                    raise ValueError("target_clean_embeddings is required when contaminated samples are not paired with clean samples.")
                contaminated_targets = clean.copy()
            else:
                contaminated_targets = self._as_2d(target_clean_embeddings).astype(np.float32)
                if len(contaminated_targets) != len(contaminated):
                    raise ValueError("target_clean_embeddings must match contaminated_context_embeddings length.")

        self.latent_dim = clean.shape[1]
        device = self._resolve_device(self.config.device)
        torch.manual_seed(self.config.seed)
        self.model = ContaminationDenoisingHead(self.latent_dim, self.config.hidden_dim, self.config.dropout).to(device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        inputs = np.concatenate([clean, contaminated], axis=0)
        labels = np.concatenate([np.zeros(len(clean), dtype=np.float32), np.ones(len(contaminated), dtype=np.float32)])
        targets = np.concatenate([clean, contaminated_targets], axis=0)
        rng = np.random.default_rng(self.config.seed)

        self.model.train()
        for _ in range(self.config.epochs):
            order = rng.permutation(len(inputs))
            epoch_loss = 0.0
            epoch_cls = 0.0
            epoch_denoise = 0.0
            seen = 0
            for start in range(0, len(order), self.config.batch_size):
                indices = order[start : start + self.config.batch_size]
                batch_inputs = torch.from_numpy(inputs[indices]).to(device)
                batch_labels = torch.from_numpy(labels[indices]).to(device)
                batch_targets = torch.from_numpy(targets[indices]).to(device)

                optimizer.zero_grad(set_to_none=True)
                denoised, logits = self.model(batch_inputs)
                classification_loss = F.binary_cross_entropy_with_logits(logits, batch_labels)
                denoise_loss = F.mse_loss(denoised, batch_targets)
                loss = classification_loss + self.config.denoise_weight * denoise_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                batch_size = len(indices)
                epoch_loss += float(loss.item()) * batch_size
                epoch_cls += float(classification_loss.item()) * batch_size
                epoch_denoise += float(denoise_loss.item()) * batch_size
                seen += batch_size

            self.training_history["loss"].append(epoch_loss / max(seen, 1))
            self.training_history["classification_loss"].append(epoch_cls / max(seen, 1))
            self.training_history["denoise_loss"].append(epoch_denoise / max(seen, 1))

        self._calibrate(clean, device)
        return self

    def score(self, full_embedding: np.ndarray, context_embedding: np.ndarray) -> MemorylessContextResult:
        if self.model is None or self.reconstruction_threshold is None or self.probability_threshold is None:
            raise RuntimeError("Memoryless context substitutor has not been fitted.")

        full = np.asarray(full_embedding, dtype=np.float32).reshape(-1)
        context = np.asarray(context_embedding, dtype=np.float32).reshape(-1)
        if len(full) != len(context):
            raise ValueError("full_embedding and context_embedding must have the same latent dimension.")

        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(context.reshape(1, -1)).to(device)
            denoised, logits = self.model(tensor)
            denoised_np = denoised.cpu().numpy()[0].astype(np.float32)
            probability = float(torch.sigmoid(logits).cpu().numpy()[0])

        reconstruction_error = float(np.linalg.norm(denoised_np - context))
        reconstruction_confidence = self._confidence_over_threshold(reconstruction_error, self.reconstruction_threshold)
        probability_confidence = self._confidence_over_threshold(probability, self.probability_threshold)
        contamination_confidence = max(probability, probability_confidence, reconstruction_confidence)
        if contamination_confidence < self.config.min_contamination_confidence:
            contamination_confidence = 0.0

        original_score = float(np.linalg.norm(full - context))
        denoised_score = float(np.linalg.norm(full - denoised_np))
        final_score = (1.0 - contamination_confidence) * original_score
        final_score += contamination_confidence * (self.config.score_boost * denoised_score)

        return MemorylessContextResult(
            original_score=original_score,
            denoised_score=denoised_score,
            reconstruction_error=reconstruction_error,
            contamination_probability=probability,
            contamination_confidence=float(contamination_confidence),
            final_score=float(final_score),
            denoised_context_embedding=denoised_np,
        )

    def transform_contexts(self, context_embeddings: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Memoryless context substitutor has not been fitted.")
        contexts = self._as_2d(context_embeddings).astype(np.float32)
        device = next(self.model.parameters()).device
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(contexts), self.config.batch_size):
                batch = torch.from_numpy(contexts[start : start + self.config.batch_size]).to(device)
                denoised, _ = self.model(batch)
                outputs.append(denoised.cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)

    def _calibrate(self, clean: np.ndarray, device: torch.device) -> None:
        if self.model is None:
            raise RuntimeError("Memoryless context substitutor has not been fitted.")
        self.model.eval()
        errors = []
        probabilities = []
        with torch.no_grad():
            for start in range(0, len(clean), self.config.batch_size):
                batch_np = clean[start : start + self.config.batch_size]
                batch = torch.from_numpy(batch_np).to(device)
                denoised, logits = self.model(batch)
                errors.extend(torch.linalg.norm(denoised - batch, dim=1).cpu().numpy().tolist())
                probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        self.reconstruction_threshold = float(np.percentile(np.asarray(errors), self.config.threshold_percentile))
        self.probability_threshold = float(np.percentile(np.asarray(probabilities), self.config.threshold_percentile))

    def _synthesize_contaminated_pairs(self, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.config.seed)
        noise = rng.normal(0.0, self.config.synthetic_noise_scale, size=clean.shape).astype(np.float32)
        shift_direction = rng.normal(0.0, 1.0, size=clean.shape).astype(np.float32)
        shift_norm = np.linalg.norm(shift_direction, axis=1, keepdims=True)
        shift_direction = shift_direction / np.clip(shift_norm, 1e-8, None)
        contaminated = clean + noise + self.config.synthetic_shift_scale * shift_direction
        return contaminated.astype(np.float32), clean.copy()

    @staticmethod
    def _as_2d(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim == 1:
            return array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError("Expected embeddings with shape (n, dim) or (dim,).")
        return array

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        if device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_name)

    @staticmethod
    def _confidence_over_threshold(value: float, threshold: float) -> float:
        if threshold <= 1e-8:
            return 1.0 if value > threshold else 0.0
        return float(np.clip((value - threshold) / threshold, 0.0, 1.0))
