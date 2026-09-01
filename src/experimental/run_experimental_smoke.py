"""Smoke test for isolated NCAD-CS novelty experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experimental.causal_counterfactual import CounterfactualContextSubstitutor
from src.experimental.hopfield_context import HopfieldContextMemory
from src.experimental.mamba_ssm_encoder import ExperimentalSSMContextEncoder
from src.experimental.memoryless_context import MemorylessContextConfig, MemorylessContextSubstitutor


def main() -> None:
    rng = np.random.default_rng(42)
    normal_context = rng.normal(0.0, 1.0, size=(128, 16)).astype(np.float32)
    full_embeddings = normal_context + rng.normal(0.0, 0.05, size=normal_context.shape).astype(np.float32)
    contaminated_context = normal_context + rng.normal(0.0, 0.65, size=normal_context.shape).astype(np.float32)

    hopfield = HopfieldContextMemory().fit(normal_context)
    hopfield_result = hopfield.retrieve(contaminated_context[0])

    causal = CounterfactualContextSubstitutor().fit(normal_context, full_embeddings)
    causal_result = causal.score(
        full_embedding=full_embeddings[0],
        observed_context_embedding=contaminated_context[0],
        intervened_context_embedding=hopfield_result.retrieved_embedding,
    )

    memoryless = MemorylessContextSubstitutor(
        MemorylessContextConfig(epochs=8, hidden_dim=32, batch_size=64, device="cpu")
    ).fit(normal_context, contaminated_context, normal_context)
    memoryless_result = memoryless.score(full_embeddings[0], contaminated_context[0])

    ssm = ExperimentalSSMContextEncoder(input_dim=4, latent_dim=16, hidden_dim=32, layers=2)
    latent = ssm(torch.randn(4, 32, 4))

    summary = {
        "hopfield_distance": round(hopfield_result.distance, 6),
        "hopfield_confidence": round(hopfield_result.confidence, 6),
        "causal_final_score": round(causal_result.final_score, 6),
        "causal_intervention_confidence": round(causal_result.intervention_confidence, 6),
        "memoryless_final_score": round(memoryless_result.final_score, 6),
        "memoryless_contamination_probability": round(memoryless_result.contamination_probability, 6),
        "ssm_latent_shape": list(latent.shape),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
