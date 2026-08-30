"""Tests for isolated novelty research modules in src/experimental."""

import numpy as np
import pytest
import torch

from src.experimental.causal_counterfactual import (
    CounterfactualConfig,
    CounterfactualContextResult,
    CounterfactualContextSubstitutor,
)
from src.experimental.hopfield_context import (
    HopfieldContextConfig,
    HopfieldContextMemory,
    HopfieldRetrieval,
)
from src.experimental.memoryless_context import (
    MemorylessContextConfig,
    MemorylessContextResult,
    MemorylessContextSubstitutor,
)
from src.experimental.selective_ssm_encoder import (
    ExperimentalSSMContextEncoder,
    SelectiveSSMContextEncoder,
    SelectiveStateSpaceBlock,
)


def test_hopfield_context_memory():
    rng = np.random.default_rng(42)
    normal_context = rng.normal(0.0, 1.0, size=(64, 16)).astype(np.float32)
    memory = HopfieldContextMemory(HopfieldContextConfig(inverse_temperature=10.0)).fit(normal_context)
    
    query = normal_context[0] + rng.normal(0.0, 0.1, size=(16,)).astype(np.float32)
    result = memory.retrieve(query)
    
    assert isinstance(result, HopfieldRetrieval)
    assert result.retrieved_embedding.shape == (16,)
    assert result.distance >= 0.0
    assert 0.0 <= result.confidence <= 1.0


def test_causal_counterfactual_substitutor():
    rng = np.random.default_rng(42)
    contexts = rng.normal(0.0, 1.0, size=(64, 16)).astype(np.float32)
    fulls = contexts * 1.5 + rng.normal(0.0, 0.05, size=contexts.shape).astype(np.float32)
    
    substitutor = CounterfactualContextSubstitutor(CounterfactualConfig(ridge_alpha=1e-3)).fit(contexts, fulls)
    
    result = substitutor.score(
        full_embedding=fulls[0],
        observed_context_embedding=contexts[0] + 0.5,
        intervened_context_embedding=contexts[0],
    )
    
    assert isinstance(result, CounterfactualContextResult)
    assert result.final_score >= 0.0
    assert result.expected_full_embedding.shape == (16,)


def test_memoryless_context_substitutor():
    rng = np.random.default_rng(42)
    normal_ctx = rng.normal(0.0, 1.0, size=(32, 16)).astype(np.float32)
    contam_ctx = normal_ctx + rng.normal(0.0, 0.5, size=normal_ctx.shape).astype(np.float32)
    full_emb = normal_ctx + rng.normal(0.0, 0.05, size=normal_ctx.shape).astype(np.float32)
    
    config = MemorylessContextConfig(epochs=3, hidden_dim=16, batch_size=16, device="cpu")
    substitutor = MemorylessContextSubstitutor(config).fit(normal_ctx, contam_ctx, full_emb)
    
    result = substitutor.score(full_emb[0], contam_ctx[0])
    assert isinstance(result, MemorylessContextResult)
    assert 0.0 <= result.contamination_probability <= 1.0


def test_experimental_selective_ssm_encoder():
    encoder = ExperimentalSSMContextEncoder(input_dim=4, latent_dim=16, hidden_dim=32, layers=2)
    x = torch.randn(4, 32, 4)
    latent = encoder(x)
    
    assert latent.shape == (4, 16)
    assert torch.isfinite(latent).all()

    # Verify backward-compatibility alias
    alias_encoder = SelectiveSSMContextEncoder(input_dim=4, latent_dim=16, hidden_dim=32, layers=2)
    assert isinstance(alias_encoder, ExperimentalSSMContextEncoder)

