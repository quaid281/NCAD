import pytest
import numpy as np

from src.models.successor_memory import CounterfactualSuccessorMemory, SuccessorMemoryConfig


def test_successor_memory():
    n_samples = 100
    latent_dim = 8
    successor_len = 16
    n_features = 2
    
    # Context embeddings
    context_embeddings = np.random.randn(n_samples, latent_dim).astype(np.float32)
    # Successor windows: shape (n_samples, successor_len, n_features)
    successors = np.random.randn(n_samples, successor_len, n_features).astype(np.float32)
    
    config = SuccessorMemoryConfig(
        n_neighbors=3,
        max_memory_windows=50,
        context_percentile=95.0,
        seed=42
    )
    
    memory = CounterfactualSuccessorMemory(config)
    
    # Fit memory
    memory.fit(context_embeddings, successors)
    
    assert memory.context_embeddings.shape[0] == 50  # capped by max_memory_windows
    assert memory.successor_windows.shape[0] == 50
    assert memory.context_threshold > 0.0
    
    # Query memory
    query_embeddings = np.random.randn(5, latent_dim).astype(np.float32)
    observed_successors = np.random.randn(5, successor_len, n_features).astype(np.float32)
    
    res = memory.query(query_embeddings, observed_successors)
    
    assert res.context_distances.shape == (5,)
    assert res.successor_scores.shape == (5,)
    assert res.neighbor_indices.shape == (5, 3)
    assert not np.any(np.isnan(res.successor_scores))
