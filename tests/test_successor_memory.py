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


def test_successor_memory_latent_space():
    n_samples = 100
    latent_dim = 16

    context_embeddings = np.random.randn(n_samples, latent_dim).astype(np.float32)
    # Successor representations: 2D array of shape (n_samples, latent_dim)
    latent_successors = np.random.randn(n_samples, latent_dim).astype(np.float32)

    config = SuccessorMemoryConfig(
        n_neighbors=3,
        max_memory_windows=50,
        context_percentile=95.0,
        seed=42,
    )
    memory = CounterfactualSuccessorMemory(config)
    memory.fit(context_embeddings, latent_successors)

    assert memory.successor_windows.shape == (50, latent_dim)

    query_embeddings = np.random.randn(5, latent_dim).astype(np.float32)
    observed_latent_successors = np.random.randn(5, latent_dim).astype(np.float32)

    res = memory.query(query_embeddings, observed_latent_successors)

    assert res.context_distances.shape == (5,)
    assert res.successor_scores.shape == (5,)
    assert res.expected_successors.shape == (5, latent_dim)
    assert not np.any(np.isnan(res.successor_scores))


def test_robust_pca_decomposition():
    from src.models.successor_memory import robust_pca

    # Create low-rank matrix + sparse outliers
    rng = np.random.default_rng(42)
    u = rng.standard_normal((30, 2))
    v = rng.standard_normal((2, 10))
    low_rank = u @ v  # rank-2 matrix
    sparse = np.zeros((30, 10))
    sparse[5, 2] = 10.0
    sparse[15, 7] = -12.0
    X = low_rank + sparse

    L, S = robust_pca(X)

    assert L.shape == (30, 10)
    assert S.shape == (30, 10)
    assert not np.any(np.isnan(L))
    assert not np.any(np.isnan(S))
    # Verify sparse outlier recovery
    assert np.abs(S[5, 2]) > 1.0
    assert np.abs(S[15, 7]) > 1.0


def test_successor_memory_rpca_sanitization():
    n_samples = 50
    latent_dim = 8

    rng = np.random.default_rng(42)
    context_embeddings = rng.standard_normal((n_samples, latent_dim)).astype(np.float32)
    successors = rng.standard_normal((n_samples, latent_dim)).astype(np.float32)
    # Add a large sparse outlier spike to training successors
    successors[10, 3] += 50.0

    config = SuccessorMemoryConfig(
        n_neighbors=3,
        max_memory_windows=50,
        context_percentile=95.0,
        seed=42,
        use_rpca_sanitization=True,
    )
    memory = CounterfactualSuccessorMemory(config)
    memory.fit(context_embeddings, successors)

    assert memory.sparse_outliers is not None
    assert memory.sparse_outliers.shape == (50, latent_dim)
    assert not np.any(np.isnan(memory.successor_windows))


