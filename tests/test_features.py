import pytest
import numpy as np

from src.features.features import FeatureConfig, NCADFeatureExtractor


def test_feature_extractor():
    config = FeatureConfig(max_features=8)
    extractor = NCADFeatureExtractor(config)
    
    # 1D time series signal
    signal = np.sin(np.linspace(0, 100, 1000)).astype(np.float32)
    
    # Fit extractor
    extractor.fit(signal)
    
    # Transform signal
    features = extractor.transform(signal)
    
    # Expect output shape to be (N, max_features)
    assert features.shape == (1000, 8)
    assert not np.any(np.isnan(features))


def test_feature_extractor_delay_embedding():
    config = FeatureConfig(
        max_features=16,
        use_delay_embedding=True,
        delay_embedding_dim=4,
        delay_lag=5,
    )
    extractor = NCADFeatureExtractor(config)
    signal = np.cos(np.linspace(0, 50, 500)).astype(np.float32)

    features = extractor.fit_transform(signal)

    assert features.shape == (500, 16)
    assert not np.any(np.isnan(features))
    assert any("delay_coord" in name for name in extractor.feature_names_)


def test_suffix_perturbation_causality():
    """Verify that altering future suffix observations has ZERO effect on past prefix features."""
    np.random.seed(42)
    n_points = 200
    split_idx = 100
    x_base = np.random.randn(n_points).astype(np.float32)

    # Perturb only the future suffix [split_idx:]
    x_perturbed = x_base.copy()
    x_perturbed[split_idx:] += 50.0 + 10.0 * np.random.randn(n_points - split_idx).astype(np.float32)

    fe1 = NCADFeatureExtractor(FeatureConfig())
    fe2 = NCADFeatureExtractor(FeatureConfig())

    # Fit on unperturbed and transform both
    fe1.fit(x_base)
    f_base = fe1.transform(x_base)
    f_perturbed = fe1.transform(x_perturbed)

    # Prefix features [0 : split_idx] must be IDENTICAL (exact zero future leakage)
    prefix_diff = np.abs(f_base[:split_idx] - f_perturbed[:split_idx])
    assert np.max(prefix_diff) < 1e-5, f"Future leakage detected! Max prefix diff: {np.max(prefix_diff)}"


def test_pre_normalization_feature_selection():
    """Verify feature selection ranks by pre-normalization variance and filters constants."""
    np.random.seed(42)
    n_points = 200
    sig = np.random.randn(n_points) * 10.0
    fe = NCADFeatureExtractor(FeatureConfig(max_features=10))
    transformed = fe.fit_transform(sig)

    assert transformed.shape == (n_points, 10)
    assert fe.selected_indices_ is not None
    assert 0 in fe.selected_indices_
    for idx in fe.selected_indices_:
        assert np.var(transformed[:, fe.selected_indices_ == idx]) > 0.0


