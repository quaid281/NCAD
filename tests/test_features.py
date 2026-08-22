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

