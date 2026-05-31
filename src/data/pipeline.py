"""Feature engineering and windowing pipeline for NCAD-CS.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional
from src.features.features import FeatureConfig, NCADFeatureExtractor
from src.data.data_loader import DataLoader


class NCADPipeline:
    """End-to-end data preparation pipeline wrapping feature extraction and windowing."""

    def __init__(self, max_features_per_channel: int = 64):
        self.max_features_per_channel = max_features_per_channel
        self.extractors: List[NCADFeatureExtractor] = []
        self.signal_indices: Optional[List[int]] = None

    def fit_prepare_train(
        self,
        train_data: np.ndarray,
        window_size: int,
        step: int = 1,
        signal_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Fits the feature extractors on the training data and returns sliding windows."""
        if train_data.ndim == 1:
            self.signal_indices = None
            extractor = NCADFeatureExtractor(FeatureConfig(max_features=self.max_features_per_channel))
            features = extractor.fit_transform(train_data)
            self.extractors = [extractor]
        else:
            if signal_indices is None:
                signal_indices = list(range(train_data.shape[1]))
            self.signal_indices = signal_indices
            self.extractors = []
            features_list = []
            for idx in self.signal_indices:
                extractor = NCADFeatureExtractor(FeatureConfig(max_features=self.max_features_per_channel))
                feat = extractor.fit_transform(train_data[:, idx])
                self.extractors.append(extractor)
                features_list.append(feat)
            features = np.concatenate(features_list, axis=1)

        return DataLoader.create_windows(features, window_size, step)

    def prepare_windows(
        self,
        test_data: np.ndarray,
        window_size: int,
        step: int = 1,
        signal_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Transforms the test data using the already fitted extractors and returns sliding windows."""
        if not self.extractors:
            raise RuntimeError("Pipeline must be fitted using fit_prepare_train first.")

        if test_data.ndim == 1:
            features = self.extractors[0].transform(test_data)
        else:
            indices = signal_indices if signal_indices is not None else self.signal_indices
            if indices is None:
                indices = list(range(test_data.shape[1]))
            if len(indices) != len(self.extractors):
                raise ValueError("Number of signal indices does not match fitted extractors.")
            
            features_list = []
            for i, idx in enumerate(indices):
                feat = self.extractors[i].transform(test_data[:, idx])
                features_list.append(feat)
            features = np.concatenate(features_list, axis=1)

        return DataLoader.create_windows(features, window_size, step)
