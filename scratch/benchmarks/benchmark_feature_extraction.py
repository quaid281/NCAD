import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
import numpy as np
from src.features.features import FeatureConfig, NCADFeatureExtractor

def test_feature_perf():
    np.random.seed(42)
    raw_signal = np.sin(np.linspace(0, 100, 20000)) + 0.1 * np.random.randn(20000)
    extractor = NCADFeatureExtractor(FeatureConfig())
    
    t0 = time.time()
    features = extractor.fit_transform(raw_signal)
    t1 = time.time()
    
    print(f"Current extraction on 20,000 points took: {t1 - t0:.4f}s")
    print(f"Shape: {features.shape}, finite: {np.all(np.isfinite(features))}")

if __name__ == "__main__":
    test_feature_perf()
