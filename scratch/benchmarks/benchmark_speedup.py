import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
import numpy as np
from scratch.benchmarks.test_slope_vec import rolling_slope_legacy, rolling_slope_vectorized
from scratch.benchmarks.test_fft_vec import fft_features_legacy, fft_features_vectorized

np.random.seed(42)
test_vals = np.random.randn(30000)

t0 = time.time()
rolling_slope_legacy(test_vals, 30)
t_slope_leg = time.time() - t0

t0 = time.time()
rolling_slope_vectorized(test_vals, 30)
t_slope_vec = time.time() - t0

t0 = time.time()
fft_features_legacy(test_vals, 64)
t_fft_leg = time.time() - t0

t0 = time.time()
fft_features_vectorized(test_vals, 64)
t_fft_vec = time.time() - t0

print(f"Rolling slope: {t_slope_leg:.4f}s -> {t_slope_vec:.4f}s ({t_slope_leg / t_slope_vec:.1f}x speedup)")
print(f"FFT features:  {t_fft_leg:.4f}s -> {t_fft_vec:.4f}s ({t_fft_leg / t_fft_vec:.1f}x speedup)")
