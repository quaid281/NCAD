import numpy as np

def rolling_slope_legacy(values: np.ndarray, window: int) -> np.ndarray:
    slopes = np.zeros(len(values), dtype=np.float64)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        y_values = values[start : index + 1]
        if len(y_values) < 2:
            continue
        x_values = np.arange(len(y_values), dtype=np.float64)
        x_centered = x_values - np.mean(x_values)
        denominator = np.sum(x_centered * x_centered)
        if denominator > 1e-12:
            slopes[index] = np.sum(x_centered * (y_values - np.mean(y_values))) / denominator
    return slopes

def rolling_slope_vectorized(values: np.ndarray, window: int) -> np.ndarray:
    n_points = len(values)
    slopes = np.zeros(n_points, dtype=np.float64)
    if n_points < 2 or window < 2:
        return slopes

    # Full window kernel
    w = min(window, n_points)
    x = np.arange(w, dtype=np.float64)
    x_centered = x - np.mean(x)
    denom = np.sum(x_centered ** 2)
    
    if denom > 1e-12:
        kernel = x_centered / denom  # shape (w,)
        # Causal linear convolution: y[t - w + 1 : t + 1] * kernel
        # np.convolve(values, kernel[::-1], mode='full')
        conv = np.convolve(values, kernel[::-1], mode="full")[:n_points]
        slopes[w - 1:] = conv[w - 1:]

    # Handle boundary warmup points (t < w - 1)
    for index in range(1, min(w - 1, n_points)):
        y_values = values[: index + 1]
        x_val = np.arange(len(y_values), dtype=np.float64)
        x_c = x_val - np.mean(x_val)
        d = np.sum(x_c ** 2)
        if d > 1e-12:
            slopes[index] = np.sum(x_c * y_values) / d

    return slopes

np.random.seed(42)
test_vals = np.random.randn(5000)
s_leg = rolling_slope_legacy(test_vals, 30)
s_vec = rolling_slope_vectorized(test_vals, 30)
diff = np.max(np.abs(s_leg - s_vec))
print(f"Max absolute difference in rolling slope: {diff:.2e}")
assert diff < 1e-10, "Rolling slope mismatch!"
print("Rolling slope vectorization verified successfully!")
