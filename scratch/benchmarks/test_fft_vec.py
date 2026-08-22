import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def fft_features_legacy(values: np.ndarray, fft_window: int = 64):
    n_points = len(values)
    window = min(fft_window, n_points)
    dominant_freq = np.zeros(n_points)
    spectral_entropy = np.zeros(n_points)
    spectral_power = np.zeros(n_points)

    for index in range(n_points):
        start = max(0, index - window + 1)
        segment = values[start : index + 1]
        if len(segment) < 8:
            continue
        fft_values = np.fft.rfft(segment - np.mean(segment))
        power = np.abs(fft_values) ** 2
        spectral_power[index] = float(np.sum(power))
        if len(power) > 1:
            dominant_freq[index] = float(np.argmax(power[1:]) + 1)
        total_power = np.sum(power)
        if total_power > 1e-12:
            normalized_power = power / total_power
            spectral_entropy[index] = -float(np.sum(normalized_power * np.log2(normalized_power + 1e-12)))

    return dominant_freq, spectral_entropy, spectral_power

def fft_features_vectorized(values: np.ndarray, fft_window: int = 64):
    n_points = len(values)
    window = min(fft_window, n_points)
    dominant_freq = np.zeros(n_points, dtype=np.float64)
    spectral_entropy = np.zeros(n_points, dtype=np.float64)
    spectral_power = np.zeros(n_points, dtype=np.float64)

    if n_points < 8:
        return dominant_freq, spectral_entropy, spectral_power

    # 1. Warmup for index in range(7, window - 1)
    for index in range(7, min(window - 1, n_points)):
        segment = values[: index + 1]
        fft_values = np.fft.rfft(segment - np.mean(segment))
        power = np.abs(fft_values) ** 2
        spectral_power[index] = float(np.sum(power))
        if len(power) > 1:
            dominant_freq[index] = float(np.argmax(power[1:]) + 1)
        total_power = np.sum(power)
        if total_power > 1e-12:
            normalized_power = power / total_power
            spectral_entropy[index] = -float(np.sum(normalized_power * np.log2(normalized_power + 1e-12)))

    # 2. Vectorized batch FFT for full windows index in range(window - 1, n_points)
    if n_points >= window:
        # shape: (n_points - window + 1, window)
        strided_windows = sliding_window_view(values, window_shape=window)
        centered = strided_windows - np.mean(strided_windows, axis=1, keepdims=True)
        fft_vals = np.fft.rfft(centered, axis=1)
        power = np.abs(fft_vals) ** 2  # (M, n_freqs)
        
        tot_power = np.sum(power, axis=1)
        spectral_power[window - 1 :] = tot_power
        
        if power.shape[1] > 1:
            dominant_freq[window - 1 :] = np.argmax(power[:, 1:], axis=1) + 1
            
        eps = 1e-12
        norm_power = np.divide(power, tot_power[:, None] + eps, out=np.zeros_like(power), where=tot_power[:, None] > eps)
        log_term = np.log2(norm_power + eps)
        spectral_entropy[window - 1 :] = -np.sum(np.where(norm_power > 0, norm_power * log_term, 0.0), axis=1)

    return dominant_freq, spectral_entropy, spectral_power

np.random.seed(42)
test_vals = np.random.randn(5000)
df_leg, se_leg, sp_leg = fft_features_legacy(test_vals, 64)
df_vec, se_vec, sp_vec = fft_features_vectorized(test_vals, 64)

print(f"Max diff dominant freq: {np.max(np.abs(df_leg - df_vec)):.2e}")
print(f"Max diff spectral entropy: {np.max(np.abs(se_leg - se_vec)):.2e}")
print(f"Max diff spectral power: {np.max(np.abs(sp_leg - sp_vec)):.2e}")

assert np.allclose(df_leg, df_vec)
assert np.allclose(se_leg, se_vec, atol=1e-6)
assert np.allclose(sp_leg, sp_vec, atol=1e-6)
print("FFT vectorization verified successfully!")
