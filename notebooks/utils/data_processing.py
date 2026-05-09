import numpy as np
from scipy import stats
import pywt
import pandas as pd

class TemporalFeatureExtractor:
    """Extract rich temporal features from time series data."""
    
    def __init__(self, window_sizes=[10, 30, 60], fft_components=5, long_window_size=150):
        """
        Initialize the feature extractor.
        
        Args:
            window_sizes: List of window sizes for rolling statistics
            fft_components: Number of frequency components to extract
        """
        self.window_sizes = window_sizes
        self.fft_components = fft_components
        self.long_window_size = long_window_size # <<< Store the new parameter
    
    def extract_features(self, time_series):
        """
        Extract comprehensive temporal features from a time series.

        Args:
            time_series: 1D numpy array containing the telemetry signal

        Returns:
            2D numpy array with shape (len(time_series), n_features) or None if error.
        """
        n = len(time_series)
        if n == 0:
            print("Warning: Input time series is empty.")
            return None

        # Original time series as the first feature
        features = [time_series]

        # --- Call helper methods to generate features ---
        # 1. Rolling statistics
        try: features.extend(self._compute_rolling_stats(time_series, n))
        except Exception as e: print(f"Error in _compute_rolling_stats: {e}")

        # 2. Frequency domain features
        try: features.extend(self._compute_frequency_features(time_series, n))
        except Exception as e: print(f"Error in _compute_frequency_features: {e}")

        # 3. Trend indicators
        try: features.extend(self._compute_trend_features(time_series, n))
        except Exception as e: print(f"Error in _compute_trend_features: {e}")

        # 4. Complexity and entropy features
        try: features.extend(self._compute_complexity_features(time_series, n))
        except Exception as e: print(f"Error in _compute_complexity_features: {e}")

        # 5. Wavelet features
        try: features.extend(self._compute_wavelet_features(time_series, n))
        except Exception as e: print(f"Error in _compute_wavelet_features: {e}")

        # 6. << NEW >> Long-term stats
        try: features.extend(self._compute_long_term_stats(time_series, n)) # <<< Call new helper
        except Exception as e: print(f"Error in _compute_long_term_stats: {e}")

        try: features.extend(self._compute_enhanced_spectral_features(time_series, n))
        except Exception as e: print(f"Error in _compute_enhanced_spectral_features: {e}")
        # --- End helper method calls ---

        # --- Stack features into a 2D array ---
        # Filter out None or incorrectly lengthed arrays before stacking
        valid_features = []
        expected_len = n
        for i, f in enumerate(features):
             if f is not None and isinstance(f, np.ndarray) and len(f) == expected_len:
                 valid_features.append(f)
             else:
                  print(f"Warning: Feature at index {i} is invalid (None or wrong length: {len(f) if f is not None else 'None'}). Skipping.")

        if not valid_features:
             print("Error: No valid features generated.")
             return None

        try:
             result = np.column_stack(valid_features)
             print(f"Total Extracted Features: {result.shape[1]}") # Add print statement
             # Check for NaNs/Infs in the final stacked array
             if np.any(~np.isfinite(result)):
                  print("Warning: Non-finite values found in final feature matrix. Attempting nan_to_num.")
                  result = np.nan_to_num(result, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
             return result
        except ValueError as e:
             print(f"Critical Error stacking features: {e}")
             # Print lengths for debugging
             for i, f in enumerate(valid_features):
                  print(f"  Feature {i} length: {len(f)}, shape: {f.shape}, dtype: {f.dtype}")
             return None
        except Exception as e:
             print(f"Unexpected error during feature stacking: {e}")
             return None


    # Pass 'n' to helpers to avoid recalculating len() repeatedly
    def _compute_rolling_stats(self, time_series, n):
        """Compute rolling statistics for given window sizes."""
        all_rolling_features = []
        for window in self.window_sizes:
            # ... (keep existing rolling stats calculation logic for each window) ...
            # Ensure this function returns a list of numpy arrays
            rolling_mean = np.zeros(n)
            rolling_std = np.zeros(n)
            rolling_min = np.zeros(n)
            rolling_max = np.zeros(n)
            rolling_median = np.zeros(n)
            rolling_skew = np.zeros(n)
            rolling_kurtosis = np.zeros(n)

            # Compute for each position
            for i in range(n):
                start = max(0, i - window + 1)
                window_data = time_series[start:i+1]
                len_wd = len(window_data)

                if len_wd > 0:
                    rolling_mean[i] = np.mean(window_data)
                    std_val = np.std(window_data) # Calculate std once
                    rolling_std[i] = std_val
                    rolling_min[i] = np.min(window_data)
                    rolling_max[i] = np.max(window_data)
                    rolling_median[i] = np.median(window_data)
                    
                    # Explicitly handle zero variance for skew/kurtosis
                    if std_val > 1e-9 and len_wd > 2:
                        try:
                            rolling_skew[i] = stats.skew(window_data)
                            rolling_kurtosis[i] = stats.kurtosis(window_data)
                        except RuntimeWarning: # Catch warning just in case
                            rolling_skew[i] = 0.0
                            rolling_kurtosis[i] = 0.0
                    else:
                        # If variance is zero, skew is 0 and kurtosis is undefined (or -3, let's use 0)
                        rolling_skew[i] = 0.0
                        rolling_kurtosis[i] = 0.0
                # else: values remain 0

            diff_from_mean = time_series - rolling_mean
            diff_from_median = time_series - rolling_median
            # Add epsilon to rolling_std before division
            z_score = diff_from_mean / (rolling_std + 1e-10)
            range_val = rolling_max - rolling_min
            # Add epsilon to range_val before division
            range_relative = np.divide(time_series - rolling_min, range_val + 1e-10, out=np.zeros_like(time_series), where=range_val > 1e-10)

            all_rolling_features.extend([
                rolling_mean, rolling_std, rolling_min, rolling_max,
                rolling_median, rolling_skew, rolling_kurtosis,
                diff_from_mean, diff_from_median, z_score, range_relative
            ])
        return all_rolling_features

    # <<< NEW HELPER METHOD >>>
    def _compute_long_term_stats(self, time_series, n):
        """Compute long-term rolling statistics using pandas."""
        if n < 2: # Need at least 2 points for std dev
            print("Warning: Not enough data points for long-term stats.")
            return [np.zeros(n)] # Return array of zeros matching input length

        features = []
        ts_series = pd.Series(time_series)

        # Calculate long-term rolling std dev
        # Use min_periods=2 for std dev calculation
        long_rolling_std = ts_series.rolling(window=self.long_window_size, min_periods=2).std()

        # Convert back to numpy and handle NaNs (which occur for min_periods not met)
        long_rolling_std_np = np.nan_to_num(long_rolling_std.to_numpy(), nan=0.0) # Replace NaN with 0

        features.append(long_rolling_std_np)

        # Example: Add long-term rolling mean as well
        # long_rolling_mean = ts_series.rolling(window=self.long_window_size, min_periods=1).mean()
        # features.append(np.nan_to_num(long_rolling_mean.to_numpy(), nan=0.0))

        print(f"  Computed long-term stats (window={self.long_window_size})")
        return features

    # --- Other helper methods (_compute_frequency_features, etc.) ---
    # Pass 'n' to them as well if needed for pre-allocation
    def _compute_frequency_features(self, time_series, n):
        # ... (existing logic) ...
        # Ensure it returns a list of numpy arrays
        features = []
        fft_power = np.zeros((n, self.fft_components))
        fft_phase = np.zeros((n, self.fft_components))
        fft_window = min(64, n // 2 if n > 0 else 64) # Handle n=0

        for i in range(n):
             start = max(0, i - fft_window + 1)
             window_data = time_series[start:i+1]
             if len(window_data) >= 8:
                 # ... (FFT calculation) ...
                 fft_result = np.fft.rfft(window_data)
                 magnitudes = np.abs(fft_result)
                 phases = np.angle(fft_result)
                 if len(magnitudes) > 1:
                     sorted_indices = np.argsort(magnitudes[1:])[::-1]
                     for j in range(min(self.fft_components, len(sorted_indices))):
                         idx = sorted_indices[j] + 1
                         fft_power[i, j] = magnitudes[idx]
                         fft_phase[i, j] = phases[idx]

        dominant_freq = np.zeros(n)
        spectral_entropy = np.zeros(n)
        for i in range(n):
            start = max(0, i - fft_window + 1)
            window_data = time_series[start:i+1]
            if len(window_data) >= 8:
                # ... (power spec, dom freq, entropy calc) ...
                fft_result = np.fft.rfft(window_data)
                power = np.abs(fft_result)**2
                if len(power) > 1: dominant_freq[i] = np.argmax(power[1:]) + 1
                power_sum = np.sum(power)
                if power_sum > 1e-10:
                    normalized_power = power / power_sum
                    spectral_entropy[i] = -np.sum(normalized_power * np.log2(normalized_power + 1e-10))

        features.extend([fft_power[:, i] for i in range(self.fft_components)])
        features.extend([fft_phase[:, i] for i in range(self.fft_components)])
        features.extend([dominant_freq, spectral_entropy])
        return features

    def _compute_enhanced_spectral_features(self, time_series, n):
        """
        Extract enhanced spectral features using Short-Time Fourier Transform (STFT)
        for multimodal detection.
        
        Args:
            time_series: 1D numpy array of telemetry values
            n: Length of the time series
            
        Returns:
            List of numpy arrays with spectral features
        """
        import numpy as np
        from scipy import signal
        
        features = []
        
        # Parameters for spectrogram calculation
        window_sizes = [64, 128, 256]  # Multiple window sizes for multi-resolution analysis
        step_sizes = [16, 32, 64]     # Corresponding step sizes
        min_window_size = min(window_sizes)
        
        if n < min_window_size:
            print(f"Warning: Time series too short ({n} points) for spectral analysis with min window size {min_window_size}")
            # Return empty feature arrays of correct length
            return [np.zeros(n) for _ in range(5)]  # 5 placeholder features
        
        # Compute spectrograms at multiple resolutions and extract features
        for win_size, step_size in zip(window_sizes, step_sizes):
            if n >= win_size:
                # Initialize feature arrays
                spectral_energy = np.zeros(n)
                spectral_entropy = np.zeros(n)
                spectral_rolloff = np.zeros(n)
                spectral_flatness = np.zeros(n)
                
                # Calculate spectrogram with overlapping windows
                for i in range(0, n - win_size + 1, max(1, step_size)):
                    end_idx = i + win_size
                    segment = time_series[i:end_idx]
                    
                    # Apply window function to reduce spectral leakage
                    windowed_segment = segment * signal.windows.hann(len(segment))
                    
                    # Calculate FFT
                    segment_fft = np.fft.rfft(windowed_segment)
                    segment_power = np.abs(segment_fft)**2
                    
                    # Skip if segment has no power
                    if np.sum(segment_power) < 1e-10:
                        continue
                    
                    # Normalize power spectrum
                    normalized_power = segment_power / np.sum(segment_power)
                    
                    # Extract spectral features
                    # 1. Total energy
                    total_energy = np.sum(segment_power)
                    
                    # 2. Spectral entropy (complexity)
                    entropy = -np.sum(normalized_power * np.log2(normalized_power + 1e-10))
                    
                    # 3. Spectral rolloff (frequency below which X% of energy is contained)
                    cumsum = np.cumsum(segment_power)
                    rolloff_idx = np.argmax(cumsum >= 0.85 * cumsum[-1]) if len(cumsum) > 0 else 0
                    rolloff = rolloff_idx / len(segment_power) if len(segment_power) > 0 else 0
                    
                    # 4. Spectral flatness (geometric mean / arithmetic mean)
                    # High flatness = noise-like, low flatness = tonal
                    with np.errstate(divide='ignore', invalid='ignore'):
                        flatness = np.exp(np.mean(np.log(segment_power + 1e-10))) / (np.mean(segment_power) + 1e-10)
                        flatness = np.nan_to_num(flatness, nan=0.0)
                    
                    # Assign features to all points in the window
                    for j in range(i, min(end_idx, n)):
                        spectral_energy[j] = total_energy
                        spectral_entropy[j] = entropy
                        spectral_rolloff[j] = rolloff
                        spectral_flatness[j] = flatness
                
                # Add extracted features
                features.extend([spectral_energy, spectral_entropy, spectral_rolloff, spectral_flatness])
                
                # Calculate relative energy in different frequency bands
                band_energy_ratios = self._calculate_frequency_band_ratios(time_series, n)
                features.extend(band_energy_ratios)
        
        return features

    def _calculate_frequency_band_ratios(self, time_series, n):
        """
        Calculate energy ratios in different frequency bands using bandpass filtering.
        
        Args:
            time_series: 1D numpy array of telemetry values
            n: Length of the time series
            
        Returns:
            List of numpy arrays with band energy ratios
        """
        import numpy as np
        from scipy import signal
        
        # Define band boundaries (normalized frequencies 0-1)
        # Low: 0-0.1, Mid-low: 0.1-0.3, Mid-high: 0.3-0.6, High: 0.6-1.0
        bands = [(0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0)]
        
        if n < 32:  # Need minimum length for meaningful filtering
            return [np.zeros(n) for _ in range(len(bands) + 2)]  # bands + 2 ratio features
        
        # Design filters for each band
        filters = []
        for low, high in bands:
            if low == 0:
                # Lowpass filter
                b, a = signal.butter(4, high, btype='lowpass')
            elif high == 1.0:
                # Highpass filter
                b, a = signal.butter(4, low, btype='highpass')
            else:
                # Bandpass filter
                b, a = signal.butter(4, [low, high], btype='bandpass')
            filters.append((b, a))
        
        # Apply filters and calculate energy in each band
        band_energies = []
        for b, a in filters:
            try:
                filtered = signal.filtfilt(b, a, time_series)
                energy = filtered**2
                band_energies.append(energy)
            except Exception as e:
                print(f"Error in frequency band filtering: {e}")
                band_energies.append(np.zeros_like(time_series))
        
        # Calculate total energy at each point
        total_energy = np.sum(np.array(band_energies), axis=0)
        total_energy = np.where(total_energy > 1e-10, total_energy, 1e-10)  # Avoid division by zero
        
        # Calculate energy ratios for each band
        band_ratios = []
        for energy in band_energies:
            ratios = energy / total_energy
            band_ratios.append(ratios)
        
        # Calculate additional ratio features
        low_to_high_ratio = band_energies[0] / (band_energies[-1] + 1e-10)
        mid_to_rest_ratio = (band_energies[1] + band_energies[2]) / (total_energy + 1e-10)
        
        band_ratios.extend([low_to_high_ratio, mid_to_rest_ratio])
        
        return band_ratios

    def _compute_trend_features(self, time_series, n):
        # ... (existing logic) ...
        # Ensure it returns a list of numpy arrays
        features = []
        slope_windows = [5, 10, 20, 40]
        for window in slope_windows:
            slopes = np.zeros(n)
            for i in range(n):
                start = max(0, i - window + 1)
                window_data = time_series[start:i+1]
                if len(window_data) >= 3:
                    x = np.arange(len(window_data))
                    A = np.vstack([x, np.ones(len(x))]).T
                    try: slope, _ = np.linalg.lstsq(A, window_data, rcond=None)[0]
                    except np.linalg.LinAlgError: slope = 0.0
                    slopes[i] = slope
            features.append(slopes)

        cumsum = np.zeros(n)
        if n > 0: cumsum = np.cumsum(time_series) / np.arange(1, n + 1)
        features.append(cumsum)

        for window in [3, 7, 15]:
            roc = np.zeros(n)
            # Prepend first value to handle i=0 case correctly
            ts_padded = np.pad(time_series, (window, 0), mode='edge')
            for i in range(n):
                 # Efficient calculation using padded array avoids boundary checks inside loop
                 # Careful with indices: original index i corresponds to ts_padded[i+window]
                 start_idx_padded = i + window - (window) # Simplified: i
                 if ts_padded[start_idx_padded] != 0: # Avoid division by zero if start value is 0
                     # Use i - (i - window + 1) + 1 = window as denominator? No, time diff is i - start
                     time_diff = window # If using fixed lookback window
                     # Or time_diff = i - max(0, i - window + 1) ? No, use fixed window diff
                     # roc[i] = (time_series[i] - time_series[max(0, i-window)]) / window # Fixed window lookback
                     roc[i] = (ts_padded[i+window] - ts_padded[i]) / window # Difference over 'window' points
                 # else: roc[i] remains 0
            features.append(roc)

        return features


    def _compute_complexity_features(self, time_series, n):
        # ... (existing logic) ...
        # Ensure it returns a list of numpy arrays
        features = []
        for window in [20, 40, 80]:
            entropy = np.zeros(n)
            for i in range(n):
                 start = max(0, i - window + 1)
                 window_data = time_series[start:i+1]
                 if len(window_data) >= 10:
                     hist, _ = np.histogram(window_data, bins=10)
                     hist_sum = np.sum(hist)
                     if hist_sum > 0:
                         hist = hist / hist_sum
                         entropy[i] = -np.sum(hist * np.log2(hist + 1e-10))
            features.append(entropy)

        for window in [10, 20, 40]:
            dfa = np.zeros(n)
            for i in range(n):
                start = max(0, i - window + 1)
                window_data = time_series[start:i+1]
                if len(window_data) >= 10:
                    x = np.arange(len(window_data))
                    A = np.vstack([x, np.ones(len(x))]).T
                    try:
                        slope, intercept = np.linalg.lstsq(A, window_data, rcond=None)[0]
                        trend = slope * x + intercept
                        detrended = window_data - trend
                        dfa[i] = np.std(detrended)
                    except np.linalg.LinAlgError:
                        dfa[i] = np.std(window_data) # Fallback to regular std dev
            features.append(dfa)
        return features

    def _compute_wavelet_features(self, time_series, n):
        # ... (existing logic) ...
        # Ensure it returns a list of numpy arrays
        features = []
        wavelet = 'cmor1.0-1.0'
        scales = np.arange(1, 16) # Consider adjusting scales
        min_len_for_cwt = 32 # CWT often needs a minimum length

        if n < min_len_for_cwt:
            print(f"Warning: Time series length ({n}) too short for CWT. Padding.")
            padding = min_len_for_cwt - n
            # Pad with edge value to minimize discontinuity
            padded_ts = np.pad(time_series, (0, padding), mode='edge')
        else:
            padded_ts = time_series

        try:
             # Use pywt.wavedec for Discrete Wavelet Transform (often more stable/faster)
             # Or stick to CWT if continuous analysis is desired
             coeffs, freqs = pywt.cwt(padded_ts, scales, wavelet) # Use padded_ts

             # Extract features - careful with alignment if padded
             wavelet_energy = np.zeros((n, len(scales))) # Size n, not padded length

             # coeffs shape: (n_scales, padded_length)
             coeffs_abs_sq = np.abs(coeffs)**2

             for i in range(n): # Iterate up to original length
                 if i < coeffs_abs_sq.shape[1]: # Check bounds against coefficient matrix width
                     for j in range(len(scales)):
                         wavelet_energy[i, j] = coeffs_abs_sq[j, i]
                 # else: Keep energy as 0 for points beyond original length if any issue

             # Use top 3 scales based on total energy across original length
             scale_energy = np.sum(wavelet_energy, axis=0)
             if len(scale_energy) > 0:
                 top_scales_indices = np.argsort(scale_energy)[-min(3, len(scales)):] # Get top 3 or fewer if fewer scales
                 for scale_idx in top_scales_indices:
                     features.append(wavelet_energy[:, scale_idx])                 # Add ratio between scales
                 if len(top_scales_indices) >= 2:
                     # Take two highest energy scales
                     scale1_idx = top_scales_indices[-1]
                     scale2_idx = top_scales_indices[-2]
                     scale1_energy = wavelet_energy[:, scale1_idx]
                     scale2_energy = wavelet_energy[:, scale2_idx]
                     
                     # OLD: scale_ratio = scale1_energy / (scale2_energy + 1e-10)
                     
                     # NEW: More stable normalized ratio
                     total_energy_of_pair = scale1_energy + scale2_energy
                     # Use np.divide for safe division where denominator can be zero
                     scale_ratio = np.divide(scale1_energy, total_energy_of_pair, 
                                             out=np.full_like(scale1_energy, 0.5), # Default to 0.5 if total energy is 0
                                             where=total_energy_of_pair > 1e-10)
                     
                     features.append(scale_ratio)
             else:
                  print("Warning: No scale energy calculated for wavelet features.")

        except Exception as e:
             print(f"Error during CWT processing: {e}. Skipping wavelet features.")
             # Add placeholder arrays of zeros if skipping to maintain structure
             num_wavelet_features_expected = 3 + (1 if len(scales) >= 2 else 0)
             for _ in range(num_wavelet_features_expected):
                  features.append(np.zeros(n))

        return features

def normalize_feature_set(features, method='zscore', stats=None):
    """
    Normalize a 2D feature set using specified method and stats.

    Args:
        features: 2D numpy array (n_samples, n_features).
        method: 'zscore' or 'minmax'.
        stats: Dictionary containing pre-calculated stats ('mean', 'std' for zscore; 
               'min', 'max' for minmax). If None, stats are calculated.

    Returns:
        If stats is None: (normalized_features, calculated_stats)
        If stats is provided: normalized_features
    """
    if features is None or features.shape[0] == 0:
         return features, {} if stats is None else features # Return empty/original if no data

    n_samples, n_features = features.shape
    normalized = np.zeros_like(features, dtype=np.float64) # Use float64 for precision
    calculated_stats = {'method': method}

    is_training = (stats is None) # Flag to determine if we calculate or apply stats

    if is_training:
        if method == 'zscore':
            calculated_stats['mean'] = np.mean(features, axis=0)
            calculated_stats['std'] = np.std(features, axis=0)
        elif method == 'minmax':
            calculated_stats['min'] = np.min(features, axis=0)
            calculated_stats['max'] = np.max(features, axis=0)
        else:
            raise ValueError("Unsupported normalization method")
        stats = calculated_stats # Use newly calculated stats for normalization below
    else:
         # Check if provided stats match the method
         required_keys = ['mean', 'std'] if method == 'zscore' else ['min', 'max']
         if not all(key in stats for key in required_keys):
              raise ValueError(f"Provided stats dictionary missing keys for method '{method}'")


    # Apply normalization column by column
    for i in range(n_features):
        feature_col = features[:, i]
        if method == 'zscore':
            mean = stats['mean'][i]
            std = stats['std'][i]
            if std > 1e-10: # Use a small epsilon for stability
                normalized[:, i] = (feature_col - mean) / std
            else:
                normalized[:, i] = feature_col - mean # Handle zero std dev
        elif method == 'minmax':
            min_val = stats['min'][i]
            max_val = stats['max'][i]
            range_val = max_val - min_val
            if range_val > 1e-10:
                normalized[:, i] = (feature_col - min_val) / range_val
            else:
                # Handle constant features - map to 0, 0.5, or 1? Let's map to 0.5
                normalized[:, i] = 0.5

    # Handle potential NaNs/Infs arising from edge cases (should be less likely now)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0) # Map NaN to 0, Inf to 1/0 for minmax

    # In data_processing.py -> normalize_feature_set, before returning:
    # Define reasonable bounds (e.g., -10 to 10 for z-score, depends on data)
    clip_min = -10.0
    clip_max = 10.0
    normalized = np.clip(normalized, clip_min, clip_max)
    print(f"Applied clipping to features: Min={clip_min}, Max={clip_max}") # Add log
    
    if is_training:
        return normalized, calculated_stats
    else:
        return normalized

def select_important_features(features, k=20, return_indices=False):
     """Selects top k features by variance. Optionally returns indices."""
     if k is None or k >= features.shape[1]:
          # Return all features if k is None or >= number of features
          indices = np.arange(features.shape[1])
          return features, indices if return_indices else features

     variances = np.var(features, axis=0)
     # Handle NaN variances if any occurred before selection
     variances = np.nan_to_num(variances, nan=0.0)
     top_indices = np.argsort(variances)[-k:]
     selected = features[:, top_indices]
     return (selected, top_indices) if return_indices else selected

def process_data(train_data, test_data, binary_start_idx=1, create_features=False):
    """Process train and test data with normalization"""
    
    def _process_single(data):
        # Separate continuous and binary features
        continuous_data = data[:, :binary_start_idx]
        binary_data = data[:, binary_start_idx:] if data.shape[1] > binary_start_idx else np.array([])
        
        # Normalize continuous features using z-score normalization
        # Using train data statistics for both train and test
        scaled_continuous = np.zeros_like(continuous_data)
        for i in range(binary_start_idx):
            mean = np.mean(train_data[:, i])  # Using train stats for normalization
            std = np.std(train_data[:, i])
            if std > 0:
                scaled_continuous[:, i] = (continuous_data[:, i] - mean) / std
            else:
                scaled_continuous[:, i] = continuous_data[:, i] - mean
        
        # Combine scaled continuous with binary
        if len(binary_data) > 0:
            return np.concatenate([scaled_continuous, binary_data], axis=1)
        else:
            return scaled_continuous
    
    # Process both datasets
    processed_train = _process_single(train_data)
    processed_test = _process_single(test_data)
    
    return processed_train, processed_test