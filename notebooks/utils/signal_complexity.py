import numpy as np
from scipy import stats, signal
import pywt
import warnings

# Keep existing functions intact and add enhanced ones

def detect_repetition_pattern(window, min_period=3, max_period=100, threshold=0.7):
    """
    Detect if a signal contains repeating patterns by analyzing autocorrelation.
    
    Args:
        window: 1D array of time series values
        min_period: Minimum period to detect (samples)
        max_period: Maximum period to detect (samples)
        threshold: Correlation threshold for repetition detection
        
    Returns:
        Dictionary with repetition metrics or None if no repetition detected
    """
    if len(window) < max_period * 2:
        max_period = len(window) // 2 if len(window) > 4 else 2
    
    if len(window) <= min_period * 2:
        return None
    
    # Normalize the window
    normalized = (window - np.mean(window)) / (np.std(window) + 1e-10)
    
    # Calculate autocorrelation
    try:
        autocorr = np.correlate(normalized, normalized, mode='full')
        autocorr = autocorr[len(autocorr)//2:] # Use only the positive lags
        if autocorr[0] != 0:  # Add this check
            autocorr = autocorr / autocorr[0]  # Normalize
        else:
            autocorr = np.zeros_like(autocorr)
    except:
        return None
    
    # Find peaks in autocorrelation
    try:
        peaks, _ = signal.find_peaks(autocorr, height=threshold)
        if len(peaks) == 0 or peaks[0] < min_period:
            return None
            
        # Use the first strong peak as the period
        period = peaks[0]
        if period > max_period:
            return None
            
        repetition_strength = autocorr[period]
        
        # Calculate regularity - how consistent the repetition is
        if len(peaks) >= 2:
            peak_distances = np.diff(peaks)
            period_consistency = 1.0 - min(1.0, np.std(peak_distances) / (np.mean(peak_distances) + 1e-10))
        else:
            period_consistency = 0.5  # Only one peak, can't determine consistency
            
        return {
            "detected": True,
            "period": period,
            "strength": repetition_strength,
            "consistency": period_consistency
        }
    except:
        return None

def analyze_spectral_features(window, sampling_rate=1.0):
    """
    Extract detailed spectral features from a signal window.
    
    Args:
        window: 1D array of time series values
        sampling_rate: Sampling rate of the signal (default=1.0)
        
    Returns:
        Dictionary of spectral features
    """
    if len(window) < 4:
        return {"spectral_entropy": 0, "dominant_freq": 0, "spectral_purity": 0}
    
    # Normalize window
    normalized = (window - np.mean(window)) / (np.std(window) + 1e-10)
    
    try:
        # Calculate spectral features using FFT
        fft_result = np.fft.rfft(normalized)
        magnitudes = np.abs(fft_result)
        
        # Normalize spectrum
        total_power = np.sum(magnitudes**2) + 1e-10
        normalized_spectrum = (magnitudes**2) / total_power
        
        # Spectral entropy
        spectral_entropy = -np.sum(normalized_spectrum * np.log2(normalized_spectrum + 1e-10))
        
        # Find dominant frequencies
        if len(magnitudes) > 1:
            freq_indices = np.argsort(magnitudes[1:])[-3:] + 1  # Skip DC component, get top 3
            dominant_freqs = freq_indices / len(window) * sampling_rate
            dominant_freq = dominant_freqs[-1] if len(dominant_freqs) > 0 else 0
            
            # Spectral purity (ratio of power in dominant frequency to total power)
            if total_power > 0:
                spectral_purity = magnitudes[freq_indices[-1]]**2 / total_power
            else:
                spectral_purity = 0
        else:
            dominant_freq = 0
            spectral_purity = 1  # Only DC component
            
        return {
            "spectral_entropy": spectral_entropy,
            "dominant_freq": dominant_freq,
            "spectral_purity": spectral_purity
        }
    except:
        return {"spectral_entropy": 0, "dominant_freq": 0, "spectral_purity": 0}

def detect_structural_changes(window, window_size=10):
    """
    Detect structural changes or change points in the time series.
    
    Args:
        window: 1D array of time series values
        window_size: Size of window for local statistics
        
    Returns:
        Dictionary with changepoint metrics
    """
    if len(window) < window_size * 2:
        return {"change_points": 0, "max_change_magnitude": 0}
    
    # Calculate local statistics
    changes = []
    magnitudes = []
    
    for i in range(window_size, len(window) - window_size):
        left_window = window[i - window_size:i]
        right_window = window[i:i + window_size]
        
        # Calculate statistics
        left_mean = np.mean(left_window)
        right_mean = np.mean(right_window)
        left_std = np.std(left_window) + 1e-10
        right_std = np.std(right_window) + 1e-10
        
        # Calculate change magnitude (standardized difference of means)
        pooled_std = np.sqrt((left_std**2 + right_std**2) / 2)
        change_magnitude = abs(right_mean - left_mean) / pooled_std
        
        if change_magnitude > 1.0:  # Threshold for significant change
            changes.append(i)
            magnitudes.append(change_magnitude)
    
    # Return change point metrics
    max_magnitude = max(magnitudes) if magnitudes else 0
    
    return {
        "change_points": len(changes),
        "max_change_magnitude": max_magnitude,
        "change_locations": changes
    }

def analyze_wavelet_features(window):
    """
    Extract wavelet-based features to identify different frequency components.
    
    Args:
        window: 1D array of time series values
        
    Returns:
        Dictionary of wavelet features
    """
    if len(window) < 16:
        return {"wavelet_energy_ratio": 0, "wavelet_entropy": 0}
    
    try:
        # Use wavelet transform to decompose signal
        wavelet = 'db4'  # Daubechies wavelet
        max_level = pywt.dwt_max_level(len(window), wavelet)
        level = min(3, max_level)  # Use appropriate decomposition level
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(window, wavelet, level=level)
        
        # Calculate energy at each level
        energies = [np.sum(np.square(c)) for c in coeffs]
        total_energy = sum(energies) + 1e-10
        
        # Normalized energies
        normalized_energies = [e / total_energy for e in energies]
        
        # Wavelet entropy
        wavelet_entropy = -np.sum([e * np.log2(e + 1e-10) for e in normalized_energies])
        
        # Energy ratio (detail vs approximation)
        detail_energy = sum(energies[1:])
        approx_energy = energies[0]
        energy_ratio = detail_energy / (approx_energy + 1e-10)
        
        return {
            "wavelet_energy_ratio": energy_ratio,
            "wavelet_entropy": wavelet_entropy,
            "energy_distribution": normalized_energies
        }
    except:
        return {"wavelet_energy_ratio": 0, "wavelet_entropy": 0}

def calculate_complexity_metrics(window, epsilon=1e-10):
    """
    Calculate basic complexity metrics for a time series window.
    
    Args:
        window: 1D array of time series values or 2D array with time series in first column
        epsilon: Small constant to avoid division by zero
        
    Returns:
        Dictionary of complexity metrics
    """
    # Ensure window is 1D
    if len(window.shape) > 1:
        if window.shape[1] > 0:
            window_1d = window[:, 0]
        else:
            return {"complexity_score": 0, "entropy": 0, "variance": 0}
    else:
        window_1d = window
    
    # Handle empty or single-value windows
    if len(window_1d) < 4:
        return {"complexity_score": 0, "entropy": 0, "variance": 0}
    
    # Calculate basic statistics
    mean = np.mean(window_1d)
    variance = np.var(window_1d)
    
    # Normalize window for subsequent calculations
    normalized = (window_1d - mean) / (np.std(window_1d) + epsilon)
    
    # Calculate zero-crossing rate (complexity indicator)
    zero_crossings = np.where(np.diff(np.signbit(normalized)))[0]
    zero_crossing_rate = len(zero_crossings) / (len(normalized) - 1)
    
    # Calculate entropy-based complexity
    try:
        hist, _ = np.histogram(normalized, bins=10)
        hist_sum = np.sum(hist)
        if hist_sum > 0:
            hist = hist / hist_sum
            entropy = -np.sum(hist * np.log2(hist + epsilon))
        else:
            entropy = 0
    except:
        entropy = 0
    
    # Calculate trend strength
    try:
        x = np.arange(len(window_1d))
        A = np.vstack([x, np.ones(len(x))]).T
        slope, _ = np.linalg.lstsq(A, window_1d, rcond=None)[0]
        trend_strength = abs(slope) * len(window_1d) / (np.max(window_1d) - np.min(window_1d) + epsilon)
        trend_strength = min(1.0, trend_strength)
    except:
        trend_strength = 0
    
    # Combined complexity score (0-1 range)
    complexity_score = 0.4 * min(1.0, entropy / 3.0) + 0.3 * zero_crossing_rate + 0.3 * (1 - trend_strength)
    
    return {
        "complexity_score": complexity_score,
        "entropy": entropy,
        "variance": variance,
        "zero_crossing_rate": zero_crossing_rate,
        "trend_strength": trend_strength
    }

def calculate_enhanced_complexity_metrics(window, epsilon=1e-10):
    """
    Calculate enhanced complexity metrics for a time series window.
    
    Args:
        window: 1D array of time series values or 2D array with time series in first column
        epsilon: Small constant to avoid division by zero
        
    Returns:
        Dictionary of enhanced complexity metrics
    """
    # Start with basic complexity metrics
    basic_metrics = calculate_complexity_metrics(window, epsilon)
    
    # Ensure window is 1D
    if len(window.shape) > 1:
        if window.shape[1] > 0:
            window_1d = window[:, 0]
        else:
            return basic_metrics
    else:
        window_1d = window
    
    # Handle empty or single-value windows
    if len(window_1d) < 4:
        return basic_metrics
    
    # --- Enhanced metrics ---
    enhanced_metrics = {}
    
    # 1. Repetition analysis
    repetition_info = detect_repetition_pattern(window_1d)
    if repetition_info is not None:
        enhanced_metrics.update(repetition_info)
        enhanced_metrics["has_repetition"] = True
    else:
        enhanced_metrics["has_repetition"] = False
    
    # 2. Spectral analysis
    spectral_features = analyze_spectral_features(window_1d)
    enhanced_metrics.update(spectral_features)
    
    # 3. Change point detection
    change_points = detect_structural_changes(window_1d)
    enhanced_metrics.update(change_points)
    
    # 4. Wavelet analysis
    wavelet_features = analyze_wavelet_features(window_1d)
    enhanced_metrics.update(wavelet_features)
    
    # Combine all metrics
    combined_metrics = {**basic_metrics, **enhanced_metrics}
    
    return combined_metrics

def classify_enhanced_pattern_type(metrics):
    """
    Classify the signal into more detailed pattern types based on enhanced metrics.
    
    Args:
        metrics: Dictionary of complexity metrics (enhanced)
        
    Returns:
        String indicating detailed pattern type and confidence dictionary
    """
    # Extract relevant metrics
    variance = metrics.get("variance", 0)
    entropy = metrics.get("entropy", 0)
    complexity_score = metrics.get("complexity_score", 0)
    trend_strength = metrics.get("trend_strength", 0)
    zero_crossing_rate = metrics.get("zero_crossing_rate", 0)
    has_repetition = metrics.get("has_repetition", False)
    repetition_strength = metrics.get("strength", 0)
    spectral_purity = metrics.get("spectral_purity", 0)
    change_points = metrics.get("change_points", 0)
    max_change_magnitude = metrics.get("max_change_magnitude", 0)
    wavelet_energy_ratio = metrics.get("wavelet_energy_ratio", 0)
    
    # Initialize confidence scores for different types
    confidence = {
        "constant": 0.0,
        "steady_periodic": 0.0,
        "varying_periodic": 0.0,
        "strong_trend": 0.0,
        "weak_trend": 0.0,
        "nonstationary": 0.0,
        "complex": 0.0,
        "noisy": 0.0,
        "mixed": 0.0
    }
    
    # 1. Constant signal detection
    if variance < 1e-4:
        confidence["constant"] = 0.95
    elif variance < 1e-3:
        confidence["constant"] = 0.7
    
    # 2. Periodic signal detection
    if has_repetition:
        if repetition_strength > 0.8 and spectral_purity > 0.7:
            confidence["steady_periodic"] = 0.8 + 0.2 * repetition_strength
        elif repetition_strength > 0.5:
            confidence["varying_periodic"] = 0.6 + 0.4 * repetition_strength
    elif zero_crossing_rate > 0.1 and zero_crossing_rate < 0.9 and spectral_purity > 0.5:
        confidence["varying_periodic"] = 0.5 + 0.3 * spectral_purity
    
    # 3. Trend detection
    if trend_strength > 0.8:
        confidence["strong_trend"] = 0.7 + 0.3 * trend_strength
    elif trend_strength > 0.5:
        confidence["weak_trend"] = 0.5 + 0.5 * trend_strength
    
    # 4. Changepoint & nonstationarity detection
    if change_points > 2 and max_change_magnitude > 2.0:
        confidence["nonstationary"] = 0.6 + 0.4 * min(1.0, max_change_magnitude / 5.0)
    
    # 5. Complexity & noise detection
    if complexity_score > 0.7:
        if entropy > 2.0:
            confidence["noisy"] = 0.6 + 0.4 * min(1.0, entropy / 4.0)
        else:
            confidence["complex"] = 0.7 + 0.3 * complexity_score
    
    # 6. Mixed patterns
    if len([score for score in confidence.values() if score > 0.3]) >= 2:
        max_confidence = max(confidence.values())
        second_max = sorted(confidence.values(), reverse=True)[1]
        confidence["mixed"] = 0.5 * (max_confidence + second_max)
    
    # Determine the most likely pattern type
    pattern_type = max(confidence, key=confidence.get)
    
    # Map detailed types to basic types for backward compatibility
    basic_type_mapping = {
        "constant": "constant",
        "steady_periodic": "periodic",
        "varying_periodic": "periodic",
        "strong_trend": "trending",
        "weak_trend": "trending",
        "nonstationary": "complex",
        "complex": "complex",
        "noisy": "complex",
        "mixed": "complex"
    }
    
    # Add basic type to the result
    basic_type = basic_type_mapping.get(pattern_type, "complex")
    
    return pattern_type, basic_type, confidence

def get_enhanced_threshold_params(pattern_type, confidence_scores):
    """
    Get fine-tuned threshold parameters based on detailed pattern type and confidence.
    
    Args:
        pattern_type: Detailed pattern type from classify_enhanced_pattern_type
        confidence_scores: Dictionary of confidence scores for each pattern type
        
    Returns:
        Dictionary of threshold parameters
    """
    # Start with basic parameters from the original function
    if pattern_type in ["constant", "steady_periodic"]:
        base_params = {
            "factor": 3.0,
            "k_nearest": 1,
            "force_sub_score": 0.9
        }
    elif pattern_type in ["varying_periodic"]:
        base_params = {
            "factor": 1.8,
            "k_nearest": 2,
            "force_sub_score": 0.75
        }
    elif pattern_type in ["strong_trend", "weak_trend"]:
        base_params = {
            "factor": 1.5,
            "k_nearest": 3,
            "force_sub_score": 0.7
        }
    elif pattern_type == "nonstationary":
        base_params = {
            "factor": 1.2,
            "k_nearest": 4,
            "force_sub_score": 0.6
        }
    elif pattern_type == "mixed":
        base_params = {
            "factor": 1.3,
            "k_nearest": 3,
            "force_sub_score": 0.65
        }
    else:  # complex or noisy
        base_params = {
            "factor": 1.0,
            "k_nearest": 5,
            "force_sub_score": 0.5
        }
    
    # Get confidence score for this pattern
    confidence = confidence_scores.get(pattern_type, 0.5)
    
    # Adjust parameters based on confidence
    if confidence > 0.8:
        # High confidence - use parameters as is
        return base_params
    elif confidence > 0.5:
        # Medium confidence - slightly more conservative
        return {
            "factor": base_params["factor"] * 0.9,  # Slightly reduce factor
            "k_nearest": min(5, base_params["k_nearest"] + 1),  # More neighbors
            "force_sub_score": base_params["force_sub_score"] * 0.9  # Lower force threshold
        }
    else:
        # Low confidence - more conservative
        return {
            "factor": base_params["factor"] * 0.7,  # Reduce factor more
            "k_nearest": min(5, base_params["k_nearest"] + 2),  # Even more neighbors
            "force_sub_score": base_params["force_sub_score"] * 0.8  # Lower force threshold more
        }

def calculate_adaptive_threshold_enhanced(base_threshold, metrics, min_threshold_multiplier=1.0):
    """
    Enhanced version of adaptive threshold calculation using detailed pattern analysis.
    
    Args:
        base_threshold: The original threshold value
        metrics: Dictionary of complexity metrics (enhanced)
        min_threshold_multiplier: Minimum threshold multiplier (protection)
        
    Returns:
        Adjusted threshold value and detailed explanation
    """
    # Get detailed pattern type and confidence
    pattern_type, basic_type, confidence_scores = classify_enhanced_pattern_type(metrics)
    
    # Get threshold parameters for this pattern
    params = get_enhanced_threshold_params(pattern_type, confidence_scores)
    
    # Apply the multiplier factor
    final_multiplier = params["factor"]
    
    # Ensure minimum threshold multiplier
    final_multiplier = max(min_threshold_multiplier, final_multiplier)
    
    # Calculate adaptive threshold
    adaptive_threshold = base_threshold * final_multiplier
    
    # Create explanation
    explanation = {
        "pattern_type": pattern_type,
        "basic_type": basic_type,
        "confidence": confidence_scores[pattern_type],
        "threshold_factor": final_multiplier,
        "base_threshold": base_threshold,
        "adaptive_threshold": adaptive_threshold,
        "k_recommendation": params["k_nearest"],
        "force_sub_score": params["force_sub_score"]
    }
    
    return adaptive_threshold, explanation