import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
# import tensorflow as tf # REMOVE
import torch # ADD
import torch.nn as nn # ADD
import torch.optim as optim # ADD
import torch.nn.functional as F # ADD
from models.anomaly_injector import ContextualAnomalyInjector
from models.memory_bank import ContextMemoryBank
# from models.encoder import TCNEncoder # Will be PyTorch version
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances # Keep for memory bank CPU calcs
# from sklearn.mixture import GaussianMixture # For GMM - Keep if GMM is still used elsewhere
from utils import visualizer
from utils.data_processing import TemporalFeatureExtractor, normalize_feature_set, select_important_features
from utils.visualizer import plot_anomaly_detection_results, generate_enhanced_plots_with_changes, create_substitution_diagnostics
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Import PyTorch TCNEncoder
from models.encoder import TCNEncoder # This now imports the PyTorch version

# --- Add device configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f"Using device: {DEVICE}")

def process_data(train_data, test_data, binary_start_idx=1):
    """Process train and test data with normalization (No TF/Torch specific code here)"""
    # ... (Keep existing code, it's NumPy based) ...
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


def create_sliding_windows(data, window_size, step=1):
    """Creates sliding windows using stride tricks for efficiency. (No TF/Torch specific code)"""
    # ... (Keep existing code, it's NumPy based) ...
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    n_samples, n_features = data.shape
    if n_samples < window_size:
        print(f"Warning: Data length ({n_samples}) is less than window size ({window_size}). Returning empty array.")
        return np.empty((0, window_size, n_features))

    n_windows = (n_samples - window_size) // step + 1
    data = np.ascontiguousarray(data)
    windows = np.lib.stride_tricks.as_strided(
        data,
        shape=(n_windows, window_size, n_features),
        strides=(step * data.strides[0], data.strides[0], data.strides[1])
    )
    return windows.copy()


def _generate_initial_embeddings(encoder, test_windows_np, context_size, batch_size, step): # test_windows is numpy
    """First pass: Generate context embeddings and store windows. (PyTorch version)"""
    print("First pass: Collecting embeddings and context windows (PyTorch)...")
    all_context_embeddings_list = []
    all_context_windows_list = [] # Keep as numpy
    window_start_indices = []

    encoder.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculations
        for i in range(0, len(test_windows_np), batch_size):
            batch_windows_np = test_windows_np[i:i + batch_size]
            # Prepare context part for encoder
            # Shape (batch, context_size, features)
            batch_context_np = batch_windows_np[:, :context_size, :]
            batch_context_torch = torch.tensor(batch_context_np, dtype=torch.float32).to(DEVICE)
            
            # Get latent embeddings (with_projection=False)
            # PyTorch model forward will handle the permutation internally
            z_batch_context = encoder(batch_context_torch, with_projection=False) # Output is (batch, latent_dim)
            
            z_batch_context_np = z_batch_context.cpu().numpy()

            for k in range(len(batch_windows_np)):
                idx = i + k
                window_start_idx = idx * step
                window_start_indices.append(window_start_idx)
                all_context_embeddings_list.append(z_batch_context_np[k])
                all_context_windows_list.append(batch_windows_np[k, :context_size, :]) # Store raw numpy context window

    return np.array(all_context_embeddings_list), all_context_windows_list, window_start_indices


def _get_signal_parameters(memory_bank):
    """(Keep existing code - no TF/Torch specifics)"""
    # ... (Keep existing code) ...
    signal_type = "complex"
    signal_type_confidence = 0.5
    pattern_type = None
    signal_params = {'threshold_adjustment': 0.0, 'boost_factor': 1.0} 

    if hasattr(memory_bank, 'signal_type') and memory_bank.signal_type is not None:
        signal_type = memory_bank.signal_type
    if hasattr(memory_bank, 'pattern_type') and memory_bank.pattern_type is not None:
        pattern_type = memory_bank.pattern_type
    if hasattr(memory_bank, 'type_confidence') and memory_bank.type_confidence is not None:
        signal_type_confidence = memory_bank.type_confidence

    if signal_type == "constant":
        signal_params['threshold_adjustment'] = 0.2
        signal_params['boost_factor'] = 1.3
    elif signal_type == "periodic":
        signal_params['threshold_adjustment'] = 0.1
        signal_params['boost_factor'] = 0.9

    if signal_type_confidence < 0.8:
        confidence_factor = signal_type_confidence / 0.8
        signal_params['threshold_adjustment'] *= confidence_factor
        signal_params['boost_factor'] = 1.0 + (signal_params['boost_factor'] - 1.0) * confidence_factor
    return signal_type, signal_type_confidence, pattern_type, signal_params


def _calculate_confidence_scores(min_distance, change_score, transition_score, effective_threshold,
                                 combine_indicators, enable_change_detection, enable_transition_detection,
                                 min_context_distance, debug=False):
    """
    Calculates substitution confidence using DYNAMICALLY-WEIGHTED indicators.
    The weight of each indicator is determined by its own confidence level.
    """
    # === Step 1: Calculate the individual confidence of each indicator (as before) ===
    
    # 1a. Distance Confidence
    distance_confidence = 0.0
    min_distance_finite = min_distance if np.isfinite(min_distance) else np.inf
    if min_distance_finite > min_context_distance:
        if np.isfinite(effective_threshold) and effective_threshold > 1e-9:
            if min_distance_finite > effective_threshold:
                distance_ratio = (min_distance_finite - effective_threshold) / max(effective_threshold, 1e-6)
                distance_confidence = min(0.95, 1.0 / (1.0 + np.exp(-2.5 * (distance_ratio - 0.5))))
            else:
                denominator = max(effective_threshold - min_context_distance, 1e-6)
                distance_ratio = (min_distance_finite - min_context_distance) / denominator
                distance_confidence = min(0.2, distance_ratio * 0.2)
        else:
            distance_confidence = 0.05
    
    # 1b. Change Confidence
    change_confidence = 0.0
    base_change_scaling_factor = 0.8
    if enable_change_detection and change_score is not None and change_score > 0.05:
        change_confidence = min(0.95, change_score / base_change_scaling_factor)
        
    # 1c. Transition Confidence
    transition_confidence = 0.0
    base_transition_scaling_factor = 8.0
    if enable_transition_detection and transition_score is not None and transition_score > 1.8:
        transition_confidence = min(0.95, transition_score / base_transition_scaling_factor)

    # === Step 2: Dynamically determine the weights based on the confidence scores ===
    
    # We square the confidences to make the "winner" even more influential.
    # Add a small epsilon to avoid division by zero if all confidences are 0.
    w_dist_raw = distance_confidence**2
    w_change_raw = change_confidence**2
    w_trans_raw = transition_confidence**2

    total_weight_raw = w_dist_raw + w_change_raw + w_trans_raw + 1e-9

    # Normalize to get the final weights
    weight_distance = w_dist_raw / total_weight_raw
    weight_change = w_change_raw / total_weight_raw
    weight_transition = w_trans_raw / total_weight_raw

    # === Step 3: Calculate the final confidence score using the dynamic weights ===
    
    substitution_confidence = (weight_distance * distance_confidence +
                               weight_change * change_confidence +
                               weight_transition * transition_confidence)
                               
    substitution_confidence = np.clip(substitution_confidence, 0.0, 1.0)
    
    # Optional debug output
    if debug and (distance_confidence > 0.1 or change_confidence > 0.1 or transition_confidence > 0.1):
        print(f"Dynamic Weighting Debug:")
        print(f"  Confidences: dist={distance_confidence:.3f}, change={change_confidence:.3f}, trans={transition_confidence:.3f}")
        print(f"  Raw weights: dist={w_dist_raw:.3f}, change={w_change_raw:.3f}, trans={w_trans_raw:.3f}")
        print(f"  Final weights: dist={weight_distance:.3f}, change={weight_change:.3f}, trans={weight_transition:.3f}")
        print(f"  Final confidence: {substitution_confidence:.3f}")
    
    # The original return signature asks for the individual confidences, which is still useful for debugging.
    return distance_confidence, change_confidence, transition_confidence, substitution_confidence


def _calculate_scores_pytorch(z_full, z_context, ref_embedding, distance_metric): # Inputs are PyTorch Tensors
    """Calculate distance scores (e.g., initial, substituted) using PyTorch."""
    if distance_metric == 'euclidean':
        squared_diff_orig = torch.sum(torch.square(z_full - z_context))
        initial_score = torch.sqrt(squared_diff_orig + 1e-9)
        score_with_sub = None
        if ref_embedding is not None:
            squared_diff_sub = torch.sum(torch.square(z_full - ref_embedding))
            score_with_sub = torch.sqrt(squared_diff_sub + 1e-9)
    else: # cosine
        # Normalize along feature dimension (dim=0 if single embedding, dim=1 if batch of embeddings)
        # Assuming z_full, z_context, ref_embedding are single embeddings [latent_dim]
        z_full_norm = F.normalize(z_full, p=2, dim=0)
        z_context_norm = F.normalize(z_context, p=2, dim=0)
        
        cosine_similarity_orig = torch.clamp(torch.sum(z_full_norm * z_context_norm), -1.0, 1.0)
        initial_score = (1.0 - cosine_similarity_orig)
        
        score_with_sub = None
        if ref_embedding is not None:
            ref_norm = F.normalize(ref_embedding, p=2, dim=0)
            cosine_similarity_sub = torch.clamp(torch.sum(z_full_norm * ref_norm), -1.0, 1.0)
            score_with_sub = (1.0 - cosine_similarity_sub)

    # Return as numpy scalars/arrays as original function did
    initial_score_np = initial_score.item() if initial_score.numel() == 1 else initial_score.cpu().numpy()
    score_with_sub_np = None
    if score_with_sub is not None:
        score_with_sub_np = score_with_sub.item() if score_with_sub.numel() == 1 else score_with_sub.cpu().numpy()
        
    return initial_score_np, score_with_sub_np


def score_windows_with_improved_mb_override(encoder, test_windows_np, context_size, memory_bank,
                              step, batch_size, distance_metric,
                              substitution_boost_factor=2.0, # Note: This is now unused but kept for signature consistency
                              enable_change_detection=True,
                              enable_transition_detection=True,
                              min_substitution_confidence=0.5,
                              combine_indicators=True,
                              min_context_distance=0.05,
                              max_substitution_ratio=0.3,
                              use_regime_aware_mb_threshold=False,
                              regime_std_threshold=0.20,
                              high_variance_threshold_factor=2.0,
                              use_transition_substitution_gating=False,
                              transition_gating_threshold=1.8,
                              use_complexity_thresholding=False,
                              is_ablation_run=False
                             ):
    """ (PyTorch version with TRUE substitution scoring) """
    print("Running TRUE SUBSTITUTION memory bank override scoring (PyTorch)...")
    print(f"  Scoring method: If context is abnormal, a reference context is encoded and used for contrastive score.")
    use_regime_aware_mb_threshold=False
    use_complexity_thresholding=False
    use_transition_substitution_gating=False
    print(f"  Complexity/Regime MB Threshold Adaptation: DISABLED")
    print(f"  Transition Substitution Gating: DISABLED")

    window_size = test_windows_np.shape[1]
    n_test_windows = len(test_windows_np)
    min_sub_confidence_filter = 0.33

    final_scores = [0.0] * n_test_windows
    substitution_flags = [False] * n_test_windows
    original_distances = [np.nan] * n_test_windows
    embedding_changes_list = [0.0] * n_test_windows
    transition_scores_list_mb = [0.0] * n_test_windows
    substitution_confidence_scores = [0.0] * n_test_windows
   
    # --- MODIFICATION: Name changed to be more accurate ---
    context_substitution_count = 0
    # --- END MODIFICATION ---

    total_windows_scored = 0

    global_mb_threshold = memory_bank.threshold if memory_bank and hasattr(memory_bank, 'threshold') and np.isfinite(memory_bank.threshold) else np.inf
    if not np.isfinite(global_mb_threshold):
         print("WARNING: Global Memory Bank threshold is not finite! Substitution logic will be disabled.")
    print(f"  Using Global MB Threshold for context check: {global_mb_threshold:.4f}")

    # --- First Pass: Embeddings ---
    all_context_embeddings_np, all_context_windows_np_list, window_start_indices = _generate_initial_embeddings(
        encoder, test_windows_np, context_size, batch_size, step
    )

    # --- Change and Transition Detection ---
    change_scores_np, _ = [], [] 
    if enable_change_detection and memory_bank:
        print("Detecting embedding changes...")
        # track_embedding_changes expects list of numpy embeddings
        change_scores_np, _ = memory_bank.track_embedding_changes(
            list(all_context_embeddings_np), window_size=5, sensitivity=1.5) # Convert to list
        embedding_changes_list = change_scores_np[:n_test_windows]

    window_transition_scores_np, _ = [], [] 
    if enable_transition_detection and memory_bank:
        print("Detecting transitions within context windows...")
        # detect_window_transitions expects list of numpy windows
        window_transition_scores_np, _ = memory_bank.detect_window_transitions(
            all_context_windows_np_list, n_segments=4, threshold=2.5)
        transition_scores_list_mb = window_transition_scores_np[:n_test_windows]

    # --- MODIFICATION: Entire second pass scoring logic is rewritten ---
    print("Second pass: Scoring with TRUE SUBSTITUTION logic...")
    encoder.eval()
    with torch.no_grad():
        for i in range(0, n_test_windows, batch_size):
            batch_windows_np = test_windows_np[i:i + batch_size]
            batch_full_torch = torch.tensor(batch_windows_np, dtype=torch.float32).to(DEVICE)
            batch_context_torch = batch_full_torch[:, :context_size, :]

            z_batch_full = encoder(batch_full_torch, with_projection=False)
            z_batch_context = encoder(batch_context_torch, with_projection=False)

            if not torch.all(torch.isfinite(z_batch_full)) or not torch.all(torch.isfinite(z_batch_context)):
                print(f"WARNING: Non-finite values detected in Pytorch embeddings! Batch idx: {i}. Skipping.")
                for k_skip in range(len(batch_windows_np)):
                     idx_skip = i + k_skip
                     final_scores[idx_skip] = np.nan 
                     original_distances[idx_skip] = np.nan
                     substitution_flags[idx_skip] = False
                     substitution_confidence_scores[idx_skip] = 0.0
                total_windows_scored += len(batch_windows_np) 
                continue

            for k in range(len(batch_windows_np)):
                idx = i + k
                current_z_full = z_batch_full[k]
                z_c_orig = z_batch_context[k]
                current_context_embedding_np = z_batch_context[k].cpu().numpy().reshape(1, -1)

                # Get initial score (standard contrastive)
                initial_score, _ = _calculate_scores_pytorch(current_z_full, z_c_orig, None, distance_metric)

                # Assess context normality
                min_distance_ctx_mb, context_is_abnormal, nearest_centroid_idx = np.inf, False, -1
                if memory_bank is not None and memory_bank.centroids is not None and len(memory_bank.centroids) > 0:
                    dist_func_mb = cosine_distances if distance_metric == 'cosine' else euclidean_distances
                    distances_ctx_mb = dist_func_mb(current_context_embedding_np, memory_bank.centroids)[0]
                    if len(distances_ctx_mb) > 0:
                       min_distance_ctx_mb = np.min(distances_ctx_mb)
                       original_distances[idx] = min_distance_ctx_mb
                       context_is_abnormal = min_distance_ctx_mb > global_mb_threshold
                       if context_is_abnormal:
                           nearest_centroid_idx = np.argmin(distances_ctx_mb)
                else:
                    original_distances[idx] = np.nan
               
                # Calculate substitution confidence
                change_score_val = embedding_changes_list[idx] if idx < len(embedding_changes_list) else 0.0
                transition_score_val = transition_scores_list_mb[idx] if idx < len(transition_scores_list_mb) else 0.0
                _, _, _, sub_conf = _calculate_confidence_scores(
                    min_distance_ctx_mb if np.isfinite(min_distance_ctx_mb) else 0.0,                    change_score_val, transition_score_val,
                    global_mb_threshold if np.isfinite(global_mb_threshold) else np.inf,
                    combine_indicators, enable_change_detection, enable_transition_detection,
                    min_context_distance
                )
                substitution_confidence_scores[idx] = sub_conf

                # --- CONFIDENCE-WEIGHTED SCORING ---
                final_score_val = initial_score
                substitution_flags[idx] = False

                if not is_ablation_run and context_is_abnormal and sub_conf >= min_sub_confidence_filter:
                    if (memory_bank.representative_windows is not None and
                        nearest_centroid_idx < len(memory_bank.representative_windows) and
                        memory_bank.representative_windows[nearest_centroid_idx] is not None):
                        
                        ref_context_window_np = memory_bank.representative_windows[nearest_centroid_idx]
                        ref_context_torch = torch.tensor(ref_context_window_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                        z_c_ref = encoder(ref_context_torch, with_projection=False).squeeze(0)
                        
                        _, score_with_sub = _calculate_scores_pytorch(current_z_full, z_c_orig, z_c_ref, distance_metric)

                        # --- NEW: Confidence-Weighted Blended Score ---
                        if score_with_sub is not None:
                            # The boosted score is the score against a clean context, amplified by the boost factor.
                            boosted_score = score_with_sub * substitution_boost_factor
                            # The final score is a blend, weighted by our confidence in the substitution.
                            final_score_val = (1 - sub_conf) * initial_score + sub_conf * boosted_score
                            
                            if sub_conf > 0.5: # Only flag a "true substitution" if confidence is high
                                substitution_flags[idx] = True
                                context_substitution_count += 1
                    else:
                        final_score_val = initial_score
                # --- END CONFIDENCE-WEIGHTED SCORING ---

                if not np.isfinite(final_score_val):
                     print(f"Warning: Non-finite final score at Pytorch idx {idx}. Initial={initial_score}, MinDist={min_distance_ctx_mb}. Assigning high value.")
                     finite_scores_so_far = [s for s in final_scores[:idx] if np.isfinite(s)]
                     if len(finite_scores_so_far) > 10:
                         final_score_val = np.percentile(finite_scores_so_far, 99.9) * 1.1 
                     else:
                         final_score_val = 1e9 
                     final_score_val = min(final_score_val, 1e12) 
               
                final_scores[idx] = final_score_val
                total_windows_scored += 1

    sub_percentage = (context_substitution_count / max(total_windows_scored, 1)) * 100
    print(f"Completed TRUE SUBSTITUTION scoring: {context_substitution_count} windows had their context substituted ({sub_percentage:.2f}%)")

    confidence_stats = {}
    valid_conf_scores = [s for s in substitution_confidence_scores if np.isfinite(s)]
    if valid_conf_scores:
        confidence_stats = {
            'mean': np.mean(valid_conf_scores), 'median': np.median(valid_conf_scores),
            'min': np.min(valid_conf_scores), 'max': np.max(valid_conf_scores),
            'p25': np.percentile(valid_conf_scores, 25), 'p75': np.percentile(valid_conf_scores, 75),
            'p95': np.percentile(valid_conf_scores, 95)
        }
        print(f"Substitution confidence (Diagnostic) - Mean: {confidence_stats.get('mean', np.nan):.4f}, Median: {confidence_stats.get('median', np.nan):.4f}")
    else:
        print("No valid confidence scores recorded.")

    # Return the same variables, but their meaning has changed.
    return (final_scores, substitution_flags, window_start_indices, context_substitution_count,
            total_windows_scored, original_distances, embedding_changes_list, transition_scores_list_mb,
            substitution_confidence_scores, confidence_stats)


def load_and_preprocess_data(stream, parent_dir):
    """(Keep existing code - NumPy based)"""
    # ... (Keep existing code) ...
    print(f"Loading data for channel {stream}...")
    channel_file = f'{stream}.npy'
    train_path = os.path.join(parent_dir, 'data', 'raw', 'train', channel_file)
    test_path = os.path.join(parent_dir, 'data', 'raw', 'test', channel_file)
    train_data_full = np.load(train_path)
    test_data_full = np.load(test_path)
    print(f"Loaded full data: Train shape: {train_data_full.shape}, Test shape: {test_data_full.shape}")
    if len(train_data_full.shape) > 1 and train_data_full.shape[1] > 0:
        train_telemetry = train_data_full[:, 0].flatten()
        test_telemetry = test_data_full[:, 0].flatten()
    elif len(train_data_full.shape) == 1:
        train_telemetry = train_data_full
        test_telemetry = test_data_full
    else:
        print(f"Error: Unexpected data shape for channel {stream}: {train_data_full.shape}")
        return None, None, None, None
    print(f"  Train telemetry shape: {train_telemetry.shape}")
    print(f"  Test telemetry shape: {test_telemetry.shape}")
    return train_telemetry, test_telemetry, train_data_full, test_data_full

def extract_features(train_telemetry, test_telemetry):
    """(Keep existing code - NumPy based)"""
    # ... (Keep existing code) ...
    print("Extracting and normalizing features...")
    feature_extractor = TemporalFeatureExtractor(
        window_sizes=[10, 30, 60],
        fft_components=5,
        long_window_size=150
    )
    train_features_raw = feature_extractor.extract_features(train_telemetry)
    if train_features_raw is None or train_features_raw.shape[1] == 0:
        print("Error: Feature extraction failed for training data.")
        return None, None
    NORM_METHOD = 'zscore'
    processed_train_norm, norm_stats = normalize_feature_set(
        train_features_raw,
        method=NORM_METHOD,
        stats=None
    )
    processed_train, selected_indices = select_important_features(
        processed_train_norm,
        k=None, 
        return_indices=True
    )
    test_features_raw = feature_extractor.extract_features(test_telemetry)
    if test_features_raw is None or test_features_raw.shape[1] == 0:
        print("Error: Feature extraction failed for test data.")
        return None, None
    if test_features_raw.shape[1] != train_features_raw.shape[1]:
        print(f"Warning: Mismatch in number of raw features between train and test.")
        return None, None
    processed_test_norm = normalize_feature_set(
        test_features_raw,
        method=NORM_METHOD,
        stats=norm_stats
    )
    processed_test = processed_test_norm[:, selected_indices]
    print(f"Processed data shapes: Train={processed_train.shape}, Test={processed_test.shape}")
    return processed_train, processed_test

def load_ground_truth(stream, parent_dir, test_length):
    """(Keep existing code - Pandas/NumPy based)"""
    # ... (Keep existing code) ...
    print("Loading ground truth labels...")
    labeled_anomalies_file = os.path.join(parent_dir, 'data', 'processed', 'final_predictions.csv')
    try:
        labeled_anomalies_df = pd.read_csv(labeled_anomalies_file)
    except FileNotFoundError:
        print(f"Error: Labeled anomalies file not found at {labeled_anomalies_file}")
        return None
    channel_labels_info = labeled_anomalies_df[labeled_anomalies_df['chan_id'] == stream]
    if channel_labels_info.empty:
        print(f"Warning: No label information found for channel {stream}.")
        binary_labels = np.zeros(test_length)
    else:
        anomaly_sequences_str = channel_labels_info['anomaly_sequences'].iloc[0]
        anomaly_sequences = eval(anomaly_sequences_str) if isinstance(anomaly_sequences_str, str) else []
        num_values_test = channel_labels_info['num_values'].iloc[0]
        binary_labels = np.zeros(num_values_test)
        for anomaly in anomaly_sequences:
            if isinstance(anomaly, (list, tuple)) and len(anomaly) == 2:
                start, end = anomaly
                start = max(0, start)
                end = min(num_values_test - 1, end)
                if start <= end:
                    binary_labels[start:end + 1] = 1
            elif isinstance(anomaly, int):
                if 0 <= anomaly < num_values_test:
                    binary_labels[anomaly] = 1
        if len(binary_labels) != test_length:
            print(f"Warning: Mismatch between label length and test data length. Adjusting.")
            if len(binary_labels) < test_length:
                padding = np.zeros(test_length - len(binary_labels))
                binary_labels = np.concatenate([binary_labels, padding])
            else:
                binary_labels = binary_labels[:test_length]
    print(f"Found {np.sum(binary_labels)} anomalous points out of {len(binary_labels)}")
    return binary_labels


def build_memory_bank(encoder, processed_train_np, context_size, step, batch_size, # processed_train is numpy
                      n_clusters, use_silhouette_clustering, min_clusters, max_clusters,
                      mem_bank_threshold_method, 
                      mem_bank_iqr_factor,       
                      percentile,                
                      distance_metric, use_complexity_thresholding=True,
                      constant_signal_factor=3.0, periodic_signal_factor=1.5):
    """ (PyTorch version for encoder interaction) """
    print("Building memory bank (PyTorch encoder)...")

    print(f"Creating sliding context windows (size={context_size}, step={step})...")
    normal_train_context_windows_np = create_sliding_windows(processed_train_np, context_size, step)
    if normal_train_context_windows_np.shape[0] == 0:
        print("Error: Not enough training data for memory bank context windows.")
        return None, None 

    normal_embeddings_list = []
    print(f"Encoding {normal_train_context_windows_np.shape[0]} normal context windows...")
    encoding_batch_size = batch_size * 4
    
    encoder.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculations
        try:
            for i in range(0, len(normal_train_context_windows_np), encoding_batch_size):
                batch_np = normal_train_context_windows_np[i:i + encoding_batch_size]
                batch_torch = torch.tensor(batch_np, dtype=torch.float32).to(DEVICE)
                
                # Get LATENT embeddings (with_projection=False)
                embeddings = encoder(batch_torch, with_projection=False) # Output: (batch, latent_dim)
                
                if not torch.all(torch.isfinite(embeddings)):
                     print(f"Warning: Non-finite values in Pytorch embeddings batch idx {i}. Skipping.")
                     continue 
                normal_embeddings_list.append(embeddings.cpu().numpy())
        except Exception as e:
            print(f"ERROR during Pytorch embedding encoding: {e}")
            return None, None 

    if not normal_embeddings_list: 
         print("Error: No valid Pytorch embeddings generated after encoding.")
         return None, None

    normal_embeddings_np = np.vstack(normal_embeddings_list)
    if not np.all(np.isfinite(normal_embeddings_np)):
        print("Error: Non-finite values in final stacked Pytorch embeddings. Cannot build memory bank.")
        return None, None 

    print(f"Generated {normal_embeddings_np.shape[0]} normal embeddings with dim {normal_embeddings_np.shape[1]}.")

    effective_max_clusters = max_clusters
    min_samples_needed_for_max = effective_max_clusters + 1
    if len(normal_embeddings_np) < min_samples_needed_for_max :
        effective_max_clusters = max(min_clusters, len(normal_embeddings_np) - 1)
        print(f"Warning: Reduced max_clusters to {effective_max_clusters} due to limited embeddings ({len(normal_embeddings_np)}).")
    effective_min_clusters = min(min_clusters, effective_max_clusters)

    try:
        memory_bank = ContextMemoryBank(
            normal_embeddings=normal_embeddings_np,
            normal_context_windows=normal_train_context_windows_np,
            n_clusters=n_clusters, 
            auto_cluster=use_silhouette_clustering, 
            min_clusters=effective_min_clusters, 
            max_clusters=effective_max_clusters, 
            distance_metric=distance_metric,
            use_complexity_thresholding=use_complexity_thresholding,
            constant_signal_factor=constant_signal_factor,
            periodic_signal_factor=periodic_signal_factor
        )
        if memory_bank.centroids is None or memory_bank.centroids.shape[0] == 0:
             print("Error: Memory bank clustering failed to produce centroids.")
             return None, normal_embeddings_np 
    except Exception as e:
        print(f"ERROR during ContextMemoryBank instantiation/clustering: {e}")
        return None, normal_embeddings_np 

    if normal_embeddings_np.shape[0] > 1 and memory_bank.centroids.shape[0] > 0:
         print(f"Setting memory bank threshold using method: '{mem_bank_threshold_method}', percentile value: {percentile}")
         try:
             calculated_threshold, threshold_desc = set_optimized_threshold(
                 memory_bank=memory_bank,           
                 normal_embeddings=normal_embeddings_np, 
                 distance_metric=distance_metric,   
                 method=mem_bank_threshold_method,  
                 percentile_value=percentile        
             )
             memory_bank.threshold = calculated_threshold
             memory_bank.threshold_method = threshold_desc 
             print(f"INFO: Memory Bank Threshold for channel calculated/set as: {memory_bank.threshold:.4f} using method: {memory_bank.threshold_method}")
         except Exception as e:
              print(f"ERROR during set_optimized_threshold call: {e}")
              memory_bank.threshold = np.inf
              memory_bank.threshold_method = "Error - Calculation Failed"
    else:
         print("Warning: Not enough normal embeddings or centroids generated to set threshold.")
         memory_bank.threshold = np.inf 
         memory_bank.threshold_method = "Error - Insufficient Data"

    return memory_bank, normal_embeddings_np


def build_and_train_model(train_windows_np, val_windows_np, context_size, input_dim, latent_dim,
                          epochs, batch_size, distance_metric, pooling_strategy="hybrid"):
    """Build and train the encoder model (PyTorch version).
    
    Args:
        train_windows_np: Training windows data
        val_windows_np: Validation windows data
        context_size: Size of context window
        input_dim: Input dimensions
        latent_dim: Latent dimension size
        epochs: Number of training epochs
        batch_size: Size of each training batch
        distance_metric: Distance metric to use ('cosine' or 'euclidean')
        pooling_strategy: Strategy for pooling TCN outputs. Options:
            - "last": Use only the last time step's output (most contextual)
            - "hybrid": Combine last time step with traditional pooling (max+avg)
            - "traditional": Use only max and average pooling (original approach)
    """
    print("Building and training model with PyTorch (Loss on Raw Latent Embeddings for Euclidean)...") # Changed log
    encoder = TCNEncoder(
        sequence_length=train_windows_np.shape[1], 
        input_dim=input_dim,
        latent_dim=latent_dim,
        tcn_layers=4, 
        filters=128,  
        kernel_size=5,
        pooling_strategy=pooling_strategy  # Use the parameter passed to the function
    ).to(DEVICE)
    
    injector = ContextualAnomalyInjector(
        suspect_injection_ratio=0.7,
        anomaly_types=['spike', 'level_shift', 'variance_change', 'stuck_value'],
        anomaly_probs=None,
        min_anomaly_len=5,
        max_anomaly_len=25,
        spike_factor_range=(6.0, 12.0),
        level_shift_factor_range=(4.0, 8.0),
        variance_factor_range=(4.0, 8.0)
    )

    optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
    history = {'loss': [], 'val_loss': []}

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    early_stop = False
    PATIENCE = 10

    for epoch in range(epochs):
        encoder.train() 
        
        indices = np.random.permutation(len(train_windows_np))
        epoch_loss_agg = 0.0
        num_batches = 0

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            
            if len(batch_indices) < 2 and encoder.training:
                print(f"  Skipping training batch of size {len(batch_indices)} to avoid BatchNorm error.")
                continue

            optimizer.zero_grad()
            
            batch_windows_full_np_slice = train_windows_np[batch_indices]
            batch_windows_mod_np, batch_labels_np = injector.inject_anomalies(batch_windows_full_np_slice, context_size)
            
            batch_windows_mod_torch = torch.tensor(batch_windows_mod_np, dtype=torch.float32).to(DEVICE)
            batch_labels_torch = torch.tensor(batch_labels_np, dtype=torch.float32).to(DEVICE)
            batch_windows_context_torch = batch_windows_mod_torch[:, :context_size, :] 

            z_full_latent, _ = encoder(batch_windows_mod_torch, with_projection=True)
            z_context_latent, _ = encoder(batch_windows_context_torch, with_projection=True)
            
            # --- START OF FIX ---
            if distance_metric == 'cosine':
                # Cosine distance requires normalization, this path is correct.
                z_full_latent_norm = F.normalize(z_full_latent, p=2, dim=1)
                z_context_latent_norm = F.normalize(z_context_latent, p=2, dim=1)
                cosine_similarity = torch.sum(z_full_latent_norm * z_context_latent_norm, dim=1)
                cosine_similarity = torch.clamp(cosine_similarity, -1.0, 1.0)
                distances = 1.0 - cosine_similarity
            else: # euclidean
                # Use raw (un-normalized) embeddings for Euclidean distance calculation
                distances = torch.sqrt(torch.sum(torch.square(z_full_latent - z_context_latent), dim=1) + 1e-9)
            # --- END OF FIX ---

            margin = 1.0 
            contrastive_loss = (1 - batch_labels_torch) * torch.square(distances) + \
                               batch_labels_torch * torch.square(torch.clamp(margin - distances, min=0.0))
            loss = torch.mean(contrastive_loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()

            batch_loss_val = loss.item() 
            if i == 0 and epoch % 5 == 0: 
                 print(f"  Epoch {epoch+1}, Batch 0 loss: {batch_loss_val:.6f}, distances min/max/mean: {torch.min(distances):.4f}/{torch.max(distances):.4f}/{torch.mean(distances):.4f}")

            epoch_loss_agg += batch_loss_val
            num_batches += 1

        avg_epoch_loss = epoch_loss_agg / max(num_batches, 1)
        history['loss'].append(avg_epoch_loss)

        # --- Validation Loop ---
        if val_windows_np is not None and len(val_windows_np) > 0:
            encoder.eval() 
            val_losses = []
            with torch.no_grad(): 
                for j in range(0, len(val_windows_np), batch_size):
                    batch_val_indices = np.arange(j, min(j + batch_size, len(val_windows_np)))
                    
                    batch_val_full_np_slice = val_windows_np[batch_val_indices]
                    batch_val_mod_np, batch_val_labels_np = injector.inject_anomalies(batch_val_full_np_slice, context_size)
                    
                    batch_val_mod_torch = torch.tensor(batch_val_mod_np, dtype=torch.float32).to(DEVICE)
                    batch_val_labels_torch = torch.tensor(batch_val_labels_np, dtype=torch.float32).to(DEVICE)
                    batch_val_context_torch = batch_val_mod_torch[:, :context_size, :]

                    z_val_full_latent, _ = encoder(batch_val_mod_torch, with_projection=True)
                    z_val_context_latent, _ = encoder(batch_val_context_torch, with_projection=True)
                    
                    # --- START OF FIX (Validation) ---
                    if distance_metric == 'cosine':
                        z_val_full_latent_norm = F.normalize(z_val_full_latent, p=2, dim=1)
                        z_val_context_latent_norm = F.normalize(z_val_context_latent, p=2, dim=1)
                        val_cosine_similarity = torch.sum(z_val_full_latent_norm * z_val_context_latent_norm, dim=1)
                        val_cosine_similarity = torch.clamp(val_cosine_similarity, -1.0, 1.0)
                        val_distances = 1.0 - val_cosine_similarity
                    else: # euclidean
                        # Use raw (un-normalized) embeddings for validation distance
                        val_distances = torch.sqrt(torch.sum(torch.square(z_val_full_latent - z_val_context_latent), dim=1) + 1e-9)
                    # --- END OF FIX (Validation) ---

                    val_contrastive_loss = (1 - batch_val_labels_torch) * torch.square(val_distances) + \
                                           batch_val_labels_torch * torch.square(torch.clamp(margin - val_distances, min=0.0))
                    val_loss = torch.mean(val_contrastive_loss)
                    val_losses.append(val_loss.item())

            avg_val_loss = np.mean(val_losses) if val_losses else 0
            history['val_loss'].append(avg_val_loss)

            if avg_val_loss < best_val_loss - 0.001: 
                best_val_loss = avg_val_loss
                best_model_state = encoder.state_dict()
                epochs_no_improve = 0
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}, Val Loss: {avg_val_loss:.6f} (new best)")
            else:
                epochs_no_improve += 1
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}, Val Loss: {avg_val_loss:.6f} (no improvement for {epochs_no_improve} epochs)")
                if epochs_no_improve >= PATIENCE: 
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    early_stop = True
                    break
        else:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}")

        if early_stop:
             break

    if early_stop and best_model_state is not None:
        print(f"Restoring best model with validation loss: {best_val_loss:.6f}")
        encoder.load_state_dict(best_model_state)

    return encoder, history


def aggregate_scores(test_scores, processed_test, full_window_size, context_size, step):
    """(Keep existing code - NumPy based)"""
    # ... (Keep existing code) ...
    print("Aggregating scores...")
    final_scores = np.zeros(len(processed_test))
    counts = np.zeros(len(processed_test))
    for i, score in enumerate(test_scores):
        start_idx = i * step
        end_idx = start_idx + full_window_size
        suspect_start_idx = start_idx + context_size
        suspect_end_idx = end_idx
        valid_start = max(0, suspect_start_idx)
        valid_end = min(len(final_scores), suspect_end_idx)
        if valid_start < valid_end:
            if np.isinf(score):
                large_score_value = np.percentile(test_scores[np.isfinite(test_scores)], 99) if np.any(np.isfinite(test_scores)) else 1e9
                final_scores[valid_start:valid_end] += large_score_value
            else:
                final_scores[valid_start:valid_end] += score
            counts[valid_start:valid_end] += 1
    valid_counts_mask = counts > 0
    final_scores[valid_counts_mask] /= counts[valid_counts_mask]
    if np.any(valid_counts_mask):
        first_valid_idx = np.min(np.where(valid_counts_mask)[0])
        last_valid_idx = np.max(np.where(valid_counts_mask)[0])
        mean_valid_score = np.mean(final_scores[valid_counts_mask])
        final_scores[:first_valid_idx] = final_scores[first_valid_idx] if first_valid_idx < len(final_scores) else mean_valid_score
        if last_valid_idx + 1 < len(final_scores):
            final_scores[last_valid_idx + 1:] = final_scores[last_valid_idx]
    else:
        final_scores[:] = 0
        print("Warning: No scores generated for any part of the test series.")
    return final_scores


def set_optimized_threshold(memory_bank, normal_embeddings, distance_metric='euclidean',
                         signal_type=None, min_threshold_factor=1.2,
                         method='percentile', 
                         percentile_value=99.0
                         ):
    """(Keep existing code - Scikit-learn/NumPy based)"""
    # ... (Keep existing code) ...
    if memory_bank.centroids is None or len(memory_bank.centroids) == 0:
        print("Warning: Memory bank has no centroids.")
        return np.inf, "Empty Memory Bank"
    if normal_embeddings is None or len(normal_embeddings) == 0:
        print("Warning: No normal embeddings provided for threshold optimization.")
        return np.inf, "No Normal Embeddings"
    if distance_metric == 'cosine':
        from sklearn.metrics.pairwise import cosine_distances 
        distances = cosine_distances(normal_embeddings, memory_bank.centroids)
    else:
        from sklearn.metrics.pairwise import euclidean_distances 
        distances = euclidean_distances(normal_embeddings, memory_bank.centroids)
    min_distances = np.min(distances, axis=1)
    finite_distances = min_distances[np.isfinite(min_distances)]
    if len(finite_distances) == 0:
        print("Warning: No finite distances for threshold calculation.")
        return np.inf, "No Finite Distances"
    method_description = ""
    raw_threshold = np.inf 
    if method == 'percentile':
        try:
            raw_threshold = np.percentile(finite_distances, percentile_value)
            method_description = f"Percentile ({percentile_value}%)"
            print(f"Calculating MB threshold using: {method_description}") 
        except Exception as e:
            print(f"Error calculating percentile {percentile_value}: {e}")
            method_description = f"Percentile ({percentile_value}%) Error"
            raw_threshold = np.percentile(finite_distances, 99.0) 
            method_description += " -> Fallback P99"
    elif method == 'auto' or method == 'iqr': 
        print(f"Calculating MB threshold using: Auto (Skewness/IQR/Mean+Std)") 
        p50 = np.percentile(finite_distances, 50); p75 = np.percentile(finite_distances, 75)
        p95 = np.percentile(finite_distances, 95); p99 = np.percentile(finite_distances, 99)
        mean = np.mean(finite_distances); std = np.std(finite_distances)
        from scipy import stats
        skewness = stats.skew(finite_distances) if len(finite_distances) > 8 else 0
        if skewness > 2.0:
            base_threshold = p99
            method_description = "Auto: P99 (Highly Skewed)"
        elif skewness > 1.0:
            p25 = np.percentile(finite_distances, 25) 
            iqr = p75 - p25
            base_threshold = p75 + 1.5 * iqr 
            method_description = "Auto: IQR-Based (Skewed)"
        else:
            base_threshold = mean + 3.0 * std
            method_description = "Auto: Mean+3*StdDev (Symmetric)"
        raw_threshold = base_threshold
    else:
         print(f"Warning: Unknown threshold method '{method}'. Defaulting to percentile 99.0.")
         raw_threshold = np.percentile(finite_distances, 99.0)
         method_description = "Unknown Method -> Default P99"
    
    if method != 'percentile': 
         min_threshold = p95 * min_threshold_factor # p95 was calculated in auto/iqr path
         final_threshold = max(raw_threshold, min_threshold)
         if final_threshold > raw_threshold:
             method_description += f" (MinLim->{min_threshold_factor}*P95)"
    else:
        final_threshold = raw_threshold
    
    # <<< NEW ROBUST BOUNDING LOGIC >>>
    # Define a sensible absolute minimum threshold, regardless of the metric.
    # This prevents hyper-specialized memory banks from creating unusable thresholds.
    ABSOLUTE_MIN_THRESHOLD = 0.1 # A good starting point.
    
    # Apply the absolute floor first
    floored_threshold = max(final_threshold, ABSOLUTE_MIN_THRESHOLD)
    if floored_threshold > final_threshold:
        method_description += f" (Floored to {ABSOLUTE_MIN_THRESHOLD})"

    # Then apply the metric-specific bounds as before
    if distance_metric == 'cosine':
        min_bound, max_bound = 0.01, 1.5
    else: # euclidean
        min_bound, max_bound = 0.001, np.inf
    
    bounded_threshold = np.clip(floored_threshold, min_bound, max_bound)
    if bounded_threshold != floored_threshold:
        method_description += f" (Bounded to [{min_bound:.4f}, {max_bound:.4f}])"
    # <<< END NEW LOGIC >>>
    print(f"Optimized threshold: {bounded_threshold:.4f} using {method_description}")
    return bounded_threshold, method_description

# apply_duration_based_hypothesis_testing (Keep existing, NumPy based)
def apply_duration_based_hypothesis_testing(final_scores, substitution_map, window_size=60,
                                          min_duration=300, score_smoothing_window=15,
                                          confidence_threshold=0.99, sig_level=0.01):
    import numpy as np
    import pandas as pd
    from scipy import stats, ndimage
    print("Applying self-adaptive duration-based hypothesis testing...")
    n_points = len(final_scores)
    if n_points == 0: return np.array([]), {}
    print("  Step 1: Applying temporal smoothing...")
    scores_df = pd.Series(final_scores)
    smoothed_scores = scores_df.rolling(window=score_smoothing_window, center=True, min_periods=1).mean().values
    sub_df = pd.Series(substitution_map.astype(float))
    smoothed_subs = sub_df.rolling(window=score_smoothing_window, center=True, min_periods=1).mean().values
    print("  Step 2: Establishing baseline behavior...")
    sub_gradient = np.gradient(smoothed_subs)
    abs_sub_gradient = np.abs(sub_gradient)
    sub_change_points = abs_sub_gradient > np.percentile(abs_sub_gradient, 95)
    sub_boundaries = ndimage.binary_dilation(sub_change_points, structure=np.ones(min(15, n_points//100)))
    labeled_regions, num_regions = ndimage.label(~sub_boundaries)
    region_means = np.zeros(num_regions+1); region_stds = np.zeros(num_regions+1); region_sizes = np.zeros(num_regions+1, dtype=int)
    for i in range(1, num_regions+1):
        region_mask = (labeled_regions == i)
        region_scores = smoothed_scores[region_mask]
        if len(region_scores) > 0:
            region_means[i] = np.mean(region_scores); region_stds[i] = np.std(region_scores); region_sizes[i] = len(region_scores)
    print("  Step 3: Calculating pattern deviations...")
    deviation_scores = np.zeros(n_points)
    for i in range(n_points):
        region_id = labeled_regions[i] if i < len(labeled_regions) else 0
        if region_id > 0 and region_stds[region_id] > 1e-10:
            deviation_scores[i] = (smoothed_scores[i] - region_means[region_id]) / region_stds[region_id]
        else:
            global_mean = np.mean(smoothed_scores); global_std = np.std(smoothed_scores)
            if global_std > 1e-10: deviation_scores[i] = (smoothed_scores[i] - global_mean) / global_std
    print("  Step 4: Applying hypothesis testing with duration constraints...")
    significant_deviations = np.zeros(n_points, dtype=bool); p_values = np.ones(n_points)
    for i in range(window_size, n_points):
        start_idx = i - window_size
        if start_idx < 0: continue
        window_data = smoothed_scores[start_idx:i]
        ref_start = max(0, start_idx - 2*window_size); ref_end = start_idx
        if ref_end - ref_start < window_size // 2:
            reference_mean = np.mean(smoothed_scores); reference_std = np.std(smoothed_scores)
        else:
            reference_data = smoothed_scores[ref_start:ref_end]
            reference_mean = np.mean(reference_data); reference_std = np.std(reference_data)
        if reference_std > 1e-10: normalized_window = (window_data - reference_mean) / reference_std
        else: normalized_window = window_data - reference_mean
        try:
            skewness = stats.skew(normalized_window); kurtosis = stats.kurtosis(normalized_window)
            if abs(skewness) < 0.5 and abs(kurtosis) < 1.0: t_stat, p_value = stats.ttest_1samp(normalized_window, 0)
            else: _, p_value = stats.wilcoxon(normalized_window)
            p_values[i] = p_value
            for j in range(max(0, i-5), i):
                if p_values[j] < sig_level: sig_level *= 0.95; break
            significant_deviations[i] = p_value < sig_level
        except Exception as e: print(f"    Warning: Hypothesis test failed: {e}")
    print("  Step 5: Applying duration constraints and connected component analysis...")
    morph_deviations = ndimage.binary_dilation(significant_deviations, structure=np.ones(min(15, n_points//100)))
    morph_deviations = ndimage.binary_erosion(morph_deviations, structure=np.ones(min(5, n_points//200)))
    labeled_deviations, num_deviation_regions = ndimage.label(morph_deviations)
    anomalous_regions = np.zeros(n_points, dtype=bool); valid_regions = []
    for region_id in range(1, num_deviation_regions + 1):
        region_mask = (labeled_deviations == region_id)
        region_size = np.sum(region_mask)
        if region_size >= min_duration:
            anomalous_regions[region_mask] = True
            valid_regions.append({'start': np.min(np.where(region_mask)[0]), 'end': np.max(np.where(region_mask)[0]),
                                  'size': region_size, 'mean_score': np.mean(smoothed_scores[region_mask]),
                                  'mean_deviation': np.mean(deviation_scores[region_mask])})
    metadata = {'valid_regions': valid_regions, 'num_regions_detected': len(valid_regions),
                'total_anomalous_points': np.sum(anomalous_regions), 'deviation_scores': deviation_scores,
                'significance_threshold': sig_level, 'p_values': p_values}
    print(f"  Duration-based hypothesis testing complete: {len(valid_regions)} regions detected")
    return anomalous_regions, metadata


def apply_adaptive_threshold_and_evaluate(
    final_agg_scores, binary_labels, context_abnormal_map, smoothing_window, # Changed smoothing_alpha to smoothing_window
    base_threshold_percentile, min_anomaly_cluster_size=10, strong_anomaly_factor=2.5,
    temporal_consistency_window=20, min_consistency_ratio=0.6, noise_suppression=True,
    suppress_isolated_points=True, anomaly_region_emphasis=False, use_duration_hypothesis=False,
    min_hypothesis_duration=200, hypothesis_window_size=40
):
    from scipy import ndimage # Ensure ndimage is available
    import pandas as pd # Ensure pandas is imported
    
    print("Applying DYNAMIC smoothing and adaptive thresholding...")
    
    # === Step 1: Define the dynamic smoothing parameters ===
    short_smoothing_window = smoothing_window # e.g., 10-25 (passed in as the main param)
    long_smoothing_window = short_smoothing_window * 4 # e.g., 40-100
    
    # The lookback period to decide if we are in a sustained event
    state_detection_window = long_smoothing_window 
    # The ratio of abnormal contexts needed in the lookback window to switch to long smoothing
    sustained_state_threshold = 0.7 

    mean_finite_score_agg = np.mean(final_agg_scores[np.isfinite(final_agg_scores)]) if np.any(np.isfinite(final_agg_scores)) else 0
    scores_to_smooth = np.nan_to_num(final_agg_scores, nan=mean_finite_score_agg, posinf=mean_finite_score_agg, neginf=mean_finite_score_agg)

    # === Step 2: Smooth the substitution map to get a stable state indicator ===
    # This avoids flickering between short/long smoothing
    sub_map_series = pd.Series(context_abnormal_map.astype(float))
    smoothed_sub_map = sub_map_series.rolling(window=state_detection_window, min_periods=1).mean().to_numpy()

    # === Step 3: Apply dynamic smoothing point by point ===
    n_points = len(final_agg_scores)
    final_scores_smoothed = np.zeros(n_points)
    
    # Pre-calculate short and long smoothed versions of the scores
    scores_series = pd.Series(np.nan_to_num(final_agg_scores))
    short_smoothed_scores = scores_series.rolling(window=short_smoothing_window, center=True, min_periods=1).mean().to_numpy()
    long_smoothed_scores = scores_series.rolling(window=long_smoothing_window, center=True, min_periods=1).mean().to_numpy()
    
    # --- NEW: Adaptive Weighted Smoothing ---
    # The weight for the long smoothing is the smoothed substitution map value itself.
    # This creates a gradual, responsive transition from short to long smoothing.
    weight_long = np.clip(smoothed_sub_map, 0.0, 1.0)
    weight_short = 1.0 - weight_long
    
    final_scores_smoothed = (weight_short * short_smoothed_scores) + (weight_long * long_smoothed_scores)
    
    avg_long_weight = np.mean(weight_long) if len(weight_long) > 0 else 0
    print(f"Dynamic weighted smoothing applied. Average weight for long smoothing: {avg_long_weight:.3f}")
    # --- END NEW SMOOTHING LOGIC ---
    
    # Initialize predictions early in case of errors
    predictions = np.zeros_like(final_scores_smoothed, dtype=bool) # Default predictions if thresholding fails

    threshold_info = {}
    threshold_method = "Error"
    calculated_threshold = np.inf 
    threshold_values = np.zeros_like(final_scores_smoothed)
    finite_indices = np.where(np.isfinite(final_scores_smoothed))[0]

    if len(finite_indices) > 0:
        finite_scores = final_scores_smoothed[finite_indices]
        if len(context_abnormal_map) < len(final_scores_smoothed):
             padding_len = len(final_scores_smoothed) - len(context_abnormal_map)
             aligned_map_full = np.pad(context_abnormal_map.astype(bool), (0, padding_len), 'constant', constant_values=False)
        else:
             aligned_map_full = context_abnormal_map[:len(final_scores_smoothed)].astype(bool)
        finite_context_map = aligned_map_full[finite_indices]
        normal_context_scores = finite_scores[~finite_context_map]
        threshold_method_detail = ""
        if len(normal_context_scores) > 100:  # Ensure we have enough data for robust stats
            try:
                # --- NEW, MORE ROBUST THRESHOLD LOGIC ---
                # Calculate both percentile and median-based thresholds
                p99_score = np.percentile(normal_context_scores, 99.0)  # A high, stable percentile
                median_score = np.median(normal_context_scores)
                iqr_score = np.percentile(normal_context_scores, 75) - np.percentile(normal_context_scores, 25)

                # The base threshold is the median plus a multiple of the IQR.
                # This is a classic robust statistical measure of the "upper fence" for outliers.
                raw_calculated_threshold = median_score + (3.0 * iqr_score)  # Use 3.0 * IQR for a sensitive but safe upper bound

                # As a safety net, ensure the threshold is at least the 99th percentile.
                raw_calculated_threshold = max(raw_calculated_threshold, p99_score)
                
                # The base_threshold_percentile from main.py is now used as a final check/alternative
                # This part is less critical now but can be kept.
                percentile_based_threshold = np.percentile(normal_context_scores, base_threshold_percentile)

                # Choose the *lower* of the two, to be more sensitive.
                raw_calculated_threshold = min(raw_calculated_threshold, percentile_based_threshold)
                
                calculated_threshold = raw_calculated_threshold 
                threshold_method_detail = f"Robust(Med+IQR={raw_calculated_threshold:.4f}, P99={p99_score:.4f}, Perc={percentile_based_threshold:.4f})"
                # --- END NEW LOGIC ---
            except Exception as e:
                print(f"ERROR calculating robust threshold from normal context scores: {e}")
                calculated_threshold = np.inf
                threshold_method_detail = f"RobustNormContext(Error)"
        elif len(normal_context_scores) > 0:
            try:
                # Fallback to original percentile method for smaller datasets
                raw_calculated_threshold = np.percentile(normal_context_scores, base_threshold_percentile)
                calculated_threshold = raw_calculated_threshold # Store raw before multiplication
                threshold_method_detail = f"NormContext({base_threshold_percentile:.1f}%={raw_calculated_threshold:.4f})" # Log raw
            except Exception as e:
                print(f"ERROR calculating threshold from normal context scores: {e}")
                calculated_threshold = np.inf; threshold_method_detail = f"NormContext(Error)"
        else:
            print("Warning: No finite scores for NORMAL context. Using ALL finite scores.")
            if len(finite_scores) > 0:
                 try:
                     raw_calculated_threshold = np.percentile(finite_scores, base_threshold_percentile)
                     calculated_threshold = raw_calculated_threshold # Store raw before multiplication
                     threshold_method_detail = f"AllScoresFallback({base_threshold_percentile:.1f}%={raw_calculated_threshold:.4f})" # Log raw
                 except Exception as e:
                      print(f"ERROR calculating fallback threshold: {e}")
                      calculated_threshold = np.inf; threshold_method_detail = f"AllScoresFallback(Error)"
        
        if not np.isfinite(calculated_threshold): # Check raw threshold before multiplication
             print("CRITICAL ERROR: Raw calculated threshold is non-finite. Cannot proceed.")
             return predictions, {'error': True, 'threshold_value': np.inf, 'values': threshold_values}, "Error - Non-finite raw threshold", final_scores_smoothed
        else:            # --- MODIFICATION HERE ---
            threshold_multiplier = 1.1  # Reduced from 1.8
            final_calculated_threshold = calculated_threshold * threshold_multiplier 
            print(f"Applied factor {threshold_multiplier:.1f} increase. Original Raw: {calculated_threshold:.4f}, Final Adjusted: {final_calculated_threshold:.4f}")
            threshold_method = f"Dynamic Smoothed Robust ({threshold_method_detail.strip()}) Factor {threshold_multiplier:.1f}"
            calculated_threshold = final_calculated_threshold # Update the variable to be used
            # --- END MODIFICATION ---
        threshold_info = {'threshold_value': calculated_threshold, 'values': threshold_values}
    else:
        print("Warning: No finite smoothed scores for threshold calculation.")
        return predictions, {'error': True, 'threshold_value': np.inf, 'values': threshold_values}, "Error - No Finite Smoothed Scores", final_scores_smoothed

    raw_anomalies = np.zeros_like(final_scores_smoothed, dtype=bool)
    print(f"Applying single adjusted threshold: {calculated_threshold:.4f}")
    current_threshold = calculated_threshold
    threshold_values[:] = current_threshold 
    finite_mask = np.isfinite(final_scores_smoothed)
    raw_anomalies[finite_mask] = final_scores_smoothed[finite_mask] > current_threshold
    print(f"Initial thresholding complete. Found {np.sum(raw_anomalies)} raw anomalies.")
    
    predictions = np.copy(raw_anomalies)
    if noise_suppression and len(raw_anomalies) > 0:
        print("Applying temporal consistency constraints (NO GAP FILLING)...")
        suppressed_count = 0; retained_count = 0; # added_count = 0
        for i in range(len(raw_anomalies)):
            if raw_anomalies[i]:
                score = final_scores_smoothed[i]
                is_strong_anomaly = False
                if np.isfinite(current_threshold) and current_threshold > 0:
                     is_strong_anomaly = (score > current_threshold * strong_anomaly_factor)
                if is_strong_anomaly:
                    predictions[i] = True; retained_count += 1; continue
                if suppress_isolated_points:
                    start_idx = max(0, i - temporal_consistency_window // 2)
                    end_idx = min(len(raw_anomalies), i + temporal_consistency_window // 2 + 1)
                    window_slice = raw_anomalies[start_idx:end_idx]
                    anomaly_count = np.sum(window_slice)
                    window_size_actual = len(window_slice) 
                    consistency_ratio = anomaly_count / window_size_actual if window_size_actual > 0 else 0
                    if anomaly_count >= min_anomaly_cluster_size or consistency_ratio >= min_consistency_ratio:
                        predictions[i] = True; retained_count += 1
                    else:
                        predictions[i] = False; suppressed_count += 1
            else:
                 predictions[i] = False
        if anomaly_region_emphasis: # This block is usually False by default in original code
             print("Applying gap filling...") 
             structure = np.ones(3) 
             dilated = ndimage.binary_dilation(predictions, structure=structure)
             eroded = ndimage.binary_erosion(dilated, structure=structure) 
             added_points_mask = eroded & (~predictions)
             added_count = np.sum(added_points_mask)
             predictions = eroded 
             print(f"Noise suppression results: {suppressed_count} suppressed, {retained_count} retained, {added_count} added by gap filling")
        else:
            print(f"Noise suppression results: {suppressed_count} suppressed, {retained_count} retained.")
    else:
         print("Skipping temporal consistency filtering.")

    if use_duration_hypothesis and len(raw_anomalies) > 0:
        print("Applying supplementary duration-based hypothesis testing...")
        try:
            duration_anomalies, duration_metadata = apply_duration_based_hypothesis_testing(
                final_scores_smoothed, aligned_map_full, window_size=hypothesis_window_size,
                min_duration=min_hypothesis_duration, score_smoothing_window=smoothing_window,
                confidence_threshold=0.99
            )
            if 'valid_regions' in duration_metadata and len(duration_metadata['valid_regions']) > 0:
                added_by_duration = 0
                for region in duration_metadata['valid_regions']:
                    region_mask_dur = np.zeros(len(predictions), dtype=bool)
                    start_idx = region['start']; end_idx = min(region['end'] + 1, len(predictions))
                    if start_idx < 0 or end_idx > len(predictions) or start_idx >= end_idx: continue
                    region_mask_dur[start_idx:end_idx] = True
                    overlap = np.sum(predictions & region_mask_dur)
                    region_coverage = overlap / max(1, np.sum(region_mask_dur))
                    if region['size'] >= min_anomaly_cluster_size * 2 and region_coverage < 0.3:
                        print(f"  Adding missed anomaly region from duration analysis: start={start_idx}, end={end_idx}, size={region['size']}")
                        predictions[region_mask_dur] = True; added_by_duration += np.sum(region_mask_dur)
                        if not threshold_method.endswith(" +DurHyp"): threshold_method += " +DurHyp"
                print(f"  Added {added_by_duration} points based on Duration Hypothesis testing.")
        except NameError: print("Warning: `apply_duration_based_hypothesis_testing` function not found. Skipping.")
        except Exception as e: print(f"Error during duration hypothesis testing: {e}")
    return predictions, threshold_info, threshold_method, final_scores_smoothed


def generate_plots(stream, test_telemetry, final_scores_smoothed, threshold, predictions,
                   binary_labels, substitution_map, memory_bank, threshold_method,
                   smoothing_window, channel_results_dir):
    """(Keep existing code - Matplotlib/NumPy based)"""
    # ... (Keep existing code) ...
    print("Generating visualization plots...")
    if len(test_telemetry) == 0: print("No data to plot."); return
    min_len = min(len(binary_labels), len(predictions), len(test_telemetry))
    plot_test_data = test_telemetry[:min_len].reshape(-1, 1)
    plot_scores = final_scores_smoothed[:min_len]
    plot_predictions = predictions[:min_len]
    plot_true_anomalies = binary_labels[:min_len]
    score_dist_path = os.path.join(channel_results_dir, f"{stream}_score_distribution.svg")
    visualizer.plot_score_distribution(final_scores_smoothed, threshold, score_dist_path, channel_name=stream)
    plt_fig = plot_anomaly_detection_results(
        test_data=plot_test_data, scores=plot_scores, threshold=threshold,
        predictions=plot_predictions, true_anomalies=plot_true_anomalies,
        title=f"NCAD-TCN Detector Results for {stream} (Method: {threshold_method}, DynamicSmooth: {smoothing_window})"
    )
    plot_path = os.path.join(channel_results_dir, f"{stream}_detection_ncad_tcn.svg")
    plt_fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Main plot saved to {plot_path}")
    plt.close(plt_fig)


def process_channel_enhanced(stream, full_window_size=248, context_size=224, step=1,
                    plot_results=True, epochs=20, batch_size=64, latent_dim=256, 
                    n_clusters=12, percentile=98.0, 
                    mem_bank_threshold_method='percentile', 
                    mem_bank_iqr_factor=3.0,
                    base_threshold_percentile=99.0, 
                    density_based_adjustment=False, 
                    substitution_boost_factor=2.0,   
                    use_silhouette_clustering=True,
                    min_clusters=2,
                    max_clusters=50,
                    smoothing_window=100, # Changed smoothing_alpha to smoothing_window
                    distance_metric='euclidean',
                    enable_change_detection=True,
                    enable_transition_detection=True,
                    use_complexity_thresholding=False,                    
                    use_regime_aware_mb_threshold=False,                    
                    use_transition_substitution_gating=False,                    
                    use_duration_hypothesis=False,                    
                    constant_signal_factor=3.0,                    
                    periodic_signal_factor=1.5,                    
                    regime_std_threshold=0.20,                    
                    high_variance_threshold_factor=2.5,                    
                    transition_gating_threshold=1.8,                    
                    use_multimodal_detection=False, 
                    min_hypothesis_duration=200,
                    hypothesis_window_size=40,
                    is_ablation_run=False
                    ): 
    
    print(f"\n{'='*80}")
    print(f"Processing channel {stream} with Enhanced NCAD-TCN (PyTorch Backend)")
    print(f"{'='*80}")

    channel_results_dir = os.path.join("enhanced_ncad_tcn_v3.5_results", stream) # Keep dir name for consistency
    os.makedirs(channel_results_dir, exist_ok=True)
    parent_dir = os.path.dirname(os.getcwd()) # Assumes running from 'notebooks' or similar subfolder


    train_telemetry, test_telemetry, _, _ = load_and_preprocess_data(stream, parent_dir)
    if train_telemetry is None: return None
    
    if use_multimodal_detection:
        print("Using enhanced multimodal feature extraction...")
        feature_extractor = TemporalFeatureExtractor(
            window_sizes=[10, 30, 60, 120], fft_components=8, long_window_size=150
        )
        processed_train, processed_test = extract_features(train_telemetry, test_telemetry) # This needs to use the new extractor
    else:
        processed_train, processed_test = extract_features(train_telemetry, test_telemetry) # Standard
        
    if processed_train is None: return None
    binary_labels = load_ground_truth(stream, parent_dir, len(processed_test))
    if binary_labels is None: return None

    train_windows_full_np = create_sliding_windows(processed_train, full_window_size, step)
    if train_windows_full_np.shape[0] == 0: return None
   
    val_split = 0.1
    num_val_samples = int(len(train_windows_full_np) * val_split)
    val_windows_full_np = None # Initialize
    if num_val_samples > 0 and len(train_windows_full_np) - num_val_samples > 1:
        val_windows_full_np = train_windows_full_np[-num_val_samples:]
        train_windows_full_np = train_windows_full_np[:-num_val_samples]
        print(f"Using {len(train_windows_full_np)} train windows, {len(val_windows_full_np)} val windows.")
    else:
        print(f"Using {len(train_windows_full_np)} train windows (no validation split).")

    input_dim = processed_train.shape[1] if len(processed_train.shape) > 1 else 1
      # Call PyTorch build_and_train_model
    encoder, history = build_and_train_model(
        train_windows_np=train_windows_full_np, # Pass NumPy arrays
        val_windows_np=val_windows_full_np,     # Pass NumPy arrays
        context_size=context_size, input_dim=input_dim, latent_dim=latent_dim,
        epochs=epochs, batch_size=batch_size, distance_metric=distance_metric,
        pooling_strategy="hybrid"  # Use hybrid pooling by default (last + traditional)
    )

    loss_plot_path = os.path.join(channel_results_dir, f"{stream}_loss_curve.svg")
    visualizer.plot_loss_curves(history, loss_plot_path)

    # Call PyTorch aware build_memory_bank (conditionally)
    memory_bank = None
    normal_embeddings_for_plot = None
    if not is_ablation_run:
        memory_bank, normal_embeddings_for_plot = build_memory_bank( # Renamed to avoid conflict
            encoder=encoder, processed_train_np=processed_train, context_size=context_size,
            step=step, batch_size=batch_size, n_clusters=n_clusters,
            use_silhouette_clustering=use_silhouette_clustering, min_clusters=min_clusters,
            max_clusters=max_clusters, mem_bank_threshold_method=mem_bank_threshold_method,
            mem_bank_iqr_factor=mem_bank_iqr_factor, percentile=percentile,
            distance_metric=distance_metric,
            use_complexity_thresholding=use_complexity_thresholding,
            constant_signal_factor=constant_signal_factor,
            periodic_signal_factor=periodic_signal_factor
        )
    else:
        print("\n--- SKIPPING MEMORY BANK CONSTRUCTION (Ablation Mode) ---\n")

    if memory_bank and hasattr(memory_bank, 'centroids') and memory_bank.centroids.shape[0] > 0 and \
       normal_embeddings_for_plot is not None and normal_embeddings_for_plot.shape[0] > 0:
        mb_plot_path = os.path.join(channel_results_dir, f"{stream}_memory_bank_pca.svg")
        visualizer.plot_memory_bank(memory_bank.centroids, normal_embeddings_for_plot, mb_plot_path, method='pca')
        mb_tsne_plot_path = os.path.join(channel_results_dir, f"{stream}_memory_bank_tsne.svg")
        visualizer.plot_memory_bank(memory_bank.centroids, normal_embeddings_for_plot, mb_tsne_plot_path, method='tsne')

    test_windows_full_np = create_sliding_windows(processed_test, full_window_size, step)
    if test_windows_full_np.shape[0] == 0: return None

    print(f"--- Using Improved Scoring with GLOBAL MB Threshold (Metric: {distance_metric}) (PyTorch) ---")
    # Call PyTorch aware score_windows...
    final_scores_window, substitution_flags_window, window_start_indices, bank_refs_used, \
    total_windows_scored, original_distances_window, embedding_changes_window, transition_scores_window, \
    substitution_confidence_scores_window, confidence_stats = score_windows_with_improved_mb_override(
        encoder=encoder, test_windows_np=test_windows_full_np, context_size=context_size,             
        memory_bank=memory_bank, step=step, batch_size=batch_size, distance_metric=distance_metric,        
        substitution_boost_factor=substitution_boost_factor,
        enable_change_detection=enable_change_detection,
        enable_transition_detection=enable_transition_detection,
        # Pass other relevant flags from args
        use_regime_aware_mb_threshold=use_regime_aware_mb_threshold,
        use_complexity_thresholding=use_complexity_thresholding,
        is_ablation_run=is_ablation_run
    )

    final_scores_window = np.array(final_scores_window) # ensure numpy
    if np.any(~np.isfinite(final_scores_window)):
        print(f"Warning: Non-finite scores detected after Pytorch scoring. Replacing.")
        finite_scores_win = final_scores_window[np.isfinite(final_scores_window)]
        replacement_value = np.percentile(finite_scores_win, 99.9) * 1.1 if len(finite_scores_win) > 10 else 1e9
        final_scores_window = np.nan_to_num(final_scores_window, nan=replacement_value, posinf=replacement_value, neginf=replacement_value)
    
    print("Creating point-level context abnormality map (substitution_map)...")
    substitution_map = np.zeros(len(processed_test), dtype=bool) 
    if window_start_indices is not None and substitution_flags_window is not None and len(window_start_indices) == len(substitution_flags_window):
        flagged_window_count = 0
        for i, flag in enumerate(substitution_flags_window): 
            if flag: 
                flagged_window_count += 1
                window_start = window_start_indices[i] 
                suspect_start_in_map = window_start + context_size
                suspect_end_in_map = window_start + full_window_size 
                valid_start = max(0, suspect_start_in_map)
                valid_end = min(len(substitution_map), suspect_end_in_map)
                if valid_start < valid_end: 
                    substitution_map[valid_start:valid_end] = True 
        print(f"Mapped {flagged_window_count} abnormal context window flags to point-level map.")
        print(f"Total points flagged as having abnormal context in map: {np.sum(substitution_map)}")
    # ... (rest of the substitution_map creation error handling) ...
    elif window_start_indices is None: print("Warning: window_start_indices is None. Cannot create substitution map.")
    elif substitution_flags_window is None: print("Warning: substitution_flags_window is None. Cannot create substitution map.")
    else: print(f"Warning: Mismatch lengths for substitution map. Indices: {len(window_start_indices)}, Flags: {len(substitution_flags_window)}")


    context_sub_perc = 100.0 * bank_refs_used / max(total_windows_scored, 1)
    print(f"Context Substitution Percentage (Window Level): {context_sub_perc:.2f}% ({bank_refs_used}/{total_windows_scored})")

    finite_final_scores_win = final_scores_window[np.isfinite(final_scores_window)]
    if len(finite_final_scores_win) > 0:
        print(f"Raw Window Scores - Min: {np.min(finite_final_scores_win):.4f}, Max: {np.max(finite_final_scores_win):.4f}, Mean: {np.mean(finite_final_scores_win):.4f}")
    else:
        print("Raw Window Scores - No finite scores to analyze.")

    print("--- Final Aggregation & Evaluation ---")
    final_agg_scores = aggregate_scores(final_scores_window, processed_test, full_window_size, context_size, step)

    finite_agg_scores = final_agg_scores[np.isfinite(final_agg_scores)]
    if len(finite_agg_scores) > 0:
        print(f"Aggregated Point Scores - Min: {np.min(finite_agg_scores):.4f}, Max: {np.max(finite_agg_scores):.4f}, Mean: {np.mean(finite_agg_scores):.4f}")
    else:
         print("Aggregated Point Scores - No finite scores to analyze.")
   
    predictions, threshold_info, threshold_method, final_scores_smoothed = apply_adaptive_threshold_and_evaluate(
        final_agg_scores=final_agg_scores,
        binary_labels=binary_labels,
        context_abnormal_map=substitution_map, 
        smoothing_window=smoothing_window, # Pass smoothing_window
        base_threshold_percentile=base_threshold_percentile,
        min_anomaly_cluster_size=3, 
        strong_anomaly_factor=1.8,        
        temporal_consistency_window=20,   
        min_consistency_ratio=0.6,        
        noise_suppression=True,           
        suppress_isolated_points=True,    
        anomaly_region_emphasis=False,    
        use_duration_hypothesis=use_duration_hypothesis,
        min_hypothesis_duration=min_hypothesis_duration,
        hypothesis_window_size=hypothesis_window_size
    )
    
    metrics_dict = {
        'TP': np.nan, 'TN': np.nan, 'FP': np.nan, 'FN': np.nan,
        'Precision': np.nan, 'Recall': np.nan, 'F1': np.nan
    }

    min_len_plot = len(test_telemetry)
    if substitution_map is not None: min_len_plot = min(min_len_plot, len(substitution_map))
    if final_scores_smoothed is not None: min_len_plot = min(min_len_plot, len(final_scores_smoothed))
    if predictions is not None: min_len_plot = min(min_len_plot, len(predictions))
    if binary_labels is not None: min_len_plot = min(min_len_plot, len(binary_labels))
    num_windows_plot = len(window_start_indices) if window_start_indices is not None else 0


    if predictions is not None and binary_labels is not None:
       
        min_len_eval = min(len(predictions), len(binary_labels))
        y_true = binary_labels[:min_len_eval]
        y_pred = predictions[:min_len_eval]
        if len(y_true) > 0: 
            try:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                metrics_dict = {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn, 'Precision': precision, 'Recall': recall, 'F1': f1}
                print(f"Metrics for {stream}: TP={tp}, TN={tn}, FP={fp}, FN={fn}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
            except Exception as e:
                print(f"Error calculating metrics for {stream}: {e}")
        else:
            print(f"Warning: No data points available for metric calculation for {stream}.")

    diagnostics_path = os.path.join(channel_results_dir, f"{stream}_substitution_diagnostics.svg")
    if (test_telemetry is not None and substitution_map is not None and
        original_distances_window is not None and substitution_confidence_scores_window is not None and
        window_start_indices is not None):
        create_substitution_diagnostics(
            test_telemetry=test_telemetry[:min_len_plot].reshape(-1),
            substitution_flags=substitution_map[:min_len_plot],
            true_anomalies=binary_labels[:min_len_plot] if binary_labels is not None else None,
            predictions=predictions[:min_len_plot] if predictions is not None else None,
            original_distances=original_distances_window[:num_windows_plot], 
            confidence_scores=substitution_confidence_scores_window[:num_windows_plot],
            embedding_changes=embedding_changes_window[:num_windows_plot] if embedding_changes_window is not None else None,
            transition_scores=transition_scores_window[:num_windows_plot] if transition_scores_window is not None else None,
            window_start_indices=window_start_indices, step=step,
            threshold_value=memory_bank.threshold if memory_bank and hasattr(memory_bank, 'threshold') else None,
            title=f"Substitution Decision Diagnostics for {stream}", save_path=diagnostics_path
        )
    else:
        print("Skipping substitution diagnostics plot due to missing essential data.")

    if plot_results:
        if (final_scores_smoothed is not None and predictions is not None and
            binary_labels is not None and substitution_map is not None):
            generate_enhanced_plots_with_changes(
                    stream, test_telemetry[:min_len_plot], final_scores_smoothed[:min_len_plot],
                    threshold_info, predictions[:min_len_plot], binary_labels[:min_len_plot],
                    substitution_map[:min_len_plot], 
                    original_distances_window[:num_windows_plot] if original_distances_window is not None else None, 
                    embedding_changes_window[:num_windows_plot] if embedding_changes_window is not None else None, 
                    transition_scores_window[:num_windows_plot] if transition_scores_window is not None else None,
                    memory_bank, threshold_method, smoothing_window, channel_results_dir, step # Pass smoothing_window
            )
            generate_plots(
                stream=stream, test_telemetry=test_telemetry[:min_len_plot],
                final_scores_smoothed=final_scores_smoothed[:min_len_plot],
                threshold=threshold_info.get('threshold_value', np.inf),
                predictions=predictions[:min_len_plot], binary_labels=binary_labels[:min_len_plot],
                substitution_map=substitution_map[:min_len_plot], memory_bank=memory_bank,
                threshold_method=threshold_method, smoothing_window=smoothing_window, # Pass smoothing_window
                channel_results_dir=channel_results_dir
            )
        else:
             print("Skipping main results plotting due to missing essential data.")
    return metrics_dict