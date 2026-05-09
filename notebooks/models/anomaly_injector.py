import numpy as np

class ContextualAnomalyInjector:
    """
    Injects diverse contextual anomalies into time series windows for training.

    Anomalies are primarily injected into the 'suspect' portion following the context.
    Magnitude is often scaled relative to the standard deviation of the context part.
    'stuck_value' anomalies are preferentially injected into noisier segments.
    Spike anomalies are always 1 point long.
    """
    def __init__(self,
                 suspect_injection_ratio=0.5,
                 anomaly_types=['spike', 'level_shift', 'variance_change', 'stuck_value'],
                 anomaly_probs=None, # If None, equal probability
                 min_anomaly_len=5,  # Default min length for non-spike anomalies
                 max_anomaly_len=15, # Default max length for non-spike anomalies
                 spike_factor_range=(3.0, 7.0),
                 level_shift_factor_range=(1.5, 4.0),
                 variance_factor_range=(1.5, 4.0),
                 stuck_val_prob_context=0.5,
                 stuck_value_noise_threshold=0.1):
        self.suspect_injection_ratio = suspect_injection_ratio
        self.anomaly_types = anomaly_types
        self.min_anomaly_len = min_anomaly_len # For non-spike anomalies
        self.max_anomaly_len = max_anomaly_len # For non-spike anomalies
        self.spike_factor_range = spike_factor_range
        self.level_shift_factor_range = level_shift_factor_range
        self.variance_factor_range = variance_factor_range
        self.stuck_val_prob_context = stuck_val_prob_context
        self.stuck_value_noise_threshold = stuck_value_noise_threshold

        if anomaly_probs is None:
            self.anomaly_probs = np.ones(len(anomaly_types)) / len(anomaly_types)
        elif len(anomaly_probs) == len(anomaly_types) and np.isclose(sum(anomaly_probs), 1.0):
            self.anomaly_probs = anomaly_probs
        else:
            raise ValueError("anomaly_probs must have the same length as anomaly_types and sum to 1.")

        self.max_anomaly_len = max(self.min_anomaly_len, self.max_anomaly_len)

        self.type_to_prob = dict(zip(self.anomaly_types, self.anomaly_probs))
        if 'stuck_value' in self.type_to_prob:
            self.stuck_value_idx = self.anomaly_types.index('stuck_value')
            self.prob_stuck = self.type_to_prob['stuck_value']
            self.types_no_stuck = [t for t in self.anomaly_types if t != 'stuck_value']
            probs_no_stuck_unnormalized = [self.type_to_prob[t] for t in self.types_no_stuck]
            sum_probs_no_stuck = sum(probs_no_stuck_unnormalized)
            if sum_probs_no_stuck > 1e-9:
                self.probs_no_stuck_normalized = [p / sum_probs_no_stuck for p in probs_no_stuck_unnormalized]
            else:
                 self.types_no_stuck = self.anomaly_types
                 self.probs_no_stuck_normalized = self.anomaly_probs
        else:
            self.stuck_value_idx = -1
            self.prob_stuck = 0.0
            self.types_no_stuck = self.anomaly_types
            self.probs_no_stuck_normalized = self.anomaly_probs


    def inject_anomalies(self, batch_windows_full_np, context_size):
        batch_size, full_window_size, num_features = batch_windows_full_np.shape
        modified_windows = batch_windows_full_np.copy()
        labels = np.zeros(batch_size)

        num_to_inject = int(batch_size * self.suspect_injection_ratio)
        if num_to_inject == 0 and batch_size > 0:
             num_to_inject = 1
        injection_indices = np.random.choice(batch_size, num_to_inject, replace=False)

        for idx in injection_indices:
            context_part = modified_windows[idx, :context_size, :] # Use modified_windows to get context part
            anomaly_injected_this_window = False

            # --- Determine Anomaly Start Location (common for all features in this window) ---
            # Smallest possible anomaly (a spike) is 1 point long.
            min_len_for_placement_check = 1
            suspect_window_len = full_window_size - context_size

            if suspect_window_len < min_len_for_placement_check:
                continue # Suspect window too small

            # max_start_offset allows anomaly to start anywhere in suspect window if its length is min_len_for_placement_check
            max_start_offset = suspect_window_len - min_len_for_placement_check
            
            current_start_offset = 0
            if max_start_offset > 0 : # check if max_start_offset is positive to avoid error with randint
                current_start_offset = np.random.randint(0, max_start_offset + 1)
            elif max_start_offset == 0: # if max_start_offset is 0, start_offset must be 0
                current_start_offset = 0
            else: # Should not happen due to suspect_window_len check, but as a fallback
                continue

            start_index = context_size + current_start_offset

            for feat_idx in range(num_features):
                context_std = np.std(context_part[:, feat_idx]); context_std = max(context_std, 1e-6)
                context_mean = np.mean(context_part[:, feat_idx])

                # --- Determine Anomaly Type for this feature ---
                feat_anomaly_type = ""
                is_noisy = context_std >= self.stuck_value_noise_threshold
                if not is_noisy and self.stuck_value_idx != -1 and len(self.types_no_stuck) > 0:
                    feat_anomaly_type = np.random.choice(self.types_no_stuck, p=self.probs_no_stuck_normalized)
                else:
                    feat_anomaly_type = np.random.choice(self.anomaly_types, p=self.anomaly_probs)

                # --- Determine Anomaly Length for this type and feature ---
                # Default to general min/max length
                current_min_len_for_type = self.min_anomaly_len
                current_max_len_for_type = self.max_anomaly_len
                if feat_anomaly_type == 'spike': # Spikes are always 1 point long
                    current_min_len_for_type = 1
                    current_max_len_for_type = 1

                max_len_available_from_start = full_window_size - start_index
                effective_max_len_for_type = min(current_max_len_for_type, max_len_available_from_start)

                if effective_max_len_for_type < current_min_len_for_type:
                    continue # Not enough space for this anomaly type at this start_index

                actual_anomaly_len = np.random.randint(current_min_len_for_type, effective_max_len_for_type + 1)
                end_index = start_index + actual_anomaly_len

                # --- Inject Anomaly ---
                original_feature_segment = modified_windows[idx, start_index:end_index, feat_idx].copy()
                
                if feat_anomaly_type == 'spike':
                    factor = np.random.uniform(*self.spike_factor_range); direction = np.random.choice([-1, 1])
                    magnitude = direction * factor * context_std
                    injected_segment = self._inject_spike(original_feature_segment, magnitude)
                elif feat_anomaly_type == 'level_shift':
                    factor = np.random.uniform(*self.level_shift_factor_range); direction = np.random.choice([-1, 1])
                    magnitude = direction * factor * context_std
                    injected_segment = self._inject_level_shift(original_feature_segment, magnitude)
                elif feat_anomaly_type == 'variance_change':
                    increase = np.random.choice([True, False]); factor = np.random.uniform(*self.variance_factor_range)
                    if not increase: factor = 1.0 / factor
                    injected_segment = self._inject_variance_change(original_feature_segment, factor, context_mean)
                elif feat_anomaly_type == 'stuck_value':
                    use_context_val = np.random.rand() < self.stuck_val_prob_context
                    if use_context_val and len(original_feature_segment) > 0:
                        stuck_value = original_feature_segment[0]
                    else:
                        stuck_value = np.random.choice([0.0, np.min(context_part[:, feat_idx]), np.max(context_part[:, feat_idx])])
                    injected_segment = self._inject_stuck_value(original_feature_segment, stuck_value)
                else:
                    injected_segment = original_feature_segment

                modified_windows[idx, start_index:end_index, feat_idx] = injected_segment
                anomaly_injected_this_window = True

            if anomaly_injected_this_window:
                labels[idx] = 1
        return modified_windows, labels

    def _inject_variance_change(self, segment, factor, mean_ref):
        segment_copy = segment.copy()
        segment_copy = mean_ref + factor * (segment_copy - mean_ref)
        return segment_copy

    def _inject_spike(self, segment, magnitude):
        if len(segment) == 0: return segment
        segment_copy = segment.copy()
        # Spike is injected at a random point within the segment.
        # If segment is length 1 (for spike type), spike_idx will be 0.
        spike_idx = np.random.randint(0, len(segment_copy))
        segment_copy[spike_idx] += magnitude
        return segment_copy

    def _inject_level_shift(self, segment, magnitude):
        return segment + magnitude

    def _inject_stuck_value(self, segment, value):
        segment_copy = segment.copy()
        segment_copy[:] = value
        return segment_copy
