import numpy as np
# REMOVE: import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
from utils.signal_complexity import classify_enhanced_pattern_type, calculate_enhanced_complexity_metrics, calculate_adaptive_threshold_enhanced
from scipy.spatial.distance import cdist

# Keep the custom cosine_distances function as it was
def cosine_distances(X, Y=None):
    """
    Compute cosine distances using scipy.spatial.distance.cdist for robustness.
    Distance = 1 - similarity.

    Args:
        X: array of shape (n_samples_X, n_features)
        Y: array of shape (n_samples_Y, n_features), optional. If None, computes distance with itself.

    Returns:
        Array of cosine distances.
    """
    # Handle case where Y is None (distance of X with itself)
    if Y is None:
        Y = X

    # Ensure inputs are float64 for cdist stability if needed
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    # Handle empty arrays edge case before calling cdist
    if X.shape[0] == 0 or Y.shape[0] == 0:
         return np.empty((X.shape[0], Y.shape[0]))
    if X.shape[1] != Y.shape[1]:
         raise ValueError("X and Y must have the same number of features.")

    try:
        # cdist calculates similarity, we want distance (1 - similarity)
        # It handles normalization internally and should be more robust
        return cdist(X, Y, metric='cosine')
    except ValueError as e:
         # Catch potential issues like zero vectors if cdist fails
         print(f"Warning: scipy.spatial.distance.cdist failed: {e}. Falling back to NaN.")
         # Return array of NaNs or handle as appropriate
         return np.full((X.shape[0], Y.shape[0]), np.nan)

class ContextMemoryBank:
    # --- MODIFICATION: Added `normal_context_windows` to the constructor ---
    def __init__(self, normal_embeddings, normal_context_windows, n_clusters=100, auto_cluster=True,
                 min_clusters=2, max_clusters=100, distance_metric='euclidean',
                 use_complexity_thresholding=True,
                 constant_signal_factor=3.0,
                 periodic_signal_factor=1.5):
        """
        Initializes and builds the memory bank using K-Means with optimal cluster selection.

        Args:
            normal_embeddings: Numpy array of embeddings from NORMAL context windows.
            n_clusters: Default number of clusters to use if auto_cluster is False.
            auto_cluster: Whether to automatically determine optimal number of clusters.
            min_clusters: Minimum number of clusters to try when auto_clustering.
            max_clusters: Maximum number of clusters to try when auto_clustering.
            distance_metric: Distance metric to use ('euclidean' or 'cosine').
            use_complexity_thresholding: Whether to use signal complexity for adaptive thresholding
            constant_signal_factor: Threshold multiplier for constant signals
            periodic_signal_factor: Threshold multiplier for periodic signals
        """
        self.n_clusters = 0  # Initialize
        self.auto_cluster = auto_cluster
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.distance_metric = distance_metric
        # Store new parameters
        self.use_complexity_thresholding = use_complexity_thresholding
        self.constant_signal_factor = constant_signal_factor
        self.periodic_signal_factor = periodic_signal_factor
        self.use_continuous_adaptation = True
        self.distance_history = []
        self.distance_history_max_len = 500  # Store up to 500 recent distances
        self.signal_type_cache = {}  # Cache signal type classifications
        self.last_threshold = None    # Track the last computed threshold

        # --- ADDITION: This will store the raw windows for each centroid ---
        self.representative_windows = None
        # --- END ADDITION ---

        if normal_embeddings is not None and len(normal_embeddings) > 0:
            if auto_cluster:
                # Determine optimal number of clusters
                optimal_n_clusters = self._find_optimal_clusters(normal_embeddings)
                print(f"Silhouette analysis selected {optimal_n_clusters} clusters as optimal.")
                self.n_clusters = optimal_n_clusters
            else:
                self.n_clusters = min(n_clusters, len(normal_embeddings))
        else:
            print("Warning: No normal embeddings provided for memory bank init.")

        # --- MODIFICATION: Pass raw windows to _build_bank ---
        self.centroids = self._build_bank(normal_embeddings, normal_context_windows)
        # --- END MODIFICATION ---
        self.threshold = None  # Threshold to determine if a context is "unusual" (used in first pass)
        self.threshold_method = None  # Store how the threshold was determined

    # --- Keep _find_optimal_clusters, _plot_silhouette_scores, _spherical_kmeans, _build_bank ---
    # --- set_threshold_from_data, _plot_distance_distribution methods as they were ---
    # Make sure set_threshold_from_data correctly handles the 'percentile' method you switched to.

    def _find_optimal_clusters(self, embeddings):
        """
        Find the optimal number of clusters using silhouette score with minimum enforcement.
        (Keep this method as previously defined)
        """
        # ... (implementation from previous versions) ...
        min_enforced_clusters = 5
        if len(embeddings) < self.min_clusters: return len(embeddings)
        effective_max_clusters = min(self.max_clusters, len(embeddings) - 1)
        effective_min_clusters = min(self.min_clusters, effective_max_clusters)
        if effective_max_clusters <= effective_min_clusters: return effective_min_clusters
        if (effective_max_clusters - effective_min_clusters) > 10:
            step_size = max(1, (effective_max_clusters - effective_min_clusters) // 10)
            cluster_range = list(range(effective_min_clusters, effective_max_clusters + 1, step_size))
        else:
            cluster_range = list(range(effective_min_clusters, effective_max_clusters + 1))
        print(f"Evaluating silhouette scores for {len(cluster_range)} different cluster counts...")
        silhouette_scores = []
        kmeans_models = []
        max_sample_size = 10000
        if len(embeddings) > max_sample_size:
            indices = np.random.choice(len(embeddings), max_sample_size, replace=False)
            sample_embeddings = embeddings[indices]
        else:
            sample_embeddings = embeddings
        metric = 'euclidean'
        if self.distance_metric == 'cosine': metric = 'cosine'
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(sample_embeddings)
            if len(np.unique(cluster_labels)) > 1:
                score = silhouette_score(sample_embeddings, cluster_labels, metric=metric)
                silhouette_scores.append(score)
                kmeans_models.append(kmeans)
                print(f"  Clusters: {n_clusters}, Silhouette Score: {score:.4f}")
            else:
                print(f"  Clusters: {n_clusters}, Skipped (degenerate clustering)")
                silhouette_scores.append(-1)
                kmeans_models.append(None)
        valid_scores = [s for s in silhouette_scores if s > -1]
        if valid_scores:
            self._plot_silhouette_scores(cluster_range, silhouette_scores)
            valid_indices = [i for i, s in enumerate(silhouette_scores) if s > -1]
            best_idx = valid_indices[np.argmax([silhouette_scores[i] for i in valid_indices])]
            optimal_clusters = cluster_range[best_idx]
            if optimal_clusters < min_enforced_clusters:
                higher_indices = [i for i in valid_indices if cluster_range[i] >= min_enforced_clusters]
                if higher_indices:
                    best_higher_idx = higher_indices[np.argmax([silhouette_scores[i] for i in higher_indices])]
                    print(f"Enforcing minimum of {min_enforced_clusters} clusters (original optimal was {optimal_clusters} with score {silhouette_scores[best_idx]:.4f})")
                    print(f"New optimal: {cluster_range[best_higher_idx]} clusters with score {silhouette_scores[best_higher_idx]:.4f}")
                    optimal_clusters = cluster_range[best_higher_idx]
                else:
                    print(f"No valid scores for clusters >= {min_enforced_clusters}. Keeping original optimal: {optimal_clusters}")
            return optimal_clusters
        else:
            print("No valid silhouette scores found. Using default cluster count.")
            return min(self.n_clusters, len(embeddings))

    # Add to memory_bank.py class ContextMemoryBank
    def track_embedding_changes(self, embeddings, window_size=5, sensitivity=1.5):
        """
        Track changes in context embeddings over time to detect transitions.
        
        Args:
            embeddings: List of context embeddings in temporal order
            window_size: Number of previous embeddings to consider
            sensitivity: Multiplier for change threshold
            
        Returns:
            change_scores: List of embedding change scores
            change_flags: Boolean flags indicating significant changes
        """
        if len(embeddings) < 2:
            return [0.0], [False]
            
        # Calculate distances between consecutive embeddings
        change_scores = []
        history = []
        
        for i, emb in enumerate(embeddings):
            if i > 0:
                # Calculate distance from previous embedding
                if self.distance_metric == 'cosine':
                    dist = cosine_distances(emb.reshape(1, -1), embeddings[i-1].reshape(1, -1))[0][0]
                else:
                    dist = euclidean_distances(emb.reshape(1, -1), embeddings[i-1].reshape(1, -1))[0][0]
                change_scores.append(dist)
                history.append(dist)
                
                # Keep history at window_size
                if len(history) > window_size:
                    history.pop(0)
            else:
                change_scores.append(0.0)
                
        # Set adaptive threshold based on recent history
        change_flags = []
        for i, score in enumerate(change_scores):
            if i < window_size:
                # Not enough history yet
                threshold = 0.5  # Initial threshold
            else:
                # Set threshold as multiple of median recent change
                recent_changes = change_scores[max(0, i-window_size):i]
                median_change = np.median(recent_changes) if recent_changes else 0.1
                threshold = max(0.1, median_change * sensitivity)
                
            change_flags.append(score > threshold)
            
        return change_scores, change_flags
        
    def detect_window_transitions(self, context_windows, n_segments=4, threshold=2.5):
        """
        Detect transitions within individual context windows by splitting them
        into segments and looking for significant changes between segments.
        
        Args:
            context_windows: List of context windows (time series)
            n_segments: Number of segments to divide each window into
            threshold: Z-score threshold for significant transitions
            
        Returns:
            transition_scores: Transition scores for each window
            transition_flags: Boolean flags indicating detected transitions
        """
        transition_scores = []
        transition_flags = []
        
        for window in context_windows:
            # Skip if window is too small
            if len(window) < n_segments*2:
                transition_scores.append(0.0)
                transition_flags.append(False)
                continue
                
            # Get the feature dimension (first column if multi-dimensional)
            signal = window[:, 0] if len(window.shape) > 1 else window
            
            # Divide window into segments
            segment_length = len(signal) // n_segments
            segments = []
            
            for i in range(n_segments):
                start_idx = i * segment_length
                end_idx = start_idx + segment_length
                segments.append(signal[start_idx:end_idx])
                
            # Calculate segment statistics
            seg_means = [np.mean(s) for s in segments]
            seg_stds = [max(np.std(s), 1e-6) for s in segments]  # Avoid division by zero
            
            # Detect transitions between adjacent segments
            max_transition = 0
            for i in range(1, len(segments)):
                # Calculate normalized difference (z-score)
                z_score = abs(seg_means[i] - seg_means[i-1]) / seg_stds[i-1]
                max_transition = max(max_transition, z_score)
                
            transition_scores.append(max_transition)
            transition_flags.append(max_transition > threshold)
            
        return transition_scores, transition_flags

    def adaptive_threshold(self, current_distance, context_window=None):
        """
        Dynamically adjust threshold based on recent distance history and signal characteristics.
        
        Args:
            current_distance: Current context distance value
            context_window: Raw context window data for signal type analysis
            
        Returns:
            Adjusted threshold value
        """
        # 1. Update distance history
        self.distance_history.append(current_distance)
        if len(self.distance_history) > self.distance_history_max_len:
            self.distance_history.pop(0)
            
        # Start with base threshold from initial setup
        if self.threshold is None:
            return 0.5  # Default fallback
        
        # 2. Determine signal type and appropriate adaptation parameters
        signal_type = "unknown"
        adaptation_factor = 1.0
        window_size = 100  # Number of recent points to consider
        
        if context_window is not None:
            # Use advanced signal classification
            metrics = calculate_enhanced_complexity_metrics(context_window)
            pattern_type, basic_type, confidence = classify_enhanced_pattern_type(metrics)
            signal_type = basic_type
            
            # Set adaptation parameters based on signal type
            if signal_type == "periodic":
                adaptation_factor = 0.7  # Lower threshold for periodic signals
                window_size = 50        # Use shorter window for faster adaptation
            elif signal_type == "constant":
                adaptation_factor = 1.5  # Higher threshold for constant signals
                window_size = 200       # Use longer window for stability
            elif signal_type == "complex":
                adaptation_factor = 1.0  # Standard threshold for complex signals
                window_size = 100       # Medium window size
        
        # 3. Calculate the adaptive threshold using recent history
        if len(self.distance_history) < 5:
            # Not enough history, use base threshold with type adjustment
            return self.threshold * adaptation_factor
            
        # Get recent distance values
        recent = self.distance_history[-min(window_size, len(self.distance_history)):]
        
        # Use robust statistics - median and IQR
        med_distance = np.median(recent)
        q1 = np.percentile(recent, 25)
        q3 = np.percentile(recent, 75)
        iqr = q3 - q1
        
        # Handle case where IQR is very small
        if iqr < 1e-6:
            # Fall back to standard deviation with a floor
            std_dev = max(np.std(recent), 0.01)
            adaptive_thresh = med_distance + (3.0 * std_dev * adaptation_factor)
        else:
            # Use IQR-based threshold (robust to outliers)
            adaptive_thresh = med_distance + (1.5 * iqr * adaptation_factor)
        
        print(f"  AdaptiveDebug: current_dist={current_distance:.4f}, base_thresh={self.threshold:.4f}, signal_type={signal_type}, factor={adaptation_factor:.2f}")
        if 'iqr' in locals() and iqr >= 1e-6 :
            print(f"  AdaptiveDebug: median={med_distance:.4f}, iqr={iqr:.4f}, calculated_thresh={adaptive_thresh:.4f} (IQR based)")
        elif 'std_dev' in locals():
            print(f"  AdaptiveDebug: median={med_distance:.4f}, std_dev={std_dev:.4f}, calculated_thresh={adaptive_thresh:.4f} (StdDev based)")


        # 4. Detect transition points (rapid change in distances)
        is_transition = False
        if len(recent) >= 20:
            # Calculate running average before and after current point
            before = np.mean(recent[-20:-10])
            after = np.mean(recent[-10:])
            # Calculate relative change
            relative_change = abs(after - before) / (before + 1e-10)
            # If large step change detected
            if relative_change > 0.3:  # 30% change threshold
                is_transition = True
                adaptive_thresh *= 0.5  # Halve threshold during transitions
                
        # Increase sensitivity during transitions
        if is_transition:
            adaptive_thresh *= 0.7  # Lower threshold temporarily during transitions
            print(f"  AdaptiveDebug: Transition detected! Relative change: {relative_change:.2f}. Threshold adjusted.")
            
        # 5. Apply limits to prevent extreme adaptation
        # - Never go below 50% of base threshold 
        # - Never go above 200% of base threshold
        min_thresh = self.threshold * 0.5
        max_thresh = self.threshold * 2.0
        alpha = 0.5  # Smoothing factor for adaptation

        # For periodic signals like A-3, allow even lower thresholds
        if signal_type == "periodic":
            min_thresh = self.threshold * 0.3
        
        adaptive_thresh = np.clip(adaptive_thresh, min_thresh, max_thresh)
        
        final_adaptive_thresh = np.clip(adaptive_thresh, min_thresh, max_thresh)
        if final_adaptive_thresh != adaptive_thresh: print(f"  AdaptiveDebug: Threshold clipped from {adaptive_thresh:.4f} to {final_adaptive_thresh:.4f}")
                
        # Apply smoothing if we have a previous threshold
        if self.last_threshold is not None:
            smoothed_thresh = (alpha * self.last_threshold) + ((1-alpha) * final_adaptive_thresh)
            print(f"  AdaptiveDebug: Smoothing applied. Final Thresh: {smoothed_thresh:.4f} (Prev: {self.last_threshold:.4f})")
            self.last_threshold = smoothed_thresh
            return smoothed_thresh
        else:
            self.last_threshold = final_adaptive_thresh
            return final_adaptive_thresh

    def _plot_silhouette_scores(self, cluster_range, silhouette_scores):
        """Plot silhouette scores to visualize optimal cluster selection."""
        # ... (implementation from previous versions) ...
        valid_indices = [i for i, score in enumerate(silhouette_scores) if score > -1]
        valid_clusters = [cluster_range[i] for i in valid_indices]
        valid_scores = [silhouette_scores[i] for i in valid_indices]
        if not valid_scores: return
        plt.figure(figsize=(10, 6))
        plt.plot(valid_clusters, valid_scores, '-o')
        plt.xlabel('Number of Clusters'); plt.ylabel('Silhouette Score'); plt.title('Silhouette Score by Cluster Count'); plt.grid(True)
        best_idx = np.argmax(valid_scores); best_clusters = valid_clusters[best_idx]; best_score = valid_scores[best_idx]
        plt.scatter(best_clusters, best_score, s=200, c='red', marker='*', label=f'Optimal: {best_clusters} clusters (score={best_score:.4f})'); plt.legend()
        os.makedirs('silhouette_analysis', exist_ok=True)
        plt.savefig(f'silhouette_analysis/silhouette_scores_{int(max(valid_scores)*10000)}.png', dpi=150); plt.close()
        print(f"Silhouette analysis plot saved to silhouette_analysis/")

    def _spherical_kmeans(self, embeddings, n_clusters, max_iter=100, tol=1e-4):
        """Implements spherical k-means for cosine similarity."""
        # ... (implementation from previous versions) ...
        n_samples, n_features = embeddings.shape
        norms = np.sqrt(np.sum(embeddings**2, axis=1, keepdims=True)) + 1e-10
        normalized_embeddings = embeddings / norms
        indices = np.arange(n_samples); first_idx = np.random.choice(indices); centroids = [normalized_embeddings[first_idx]]
        for _ in range(1, n_clusters):
            min_cos_dist = np.ones(n_samples)
            for centroid in centroids:
                cos_sim = np.dot(normalized_embeddings, centroid)
                cos_dist = 1.0 - cos_sim
                min_cos_dist = np.minimum(min_cos_dist, cos_dist)
            weights = min_cos_dist**2; weights /= np.sum(weights); next_idx = np.random.choice(indices, p=weights)
            centroids.append(normalized_embeddings[next_idx])
        centroids = np.array(centroids)
        for iteration in range(max_iter):
            cosine_sim = np.dot(normalized_embeddings, centroids.T); cosine_sim = np.clip(cosine_sim, -1.0, 1.0); assignments = np.argmax(cosine_sim, axis=1)
            old_centroids = centroids.copy()
            for j in range(n_clusters):
                cluster_points = normalized_embeddings[assignments == j]
                if len(cluster_points) > 0:
                    new_centroid = np.sum(cluster_points, axis=0); centroid_norm = np.sqrt(np.sum(new_centroid**2)) + 1e-10; centroids[j] = new_centroid / centroid_norm
            centroid_shift = np.sum(1.0 - np.sum(centroids * old_centroids, axis=1))
            if centroid_shift < tol: print(f"Spherical k-means converged after {iteration+1} iterations"); break
        cosine_sim = np.dot(normalized_embeddings, centroids.T); assignments = np.argmax(cosine_sim, axis=1); intra_cluster_sim = 0
        for j in range(n_clusters):
            cluster_points = normalized_embeddings[assignments == j]
            if len(cluster_points) > 0: cluster_sim = np.mean(np.dot(cluster_points, centroids[j])); intra_cluster_sim += cluster_sim * (len(cluster_points) / n_samples)
        print(f"Average intra-cluster cosine similarity: {intra_cluster_sim:.4f}")
        return centroids    # --- MODIFICATION: The _build_bank method now also saves representative raw windows ---
    def _build_bank(self, normal_embeddings, normal_context_windows):
        """
        Performs clustering to find prototypes and stores both the centroid embeddings
        and the raw context window closest to each centroid.
        """
        if self.n_clusters <= 0 or normal_embeddings is None or len(normal_embeddings) < self.n_clusters:
            if normal_embeddings is not None:
                print(f"Warning: Not enough embeddings ({len(normal_embeddings)}) for requested clusters ({self.n_clusters}).")
            return np.array([])

        print(f"Building memory bank with {self.n_clusters} clusters using {self.distance_metric} distance...")
       
        centroids = None
        labels = None

        # --- Perform Clustering ---
        if self.distance_metric == 'cosine':
            # Note: _spherical_kmeans doesn't return labels, so we calculate them after
            centroids = self._spherical_kmeans(normal_embeddings, self.n_clusters)
            # Calculate assignments to get labels
            cosine_sim = np.dot(normal_embeddings, centroids.T)
            labels = np.argmax(cosine_sim, axis=1)
            print("Memory bank built with spherical k-means (cosine distance).")
        else: # euclidean
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10, max_iter=300)
            kmeans.fit(normal_embeddings)
            centroids = kmeans.cluster_centers_
            labels = kmeans.labels_
            print("Memory bank built with standard k-means (Euclidean distance).")
       
        # --- ADDITION: Find and store the representative raw window for each centroid ---
        if centroids is not None and labels is not None:
            self.representative_windows = []
            for i in range(self.n_clusters):
                # Get all embeddings belonging to the current cluster
                cluster_member_indices = np.where(labels == i)[0]
               
                if len(cluster_member_indices) > 0:
                    cluster_embeddings = normal_embeddings[cluster_member_indices]
                   
                    # Find the embedding in this cluster that is closest to the centroid
                    if self.distance_metric == 'cosine':
                        distances_to_centroid = cosine_distances(cluster_embeddings, centroids[i].reshape(1, -1))
                    else:
                        distances_to_centroid = euclidean_distances(cluster_embeddings, centroids[i].reshape(1, -1))
                   
                    closest_member_idx_in_cluster = np.argmin(distances_to_centroid)
                   
                    # Get the original index (from before clustering) of this closest member
                    original_index = cluster_member_indices[closest_member_idx_in_cluster]
                   
                    # Use this original index to get the corresponding raw context window
                    representative_window = normal_context_windows[original_index]
                    self.representative_windows.append(representative_window)
                else:
                    # Handle empty cluster case, though unlikely with KMeans
                    print(f"Warning: Cluster {i} is empty. No representative window can be stored.")
                    # We need to append something to keep indices aligned. Let's append None.
                    self.representative_windows.append(None)

            # Convert to numpy array for consistency
            # Note: This will fail if there are Nones. Check for that.
            if any(rw is None for rw in self.representative_windows):
                print("Warning: Some representative windows are missing. This may cause issues.")
            else:
                self.representative_windows = np.array(self.representative_windows)
                print(f"Stored {len(self.representative_windows)} representative raw context windows in memory bank.")

        return centroids

    def set_threshold_from_data(self, normal_embeddings, method='iqr', factor=2.5, percentile=99.0):
        """Calculates distances of normal embeddings to nearest centroid and sets threshold."""
        # ... (implementation from previous versions - ensure 'percentile' path works correctly) ...
        if self.centroids is None or self.centroids.shape[0] == 0:
            self.threshold = np.inf; self.threshold_method = "Error - Bank Empty"; return
        if normal_embeddings is None or normal_embeddings.shape[0] == 0:
            self.threshold = np.inf; self.threshold_method = "Error - No Embeddings"; return
        if self.distance_metric == 'cosine':
            embedding_distances = cosine_distances(normal_embeddings, self.centroids)
        else:
            embedding_distances = euclidean_distances(normal_embeddings, self.centroids)
        nearest_centroid_indices = np.argmin(embedding_distances, axis=1)
        all_distances = embedding_distances[np.arange(len(normal_embeddings)), nearest_centroid_indices]
        finite_distances = all_distances[np.isfinite(all_distances)]
        if len(finite_distances) == 0:
            self.threshold = np.inf; self.threshold_method = "Error - No Finite Distances"; return

        if method == 'iqr':
            q1, q3 = np.percentile(finite_distances, [25, 75]); iqr = q3 - q1
            print(f"Distance quartiles (Q1, Q3): {q1:.4f}, {q3:.4f}, IQR: {iqr:.4f}")
            min_iqr = 1e-6
            if iqr > min_iqr:
                raw_threshold = q3 + factor * iqr; self.threshold_method = f"IQR Distance (Factor={factor})"
            else:
                print(f"Warning: IQR of distances ({iqr:.6f}) is too small (< {min_iqr})")
                raw_threshold = np.percentile(finite_distances, percentile); self.threshold_method = f"Percentile Distance (Fallback {percentile}% - Small IQR)"
        elif method == 'percentile':
            raw_threshold = np.percentile(finite_distances, percentile)
            self.threshold_method = f"Percentile Distance ({percentile}%)"
        else:
            print(f"Warning: Unknown threshold method '{method}'. Defaulting to percentile.")
            raw_threshold = np.percentile(finite_distances, percentile); self.threshold_method = f"Percentile Distance ({percentile}% - Unknown Method)"
        self.threshold = raw_threshold
        if self.distance_metric == 'cosine':
            min_cosine_thresh = 0.01; max_cosine_thresh = 1.5; original_thresh_before_bounds = self.threshold
            self.threshold = np.clip(self.threshold, min_cosine_thresh, max_cosine_thresh)
            if self.threshold != original_thresh_before_bounds: self.threshold_method += f" (Bounded to [{min_cosine_thresh:.3f}, {max_cosine_thresh:.3f}])"
        if not np.isfinite(self.threshold):
            print("Warning: Calculated threshold is non-finite. Setting to infinity.")
            self.threshold = np.inf; self.threshold_method += " (Set to Inf)"
        self._plot_distance_distribution(finite_distances)
        print(f"Memory bank threshold set to {self.threshold:.4f} using method: {self.threshold_method}")

    def _plot_distance_distribution(self, distances):
        """Plot the distribution of distances and the threshold."""
        # ... (implementation from previous versions) ...
        plt.figure(figsize=(10, 6)); plt.hist(distances, bins=50, alpha=0.7, density=True)
        plt.axvline(self.threshold, color='r', linestyle='--', label=f'Threshold: {self.threshold:.4f} ({self.threshold_method})')
        plt.xlabel('Distance to Nearest Centroid'); plt.ylabel('Density'); plt.title('Distance Distribution with Threshold'); plt.legend(); plt.grid(True)
        os.makedirs('threshold_analysis', exist_ok=True)
        plt.savefig(f'threshold_analysis/distance_distribution_{self.threshold_method.replace(" ", "_").replace("%","Perc").replace("(","").replace(")","")}.png', dpi=150); plt.close() # Made filename safer


    # <<< MODIFIED VERSION FOR IDEA 1, OPTION A >>>
    def get_ref_embedding(self, current_context_embedding, current_context_window=None, k=3,
                     use_adaptive_threshold=False, force_substitution=False,
                     # --- Heuristic 1 & Complexity Params ---
                     use_regime_aware_mb_threshold=True,
                     regime_std_threshold=0.20,
                     high_variance_threshold_factor=2.0,
                     use_complexity_thresholding=True
                    ):
        """
        Returns a reference embedding and determines if substitution is needed, incorporating
        Heuristic 1 (Regime-Aware Thresholding) and complexity analysis.

        Args:
            current_context_embedding: The embedding vector (1D numpy array) to find references for.
            current_context_window: Raw time series data (numpy array, e.g., shape [context_size, n_features])
                                      for complexity and regime analysis. Required if complexity/regime
                                      adaptation is enabled.
            k: Default number of nearest centroids to consider for weighted reference. Can be overridden
               by complexity analysis.
            use_adaptive_threshold (bool): Confusingly named legacy parameter.
                                          If False (First Pass): Substitution decision uses distance vs adaptive threshold.
                                          If True (Second Pass): Substitution decision driven ONLY by force_substitution flag.
            force_substitution (bool): Flag indicating if substitution should be forced (e.g., from confidence map).
                                       Can be overridden by high reference instability.
            use_regime_aware_mb_threshold (bool): Enable regime-based MB threshold adaptation.
            regime_std_threshold (float): Std dev threshold on raw context window to distinguish regimes.
            high_variance_threshold_factor (float): Multiplier for MB threshold in high-variance regime.
            use_complexity_thresholding (bool): Enable complexity-based threshold adaptation and k selection.

        Returns:
            tuple: (reference_embedding, needs_substitution)
                - reference_embedding (np.ndarray): Weighted reference if substituted, else original embedding.
                - needs_substitution (bool): True if substitution decision was made.
        """
        # --- Input Validation / Setup ---
        if self.centroids is None or len(self.centroids) == 0:
            # If no bank, return original embedding and False (cannot substitute)
            # print("Warning: Memory bank is empty. Cannot substitute.") # Optional warning
            return current_context_embedding, False

        # Ensure input is 2D for distance calculation, store original shape for return
        original_shape = current_context_embedding.shape
        if len(original_shape) == 1:
            current_context_embedding_2d = current_context_embedding.reshape(1, -1)
        elif len(original_shape) == 2 and original_shape[0] == 1:
             current_context_embedding_2d = current_context_embedding
        else: # Should not happen if called correctly from scoring loop
             print(f"Warning: Unexpected input embedding shape {original_shape} in get_ref_embedding.")
             # Attempt to use first row if possible, otherwise return original
             if len(original_shape) > 1 and original_shape[1] > 0:
                 current_context_embedding_2d = current_context_embedding[0].reshape(1,-1)
             else:
                 return current_context_embedding, False # Cannot process

        # --- Calculate Distances ---
        if self.distance_metric == 'cosine':
            distances = cosine_distances(current_context_embedding_2d, self.centroids)[0]
        else:
            distances = euclidean_distances(current_context_embedding_2d, self.centroids)[0]

        if len(distances) == 0:
            print("Warning: No distances calculated (centroids might be empty?).")
            return current_context_embedding, False # Return original embedding

        min_distance = np.min(distances)

        # --- Reference Stability Analysis ---
        reference_instability = 0.0
        stability_k = min(k + 2, len(distances))
        if stability_k >= 2:
            nearest_indices = np.argsort(distances)[:stability_k]
            nearest_distances = distances[nearest_indices]
            top_centroids = self.centroids[nearest_indices]

            # Calculate variance among the nearest centroids
            centroid_variance = np.var(top_centroids, axis=0).mean()

            # Calculate coefficient of variation of distances
            mean_nearest_dist = np.mean(nearest_distances)
            if mean_nearest_dist > 1e-9:
                distance_variation = np.std(nearest_distances) / mean_nearest_dist
            else:
                distance_variation = 0

            # Combined instability score (heuristic)
            # Clip individual components to avoid extreme values disproportionately affecting score
            reference_instability = 0.5 * min(1.0, centroid_variance * 10) + 0.5 * min(1.0, distance_variation)

            # Optional: Log high instability
            # if reference_instability > 0.5:
            #    print(f"  High reference instability detected: {reference_instability:.3f}")

        # --- Calculate Base Adaptive Threshold (Complexity) ---
        base_adaptive_threshold = self.threshold # Start with static threshold from bank setup
        threshold_explanation_list = [f"BaseThresh={self.threshold:.4f}"] # Start building explanation
        effective_k = k # Default k
        signal_type = "unknown"

        if use_complexity_thresholding and current_context_window is not None and self.threshold is not None:
            try:
                complexity_metrics = calculate_enhanced_complexity_metrics(current_context_window)
                pattern_type, basic_type, confidence_scores = classify_enhanced_pattern_type(complexity_metrics)
                signal_type = basic_type

                adaptive_thresh_complex, explanation = calculate_adaptive_threshold_enhanced(
                    self.threshold,
                    complexity_metrics,
                    min_threshold_multiplier=1.0
                )
                base_adaptive_threshold = adaptive_thresh_complex
                effective_k = explanation.get('k_recommendation', k)
                threshold_explanation_list.append(f"ComplexAdj({pattern_type[:5]},fac={explanation['threshold_factor']:.2f})->{base_adaptive_threshold:.4f}")
            except Exception as e:
                print(f"Warning: Error during complexity analysis: {e}. Using base threshold.")


        # --- Apply Regime-Aware Adaptation (Heuristic 1) ---
        final_adaptive_threshold = base_adaptive_threshold # Initialize with (potentially complexity-adjusted) threshold

        if use_regime_aware_mb_threshold and current_context_window is not None and base_adaptive_threshold is not None:
            try:
                # Calculate std dev of the first feature in the raw context window
                if current_context_window.shape[1] > 0:
                    context_std_dev = np.std(current_context_window[:, 0])
                else:
                    context_std_dev = 0.0

                is_high_variance_regime = context_std_dev >= regime_std_threshold

                if is_high_variance_regime:
                    regime_factor = high_variance_threshold_factor
                    final_adaptive_threshold = base_adaptive_threshold * regime_factor # Apply regime factor ON TOP of base (potentially complexity-adjusted)
                    threshold_explanation_list.append(f"HiVarAdj(std={context_std_dev:.3f},fac={regime_factor:.2f})->{final_adaptive_threshold:.4f}")
                    # else: threshold remains base_adaptive_threshold
            except Exception as e:
                print(f"Warning: Error during regime std dev calculation: {e}. Skipping regime adjustment.")
                # final_adaptive_threshold remains base_adaptive_threshold

        # --- Make Substitution Decision ---
        is_unusual = False # Initialize
        # Determine pass based on confusing legacy parameter name
        is_first_pass_decision = not use_adaptive_threshold

        if is_first_pass_decision:
            # --- First Pass ---
            reason = "None"
            if force_substitution:
                 if reference_instability > 0.7 and self.threshold is not None and min_distance < (self.threshold * 1.2):
                     is_unusual = False
                     reason = "ForceDenied(Instability)"
                 else:
                     is_unusual = True
                     reason = "Forced"
            elif final_adaptive_threshold is not None and min_distance > final_adaptive_threshold:
                 is_unusual = True
                 reason = f"Dist({min_distance:.3f})>Thresh({final_adaptive_threshold:.3f})"

            # Optional: Log first pass decisions leading to substitution
            # if is_unusual and reason != "Forced":
            #     print(f"  Sub decision (1st Pass): YES. Reason: {reason}. Signal: {signal_type}. Thresh Exp: [{' | '.join(threshold_explanation_list)}]")

        else:
            # --- Second Pass ---
            # Decision driven ONLY by force_substitution (potentially overridden by instability)
            if force_substitution:
                if reference_instability > 0.7 and self.threshold is not None and min_distance < (self.threshold * 1.2):
                    is_unusual = False
                    # print("  Sub decision (2nd Pass): NO (Force Denied by Instability)") # Optional log
                else:
                    is_unusual = True
                    # print("  Sub decision (2nd Pass): YES (Forced by Map)") # Optional log
            # else: is_unusual remains False

        # --- Generate Output ---
        needs_substitution = is_unusual # Final decision

        if needs_substitution:
            # Find k nearest centroids (use effective_k from complexity analysis)
            k_to_use = min(effective_k, len(distances))
            if k_to_use == 0 : # Should not happen if distances exist, but safety check
                 print("Warning: k_to_use is 0 in get_ref_embedding. Returning original.")
                 return current_context_embedding, False

            nearest_k_indices = np.argsort(distances)[:k_to_use]
            nearest_k_distances = distances[nearest_k_indices]

            # Calculate weights (inverse of distance, handle zero distance)
            weights = 1.0 / (nearest_k_distances + 1e-9) # Add epsilon for stability
            weights_sum = np.sum(weights)
            if weights_sum > 1e-9:
                weights = weights / weights_sum # Normalize weights
            else:
                 # Handle case where all nearest distances are huge -> equal weights
                 weights = np.ones_like(weights) / k_to_use


            # Compute weighted centroid
            weighted_ref = np.sum(weights[:, np.newaxis] * self.centroids[nearest_k_indices], axis=0)

            # Ensure returned embedding has same dimension as input
            return weighted_ref.astype(current_context_embedding.dtype), True

        else:
            # Return original embedding (ensure it's 1D numpy array as expected by scoring)
            return current_context_embedding_2d[0].astype(current_context_embedding.dtype), False