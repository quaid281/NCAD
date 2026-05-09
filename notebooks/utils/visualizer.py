import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
from sklearn.metrics.pairwise import euclidean_distances # Needed for cluster assignment
import matplotlib.cm as cm # Needed for colormaps
from matplotlib.gridspec import GridSpec # Needed for subplot grid layout
# Create custom colorbar
from matplotlib.colors import LinearSegmentedColormap

# Optional: Set a consistent style
# plt.style.use('seaborn-v0_8-whitegrid')
# Define standard font sizes for consistency
SUptitle_FONTSIZE = 18
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 12
TICK_FONTSIZE = 12
SAVE_DPI = 300 # Standard DPI for publications


# Add to visualizer.py
def generate_enhanced_plots_with_changes(stream, test_telemetry, final_scores_smoothed, threshold_info,
                                        predictions, binary_labels, substitution_map,
                                        original_distances, embedding_changes, transition_scores,
                                        memory_bank, threshold_method, smoothing_window,
                                        channel_results_dir, step=1):
    """Generate enhanced visualization that includes embedding changes and transitions."""
    # Create standard plots first
   
    # Add a new visualization specifically for the change metrics
    change_analysis_path = os.path.join(channel_results_dir, f"{stream}_change_analysis.svg")
   
    plt.figure(figsize=(12, 15))
   
    # Plot 1: Original data
    plt.subplot(5, 1, 1)
    plt.plot(test_telemetry[:len(binary_labels)], label='Telemetry')
    plt.title('Original Data', fontsize=12)
    plt.legend()
   
    # Plot 2: Memory bank distances
    plt.subplot(5, 1, 2)
    distance_plot = np.zeros(len(binary_labels))
    for i, dist in enumerate(original_distances):
        idx = i * step
        if idx < len(distance_plot):
            end_idx = min(idx + step, len(distance_plot))
            distance_plot[idx:end_idx] = dist
   
    plt.plot(distance_plot, 'r-', label='Distance to Memory Bank')
    if memory_bank and hasattr(memory_bank, 'threshold'):
        plt.axhline(y=memory_bank.threshold, color='k', linestyle='--',
                  label=f'Threshold: {memory_bank.threshold:.4f}')
    plt.title('Distance to Memory Bank', fontsize=12)
    plt.legend()
   
    # Plot 3: Embedding changes
    plt.subplot(5, 1, 3)
    change_plot = np.zeros(len(binary_labels))
    for i, change in enumerate(embedding_changes):
        idx = i * step
        if idx < len(change_plot):
            end_idx = min(idx + step, len(change_plot))
            change_plot[idx:end_idx] = change
   
    plt.plot(change_plot, 'g-', label='Embedding Change Rate')
    plt.title('Embedding Changes Over Time', fontsize=12)
    plt.legend()
   
    # Plot 4: Window transitions
    plt.subplot(5, 1, 4)
    transition_plot = np.zeros(len(binary_labels))
    for i, trans in enumerate(transition_scores):
        idx = i * step
        if idx < len(transition_plot):
            end_idx = min(idx + step, len(transition_plot))
            transition_plot[idx:end_idx] = trans
   
    plt.plot(transition_plot, 'b-', label='Window Transition Scores')
    plt.title('Context Window Transitions', fontsize=12)
    plt.legend()
   
    # Plot 5: Anomalies and substitutions
    plt.subplot(5, 1, 5)
    plt.plot(binary_labels, 'g-', drawstyle='steps-post', label='True Anomalies')
    plt.plot(substitution_map, 'r-', drawstyle='steps-post', alpha=0.6, label='Substitutions')
    plt.title('Anomalies vs. Substitutions', fontsize=12)
    plt.xlabel('Time Step')
    plt.legend()
   
    plt.suptitle(f'Change-Based Analysis for {stream}', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(change_analysis_path, dpi=150)
    plt.close()    

def plot_anomaly_detection_results(test_data, scores, threshold, predictions,
                                   true_anomalies=None,
                                   title="Anomaly Detection Results",
                                   save_path=None): # Added save_path option
    """
    Creates a publication-ready visualization of anomaly detection results
    with increased font sizes and optional saving at high DPI.

    Args:
        test_data (np.ndarray): The input telemetry data (using first column if multi-dim).
        scores (np.ndarray): Anomaly scores corresponding to the test data.
        threshold (float or dict): The anomaly threshold(s).
        predictions (np.ndarray): Binary predictions (0=normal, 1=anomaly).
        true_anomalies (np.ndarray, optional): Ground truth binary labels. Defaults to None.
        title (str, optional): Overall title for the figure. Defaults to "Anomaly Detection Results".
        save_path (str, optional): If provided, saves the figure to this path. Defaults to None.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    fig = plt.figure(figsize=(15, 10)) # Keep figsize or adjust as needed

    # --- Subplot 1: Original Data ---
    ax1 = plt.subplot(3, 1, 1)
    # Use only the first dimension if test_data is multi-dimensional for this plot
    data_to_plot = test_data[:, 0] if test_data.ndim > 1 else test_data
    ax1.plot(data_to_plot, label='Telemetry')
    ax1.set_title('Original Telemetry Data', fontsize=TITLE_FONTSIZE)
    ax1.set_ylabel('Value', fontsize=LABEL_FONTSIZE) # Added Y Label
    ax1.legend(fontsize=LEGEND_FONTSIZE)
    ax1.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    plt.setp(ax1.get_xticklabels(), visible=False) # Hide x-ticks labels since it's shared

    # --- Subplot 2: Scores and Threshold ---
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(scores, 'b-', label='Anomaly Scores')

    # Handle single or multiple thresholds
    if isinstance(threshold, dict):
        first_key = next(iter(threshold.keys()), None) # Get first key for labeling logic
        for i, ((start, end), thresh) in enumerate(threshold.items()):
            label = f'Threshold ({start}-{end}): {thresh:.4f}' if i == 0 else f'{thresh:.4f}'
            ax2.plot(range(start, end + 1), [thresh] * (end - start + 1), 'r--',
                     label=label) # Plot segment, label first segment fully
    else:
        # For global thresholding
        ax2.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')

    ax2.set_title('Anomaly Scores and Threshold', fontsize=TITLE_FONTSIZE)
    ax2.set_ylabel('Score', fontsize=LABEL_FONTSIZE) # Added Y Label
    ax2.legend(fontsize=LEGEND_FONTSIZE)
    ax2.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    plt.setp(ax2.get_xticklabels(), visible=False) # Hide x-ticks labels since it's shared

    # --- Subplot 3: Anomalies ---
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    # Ensure predictions is integer type for clear plotting if not already
    ax3.plot(predictions.astype(int), 'r-', drawstyle='steps-post', label='Detected Anomalies') # steps-post often clearer for binary flags

    # Plot ground truth if available
    if true_anomalies is not None:
        # Multiply by a factor > 1 to visually separate true/detected when they overlap
        ax3.plot(true_anomalies.astype(int) * 1.1, 'g-', alpha=0.7, drawstyle='steps-post', label='True Anomalies') # steps-post
        ax3.set_yticks([0, 1, 1.1]) # Adjust yticks based on multiplication factor
        ax3.set_yticklabels(['Normal', 'Detected', 'True'], fontsize=TICK_FONTSIZE) # Apply fontsize here too
    else:
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(['Normal', 'Detected'], fontsize=TICK_FONTSIZE) # Apply fontsize here too

    ax3.set_title('Anomaly Detection Results', fontsize=TITLE_FONTSIZE)
    ax3.set_xlabel('Time Step / Sample Index', fontsize=LABEL_FONTSIZE) # Added X Label to the bottom plot
    ax3.set_ylabel('Status', fontsize=LABEL_FONTSIZE) # Added Y Label
    ax3.legend(fontsize=LEGEND_FONTSIZE)
    ax3.tick_params(axis='x', which='major', labelsize=TICK_FONTSIZE) # Apply x tick size only (y handled by set_yticklabels)

    # --- Final Adjustments ---
    # Overall Title
    plt.suptitle(title, fontsize=SUptitle_FONTSIZE, y=0.98) # Adjust y position if needed

    # Adjust layout tightly
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # rect leaves space for suptitle [left, bottom, right, top]
    # plt.subplots_adjust(top=0.92, hspace=0.3) # Alternative or additional adjustment if tight_layout isn't enough, adjust hspace


    # --- Saving ---
    if save_path:
        plt.savefig(save_path, dpi=SAVE_DPI, bbox_inches='tight')
        print(f"Figure saved to {save_path} with DPI={SAVE_DPI}")
        plt.close(fig) # Close the figure after saving to free memory

    else:
         # If not saving, the calling code is responsible for showing or saving the fig
        pass # Figure object will be returned

    # Return the figure object (especially useful if not saving within the function)
    # Note: If save_path was provided, the figure is closed above.
    # Returning it might lead to errors if used later unless saving failed.
    # Consider only returning if save_path is None.
    if save_path is None:
        return fig
    else:
        # If saved and closed, maybe return None or the path? Returning None is safer.
        return None # Indicate figure was handled (saved and closed)


def ensure_dir(file_path):
    """Ensures the directory for a file path exists."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def plot_loss_curves(history, save_path):
    """Plots training and validation loss curves."""
    ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(history.get('loss', [])) + 1)

    if 'loss' in history and len(history['loss']) > 0:
        ax.plot(epochs, history['loss'], 'bo-', label='Training Loss')
    if 'val_loss' in history and len(history['val_loss']) > 0:
        # Ensure val_loss has same length as epochs for plotting
        val_loss = history['val_loss']
        if len(val_loss) == len(epochs):
                ax.plot(epochs, val_loss, 'ro--', label='Validation Loss')
        else:
                warnings.warn(f"Length mismatch: {len(epochs)} epochs vs {len(val_loss)} val_loss entries. Skipping val_loss plot.")


    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Loss curve saved to {save_path}")
    plt.close(fig)

def plot_embeddings(embeddings, labels, title, save_path, method='pca', sample_size=5000):
    """Plots 2D representation of embeddings using PCA or t-SNE."""
    ensure_dir(save_path)
    if embeddings is None or len(embeddings) == 0:
        print("No embeddings provided for plotting.")
        return
    if labels is None or len(labels) != len(embeddings):
        print("Warning: Invalid or missing labels for embeddings plot. Plotting all as one color.")
        labels = np.zeros(len(embeddings)) # Default labels if missing/invalid

    # Sample data if too large (t-SNE can be slow)
    if len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]

    if embeddings.shape[1] > 2:
        print(f"Reducing embedding dimensions using {method.upper()}...")
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        elif method == 'tsne':
            perplexity = min(30.0, max(5.0, len(embeddings) / 5.0 - 1)) # Adjust perplexity based on sample size
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=300)
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")
           
        embeddings_2d = reducer.fit_transform(embeddings)
       
    elif embeddings.shape[1] == 2:
        embeddings_2d = embeddings
    else:
        print(f"Cannot plot embeddings with {embeddings.shape[1]} dimensions. Need 2 or more.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    unique_labels = np.unique(labels)

    for label_val in unique_labels:
        subset_indices = (labels == label_val)
        label_name = f'Class {int(label_val)}' # Basic label naming
        if int(label_val) == 0: label_name = 'Normal (0)'
        if int(label_val) == 1: label_name = 'Anomaly (1)'
        # Add more specific names if available

        ax.scatter(embeddings_2d[subset_indices, 0],
                    embeddings_2d[subset_indices, 1],
                    label=label_name, alpha=0.6, s=20) # Smaller points

    ax.set_title(title)
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Embeddings plot saved to {save_path}")
    plt.close(fig)


def plot_memory_bank(bank_centroids, normal_embeddings, save_path, method='pca', sample_size=5000):
    """
    Plots 2D representation of memory bank centroids and normal embeddings,
    coloring normal embeddings by their closest centroid.
    """
    ensure_dir(save_path)
    if bank_centroids is None or len(bank_centroids) == 0:
        print("No centroids provided for memory bank plot.")
        return
    if normal_embeddings is None or len(normal_embeddings) == 0:
          print("No normal embeddings provided for memory bank plot.")
          return

    # Sample normal embeddings if too large
    if len(normal_embeddings) > sample_size:
        indices = np.random.choice(len(normal_embeddings), sample_size, replace=False)
        sampled_embeddings = normal_embeddings[indices]
        print(f"Sampled {sample_size} normal embeddings for plotting.")
    else:
        sampled_embeddings = normal_embeddings

    # Assign sampled embeddings to nearest centroid
    if sampled_embeddings.shape[0] > 0 and bank_centroids.shape[0] > 0:
        distances = euclidean_distances(sampled_embeddings, bank_centroids)
        cluster_assignments = np.argmin(distances, axis=1) # Index of the closest centroid
        n_clusters = bank_centroids.shape[0]
        print(f"Assigned {len(sampled_embeddings)} points to {n_clusters} clusters.")
    else:
        cluster_assignments = np.array([])
        n_clusters = 0

    # Combine for dimensionality reduction
    # Avoid combining if sampled_embeddings is empty
    if sampled_embeddings.shape[0] == 0:
            print("No sampled embeddings to plot after potential filtering.")
            # Handle centroids separately if needed, or return
            if bank_centroids.shape[0] == 0:
                return # Nothing to plot
            combined = bank_centroids
            embeddings_2d = np.empty((0, 2)) # Empty array for embeddings
    else:
            combined = np.vstack([sampled_embeddings, bank_centroids])


    # Perform dimensionality reduction
    if combined.shape[1] > 2:
        print(f"Reducing memory bank dimensions using {method.upper()}...")
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        elif method == 'tsne':
            # Adjust perplexity calculation to handle small combined sizes
            perplexity_max = max(0.0, combined.shape[0] - 1.0) # Cannot be >= n_samples
            perplexity = min(30.0, max(5.0, perplexity_max / 3.0)) # Adjust logic as needed
            if perplexity < 5.0 or combined.shape[0] <= 1: # Check if TSNE is feasible
                    print(f"Warning: Not enough samples ({combined.shape[0]}) for reliable t-SNE with perplexity {perplexity}. Using PCA instead.")
                    reducer = PCA(n_components=2, random_state=42)
                    method = 'pca' # Update method label
            else:
                reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=300, init='pca', learning_rate='auto') # Added init and lr
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")

        # Ensure data is float64 for some TSNE versions
        combined_2d = reducer.fit_transform(combined.astype(np.float64))


    elif combined.shape[1] == 2:
        combined_2d = combined
    else:
        print(f"Cannot plot memory bank with {combined.shape[1]} dimensions.")
        return

    # Separate reduced embeddings and centroids
    # Adjust separation logic based on whether embeddings were included
    if sampled_embeddings.shape[0] > 0:
        embeddings_2d = combined_2d[:-len(bank_centroids)]
        centroids_2d = combined_2d[-len(bank_centroids):]
    else: # Only centroids were processed
        embeddings_2d = np.empty((0, 2))
        centroids_2d = combined_2d


    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 10)) # Slightly larger plot

    # Choose colormap based on number of clusters
    if n_clusters > 0 and embeddings_2d.shape[0] > 0: # Check if there are embeddings to color
        if n_clusters <= 10:
            colors = cm.get_cmap('tab10', n_clusters)
        elif n_clusters <= 20:
            colors = cm.get_cmap('tab20', n_clusters)
        else: # Use a continuous map like viridis if many clusters
            colors = cm.get_cmap('viridis', n_clusters)

        # Plot normal embeddings colored by cluster assignment
        for i in range(n_clusters):
            cluster_mask = (cluster_assignments == i)
            # Only plot if this cluster has members in the sample
            if np.any(cluster_mask):
                ax.scatter(embeddings_2d[cluster_mask, 0],
                            embeddings_2d[cluster_mask, 1],
                            # label=f'Cluster {i}', # Avoid legend clutter
                            alpha=0.5, # Make slightly less transparent
                            s=25,     # Slightly larger points
                            color=colors(i / (n_clusters -1) if n_clusters > 1 else 0)) # Normalize index for continuous maps

    # Plot centroids on top (always plot if they exist)
    if centroids_2d.shape[0] > 0:
            ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                    label='Bank Centroids', # Keep this legend entry
                    alpha=0.95,
                    s=80, # Make centroids more prominent (Increased size)
                    c='red',
                    marker='X',
                    edgecolors='black', # Add edge color for visibility
                    linewidth=0.8)

    # --- Adjust Font Sizes Here ---
    TITLE_FONTSIZE = 18
    LABEL_FONTSIZE = 16
    LEGEND_FONTSIZE = 14
    TICK_FONTSIZE = 12

    ax.set_title(f'Memory Bank Centroids and Normal Embeddings (Colored by Cluster Assignment, Method: {method.upper()})',
                    fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=LABEL_FONTSIZE)

    # Increase legend font size (only shows 'Bank Centroids' in your current setup)
    if centroids_2d.shape[0] > 0: # Only add legend if centroids were plotted
        ax.legend(fontsize=LEGEND_FONTSIZE)

    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)

    ax.grid(True)

    plt.tight_layout() # Adjust layout to prevent labels overlapping

    # --- Save with Higher DPI ---
    SAVE_DPI = 300 # Increase DPI for better quality in papers
    plt.savefig(save_path, dpi=SAVE_DPI, bbox_inches='tight') # Use bbox_inches='tight' to include labels properly
    print(f"Memory bank plot saved to {save_path} with DPI={SAVE_DPI}")
    plt.close(fig) # Ensure figure is closed

   
def plot_score_distribution(scores, threshold, save_path, channel_name=""):
    """Plots the distribution of anomaly scores and the threshold."""
    ensure_dir(save_path)
    if scores is None or len(scores) == 0:
        print("No scores provided for distribution plot.")
        return
   
    finite_scores = scores[np.isfinite(scores)]
    if len(finite_scores) == 0:
        print("No finite scores to plot distribution.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(finite_scores, bins=50, alpha=0.7, label='Score Distribution', density=True)

    if np.isfinite(threshold):
        ax.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')

    ax.set_title(f'Anomaly Score Distribution for {channel_name}')
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Density')
    # Consider log scale if distribution is highly skewed
    # ax.set_yscale('log')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Score distribution plot saved to {save_path}")
    plt.close(fig)


# Helper function for plotting with adaptive threshold
def plot_anomaly_detection_with_adaptive_threshold(test_data, scores, threshold_values, predictions,
                                               true_anomalies=None, substitution_map=None,
                                               title="Enhanced Anomaly Detection Results"):
    """Creates an enhanced visualization showing adaptive thresholds."""
    fig = plt.figure(figsize=(15, 12))  # Taller to accommodate more subplots
   
    # --- Subplot 1: Original Data ---
    ax1 = plt.subplot(5, 1, 1)  # Now 5 rows instead of 3
    data_to_plot = test_data[:, 0] if test_data.ndim > 1 else test_data
    ax1.plot(data_to_plot, label='Telemetry')
    ax1.set_title('Original Telemetry Data', fontsize=12)
    ax1.set_ylabel('Value', fontsize=10)
    ax1.legend(fontsize=10)
    ax1.tick_params(axis='both', which='major', labelsize=9)
    plt.setp(ax1.get_xticklabels(), visible=False)
   
    # --- Subplot 2: Substitution Map ---
    ax2 = plt.subplot(5, 1, 2, sharex=ax1)
    if substitution_map is not None:
        ax2.plot(substitution_map.astype(int), 'b-', drawstyle='steps-post',
               label=f'Memory Bank Substitutions ({np.sum(substitution_map)} points)')
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['No', 'Yes'])
    else:
        ax2.text(0.5, 0.5, 'No substitution data available',
                ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Context Substitutions', fontsize=12)
    ax2.set_ylabel('Substituted', fontsize=10)
    ax2.legend(fontsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=9)
    plt.setp(ax2.get_xticklabels(), visible=False)
   
    # --- Subplot 3: Scores and Adaptive Threshold ---
    ax3 = plt.subplot(5, 1, 3, sharex=ax1)
    ax3.plot(scores, 'b-', label='Anomaly Scores')
   
    # Plot adaptive threshold if available
    if threshold_values is not None:
        ax3.plot(threshold_values, 'r--', label='Adaptive Threshold')
    else:
        # If threshold_values not provided, just show a message
        ax3.text(0.5, 0.8, 'Adaptive threshold values not available',
                ha='center', va='center', transform=ax3.transAxes)
       
    ax3.set_title('Anomaly Scores and Adaptive Threshold', fontsize=12)
    ax3.set_ylabel('Score', fontsize=10)
    ax3.legend(fontsize=10)
    ax3.tick_params(axis='both', which='major', labelsize=9)
    plt.setp(ax3.get_xticklabels(), visible=False)
   
    # --- Subplot 4: Model Predictions ---
    ax4 = plt.subplot(5, 1, 4, sharex=ax1)
    ax4.plot(predictions.astype(int), 'r-', drawstyle='steps-post', label='Detected Anomalies')
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Normal', 'Anomaly'])
    ax4.set_title('Model Predictions', fontsize=12)
    ax4.set_ylabel('Status', fontsize=10)
    ax4.legend(fontsize=10)
    ax4.tick_params(axis='both', which='major', labelsize=9)
    plt.setp(ax4.get_xticklabels(), visible=False)
   
    # --- Subplot 5: Ground Truth ---
    ax5 = plt.subplot(5, 1, 5, sharex=ax1)
    if true_anomalies is not None:
        ax5.plot(true_anomalies.astype(int), 'g-', drawstyle='steps-post', label='True Anomalies')
        ax5.set_yticks([0, 1])
        ax5.set_yticklabels(['Normal', 'Anomaly'])
    else:
        ax5.text(0.5, 0.5, 'No ground truth available',
                ha='center', va='center', transform=ax5.transAxes)
    ax5.set_title('Ground Truth', fontsize=12)
    ax5.set_xlabel('Time Step / Sample Index', fontsize=10)
    ax5.set_ylabel('Status', fontsize=10)
    ax5.legend(fontsize=10)
    ax5.tick_params(axis='both', which='major', labelsize=9)
   
    # --- Final Adjustments ---
    plt.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
   
    return fig

def create_substitution_diagnostics(
    # --- Point-Level Data ---
    test_telemetry,
    substitution_flags, # Point-level boolean map where substitution was applied
    true_anomalies=None,
    predictions=None,
    # --- Window-Level Data ---
    original_distances=None,
    confidence_scores=None,
    embedding_changes=None,
    transition_scores=None,
    # --- Metadata ---
    window_start_indices=None,
    step=1,
    threshold_value=None, # Single threshold value from memory bank
    # --- Plotting ---
    title="Substitution Decision Diagnostics",
    save_path=None
):
    """
    Create enhanced diagnostic visualization showing how substitution decisions were made,
    accurately mapping window-level metrics to point-level representation.

    Args:
        test_telemetry (np.ndarray): Original telemetry data (point-level).
        substitution_flags (np.ndarray): Boolean array indicating where substitution occurred (point-level).
        true_anomalies (np.ndarray, optional): Ground truth anomaly labels (point-level).
        predictions (np.ndarray, optional): Model predictions (point-level).
        original_distances (np.ndarray, optional): Array of distances from context to memory bank (window-level).
        confidence_scores (np.ndarray, optional): Confidence scores for substitution decisions (window-level).
        embedding_changes (np.ndarray, optional): Change scores for embeddings (window-level).
        transition_scores (np.ndarray, optional): Transition scores for context windows (window-level).
        window_start_indices (list or np.ndarray, optional): Start index for each window in test_telemetry. Required for mapping.
        step (int): Step size used for windowing. Required for mapping.
        threshold_value (float, optional): The memory bank threshold used for substitution decisions.
        title (str): Plot title.
        save_path (str, optional): Path to save the plot.
    """
    min_len = len(test_telemetry)
    if min_len == 0:
        print("Warning: test_telemetry is empty. Cannot generate diagnostics plot.")
        return

    # --- Map Window-Level Data to Point Level using window_start_indices ---
    mapped_distances = np.full(min_len, np.nan) # Use NaN for unmapped areas
    mapped_confidence = np.full(min_len, np.nan)
    mapped_changes = np.full(min_len, np.nan) if embedding_changes is not None else None
    mapped_transitions = np.full(min_len, np.nan) if transition_scores is not None else None

    if window_start_indices is not None:
        num_windows = len(window_start_indices)

        # Check length consistency
        if original_distances is not None and len(original_distances) != num_windows:
            print(f"Warning: Length mismatch between original_distances ({len(original_distances)}) and window_start_indices ({num_windows}). Skipping distance mapping.")
            original_distances = None
        if confidence_scores is not None and len(confidence_scores) != num_windows:
            print(f"Warning: Length mismatch between confidence_scores ({len(confidence_scores)}) and window_start_indices ({num_windows}). Skipping confidence mapping.")
            confidence_scores = None
        if embedding_changes is not None and len(embedding_changes) != num_windows:
            print(f"Warning: Length mismatch between embedding_changes ({len(embedding_changes)}) and window_start_indices ({num_windows}). Skipping changes mapping.")
            embedding_changes = None
            mapped_changes = None
        if transition_scores is not None and len(transition_scores) != num_windows:
            print(f"Warning: Length mismatch between transition_scores ({len(transition_scores)}) and window_start_indices ({num_windows}). Skipping transitions mapping.")
            transition_scores = None
            mapped_transitions = None

        # Perform mapping
        for i in range(num_windows):
            start_idx = window_start_indices[i]
            # Map the value to the points corresponding to the window's step
            end_idx = min(start_idx + step, min_len)

            if start_idx >= min_len: continue # Skip if window starts beyond the telemetry length

            if original_distances is not None:
                mapped_distances[start_idx:end_idx] = original_distances[i]
            if confidence_scores is not None:
                mapped_confidence[start_idx:end_idx] = confidence_scores[i]
            if mapped_changes is not None and embedding_changes is not None:
                mapped_changes[start_idx:end_idx] = embedding_changes[i]
            if mapped_transitions is not None and transition_scores is not None:
                mapped_transitions[start_idx:end_idx] = transition_scores[i]
    else:
        print("Warning: window_start_indices not provided. Cannot map window-level metrics for plotting.")
        # Set window-level data to None so plots don't attempt to use it
        original_distances = None
        confidence_scores = None
        embedding_changes = None
        transition_scores = None
        mapped_distances = None
        mapped_confidence = None
        mapped_changes = None
        mapped_transitions = None


    # --- Create Figure and GridSpec ---
    n_plots = 4 # Telemetry, Distance/Confidence, Substitution Map, Confidence Breakdown
    if mapped_changes is not None: n_plots += 1
    if mapped_transitions is not None: n_plots += 1
    if true_anomalies is not None or predictions is not None: n_plots += 1

    figsize = (15, n_plots * 2.5) # Adjust height based on number of plots
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_plots, 1, figure=fig, height_ratios=[1] * n_plots) # Equal height ratios
    plot_idx = 0

    # --- Plot 1: Original telemetry ---
    ax1 = fig.add_subplot(gs[plot_idx])
    plot_idx += 1
    ax1.plot(test_telemetry, label='Telemetry', linewidth=1.0)
    ax1.set_title('Original Telemetry Data', fontsize=14)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Plot 2: Distance with threshold and confidence ---
    ax2 = fig.add_subplot(gs[plot_idx], sharex=ax1)
    plot_idx += 1
    lines = [] # For combined legend

    if mapped_distances is not None:
        line1 = ax2.plot(mapped_distances, 'r-', alpha=0.7, label='Distance to Memory Bank', linewidth=1.5)
        lines.extend(line1)
        if threshold_value is not None and np.isfinite(threshold_value):
            ax2.axhline(y=threshold_value, color='k', linestyle='--', linewidth=1.2,
                       label=f'MB Threshold: {threshold_value:.4f}')
            # Add threshold to legend manually if needed, or rely on the axhline label
    else:
        ax2.text(0.5, 0.5, 'Distance data not available', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)

    ax2.set_ylabel('Distance', color='r', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='r', labelsize=10)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Confidence on second y-axis
    ax2b = ax2.twinx()
    if mapped_confidence is not None:
        line2 = ax2b.plot(mapped_confidence, 'b-', alpha=0.5, label='Substitution Confidence', linewidth=1.5)
        lines.extend(line2)
        # Optionally add confidence threshold lines if meaningful
        # ax2b.axhline(y=0.3, color='b', linestyle=':', alpha=0.7, label='Min Confidence (0.3)')
        ax2b.set_ylim([0, 1.05]) # Allow slight overshoot visually
    else:
         ax2b.text(0.5, 0.5, 'Confidence data not available', horizontalalignment='center', verticalalignment='center', transform=ax2b.transAxes)

    ax2b.set_ylabel('Confidence', color='b', fontsize=12)
    ax2b.tick_params(axis='y', labelcolor='b', labelsize=10)

    # Combine legends
    if lines: # Only create legend if there's something to show
        labels = [l.get_label() for l in lines]
        # Add threshold label if it exists
        thresh_label = next((line.get_label() for line in ax2.get_lines() if line.get_linestyle() == '--'), None)
        if thresh_label:
            lines.append(ax2.get_lines()[1]) # Assumes threshold is the second line plotted on ax2
            labels.append(thresh_label)
        ax2.legend(lines, labels, loc='upper right', fontsize=11)
    ax2.set_title('Distance to Memory Bank vs Substitution Confidence', fontsize=14)

    # --- Plot 3: Substitution map (Point-Level) ---
    ax3 = fig.add_subplot(gs[plot_idx], sharex=ax1)
    plot_idx += 1
    # <<< FIX: Calculate points and percentage from the point-level flags array >>>
    sub_points = np.sum(substitution_flags[:min_len]) # Count True values in the passed point-level array
    sub_percentage = (sub_points / min_len) * 100 if min_len > 0 else 0
    ax3.plot(substitution_flags[:min_len].astype(int), 'k-', drawstyle='steps-post', linewidth=1.5,
             label=f'Substituted ({sub_points} points, {sub_percentage:.2f}%)') # Use correct count/percentage
    # <<< END FIX >>>
    ax3.set_title(f'Memory Bank Substitutions (Applied)', fontsize=14)
    ax3.set_ylabel('Used', fontsize=12)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['No', 'Yes'], fontsize=10)
    ax3.legend(loc='upper right', fontsize=11)
    plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.grid(True, linestyle='--', alpha=0.6)

    # --- Plot 4: High-Confidence vs Low-Confidence substitutions ---
    ax4 = fig.add_subplot(gs[plot_idx], sharex=ax1)
    plot_idx += 1
    if mapped_confidence is not None:
        # Ensure mapped_confidence is aligned with substitution_flags before masking
        aligned_confidence = mapped_confidence[:min_len]
        aligned_flags = substitution_flags[:min_len]

        high_conf_mask = aligned_flags & (aligned_confidence >= 0.6)
        med_conf_mask = aligned_flags & (aligned_confidence >= 0.3) & (aligned_confidence < 0.6)
        low_conf_mask = aligned_flags & (aligned_confidence < 0.3)
        # Handle NaNs in confidence - treat as low confidence for this plot? Or exclude? Let's exclude.
        nan_mask = np.isnan(aligned_confidence)
        high_conf_mask = high_conf_mask & ~nan_mask
        med_conf_mask = med_conf_mask & ~nan_mask
        low_conf_mask = low_conf_mask & ~nan_mask

        high_count = np.sum(high_conf_mask)
        med_count = np.sum(med_conf_mask)
        low_count = np.sum(low_conf_mask)

        ax4.plot(high_conf_mask.astype(int) * 1.0, 'g-', drawstyle='steps-post', linewidth=1.5,
                 label=f'High Confidence (≥0.6): {high_count} points')
        ax4.plot(med_conf_mask.astype(int) * 0.6, 'y-', drawstyle='steps-post', linewidth=1.5,
                 label=f'Medium Confidence (0.3-0.6): {med_count} points')
        ax4.plot(low_conf_mask.astype(int) * 0.3, 'r-', drawstyle='steps-post', linewidth=1.5,
                 label=f'Low Confidence (<0.3): {low_count} points')

        ax4.set_yticks([0, 0.3, 0.6, 1.0])
        ax4.set_yticklabels(['None', 'Low', 'Medium', 'High'], fontsize=10)
        ax4.legend(fontsize=11)
    else:
        ax4.text(0.5, 0.5, 'Confidence data not available for breakdown', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)

    ax4.set_title('Substitution Confidence Breakdown', fontsize=14)
    ax4.set_ylabel('Confidence Level', fontsize=12)
    plt.setp(ax4.get_xticklabels(), visible=False)
    ax4.grid(True, linestyle='--', alpha=0.6)

    # --- Plot 5: Embedding changes (Optional) ---
    if mapped_changes is not None:
        ax5 = fig.add_subplot(gs[plot_idx], sharex=ax1)
        plot_idx += 1
        ax5.plot(mapped_changes, 'g-', label='Embedding Change Rate', linewidth=1.0)
        # Example threshold line - adjust value as needed based on your change detection logic
        ax5.axhline(y=0.2, color='g', linestyle=':', alpha=0.7, linewidth=1.0, label='Change Threshold (0.2)')
        ax5.set_title('Embedding Changes Over Time', fontsize=14)
        ax5.set_ylabel('Change Rate', fontsize=12)
        ax5.legend(fontsize=11)
        plt.setp(ax5.get_xticklabels(), visible=False)
        ax5.grid(True, linestyle='--', alpha=0.6)
        ax5.tick_params(axis='both', which='major', labelsize=10)


    # --- Plot 6: Window transitions (Optional) ---
    if mapped_transitions is not None:
        ax6 = fig.add_subplot(gs[plot_idx], sharex=ax1)
        plot_idx += 1
        ax6.plot(mapped_transitions, 'b-', label='Window Transition Scores', linewidth=1.0)
         # Example threshold line - adjust value as needed based on your transition detection logic
        ax6.axhline(y=2.5, color='b', linestyle=':', alpha=0.7, linewidth=1.0, label='Transition Threshold (2.5)')
        ax6.set_title('Context Window Transitions', fontsize=14)
        ax6.set_ylabel('Transition Score', fontsize=12)
        ax6.legend(fontsize=11)
        plt.setp(ax6.get_xticklabels(), visible=False)
        ax6.grid(True, linestyle='--', alpha=0.6)
        ax6.tick_params(axis='both', which='major', labelsize=10)


    # --- Plot 7: Anomalies vs Predictions (Optional) ---
    if true_anomalies is not None or predictions is not None:
        ax7 = fig.add_subplot(gs[plot_idx], sharex=ax1)
        plot_idx += 1
        title_str = 'Anomalies vs. Predictions'
        overlap_str = ""

        if true_anomalies is not None:
            true_anomalies_plot = true_anomalies[:min_len]
            ax7.plot(true_anomalies_plot.astype(int), 'g-', drawstyle='steps-post', linewidth=1.5, label='True Anomalies')

            # Calculate overlap with point-level substitution map
            overlap_mask = (true_anomalies_plot == 1) & (substitution_flags[:min_len] == 1)
            overlap_count = np.sum(overlap_mask)
            total_true_anomalies = np.sum(true_anomalies_plot)
            if total_true_anomalies > 0:
                overlap_percent = (overlap_count / total_true_anomalies) * 100
                overlap_str = f" (Anomaly-Substitution Overlap: {overlap_count}/{total_true_anomalies}, {overlap_percent:.1f}%)"
            else:
                overlap_str = " (No True Anomalies)"


        if predictions is not None:
            predictions_plot = predictions[:min_len]
            y_offset = 1.1 if true_anomalies is not None else 1.0 # Offset if plotting both
            ax7.plot(predictions_plot.astype(float) * y_offset, 'r-', drawstyle='steps-post', linewidth=1.5,
                     label='Predictions')
            if true_anomalies is not None:
                ax7.set_yticks([0, 1, y_offset])
                ax7.set_yticklabels(['Normal', 'True', 'Pred'], fontsize=10)
            else:
                ax7.set_yticks([0, 1])
                ax7.set_yticklabels(['Normal', 'Anomaly'], fontsize=10)
        elif true_anomalies is not None: # Only true anomalies plotted
            ax7.set_yticks([0, 1])
            ax7.set_yticklabels(['Normal', 'Anomaly'], fontsize=10)


        ax7.set_title(title_str + overlap_str, fontsize=14)
        ax7.set_ylabel('Status', fontsize=12)
        ax7.legend(fontsize=11)
        ax7.grid(True, linestyle='--', alpha=0.6)
        ax7.tick_params(axis='x', which='major', labelsize=10) # Show x labels only on last plot


    # --- Final Adjustments ---
    plt.suptitle(title, fontsize=16, y=0.99) # Adjust y position if needed
    fig.align_ylabels() # Align y-labels across subplots
    plt.tight_layout(rect=[0, 0.02, 1, 0.96]) # Adjust rect to prevent title overlap

    # --- Save or Show ---
    if save_path:
        ensure_dir(save_path) # Make sure directory exists
        plt.savefig(save_path, dpi=300, bbox_inches='tight') # Higher DPI, tight bbox
        print(f"Substitution diagnostic visualization saved to {save_path}")
        plt.close(fig) # Close the figure after saving
    else:
        # If not saving, return the figure object for display/further manipulation
        return fig
