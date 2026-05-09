import os
import gc
import time
import sys
import argparse
import numpy as np
import pandas as pd
import torch # Import PyTorch
from datetime import datetime
import matplotlib.pyplot as plt
import warnings # Import warnings
from train import process_channel_enhanced

def check_channel_data(channel_list, data_dir):
    """Check if all channels have the required data files."""
    missing_train = []
    missing_test = []
    
    for channel in channel_list:
        channel_file = f'{channel}.npy'
        train_path = os.path.join(data_dir, 'train', channel_file)
        test_path = os.path.join(data_dir, 'test', channel_file)
        
        if not os.path.exists(train_path):
            missing_train.append(channel)
        if not os.path.exists(test_path):
            missing_test.append(channel)
    
    return missing_train, missing_test


def parse_args():
    """Parse command-line arguments for flexible channel selection."""
    parser = argparse.ArgumentParser(
        description="NCAD-TCN Enhanced Anomaly Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run on all available channels
  python main.py --channel A-1             # Run on single channel A-1
  python main.py --channels A-1 A-2 D-4    # Run on specific channels
  python main.py --list-channels           # List all available channels
  python main.py --all                     # Explicitly run on all channels (same as default)

Note: The script will automatically check for available data files and skip channels 
with missing train/test data.
        """
    )
    
    parser.add_argument(
        '--channel', 
        type=str, 
        default=None,
        help='Run on a single specific channel (e.g., "A-1", "D-4")'
    )
    
    parser.add_argument(
        '--channels', 
        type=str, 
        nargs='+',
        default=None,
        help='Run on multiple specific channels (e.g., --channels A-1 A-2 D-4)'
    )
    
    parser.add_argument(
        '--all', 
        action='store_true',
        help='Run on all available channels (default behavior)'
    )
    
    parser.add_argument(
        '--list-channels', 
        action='store_true',
        help='List all available channels and exit'
    )
    
    parser.add_argument(
        '--ablation-no-cmb',
        action='store_true',
        help='Run in ablation mode: disables the Context Memory Bank and all substitution logic.'
    )
    
    return parser.parse_args()


def main():
    """Main execution function for NCAD-TCN-GMM approach."""
    # Parse command-line arguments
    args = parse_args()
    
    # Enable garbage collection for better memory management
    print("Initializing memory management...")
    gc.enable()
    gc.collect()  # Initial garbage collection
    
    # --- ADD ABLATION MODE DETECTION ---
    is_ablation_run = args.ablation_no_cmb
    if is_ablation_run:
        print("\n" + "!"*30)
        print("!!! RUNNING IN ABLATION MODE !!!")
        print("!!! CMB and Substitution Logic will be DISABLED. !!!")
        print("!"*30 + "\n")
    # --- END ABLATION BLOCK ---
    
    # --- Configurable Parameters ---
    context_size = 284
    suspect_size = 16
    step = 1
    latent_dim = 16
    n_clusters = 100  # Changed from 12 to 100 for much larger number of clusters
    epochs = 40 # Reduced for faster testing, original was 20
    batch_size = 32
    distance_metric = 'euclidean'
    run_description = f"NCAD-TCN_GlobalPercentile_Enhanced_{distance_metric}_DistanceCheck_PyTorch" # Added PyTorch to desc
    if is_ablation_run:
        run_description += "_Ablation_No_CMB"  # Append to the run name for clarity

    # ... (other flags mostly disabled or standard) ...

    # Percentile for Memory Bank (Keep the 99.9 setting)
    mem_bank_threshold_method = 'percentile' # GOOD: Ensures percentile method is used
    mem_bank_fallback_percentile = 99.0
    use_silhouette_clustering = True  # Keep this on
    min_clusters = 1
    max_clusters = 150  # Also increase the search space for silhouette

    # Final Score Thresholding Params (Adaptive) - Base for Normal Scores
    low_vol_perc = 98.5  # Example
    med_vol_perc = 99.0  # Example
    high_vol_perc = 99.5 # Example
    default_final_perc = 99.5 # Significantly lower than 99.9    # Substitution parameters (boost factor only used if scoring logic uses it)
    substitution_boost_factor = 2.5 # GOOD: Increased as recommended for Step 2 (suggested 2.0-2.5)

    # Smoothing and Thresholding Params
    smoothing_window = 50  # REVERTED: Back to smoothing_window, e.g. 10 or 50

    # --- Derived Parameters ---
    full_window_size = context_size + suspect_size
    print(f"Derived Parameters: full_window_size = {full_window_size}") # Verify 

    # Set random seeds for reproducibility
    torch.manual_seed(42) # PyTorch seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42) # PyTorch CUDA seed
    np.random.seed(42)
    # Suppress specific warnings if needed
    # warnings.filterwarnings("ignore", category=ConvergenceWarning)
    # Suppress RuntimeWarning from skew/kurtosis in data_processing
    warnings.filterwarnings("ignore", category=RuntimeWarning, module='scipy.stats')


    # --- Path Definitions ---
    parent_dir = os.path.dirname(os.getcwd())
    if os.path.basename(os.getcwd()) == 'notebooks':
         parent_dir = os.path.dirname(parent_dir)

    # Define base data directory more robustly
    base_data_dir = os.path.join(parent_dir, 'Single Pass TCN Autoencoder', 'data')
    train_data_dir = os.path.join(base_data_dir, 'raw', 'train')
    labeled_anomalies_file = os.path.join(base_data_dir, 'processed', 'final_predictions.csv')
    master_results_path = os.path.join(parent_dir, 'Single Pass TCN Autoencoder', "master_results.csv")

    # --- Run Setup ---
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(parent_dir, "run_results", run_timestamp)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Run ID: {run_timestamp}")
    print(f"Results will be saved in: {results_dir}")
    consolidated_results_file = os.path.join(results_dir, f"{run_timestamp}_consolidated_results.csv")    # --- Load Channel List ---
    labeled_anomalies = pd.read_csv(labeled_anomalies_file)
    all_channels = sorted(labeled_anomalies['chan_id'].unique())
    print(f"Found {len(all_channels)} channels in the labeled anomalies file: {all_channels[:5]}...")

    # Handle --list-channels argument
    if args.list_channels:
        print("\nAvailable channels:")
        for i, channel in enumerate(all_channels, 1):
            print(f"  {i:2d}. {channel}")
        print(f"\nTotal: {len(all_channels)} channels")
        return None, None

    # --- Check for available data files ---
    print("Checking for available data files for all channels...")
    raw_data_dir = os.path.join(base_data_dir, 'raw')
    missing_train, missing_test = check_channel_data(all_channels, raw_data_dir)

    if missing_train:
        print(f"Warning: {len(missing_train)} channels are missing train data: {missing_train}")
    if missing_test:
        print(f"Warning: {len(missing_test)} channels are missing test data: {missing_test}")
    
    # Filter channels to only include those with available data
    valid_channels = [ch for ch in all_channels if ch not in missing_train and ch not in missing_test]
    if len(valid_channels) < len(all_channels):
        print(f"Filtered to {len(valid_channels)} channels with available data (out of {len(all_channels)} total).")

    # --- Channel Selection Based on Arguments ---
    if args.channel:
        # Single channel specified
        if args.channel not in valid_channels:
            if args.channel in all_channels:
                print(f"Error: Channel '{args.channel}' is missing required data files.")
            else:
                print(f"Error: Channel '{args.channel}' not found in the labeled anomalies file.")
                print(f"Available channels: {', '.join(all_channels[:10])}{'...' if len(all_channels) > 10 else ''}")
            return None, None
        channels = [args.channel]
        print(f"Running on single channel: {args.channel}")
        
    elif args.channels:
        # Multiple channels specified
        invalid_channels = [ch for ch in args.channels if ch not in valid_channels]
        if invalid_channels:
            missing_from_data = [ch for ch in invalid_channels if ch in all_channels]
            not_in_file = [ch for ch in invalid_channels if ch not in all_channels]
            
            if missing_from_data:
                print(f"Error: These channels are missing required data files: {missing_from_data}")
            if not_in_file:
                print(f"Error: These channels were not found in the labeled anomalies file: {not_in_file}")
                print(f"Available channels: {', '.join(all_channels[:10])}{'...' if len(all_channels) > 10 else ''}")
            return None, None
        
        channels = [ch for ch in args.channels if ch in valid_channels]
        print(f"Running on {len(channels)} specified channels: {channels}")
        
    else:
        # Default: use all valid channels (or --all was specified)
        channels = valid_channels
        print(f"Running on all {len(channels)} available channels")

    # Legacy single channel selection (commented out for reference)
    # Uncomment and modify the line below to run only one specific channel
    # channels = ["A-1"]  # Replace "A-1" with your desired channel
    # channels = ["A-3"]  # Example: run only A-3
    # channels = ["D-4"]  # Example: run only D-4
    print(f"Final selected channels for processing: {channels[:5]}{'...' if len(channels) > 5 else ''} (Total: {len(channels)})")

    volatilities = {}
    print("\nCalculating training data volatility for threshold adaptation...")
    for stream in channels:
        channel_file = f'{stream}.npy'
        train_path = os.path.join(train_data_dir, channel_file)
        train_data_full = np.load(train_path)

        if len(train_data_full.shape) > 1 and train_data_full.shape[1] > 0:
            train_telemetry = train_data_full[:, 0].flatten()
        elif len(train_data_full.shape) == 1:
            train_telemetry = train_data_full
        else:
            print(f"Warning: Skipping channel {stream} due to unexpected shape {train_data_full.shape}")
            continue

        # Calculate volatility (standard deviation)
        volatility = np.std(train_telemetry) if np.std(train_telemetry) > 1e-9 else 0.0
        volatilities[stream] = volatility

    # Create DataFrame and determine quantiles
    volatility_df = pd.DataFrame(list(volatilities.items()), columns=['channel', 'volatility'])
    volatility_df = volatility_df[volatility_df['volatility'] >= 0] # Exclude errors

    adaptive_percentiles = {}
    if not volatility_df.empty:
        # Using terciles (33rd, 66th percentiles) to split into low, medium, high
        q1, q2 = volatility_df['volatility'].quantile([0.33, 0.66])
        print(f"Volatility Quantiles (33rd, 66th): {q1:.4f}, {q2:.4f}")
        print(f"Mapping to Percentiles (Low/Med/High): {low_vol_perc}% / {med_vol_perc}% / {high_vol_perc}%")

        for _, row in volatility_df.iterrows():
            vol = row['volatility']
            if vol <= q1:
                adaptive_percentiles[row['channel']] = low_vol_perc
            elif vol <= q2:
                adaptive_percentiles[row['channel']] = med_vol_perc
            else:
                adaptive_percentiles[row['channel']] = high_vol_perc

        print("Example mapping (first 5):")
        for ch in list(adaptive_percentiles.keys())[:5]:
                print(f"  {ch}: Volatility={volatilities.get(ch, 'N/A'):.4f} -> Percentile={adaptive_percentiles[ch]}%")
    else:
        print("No valid volatilities calculated. Using fixed default percentile for all channels.")
        # Set default percentiles for all channels
        for ch in channels:
            adaptive_percentiles[ch] = default_final_perc

    # Define NEW columns for the run summary, including metrics
    run_summary_columns = [
        'channel', 'run_description',
        'context_size', 'suspect_size', # Keep key parameters
        # Add Metric Columns
        'TP', 'TN', 'FP', 'FN',
        'Precision', 'Recall', 'F1',
        'Status' # Add a status column (Success/Failure)
    ]
    run_summary_df = pd.DataFrame(columns=run_summary_columns)

    # --- Processing Loop ---
    print(f"\nProcessing {len(channels)} channels with {run_description}...")
    overall_start_time = time.time()

    for i, stream in enumerate(channels):
        print(f"\n[{i+1}/{len(channels)}] Processing {stream}...")
        channel_metrics = None # Initialize metrics for this channel
        status = "Success"     # Default status

        try:
            if stream in adaptive_percentiles:
                target_final_percentile = adaptive_percentiles[stream]
                print(f"  Using adaptive final threshold percentile: {target_final_percentile}% (based on volatility)")
            else:
                target_final_percentile = default_final_perc # Use default fallback
                print(f"  Using default final threshold percentile: {target_final_percentile}%")

            # Force garbage collection before processing to free memory
            gc.collect()

            channel_metrics = process_channel_enhanced (
                # --- Core Identification ---
                stream=stream,

                # --- Windowing ---
                full_window_size=full_window_size,
                context_size=context_size,
                step=step,

                # --- Model & Training ---
                epochs=epochs,
                batch_size=batch_size,
                latent_dim=latent_dim,

                # --- Memory Bank ---
                n_clusters=n_clusters,
                percentile=mem_bank_fallback_percentile,         # CORRECT: Pass the percentile value
                mem_bank_threshold_method=mem_bank_threshold_method, # CORRECT: Pass the method name
                use_silhouette_clustering=use_silhouette_clustering,
                min_clusters=min_clusters,
                max_clusters=max_clusters,

                # --- Final Thresholding (Context-Modulated Approach) ---
                base_threshold_percentile=target_final_percentile,
                density_based_adjustment=False,

                # --- Substitution Scoring ---
                substitution_boost_factor=substitution_boost_factor,

                # --- Smoothing ---
                smoothing_window=smoothing_window,

                # --- Core Params ---
                distance_metric=distance_metric,

                # --- Feature Flags ---
                enable_change_detection=True,
                enable_transition_detection=True,

                # --- DISABLED HEURISTICS ---
                use_complexity_thresholding=False,
                use_regime_aware_mb_threshold=False,
                use_transition_substitution_gating=False,
                use_duration_hypothesis=False,
                use_multimodal_detection=False,

                # --- DISABLED HEURISTICS Values ---
                constant_signal_factor=3.0,
                periodic_signal_factor=1.5,

                # --- Plotting ---
                plot_results=True,
                
                # --- Ablation Mode ---
                is_ablation_run=is_ablation_run
            )

            # --- Prepare result row (now includes metrics) ---
            result_row = {
                'channel': stream,
                'run_description': run_description,
                'context_size': context_size,
                'suspect_size': suspect_size,
                # Populate metrics using .get() for safety if channel_metrics is None
                'TP': channel_metrics.get('TP', np.nan) if channel_metrics else np.nan,
                'TN': channel_metrics.get('TN', np.nan) if channel_metrics else np.nan,
                'FP': channel_metrics.get('FP', np.nan) if channel_metrics else np.nan,
                'FN': channel_metrics.get('FN', np.nan) if channel_metrics else np.nan,
                'Precision': channel_metrics.get('Precision', np.nan) if channel_metrics else np.nan,
                'Recall': channel_metrics.get('Recall', np.nan) if channel_metrics else np.nan,
                'F1': channel_metrics.get('F1', np.nan) if channel_metrics else np.nan,
                'Status': status
            }
            # Add to run-specific summary
            run_summary_df.loc[len(run_summary_df)] = result_row

        except Exception as e:
            print(f"ERROR: Channel {stream} processing failed with exception: {str(e)}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            status = "Failure" # Update status
            # Log the error and add a failed row to the summary
            failed_row = {col: np.nan for col in run_summary_columns} # Fill with NaN
            failed_row['channel'] = stream
            failed_row['run_description'] = run_description
            failed_row['Status'] = status
            run_summary_df.loc[len(run_summary_df)] = failed_row

            # Continue with the next channel
            continue


    # --- Final Calculations and Summary ---
    total_time = time.time() - overall_start_time
    print(f"\nTotal processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    # Save final run summary (which now includes metrics)
    run_summary_path = os.path.join(results_dir, f"{run_timestamp}_run_summary.csv")
    run_summary_df.to_csv(run_summary_path, index=False)
    print(f"\nRun summary with metrics saved to {run_summary_path}")

    # --- Final Output ---
    # Removed reference to consolidated_results_file as we are using run_summary_df
    # print(f"\nConsolidated results for this run saved to {consolidated_results_file}")
    print(f"Run summary saved to {run_summary_path}")
    print(f"Aggregate visualizations saved in: {results_dir}")
    # Corrected path for channel-specific results based on train.py
    print(f"Channel-specific visualizations saved in: enhanced_ncad_tcn_v3.5_results/ (relative path)")
    print("\nProcessing complete!")

    # Return run_summary_df instead of results_df if needed elsewhere
    return run_timestamp, run_summary_df

if __name__ == "__main__":
    result = main()
    if result is not None:
        run_timestamp, run_summary_df = result
        print(f"\nRun completed successfully: {run_timestamp}")
    else:
        print("\nProgram exited early.")