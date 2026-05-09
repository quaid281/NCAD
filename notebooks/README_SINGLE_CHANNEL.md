# NCAD-TCN Enhanced Anomaly Detection Pipeline

This enhanced version of the NCAD-TCN pipeline includes flexible single-channel processing capabilities and numerous improvements for better anomaly detection sensitivity.

## Key Features

- **Flexible Channel Selection**: Run on single channels, multiple specific channels, or all channels
- **Enhanced Memory Bank**: Stores both centroids and representative context windows
- **Robust Feature Extraction**: Improved stability for constant/near-constant signals
- **Dynamic Thresholding**: Adaptive thresholding based on signal volatility
- **Improved Model Architecture**: LayerNorm instead of BatchNorm, enhanced pooling strategies
- **Dynamic Smoothing**: Context-aware smoothing window selection
- **Command-line Interface**: Easy-to-use command-line arguments for channel selection

## Usage

### Command Line Options

```bash
# Run on all available channels (default)
python main.py

# Run on a single channel
python main.py --channel A-1

# Run on multiple specific channels
python main.py --channels A-1 A-2 D-4

# List all available channels
python main.py --list-channels

# Get help on all options
python main.py --help
```

### Windows Batch Scripts

For Windows users, convenient batch scripts are provided:

```batch
# Run on a single channel
run_single_channel.bat A-1

# List all available channels
list_channels.bat
```

### Legacy Method (still supported)

You can still use the legacy method by editing the main.py file directly:

```python
# Uncomment and modify this line in main.py around line 150:
channels = ["A-1"]  # Replace "A-1" with your desired channel
```

## Examples

### Example 1: Quick single-channel test
```bash
python main.py --channel A-1
```

### Example 2: Process a few specific channels
```bash
python main.py --channels A-1 A-2 D-4 E-1
```

### Example 3: Find available channels first
```bash
python main.py --list-channels
```
Output:
```
Available channels:
   1. A-1
   2. A-2
   3. A-3
   ...
```

### Example 4: Full batch processing (default)
```bash
python main.py
```

## Data Requirements

The pipeline expects the following directory structure:
```
data/
├── raw/
│   ├── train/
│   │   ├── A-1.npy
│   │   ├── A-2.npy
│   │   └── ...
│   └── test/
│       ├── A-1.npy
│       ├── A-2.npy
│       └── ...
└── processed/
    └── final_predictions.csv
```

The script will automatically check for missing data files and skip channels that don't have both train and test data available.

## Key Enhancements

### 1. Memory Bank Improvements
- **Context Window Storage**: Stores representative raw context windows alongside centroids
- **Robust Scoring**: Enhanced scoring logic using both centroid and context window similarities

### 2. Feature Extraction Robustness
- **Constant Signal Handling**: Improved stability for signals with little to no variation
- **Wavelet Features**: Enhanced wavelet decomposition with better error handling
- **Rolling Statistics**: More robust calculation of rolling statistics

### 3. Model Architecture Updates
- **LayerNorm**: Replaced BatchNorm1d with LayerNorm for better stability
- **Enhanced Pooling**: Added "last", "hybrid", and "traditional" pooling strategies
- **Increased Capacity**: Spectral pathway now uses 128 units (was 64)

### 4. Dynamic Thresholding
- **Volatility-Based**: Thresholds adapt based on training data volatility
- **Robust Methods**: Uses median + IQR for outlier-resistant thresholding
- **Minimum Floor**: Prevents unusably low thresholds

### 5. Dynamic Smoothing and Weighting
- **Context-Aware Smoothing**: Smoothing window adapts based on context abnormality
- **Confidence Weighting**: Confidence scores determine their own weights dynamically

## Output

The pipeline generates:
- **Run Summary**: CSV file with metrics for each processed channel
- **Visualizations**: Channel-specific plots saved in `enhanced_ncad_tcn_v3.5_results/`
- **Detailed Logs**: Processing logs and error information

## Error Handling

The pipeline includes robust error handling:
- Missing data files are automatically detected and reported
- Invalid channel names are caught and reported with suggestions
- Processing errors for individual channels don't stop the overall run
- Failed channels are logged in the run summary with status information

## Performance Tips

1. **Single Channel Testing**: Use `--channel` for quick testing and debugging
2. **Memory Management**: The pipeline includes automatic garbage collection
3. **Batch Processing**: For production runs, use the default (all channels) mode
4. **Resource Monitoring**: Watch memory usage during large batch runs

## Configuration

Key parameters can be modified in the main() function:
- `context_size`: Size of context window (default: 284)
- `suspect_size`: Size of suspect window (default: 16)
- `epochs`: Training epochs (default: 40)
- `n_clusters`: Number of memory bank clusters (default: 12)
- `latent_dim`: Latent dimension size (default: 16)

## Troubleshooting

### Common Issues

1. **Channel not found**: Use `--list-channels` to see available channels
2. **Missing data files**: Check that both train/A-1.npy and test/A-1.npy exist
3. **Memory errors**: Try processing fewer channels at once or increase system memory
4. **Import errors**: Ensure all required packages are installed (torch, numpy, pandas, etc.)

### Getting Help

```bash
python main.py --help
```

This will show all available options and examples.
