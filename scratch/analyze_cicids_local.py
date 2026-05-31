import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

project_root = Path.cwd().resolve()
sys.path.insert(0, str(project_root))

from src.models.tcn_encoder import HybridTCNEncoder
from src.data.data_loader import DataLoader
from src.models.successor_memory import CounterfactualSuccessorMemory, SuccessorMemoryConfig
from src.models.anomaly_injector import ContextualAnomalyInjector, AnomalyInjectionConfig
from src.utils.event_fusion import (
    robust_stats,
    positive_robust_z,
    fuse_evidence_scores,
    moving_average,
    aggregate_window_scores,
    event_level_filter,
    compute_metrics,
)
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
cicids_dir = project_root / 'mTSBench_data' / 'cicids'
train_df = pd.read_csv(cicids_dir / 'cicids_0_train.csv')
val_df = pd.read_csv(cicids_dir / 'cicids_0_val.csv')
test_df = pd.read_csv(cicids_dir / 'cicids_0_test.csv')

numeric_cols = train_df.select_dtypes(include='number').columns.tolist()
if 'is_anomaly' in numeric_cols:
    numeric_cols.remove('is_anomaly')

constant_cols = [c for c in numeric_cols if train_df[c].std() == 0]
feature_cols = [c for c in numeric_cols if c not in constant_cols]

scaler = StandardScaler().fit(train_df[feature_cols])

train_scaled = scaler.transform(train_df[feature_cols])
val_scaled = scaler.transform(val_df[feature_cols])
test_scaled = scaler.transform(test_df[feature_cols])

context_size = 284
suspect_size = 16
window_size = context_size + suspect_size
step = 10

train_windows = DataLoader.create_windows(train_scaled, window_size, step)
val_windows = DataLoader.create_windows(val_scaled, window_size, step)
test_windows = DataLoader.create_windows(test_scaled, window_size, step)

# Split train/val
training_data = train_windows

# Check different local deviation combinations
def get_local_dev(windows, context_size, method="univariate", tail_size=64):
    raw_values = np.asarray(windows, dtype=np.float64) # (batch, window_size, n_features)
    
    if method == "univariate":
        # Feature 0 only
        raw_values = raw_values[:, :, 0:1]
    
    context_tail = raw_values[:, max(0, context_size - tail_size) : context_size, :]
    suspects = raw_values[:, context_size:, :]
    
    medians = np.median(context_tail, axis=1) # (batch, n_features)
    mad = np.median(np.abs(context_tail - medians[:, None, :]), axis=1) # (batch, n_features)
    scale = np.maximum(1.4826 * mad, 1e-4) # (batch, n_features)
    
    point_z = np.max(np.abs((suspects - medians[:, None, :]) / scale[:, None, :]), axis=1) # (batch, n_features)
    mean_shift = np.abs(np.mean(suspects, axis=1) - medians) / scale # (batch, n_features)
    feature_scores = np.maximum(point_z, mean_shift) # (batch, n_features)
    
    if method == "univariate":
        return np.max(feature_scores, axis=1).astype(np.float32)
    elif method == "multivariate_max":
        return np.max(feature_scores, axis=1).astype(np.float32)
    elif method == "multivariate_mean":
        return np.mean(feature_scores, axis=1).astype(np.float32)
    elif method == "multivariate_rms":
        return np.sqrt(np.mean(feature_scores**2, axis=1)).astype(np.float32)
    else:
        raise ValueError(f"Unknown method {method}")

test_labels = test_df['is_anomaly'].to_numpy()
smoothing_window = 12
sample_indices = np.arange(len(train_windows)) # simple mock sample indices

def evaluate_local(method_name):
    # Train stats
    train_dev = get_local_dev(train_windows, context_size, method=method_name)
    stats = robust_stats(train_dev)
    
    # Test stats
    test_dev = get_local_dev(test_windows, context_size, method=method_name)
    test_z = positive_robust_z(test_dev, stats)
    
    # Eval
    point_scores, valid_mask = aggregate_window_scores(
        test_z, n_points=len(test_df), context_size=context_size, suspect_size=suspect_size, step=10
    )
    smoothed = moving_average(point_scores, smoothing_window)
    
    best_f1 = 0
    best_th = 0
    best_metrics = {}
    for th in np.linspace(np.percentile(smoothed[valid_mask], 50), np.percentile(smoothed[valid_mask], 99.9), 200):
        preds = event_level_filter(smoothed, th, valid_mask, min_run=2, extreme_factor=1.75) * valid_mask.astype(np.float32)
        m = compute_metrics(test_labels, preds, valid_mask=valid_mask)
        f1 = m.get('f1', 0.0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_metrics = m
    print(f"[{method_name}] Best Test F1: {best_f1:.4f} at threshold {best_th:.4f} (TP: {best_metrics.get('tp')}, FP: {best_metrics.get('fp')})")

print("=== CICIDS Local Deviation Combinations ===")
evaluate_local("univariate")
evaluate_local("multivariate_max")
evaluate_local("multivariate_mean")
evaluate_local("multivariate_rms")
