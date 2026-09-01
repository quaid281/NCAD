import sys
import os
from pathlib import Path
sys.path.append('c:/Users/andre/OneDrive/Desktop/NCAD_CS')
import numpy as np
import pandas as pd
import torch

from src.scoring.event_fusion import robust_stats, positive_robust_z, local_deviation_scores, reconstruction_deviation_scores

# Load CalIt2 validation dataset to analyze
calit2_dir = Path('c:/Users/andre/OneDrive/Desktop/NCAD_CS/mTSBench_data/CalIt2')
val_df = pd.read_csv(calit2_dir / 'CalIt2_traffic_val.csv')
feature_cols = ['in_count', 'out_count']

# Let's check the scaling of val_df
from sklearn.preprocessing import StandardScaler
train_df = pd.read_csv(calit2_dir / 'CalIt2_traffic_train.csv')
scaler = StandardScaler()
scaler.fit(train_df[feature_cols])
val_scaled = scaler.transform(val_df[feature_cols])

from src.data.data_loader import DataLoader
context_size = 284
suspect_size = 16
window_size = context_size + suspect_size
step = 1

val_windows = DataLoader.create_windows(val_scaled, window_size, step)

# Let's print local deviation scores for feature 0, feature 1, and max
raw_f0 = val_windows[:, :, 0]
raw_f1 = val_windows[:, :, 1]

def local_dev_single_feature(windows_feat, context_size, tail_size=64):
    context_tail = windows_feat[:, max(0, context_size - tail_size) : context_size]
    suspects = windows_feat[:, context_size:]
    medians = np.median(context_tail, axis=1)
    mad = np.median(np.abs(context_tail - medians[:, None]), axis=1)
    scale = np.maximum(1.4826 * mad, 1e-4)
    point_z = np.max(np.abs((suspects - medians[:, None]) / scale[:, None]), axis=1)
    mean_shift = np.abs(np.mean(suspects, axis=1) - medians) / scale
    return np.maximum(point_z, mean_shift)

ld_f0 = local_dev_single_feature(raw_f0, context_size)
ld_f1 = local_dev_single_feature(raw_f1, context_size)
ld_max = np.maximum(ld_f0, ld_f1)

print("Local deviation stats for CalIt2 validation windows:")
for name, ld in [("Feature 0 (in_count)", ld_f0), ("Feature 1 (out_count)", ld_f1), ("Max of both", ld_max)]:
    print(f"{name}:")
    print(f"  Min:  {ld.min():.4f}")
    print(f"  Mean: {ld.mean():.4f}")
    print(f"  Max:  {ld.max():.4f}")
    print(f"  p50:  {np.percentile(ld, 50):.4f}")
    print(f"  p90:  {np.percentile(ld, 90):.4f}")
    print(f"  p99:  {np.percentile(ld, 99):.4f}")
