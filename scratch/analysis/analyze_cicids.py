import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Resolve paths
project_root = Path.cwd().resolve()
sys.path.insert(0, str(project_root))

from src.models.encoders.tcn_encoder import HybridTCNEncoder
from src.data.data_loader import DataLoader
from src.models.memory.successor_memory import CounterfactualSuccessorMemory, SuccessorMemoryConfig
from src.models.losses.anomaly_injector import ContextualAnomalyInjector, AnomalyInjectionConfig
from src.scoring.event_fusion import (
    robust_stats,
    positive_robust_z,
    local_deviation_scores,
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
train_path = cicids_dir / 'cicids_0_train.csv'
val_path = cicids_dir / 'cicids_0_val.csv'
test_path = cicids_dir / 'cicids_0_test.csv'

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

numeric_cols = train_df.select_dtypes(include='number').columns.tolist()
if 'is_anomaly' in numeric_cols:
    numeric_cols.remove('is_anomaly')

constant_cols = [c for c in numeric_cols if train_df[c].std() == 0]
feature_cols = [c for c in numeric_cols if c not in constant_cols]

scaler = StandardScaler()
scaler.fit(train_df[feature_cols])

train_scaled = scaler.transform(train_df[feature_cols])
val_scaled = scaler.transform(val_df[feature_cols])
test_scaled = scaler.transform(test_df[feature_cols])

context_size = 284
suspect_size = 16
window_size = context_size + suspect_size
step = 10  # Use step=10 as in notebook

train_windows = DataLoader.create_windows(train_scaled, window_size, step)
val_windows = DataLoader.create_windows(val_scaled, window_size, step)
test_windows = DataLoader.create_windows(test_scaled, window_size, step)

# Train model (fewer epochs to save time, 2 epochs is enough to check)
epochs = 2
batch_size = 32
learning_rate = 1e-3
weight_decay = 1e-5
margin = 1.0
val_split = 0.1
seed = 42

np.random.seed(seed)
torch.manual_seed(seed)

def split_train_validation(windows, val_split=0.1, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(windows))
    rng.shuffle(indices)
    n_val = int(len(indices) * val_split)
    return windows[indices[n_val:]], windows[indices[:n_val]]

training_data, val_data = split_train_validation(train_windows, val_split, seed)

input_dim = len(feature_cols)
model = HybridTCNEncoder(
    input_dim=input_dim,
    latent_dim=16,
    filters=64,
    tcn_layers=4,
    kernel_size=5,
    dropout=0.20
).to(device)

def contrastive_loss(z_full, z_context, labels, margin=1.0):
    distances = torch.linalg.norm(z_full - z_context, dim=1)
    positive_loss = (1.0 - labels) * distances.pow(2)
    negative_loss = labels * torch.nn.functional.relu(margin - distances).pow(2)
    return torch.mean(positive_loss + negative_loss)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
injector = ContextualAnomalyInjector(AnomalyInjectionConfig(injection_ratio=0.70), seed=seed)

print("Training model for 2 epochs...")
for epoch in range(1, epochs + 1):
    model.train()
    epoch_indices = np.random.permutation(len(training_data))
    for batch_start in range(0, len(epoch_indices), batch_size):
        batch_indices = epoch_indices[batch_start : batch_start + batch_size]
        clean_batch = training_data[batch_indices]
        modified_batch, labels = injector.inject_batch(clean_batch, context_size)
        full_tensor = torch.from_numpy(modified_batch).float().to(device)
        context_tensor = torch.from_numpy(clean_batch[:, :context_size]).float().to(device)
        label_tensor = torch.from_numpy(labels).float().to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = contrastive_loss(model(full_tensor), model(context_tensor), label_tensor, margin=margin)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

def encode_windows(model, windows, batch_size=32):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size]).float().to(device)
            embeddings.append(model(batch).cpu().numpy())
    return np.concatenate(embeddings, axis=0).astype(np.float32)

train_contexts = train_windows[:, :context_size]
train_successors = train_windows[:, context_size:]
train_context_embeddings = encode_windows(model, train_contexts)

memory = CounterfactualSuccessorMemory(
    SuccessorMemoryConfig(n_neighbors=8, max_memory_windows=5000, context_percentile=99.0, seed=42)
)
memory.fit(train_context_embeddings, train_successors)

# Check component stats
train_local_scores = local_deviation_scores(train_windows, context_size, tail_size=64)
calibration_local_scores = train_local_scores[memory.sample_indices]
successor_stats = robust_stats(memory.calibration_successor_scores)
local_stats = robust_stats(calibration_local_scores)

# Let's inspect test scores and components
contexts = test_windows[:, :context_size]
observed_successors = test_windows[:, context_size:]
context_embeddings = encode_windows(model, contexts)
query = memory.query(context_embeddings, observed_successors)

local_raw_scores = local_deviation_scores(test_windows, context_size, tail_size=64)
successor_z = positive_robust_z(query.successor_scores, successor_stats)
local_z = positive_robust_z(local_raw_scores, local_stats)
context_ratio = query.context_distances / float(memory.context_threshold)

fused_scores = fuse_evidence_scores(successor_z, local_z, context_ratio)

# Let's analyze what happens with a multivariate local deviation
def local_deviation_scores_multivariate(windows: np.ndarray, context_size: int, tail_size: int = 64) -> np.ndarray:
    raw_values = np.asarray(windows, dtype=np.float64) # (batch, window_size, n_features)
    context_tail = raw_values[:, max(0, context_size - tail_size) : context_size, :]
    suspects = raw_values[:, context_size:, :]
    
    medians = np.median(context_tail, axis=1) # (batch, n_features)
    mad = np.median(np.abs(context_tail - medians[:, None, :]), axis=1) # (batch, n_features)
    scale = np.maximum(1.4826 * mad, 1e-4) # (batch, n_features)
    
    point_z = np.max(np.abs((suspects - medians[:, None, :]) / scale[:, None, :]), axis=1) # (batch, n_features)
    mean_shift = np.abs(np.mean(suspects, axis=1) - medians) / scale # (batch, n_features)
    
    feature_scores = np.maximum(point_z, mean_shift) # (batch, n_features)
    return np.max(feature_scores, axis=1).astype(np.float32)

local_raw_scores_multi = local_deviation_scores_multivariate(test_windows, context_size, tail_size=64)
train_local_scores_multi = local_deviation_scores_multivariate(train_windows, context_size, tail_size=64)
calibration_local_scores_multi = train_local_scores_multi[memory.sample_indices]
local_stats_multi = robust_stats(calibration_local_scores_multi)
local_z_multi = positive_robust_z(local_raw_scores_multi, local_stats_multi)

fused_scores_multi = fuse_evidence_scores(successor_z, local_z_multi, context_ratio)

# Let's compute F1 for both
smoothing_window = 12
test_labels = test_df['is_anomaly'].to_numpy()

def evaluate(window_scores, name):
    point_scores, valid_mask = aggregate_window_scores(
        window_scores, n_points=len(test_df), context_size=context_size, suspect_size=suspect_size, step=step
    )
    smoothed = moving_average(point_scores, smoothing_window)
    # Search threshold
    best_f1 = 0
    best_th = 0
    for th in np.linspace(np.percentile(smoothed[valid_mask], 50), np.percentile(smoothed[valid_mask], 99.9), 200):
        preds = event_level_filter(smoothed, th, valid_mask, min_run=2, extreme_factor=1.75) * valid_mask.astype(np.float32)
        m = compute_metrics(test_labels, preds, valid_mask=valid_mask)
        f1 = m.get('f1', 0.0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_metrics = m
    print(f"\n[{name}] Best Test F1: {best_f1:.4f} at threshold {best_th:.4f}")
    print(f"Metrics: {best_metrics}")

print("=== Evaluation ===")
evaluate(fused_scores, "Original Fused")
evaluate(fused_scores_multi, "Multivariate Local Fused")
evaluate(successor_z, "Successor Z Only")
evaluate(local_z, "Univariate Local Z Only")
evaluate(local_z_multi, "Multivariate Local Z Only")
