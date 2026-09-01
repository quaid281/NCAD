import sys
import os
from pathlib import Path
sys.path.append('c:/Users/andre/OneDrive/Desktop/NCAD_CS')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.encoders.selective_ssm_encoder import SelectiveSSMContextEncoder
from src.models.losses.anomaly_injector import ContextualAnomalyInjector, AnomalyInjectionConfig
from src.models.memory.successor_memory import CounterfactualSuccessorMemory, SuccessorMemoryConfig
from src.scoring.event_fusion import (
    robust_stats, positive_robust_z, local_deviation_scores, 
    reconstruction_deviation_scores, fuse_evidence_scores,
    adaptive_elbow_score_floor, event_level_filter, compute_metrics,
    moving_average, aggregate_window_scores
)
from src.data.data_loader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

calit2_dir = Path('c:/Users/andre/OneDrive/Desktop/NCAD_CS/mTSBench_data/CalIt2')
train_df = pd.read_csv(calit2_dir / 'CalIt2_traffic_train.csv')
val_df = pd.read_csv(calit2_dir / 'CalIt2_traffic_val.csv')
test_df = pd.read_csv(calit2_dir / 'CalIt2_traffic_test.csv')

feature_cols = ['in_count', 'out_count']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_df[feature_cols])

train_features = scaler.transform(train_df[feature_cols])
val_features = scaler.transform(val_df[feature_cols])
test_features = scaler.transform(test_df[feature_cols])

context_size = 284
suspect_size = 16
window_size = context_size + suspect_size
step = 1

train_windows = DataLoader.create_windows(train_features, window_size, step)
val_windows = DataLoader.create_windows(val_features, window_size, step)
test_windows = DataLoader.create_windows(test_features, window_size, step)

model = SelectiveSSMContextEncoder(
    input_dim=len(feature_cols),
    latent_dim=16,
    hidden_dim=64,
    layers=4,
    dropout=0.10
).to(device)

def contrastive_loss(z_full, z_clean, labels, margin=1.0):
    dist = torch.norm(z_full - z_clean, p=2, dim=1)
    loss = (1.0 - labels) * (dist ** 2) + labels * torch.clamp(margin - dist, min=0.0) ** 2
    return loss.mean()

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
injector = ContextualAnomalyInjector(AnomalyInjectionConfig(injection_ratio=0.70), seed=42)

print("Training model...")
model.train()
for epoch in range(10):  # Train for 10 epochs
    epoch_indices = np.random.permutation(len(train_windows))
    for start in range(0, len(epoch_indices), 32):
        batch_idx = epoch_indices[start : start + 32]
        clean_batch = train_windows[batch_idx]
        modified_batch, labels = injector.inject_batch(clean_batch, context_size)
        
        full_tensor = torch.from_numpy(modified_batch).float().to(device)
        context_tensor = torch.from_numpy(clean_batch[:, :context_size]).float().to(device)
        label_tensor = torch.from_numpy(labels).float().to(device)
        
        optimizer.zero_grad(set_to_none=True)
        loss = contrastive_loss(model(full_tensor), model(context_tensor), label_tensor)
        loss.backward()
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

# Successor memory
memory = CounterfactualSuccessorMemory(
    SuccessorMemoryConfig(n_neighbors=8, max_memory_windows=5000, context_percentile=99.0, seed=42)
)
memory.fit(train_context_embeddings, train_successors)

# CRITICAL FIX: Add calibration_expected_successors to the memory fit calibration phase!
# Since we didn't modify successor_memory.py yet, we can query it on the calibration set using leave-one-out manually:
n_neighbors = memory.config.n_neighbors + 1
distances, indices = memory.neighbor_model.kneighbors(memory.context_embeddings, n_neighbors=n_neighbors)

clean_distances = []
clean_indices = []
for row_index, row_indices in enumerate(indices):
    keep = row_indices[row_indices != row_index]
    if len(keep) == 0:
        keep = row_indices[:1]
    keep = keep[: memory.config.n_neighbors]
    clean_indices.append(keep)
    distance_lookup = {int(index): float(distance) for index, distance in zip(row_indices, distances[row_index])}
    clean_distances.append([distance_lookup.get(int(index), 0.0) for index in keep])

max_len = max(len(row) for row in clean_indices)
padded_indices = np.zeros((len(clean_indices), max_len), dtype=np.int64)
padded_distances = np.zeros((len(clean_distances), max_len), dtype=np.float32)
for row_index, row in enumerate(clean_indices):
    padded_indices[row_index, : len(row)] = row
    padded_indices[row_index, len(row) :] = row[-1]
    padded_distances[row_index, : len(row)] = clean_distances[row_index]
    padded_distances[row_index, len(row) :] = clean_distances[row_index][-1]

# Re-run leave-one-out scoring to get the true calibration expected successors
neighbor_successors = memory.successor_windows[padded_indices]
calibration_expected_successors = np.median(neighbor_successors, axis=1).astype(np.float32)

# Check calibration
train_local_scores = local_deviation_scores(train_windows, context_size, tail_size=64)
calibration_local_scores = train_local_scores[memory.sample_indices]

# Use the manually computed leave-one-out expected successors
train_recon_scores = reconstruction_deviation_scores(
    train_successors[memory.sample_indices],
    calibration_expected_successors
)

successor_stats = robust_stats(memory.calibration_successor_scores)
local_stats = robust_stats(calibration_local_scores)
recon_stats = robust_stats(train_recon_scores)

print(f"Calibrated Recon Stats: median={recon_stats.median:.4f}, iqr={recon_stats.iqr:.4f}")

def compute_anomaly_scores(windows):
    contexts = windows[:, :context_size]
    observed_successors = windows[:, context_size:]
    context_embeddings = encode_windows(model, contexts)
    query = memory.query(context_embeddings, observed_successors)
    
    local_raw_scores = local_deviation_scores(windows, context_size, tail_size=64)
    successor_z = positive_robust_z(query.successor_scores, successor_stats)
    local_z = positive_robust_z(local_raw_scores, local_stats)
    
    recon_raw = reconstruction_deviation_scores(observed_successors, query.expected_successors)
    recon_z = positive_robust_z(recon_raw, recon_stats)
    
    if float(memory.context_threshold) <= 1e-6:
        context_ratio = np.ones_like(query.context_distances, dtype=np.float32)
    else:
        context_ratio = query.context_distances / float(memory.context_threshold)
        
    window_scores = fuse_evidence_scores(
        successor_z=successor_z,
        local_z=local_z,
        context_ratio=context_ratio,
        reconstruction_z=recon_z,
        successor_weight=1.0,
        local_weight=0.80,
        context_weight=0.35,
        reconstruction_weight=0.60,
    )
    return window_scores, successor_z, local_z, recon_z

val_window_scores, val_sz, val_lz, val_rz = compute_anomaly_scores(val_windows)
test_window_scores, test_sz, test_lz, test_rz = compute_anomaly_scores(test_windows)

print(f"Val Components (Post-Fix):")
print(f"  successor_z: median={np.median(val_sz):.4f}, max={np.max(val_sz):.4f}")
print(f"  local_z:     median={np.median(val_lz):.4f}, max={np.max(val_lz):.4f}")
print(f"  recon_z:     median={np.median(val_rz):.4f}, max={np.max(val_rz):.4f}")

# Threshold & evaluate
smoothing_window = 12

def process_point_scores(window_scores, raw_len):
    point_scores, valid_mask = aggregate_window_scores(
        window_scores,
        n_points=raw_len,
        context_size=context_size,
        suspect_size=suspect_size,
        step=step,
        reducer="mean"
    )
    smoothed = moving_average(point_scores, smoothing_window)
    return smoothed, valid_mask

val_scores, val_mask = process_point_scores(val_window_scores, len(val_df))
test_scores, test_mask = process_point_scores(test_window_scores, len(test_df))

val_valid_scores = val_scores[val_mask]
floor_res = adaptive_elbow_score_floor(val_valid_scores)
unsupervised_threshold = floor_res.threshold
print(f"Unsupervised Threshold: {unsupervised_threshold:.5f} ({floor_res.selected_candidate})")

val_labels = val_df['is_anomaly'].to_numpy()
best_f1 = 0.0
best_threshold = 0.0
candidates = np.unique(np.concatenate([
    np.linspace(np.percentile(val_valid_scores, 0.5), np.percentile(val_valid_scores, 50.0), 100),
    np.linspace(np.percentile(val_valid_scores, 50.0), np.percentile(val_valid_scores, 99.99), 400)
]))

for threshold in candidates:
    preds = event_level_filter(val_scores, threshold, val_mask, min_run=2, extreme_factor=1.75)
    preds = preds * val_mask.astype(np.float32)
    metrics = compute_metrics(val_labels, preds, valid_mask=val_mask)
    if metrics.get('f1', 0.0) > best_f1:
        best_f1 = metrics['f1']
        best_threshold = threshold

print(f"Validation Optimized Threshold: {best_threshold:.5f} (Val F1: {best_f1:.4f})")

test_labels = test_df['is_anomaly'].to_numpy()
for name, threshold in [("Unsupervised", unsupervised_threshold), ("Validation-Optimized", best_threshold)]:
    preds = event_level_filter(test_scores, threshold, test_mask, min_run=2, extreme_factor=1.75)
    preds = preds * test_mask.astype(np.float32)
    metrics = compute_metrics(test_labels, preds, valid_mask=test_mask)
    print(f"=== Test Set Evaluation: {name} Threshold ({threshold:.5f}) ===")
    print(f"  Precision:        {metrics.get('precision', 0.0):.4f}")
    print(f"  Recall:           {metrics.get('recall', 0.0):.4f}")
    print(f"  F1-Score:         {metrics.get('f1', 0.0):.4f}")
    print(f"  Confusion Matrix: TP={metrics.get('tp', 0)}, TN={metrics.get('tn', 0)}, FP={metrics.get('fp', 0)}, FN={metrics.get('fn', 0)}")
