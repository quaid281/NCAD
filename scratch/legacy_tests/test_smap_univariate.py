import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.anomaly_injector import AnomalyInjectionConfig, ContextualAnomalyInjector
from src.models.tcn_encoder import contrastive_loss
from src.models.selective_ssm_encoder import SelectiveSSMContextEncoder
from src.features.features import FeatureConfig, NCADFeatureExtractor
from src.models.successor_memory import CounterfactualSuccessorMemory, SuccessorMemoryConfig
from src.data.data_loader import DataLoader
from src.utils.event_fusion import (
    adaptive_elbow_score_floor,
    aggregate_window_scores,
    compute_metrics,
    event_level_filter,
    fuse_evidence_scores,
    local_deviation_scores,
    moving_average,
    robust_stats,
    positive_robust_z,
)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load A-1 dataset
train_df = pd.read_csv(project_root / "mTSBench_data" / "SMAP" / "SMAP_A-1_train.csv")
test_df = pd.read_csv(project_root / "mTSBench_data" / "SMAP" / "SMAP_A-1_test.csv")

test_labels = test_df["is_anomaly"].to_numpy()

# Extract feature_1 (column 0 of telemetry)
train_raw = train_df["feature_1"].to_numpy()
test_raw = test_df["feature_1"].to_numpy()

# Normalize training and test raw signal with train statistics
mean_val = float(np.mean(train_raw))
std_val = float(np.std(train_raw))
if std_val < 1e-8:
    std_val = 1.0

train_normalized = (train_raw - mean_val) / std_val
test_normalized = (test_raw - mean_val) / std_val

# Fit feature extractor
feature_cfg = FeatureConfig(max_features=64)
fe = NCADFeatureExtractor(feature_cfg)

train_features = fe.fit_transform(train_normalized)
test_features = fe.transform(test_normalized)

# Set window hyperparams
context_size = 284
suspect_size = 16
window_size = context_size + suspect_size
step = 1

# Create sliding windows
train_windows = DataLoader.create_windows(train_features, window_size, step)
test_windows = DataLoader.create_windows(test_features, window_size, step)

print(f"Train windows shape: {train_windows.shape}")
print(f"Test windows shape:  {test_windows.shape}")

# Model parameters
epochs = 5
batch_size = 32
latent_dim = 64
injection_ratio = 0.5
margin = 1.0

# Split train for validation early stopping
def split_train_validation(windows, val_split=0.1, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(windows))
    rng.shuffle(indices)
    n_val = int(len(indices) * val_split)
    return windows[indices[n_val:]], windows[indices[:n_val]]

training_data, val_data = split_train_validation(train_windows, val_split=0.1, seed=42)

model = SelectiveSSMContextEncoder(
    input_dim=train_features.shape[1],
    latent_dim=latent_dim,
    hidden_dim=64,
    layers=2,
    dropout=0.25
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
injector = ContextualAnomalyInjector(AnomalyInjectionConfig(injection_ratio=injection_ratio), seed=42)

# Training loop
for epoch in range(1, epochs + 1):
    model.train()
    epoch_indices = np.random.permutation(len(training_data))
    total_loss = 0.0
    total_count = 0
    
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
        optimizer.step()
        
    print(f"Epoch {epoch:02d}/{epochs}")

# Encode windows
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
train_successor_embeddings = encode_windows(model, train_successors)

memory = CounterfactualSuccessorMemory(
    SuccessorMemoryConfig(n_neighbors=8, max_memory_windows=5000, context_percentile=99.0, seed=42)
)
memory.fit(train_context_embeddings, train_successor_embeddings)

# Local deviation score (multivariate)
def local_deviation_scores_multivariate(windows, context_size, tail_size=64, chunk_size=2048):
    n_windows = windows.shape[0]
    out = np.empty(n_windows, dtype=np.float32)
    tail_start = max(0, context_size - tail_size)
    for start in range(0, n_windows, chunk_size):
        end = min(start + chunk_size, n_windows)
        chunk = np.asarray(windows[start:end], dtype=np.float64)            # (B, T, F)
        context_tail = chunk[:, tail_start:context_size, :]                  # (B, Tc, F)
        suspects = chunk[:, context_size:, :]                                # (B, Ts, F)
        medians = np.median(context_tail, axis=1)                            # (B, F)
        mad = np.median(np.abs(context_tail - medians[:, None, :]), axis=1)  # (B, F)
        scale = np.maximum(1.4826 * mad, 1e-4)                               # (B, F)
        diff = (suspects - medians[:, None, :]) / scale[:, None, :]          # (B, Ts, F)
        point_z = np.max(np.abs(diff), axis=1)                               # (B, F)
        mean_shift = np.abs(np.mean(suspects, axis=1) - medians) / scale     # (B, F)
        per_feature = np.maximum(point_z, mean_shift)                        # (B, F)
        out[start:end] = np.max(per_feature, axis=1).astype(np.float32)
    return out

print("Computing scores...")
train_local_scores = local_deviation_scores_multivariate(train_windows, context_size)
calibration_local_scores = train_local_scores[memory.sample_indices]

successor_stats = robust_stats(memory.calibration_successor_scores)
local_stats = robust_stats(calibration_local_scores)

# Query memory for test set
test_contexts = test_windows[:, :context_size]
test_successors = test_windows[:, context_size:]
test_context_embeddings = encode_windows(model, test_contexts)
test_successor_embeddings = encode_windows(model, test_successors)

query = memory.query(test_context_embeddings, test_successor_embeddings)
test_local_scores = local_deviation_scores_multivariate(test_windows, context_size)

successor_z = positive_robust_z(query.successor_scores, successor_stats)
local_z = positive_robust_z(test_local_scores, local_stats)

if float(memory.context_threshold) <= 1e-6:
    context_ratio = np.ones_like(query.context_distances, dtype=np.float32)
else:
    context_ratio = query.context_distances / float(memory.context_threshold)
context_ratio = np.minimum(context_ratio, 3.0).astype(np.float32)

window_scores = fuse_evidence_scores(
    successor_z=successor_z,
    local_z=local_z,
    context_ratio=context_ratio,
)

# Aggregate point scores
point_scores, valid_mask = aggregate_window_scores(
    window_scores,
    n_points=len(test_df),
    context_size=context_size,
    suspect_size=suspect_size,
    step=step,
    reducer="mean",
    mapping_method="middle"
)
test_scores = moving_average(point_scores, 12)

# Unsupervised Adaptive Elbow Threshold on test scores
floor_res = adaptive_elbow_score_floor(test_scores[valid_mask])
threshold = floor_res.threshold
print(f"Elbow Threshold: {threshold:.5f} (Selected candidate: {floor_res.selected_candidate})")

# Evaluate metrics
preds = event_level_filter(test_scores, threshold, valid_mask, min_run=2, extreme_factor=1.75)
preds = preds * valid_mask.astype(np.float32)

metrics_std = compute_metrics(test_labels, preds, valid_mask=valid_mask, use_pa=False)
metrics_pa = compute_metrics(test_labels, preds, valid_mask=valid_mask, use_pa=True)

print("Standard Metrics (No PA):")
print(metrics_std)
print("Point-Adjusted Metrics (PA):")
print(metrics_pa)
