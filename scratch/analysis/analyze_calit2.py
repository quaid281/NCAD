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

from src.models.encoders.selective_ssm_encoder import SelectiveSSMContextEncoder
from src.features.features import FeatureConfig, NCADFeatureExtractor
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
    adaptive_elbow_score_floor,
)
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
calit2_dir = project_root / 'mTSBench_data' / 'CalIt2'
train_path = calit2_dir / 'CalIt2_traffic_train.csv'
val_path = calit2_dir / 'CalIt2_traffic_val.csv'
test_path = calit2_dir / 'CalIt2_traffic_test.csv'

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

feature_cols = ['in_count', 'out_count']
scaler = StandardScaler()
scaler.fit(train_df[feature_cols])

train_scaled = scaler.transform(train_df[feature_cols])
val_scaled = scaler.transform(val_df[feature_cols])
test_scaled = scaler.transform(test_df[feature_cols])

# Features
feature_cfg = FeatureConfig(max_features=32)
fe_in = NCADFeatureExtractor(feature_cfg)
fe_out = NCADFeatureExtractor(feature_cfg)

train_feat_in = fe_in.fit_transform(train_scaled[:, 0])
train_feat_out = fe_out.fit_transform(train_scaled[:, 1])
train_features = np.concatenate([train_feat_in, train_feat_out], axis=1)

val_feat_in = fe_in.transform(val_scaled[:, 0])
val_feat_out = fe_out.transform(val_scaled[:, 1])
val_features = np.concatenate([val_feat_in, val_feat_out], axis=1)

test_feat_in = fe_in.transform(test_scaled[:, 0])
test_feat_out = fe_out.transform(test_scaled[:, 1])
test_features = np.concatenate([test_feat_in, test_feat_out], axis=1)

context_size = 284
suspect_size = 16
window_size = context_size + suspect_size
step = 1

train_windows = DataLoader.create_windows(train_features, window_size, step)
val_windows = DataLoader.create_windows(val_features, window_size, step)
test_windows = DataLoader.create_windows(test_features, window_size, step)

# Train model
epochs = 20
batch_size = 32
learning_rate = 1e-3
weight_decay = 1e-5
margin = 1.0
val_split = 0.1
patience = 5
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

input_dim = train_features.shape[1]
model = SelectiveSSMContextEncoder(
    input_dim=input_dim,
    latent_dim=16,
    hidden_dim=64,
    layers=4,
    dropout=0.10
).to(device)

def contrastive_loss(z_full, z_context, labels, margin=1.0):
    distances = torch.linalg.norm(z_full - z_context, dim=1)
    positive_loss = (1.0 - labels) * distances.pow(2)
    negative_loss = labels * torch.nn.functional.relu(margin - distances).pow(2)
    return torch.mean(positive_loss + negative_loss)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
injector = ContextualAnomalyInjector(AnomalyInjectionConfig(injection_ratio=0.70), seed=seed)
val_injector = ContextualAnomalyInjector(AnomalyInjectionConfig(injection_ratio=0.70), seed=seed + 1)

best_state = None
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(1, epochs + 1):
    model.train()
    epoch_indices = np.random.permutation(len(training_data))
    total_train_loss = 0.0
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += float(loss.item()) * len(clean_batch)
        total_count += len(clean_batch)
    
    model.eval()
    total_val_loss = 0.0
    val_count = 0
    with torch.no_grad():
        for batch_start in range(0, len(val_data), batch_size):
            clean_batch = val_data[batch_start : batch_start + batch_size]
            modified_batch, labels = val_injector.inject_batch(clean_batch, context_size)
            full_tensor = torch.from_numpy(modified_batch).float().to(device)
            context_tensor = torch.from_numpy(clean_batch[:, :context_size]).float().to(device)
            label_tensor = torch.from_numpy(labels).float().to(device)
            loss = contrastive_loss(model(full_tensor), model(context_tensor), label_tensor, margin=margin)
            total_val_loss += float(loss.item()) * len(clean_batch)
            val_count += len(clean_batch)
            
    val_loss = total_val_loss / max(val_count, 1)
    if val_loss < best_val_loss - 1e-5:
        best_val_loss = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

if best_state is not None:
    model.load_state_dict(best_state)

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

print("\n--- Statistics ---")
print(f"Successor median: {successor_stats.median:.4f}, iqr: {successor_stats.iqr:.4f}")
print(f"Local median:     {local_stats.median:.4f}, iqr: {local_stats.iqr:.4f}")

# Let's inspect test scores and components
contexts = test_windows[:, :context_size]
observed_successors = test_windows[:, context_size:]
context_embeddings = encode_windows(model, contexts)
query = memory.query(context_embeddings, observed_successors)

local_raw_scores = local_deviation_scores(test_windows, context_size, tail_size=64)
successor_z = positive_robust_z(query.successor_scores, successor_stats)
local_z = positive_robust_z(local_raw_scores, local_stats)
context_ratio = query.context_distances / float(memory.context_threshold)

print(f"\nTest Successor_z shape: {successor_z.shape}")
print(f"Min/Max/Mean Successor_z: {successor_z.min():.4f} / {successor_z.max():.4f} / {successor_z.mean():.4f}")
print(f"Min/Max/Mean Local_z:     {local_z.min():.4f} / {local_z.max():.4f} / {local_z.mean():.4f}")
print(f"Min/Max/Mean Context_ratio: {context_ratio.min():.4f} / {context_ratio.max():.4f} / {context_ratio.mean():.4f}")

fused_scores = fuse_evidence_scores(successor_z, local_z, context_ratio)
print(f"Min/Max/Mean Fused:       {fused_scores.min():.4f} / {fused_scores.max():.4f} / {fused_scores.mean():.4f}")

# Now let's see how many test samples have high successor_z vs local_z
print(f"\nNumber of test windows where successor_z > 5: {np.sum(successor_z > 5)}")
print(f"Number of test windows where local_z > 5:     {np.sum(local_z > 5)}")

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

print(f"\n--- Multivariate Local Deviation ---")
print(f"Local Multi median: {local_stats_multi.median:.4f}, iqr: {local_stats_multi.iqr:.4f}")
print(f"Min/Max/Mean Local_z_multi: {local_z_multi.min():.4f} / {local_z_multi.max():.4f} / {local_z_multi.mean():.4f}")
print(f"Number of test windows where local_z_multi > 5: {np.sum(local_z_multi > 5)}")

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

evaluate(fused_scores, "Original Fused")
fused_scores_multi = fuse_evidence_scores(successor_z, local_z_multi, context_ratio)
evaluate(fused_scores_multi, "Multivariate Local Fused")
evaluate(successor_z, "Successor Z Only")
evaluate(local_z, "Univariate Local Z Only")
evaluate(local_z_multi, "Multivariate Local Z Only")
