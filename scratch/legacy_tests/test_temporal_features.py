import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

project_root = Path.cwd().resolve()
while not (project_root / "mTSBench_data").exists() and project_root != project_root.parent:
    project_root = project_root.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'notebooks_v4'))

from src.models.anomaly_injector import AnomalyInjectionConfig, ContextualAnomalyInjector
from src.models.tcn_encoder import HybridTCNEncoder, contrastive_loss
from src.features.features import FeatureConfig, NCADFeatureExtractor
from src.data.data_loader import DataLoader
from src.models.successor_memory import CounterfactualSuccessorMemory, SuccessorMemoryConfig
from src.utils.event_fusion import (
    adaptive_elbow_score_floor,
    aggregate_window_scores,
    compute_metrics,
    event_level_filter,
    fuse_evidence_scores,
    moving_average,
    robust_stats,
    positive_robust_z,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Data
calit2_dir = project_root / 'mTSBench_data' / 'CalIt2'
train_df = pd.read_csv(calit2_dir / 'CalIt2_traffic_train.csv')
val_df = pd.read_csv(calit2_dir / 'CalIt2_traffic_val.csv')
test_df = pd.read_csv(calit2_dir / 'CalIt2_traffic_test.csv')

# 2. Extract Cyclic Temporal Features
def add_temporal_features(df):
    df_new = df.copy()
    dt = pd.to_datetime(df_new['timestamp'])
    hour = dt.dt.hour + dt.dt.minute / 60.0
    day = dt.dt.dayofweek
    df_new['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df_new['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    df_new['day_sin'] = np.sin(2 * np.pi * day / 7.0)
    df_new['day_cos'] = np.cos(2 * np.pi * day / 7.0)
    return df_new

train_df = add_temporal_features(train_df)
val_df = add_temporal_features(val_df)
test_df = add_temporal_features(test_df)

temporal_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']

# 3. Scaling Data (prevent leakage)
from sklearn.preprocessing import StandardScaler
feature_cols = ['in_count', 'out_count']
scaler = StandardScaler()
scaler.fit(train_df[feature_cols])

train_scaled = scaler.transform(train_df[feature_cols])
val_scaled = scaler.transform(val_df[feature_cols])
test_scaled = scaler.transform(test_df[feature_cols])

# 4. Feature Extraction (engineered features + temporal features)
feature_cfg = FeatureConfig(max_features=32)
fe_in = NCADFeatureExtractor(feature_cfg)
fe_out = NCADFeatureExtractor(feature_cfg)

train_feat_in = fe_in.fit_transform(train_scaled[:, 0])
train_feat_out = fe_out.fit_transform(train_scaled[:, 1])
# Concatenate engineered features and temporal features
train_features = np.concatenate([train_feat_in, train_feat_out, train_df[temporal_cols].values], axis=1)

val_feat_in = fe_in.transform(val_scaled[:, 0])
val_feat_out = fe_out.transform(val_scaled[:, 1])
val_features = np.concatenate([val_feat_in, val_feat_out, val_df[temporal_cols].values], axis=1)

test_feat_in = fe_in.transform(test_scaled[:, 0])
test_feat_out = fe_out.transform(test_scaled[:, 1])
test_features = np.concatenate([test_feat_in, test_feat_out, test_df[temporal_cols].values], axis=1)

# 5. Sliding Windows
context_size = 284
suspect_size = 16
window_size = context_size + suspect_size
step = 1

train_windows = DataLoader.create_windows(train_features, window_size, step)
val_windows = DataLoader.create_windows(val_features, window_size, step)
test_windows = DataLoader.create_windows(test_features, window_size, step)

# 6. Early-Stopping and Training
def split_train_validation(windows, val_split=0.1, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(windows))
    rng.shuffle(indices)
    n_val = int(len(indices) * val_split)
    return windows[indices[n_val:]], windows[indices[:n_val]]

epochs = 15
batch_size = 32
learning_rate = 1e-3
weight_decay = 1e-5
margin = 1.0
val_split = 0.1
patience = 5
seed = 42

np.random.seed(seed)
torch.manual_seed(seed)

training_data, val_data = split_train_validation(train_windows, val_split, seed)

input_dim = train_features.shape[1]
model = HybridTCNEncoder(input_dim=input_dim, latent_dim=16, filters=64, tcn_layers=4, kernel_size=5, dropout=0.20)
model = model.to(device)

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
        
    train_loss = total_train_loss / total_count
    
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

# 7. Fit Successor Memory
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

# 8. Fixed Multivariate Local Deviation Score
def local_deviation_scores_multivariate(windows, context_size, tail_size=64, raw_indices=[0, 32]):
    scores = []
    for idx in raw_indices:
        raw_values = np.asarray(windows[:, :, idx], dtype=np.float64)
        context_tail = raw_values[:, max(0, context_size - tail_size) : context_size]
        suspects = raw_values[:, context_size:]
        medians = np.median(context_tail, axis=1)
        mad = np.median(np.abs(context_tail - medians[:, None]), axis=1)
        scale = np.maximum(1.4826 * mad, 1e-4)
        point_z = np.max(np.abs((suspects - medians[:, None]) / scale[:, None]), axis=1)
        mean_shift = np.abs(np.mean(suspects, axis=1) - medians) / scale
        scores.append(np.maximum(point_z, mean_shift))
    return np.max(scores, axis=0).astype(np.float32)

# Compute Anomaly Scores
def compute_anomaly_scores(windows, dataset_name):
    # Obtain calibration stats from training memory
    train_local_scores = local_deviation_scores_multivariate(train_windows, context_size, tail_size=64)
    calibration_local_scores = train_local_scores[memory.sample_indices]
    
    successor_stats = robust_stats(memory.calibration_successor_scores)
    local_stats = robust_stats(calibration_local_scores)
    
    # Query memory
    contexts = windows[:, :context_size]
    observed_successors = windows[:, context_size:]
    context_embeddings = encode_windows(model, contexts)
    query = memory.query(context_embeddings, observed_successors)
    
    # Z-scores
    local_raw_scores = local_deviation_scores_multivariate(windows, context_size, tail_size=64)
    successor_z = positive_robust_z(query.successor_scores, successor_stats)
    local_z = positive_robust_z(local_raw_scores, local_stats)
    
    if float(memory.context_threshold) <= 1e-6:
        context_ratio = np.ones_like(query.context_distances, dtype=np.float32)
    else:
        context_ratio = query.context_distances / float(memory.context_threshold)
        
    window_scores = fuse_evidence_scores(successor_z, local_z, context_ratio)
    return window_scores

val_window_scores = compute_anomaly_scores(val_windows, "Validation")
test_window_scores = compute_anomaly_scores(test_windows, "Test")

# 9. Aggregation & Smoothing
def process_point_scores(window_scores, raw_len):
    point_scores, valid_mask = aggregate_window_scores(
        window_scores, n_points=raw_len, context_size=context_size, suspect_size=suspect_size, step=step, reducer="mean"
    )
    smoothed = moving_average(point_scores, 12)
    return smoothed, valid_mask

val_scores, val_mask = process_point_scores(val_window_scores, len(val_df))
test_scores, test_mask = process_point_scores(test_window_scores, len(test_df))

# 10. Threshold search & Test set eval
val_valid_scores = val_scores[val_mask]
floor_res = adaptive_elbow_score_floor(val_valid_scores)
unsupervised_threshold = floor_res.threshold

val_labels = val_df['is_anomaly'].to_numpy()
best_f1 = 0.0
best_threshold = 0.0
candidates = np.linspace(np.percentile(val_valid_scores, 50.0), np.percentile(val_valid_scores, 99.9), 300)
for th in candidates:
    preds = event_level_filter(val_scores, th, val_mask, min_run=2, extreme_factor=1.75)
    metrics = compute_metrics(val_labels, preds, valid_mask=val_mask)
    if metrics.get('f1', 0.0) > best_f1:
        best_f1 = metrics['f1']
        best_threshold = th

print(f"Validation F1 optimized: {best_f1:.4f} at threshold {best_threshold:.5f}")
print(f"Unsupervised Threshold: {unsupervised_threshold:.5f}")

test_labels = test_df['is_anomaly'].to_numpy()
for name, th in [("Unsupervised Elbow", unsupervised_threshold), ("Val Optimized", best_threshold)]:
    preds = event_level_filter(test_scores, th, test_mask, min_run=2, extreme_factor=1.75)
    preds = preds * test_mask.astype(np.float32)
    metrics = compute_metrics(test_labels, preds, valid_mask=test_mask)
    print(f"\n--- Test Evaluation: {name} ({th:.5f}) ---")
    print(f"Precision: {metrics.get('precision', 0.0):.4f}")
    print(f"Recall:    {metrics.get('recall', 0.0):.4f}")
    print(f"F1-Score:  {metrics.get('f1', 0.0):.4f}")
    print(f"Conf Matrix: TP={metrics.get('tp')}, TN={metrics.get('tn')}, FP={metrics.get('fp')}, FN={metrics.get('fn')}")
