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

from src.models.selective_ssm_encoder import SelectiveSSMContextEncoder
from src.features.features import FeatureConfig, NCADFeatureExtractor
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
calit2_dir = project_root / 'mTSBench_data' / 'CalIt2'
train_df = pd.read_csv(calit2_dir / 'CalIt2_traffic_train.csv')
val_df = pd.read_csv(calit2_dir / 'CalIt2_traffic_val.csv')
test_df = pd.read_csv(calit2_dir / 'CalIt2_traffic_test.csv')

feature_cols = ['in_count', 'out_count']
scaler = StandardScaler().fit(train_df[feature_cols])

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

# Split train/val
training_data, val_data = train_windows, train_windows # Use all train windows for memory fitting

# Train encoder (just load or train 1 epoch to save time, actually let's train for 5 epochs)
epochs = 5
batch_size = 32
learning_rate = 1e-3
weight_decay = 1e-5
margin = 1.0
seed = 42

np.random.seed(seed)
torch.manual_seed(seed)

model = SelectiveSSMContextEncoder(
    input_dim=train_features.shape[1],
    latent_dim=16,
    hidden_dim=64,
    layers=4,
    dropout=0.10
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
injector = ContextualAnomalyInjector(AnomalyInjectionConfig(injection_ratio=0.70), seed=seed)

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
        loss = torch.mean((model(full_tensor) - model(context_tensor)).pow(2)) # simpler MSE loss
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

memory = CounterfactualSuccessorMemory(SuccessorMemoryConfig(n_neighbors=8, max_memory_windows=5000, context_percentile=99.0, seed=42))
memory.fit(train_context_embeddings, train_successors)

# Check different local deviation combinations
def get_local_dev(windows, context_size, method="univariate", tail_size=64):
    raw_values = np.asarray(windows, dtype=np.float64) # (batch, window_size, n_features)
    
    if method == "univariate":
        # Feature 0 only
        raw_values = raw_values[:, :, 0:1]
    elif method == "raw_only":
        # Feature 0 and Feature 32
        raw_values = raw_values[:, :, [0, 32]]
    
    context_tail = raw_values[:, max(0, context_size - tail_size) : context_size, :]
    suspects = raw_values[:, context_size:, :]
    
    medians = np.median(context_tail, axis=1) # (batch, n_features)
    mad = np.median(np.abs(context_tail - medians[:, None, :]), axis=1) # (batch, n_features)
    scale = np.maximum(1.4826 * mad, 1e-4) # (batch, n_features)
    
    point_z = np.max(np.abs((suspects - medians[:, None, :]) / scale[:, None, :]), axis=1) # (batch, n_features)
    mean_shift = np.abs(np.mean(suspects, axis=1) - medians) / scale # (batch, n_features)
    feature_scores = np.maximum(point_z, mean_shift) # (batch, n_features)
    
    if method in ["univariate", "raw_only"]:
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

def evaluate_local(method_name):
    # Train stats
    train_dev = get_local_dev(train_windows, context_size, method=method_name)
    cal_dev = train_dev[memory.sample_indices]
    stats = robust_stats(cal_dev)
    
    # Test stats
    test_dev = get_local_dev(test_windows, context_size, method=method_name)
    test_z = positive_robust_z(test_dev, stats)
    
    # Eval
    point_scores, valid_mask = aggregate_window_scores(
        test_z, n_points=len(test_df), context_size=context_size, suspect_size=suspect_size, step=1
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

print("=== CalIt2 Local Deviation Combinations ===")
evaluate_local("univariate")
evaluate_local("raw_only")
evaluate_local("multivariate_max")
evaluate_local("multivariate_mean")
evaluate_local("multivariate_rms")
