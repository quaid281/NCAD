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

from src.models.encoders.selective_ssm_encoder import SelectiveSSMContextEncoder
from src.data.data_loader import DataLoader
from src.models.memory.successor_memory import CounterfactualSuccessorMemory, SuccessorMemoryConfig
from src.models.losses.anomaly_injector import ContextualAnomalyInjector, AnomalyInjectionConfig
from src.scoring.event_fusion import (
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
creditcard_dir = project_root / 'mTSBench_data' / 'creditcard'
train_df = pd.read_csv(creditcard_dir / 'creditcard_train.csv')
val_df = pd.read_csv(creditcard_dir / 'creditcard_val.csv')
test_df = pd.read_csv(creditcard_dir / 'creditcard_test.csv')

numeric_cols = train_df.select_dtypes(include='number').columns.tolist()
if 'is_anomaly' in numeric_cols:
    numeric_cols.remove('is_anomaly')

constant_cols = [c for c in numeric_cols if train_df[c].std() == 0]
feature_cols = [c for c in numeric_cols if c not in constant_cols]

scaler = StandardScaler().fit(train_df[feature_cols])

train_scaled = scaler.transform(train_df[feature_cols])
val_scaled = scaler.transform(val_df[feature_cols])
test_scaled = scaler.transform(test_df[feature_cols])

# Short Window Configuration: context=9, suspect=1, step=1
context_size = 9
suspect_size = 1
window_size = context_size + suspect_size
step = 1

train_windows = DataLoader.create_windows(train_scaled, window_size, step)
val_windows = DataLoader.create_windows(val_scaled, window_size, step)
test_windows = DataLoader.create_windows(test_scaled, window_size, step)

# Split train/val
epochs = 3
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
model = SelectiveSSMContextEncoder(
    input_dim=input_dim,
    latent_dim=16,
    hidden_dim=64,
    layers=2, # fewer layers since sequence is very short (10)
    dropout=0.10
).to(device)

def contrastive_loss(z_full, z_context, labels, margin=1.0):
    distances = torch.linalg.norm(z_full - z_context, dim=1)
    positive_loss = (1.0 - labels) * distances.pow(2)
    new_margin = margin
    negative_loss = labels * torch.nn.functional.relu(new_margin - distances).pow(2)
    return torch.mean(positive_loss + negative_loss)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
injector = ContextualAnomalyInjector(AnomalyInjectionConfig(injection_ratio=0.70), seed=seed)

print(f"Training model with context_size={context_size}, suspect_size={suspect_size}...")
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

successor_stats = robust_stats(memory.calibration_successor_scores)

# Query Successor Memory
contexts = test_windows[:, :context_size]
observed_successors = test_windows[:, context_size:]
context_embeddings = encode_windows(model, contexts)
query = memory.query(context_embeddings, observed_successors)

successor_z = positive_robust_z(query.successor_scores, successor_stats)

val_contexts = val_windows[:, :context_size]
val_observed_successors = val_windows[:, context_size:]
val_context_embeddings = encode_windows(model, val_contexts)
val_query = memory.query(val_context_embeddings, val_observed_successors)
val_successor_z = positive_robust_z(val_query.successor_scores, successor_stats)

test_labels = test_df['is_anomaly'].to_numpy()
val_labels = val_df['is_anomaly'].to_numpy()
smoothing_window = 1  # No smoothing for point anomalies!

def evaluate_threshold_strategy(min_run, name):
    val_point_scores, val_mask = aggregate_window_scores(val_successor_z, len(val_df), context_size, suspect_size, step)
    test_point_scores, test_mask = aggregate_window_scores(successor_z, len(test_df), context_size, suspect_size, step)
    
    val_valid_scores = val_point_scores[val_mask]
    best_f1 = 0
    best_th = 0
    candidates = np.linspace(np.percentile(val_valid_scores, 1.0), np.percentile(val_valid_scores, 99.99), 500)
    for th in candidates:
        preds = event_level_filter(val_point_scores, th, val_mask, min_run=min_run, extreme_factor=1.75) * val_mask.astype(np.float32)
        m = compute_metrics(val_labels, preds, valid_mask=val_mask)
        f1 = m.get('f1', 0.0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            
    # Apply to test set
    test_preds = event_level_filter(test_point_scores, best_th, test_mask, min_run=min_run, extreme_factor=1.75) * test_mask.astype(np.float32)
    test_metrics = compute_metrics(test_labels, test_preds, valid_mask=test_mask)
    
    print(f"\n--- Strategy: {name} (min_run={min_run}) ---")
    print(f"Optimal Val Threshold: {best_th:.5f} (Val F1: {best_f1:.4f})")
    print(f"Test Precision:        {test_metrics.get('precision', 0.0):.4f}")
    print(f"Test Recall:           {test_metrics.get('recall', 0.0):.4f}")
    print(f"Test F1-score:         {test_metrics.get('f1', 0.0):.4f}")
    print(f"Confusion Matrix:      TP={test_metrics.get('tp')}, TN={test_metrics.get('tn')}, FP={test_metrics.get('fp')}, FN={test_metrics.get('fn')}")

evaluate_threshold_strategy(min_run=1, name="Point Anomaly Filter")
