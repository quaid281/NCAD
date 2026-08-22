import sys
import os
import time
from pathlib import Path
sys.path.append('c:/Users/andre/OneDrive/Desktop/NCAD_CS')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.selective_ssm_encoder import SelectiveSSMContextEncoder
from src.models.anomaly_injector import ContextualAnomalyInjector, AnomalyInjectionConfig
from src.models.successor_memory import CounterfactualSuccessorMemory, SuccessorMemoryConfig
from src.utils.event_fusion import (
    robust_stats, positive_robust_z, local_deviation_scores, 
    reconstruction_deviation_scores, fuse_evidence_scores
)
from src.data.data_loader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

calit2_dir = Path('c:/Users/andre/OneDrive/Desktop/NCAD_CS/mTSBench_data/CalIt2')
train_df = pd.read_csv(calit2_dir / 'CalIt2_traffic_train.csv')
val_df = pd.read_csv(calit2_dir / 'CalIt2_traffic_val.csv')
test_df = pd.read_csv(calit2_dir / 'CalIt2_traffic_test.csv')

feature_cols = ['in_count', 'out_count']

# Fit scaler
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

# Let's train the model for 5 epochs to get a basic representation
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

print("Training model for 5 epochs...")
model.train()
for epoch in range(5):
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

# Check calibration
train_local_scores = local_deviation_scores(train_windows, context_size, tail_size=64)
calibration_local_scores = train_local_scores[memory.sample_indices]

train_recon_scores = reconstruction_deviation_scores(
    train_successors[memory.sample_indices],
    memory.successor_windows
)

successor_stats = robust_stats(memory.calibration_successor_scores)
local_stats = robust_stats(calibration_local_scores)
recon_stats = robust_stats(train_recon_scores)

print(f"Calibration stats:")
print(f"  Successor stats: median={successor_stats.median:.4f}, iqr={successor_stats.iqr:.4f}")
print(f"  Local stats:     median={local_stats.median:.4f}, iqr={local_stats.iqr:.4f}")
print(f"  Recon stats:     median={recon_stats.median:.4f}, iqr={recon_stats.iqr:.4f}")

# Compute validation scores
contexts = val_windows[:, :context_size]
observed_successors = val_windows[:, context_size:]
context_embeddings = encode_windows(model, contexts)
query = memory.query(context_embeddings, observed_successors)

local_raw_scores = local_deviation_scores(val_windows, context_size, tail_size=64)

successor_z = positive_robust_z(query.successor_scores, successor_stats)
local_z = positive_robust_z(local_raw_scores, local_stats)

recon_raw = reconstruction_deviation_scores(observed_successors, query.expected_successors)
recon_z = positive_robust_z(recon_raw, recon_stats)

print(f"Validation score components:")
for name, z in [("successor_z", successor_z), ("local_z", local_z), ("recon_z", recon_z)]:
    print(f"  {name}: median={np.median(z):.4f}, mean={np.mean(z):.4f}, max={np.max(z):.4f}, p90={np.percentile(z, 90):.4f}")
