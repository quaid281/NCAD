import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)

import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Add required paths
project_root = Path.cwd().resolve()
while not (project_root / 'mTSBench_data').exists() and project_root != project_root.parent:
    project_root = project_root.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.anomaly_injector import AnomalyInjectionConfig, ContextualAnomalyInjector
from src.models.tcn_encoder import contrastive_loss
from src.models.selective_ssm_encoder import SelectiveSSMContextEncoder
from src.models.successor_memory import CounterfactualSuccessorMemory, SuccessorMemoryConfig
from src.data.data_loader import DataLoader
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

device = torch.device("cpu")
dataset_dir = project_root / 'mTSBench_data' / 'SMD'

context_size = 284
suspect_size = 16
window_size = context_size + suspect_size
batch_size = 64
latent_dim = 64
injection_ratio = 0.5
margin = 1.0

def local_deviation_scores_custom(windows, context_size, tail_size=64, mad_floor=0.20):
    n_windows = windows.shape[0]
    out = np.empty(n_windows, dtype=np.float32)
    tail_start = max(0, context_size - tail_size)
    chunk_size = 2048
    for start in range(0, n_windows, chunk_size):
        end = min(start + chunk_size, n_windows)
        chunk = np.asarray(windows[start:end], dtype=np.float64)
        context_tail = chunk[:, tail_start:context_size, :]
        suspects = chunk[:, context_size:, :]
        medians = np.median(context_tail, axis=1)
        mad = np.median(np.abs(context_tail - medians[:, None, :]), axis=1)
        scale = np.maximum(1.4826 * mad, mad_floor)
        diff = (suspects - medians[:, None, :]) / scale[:, None, :]
        point_z = np.max(np.abs(diff), axis=1)
        mean_shift = np.abs(np.mean(suspects, axis=1) - medians) / scale
        per_feature = np.maximum(point_z, mean_shift)
        out[start:end] = np.max(per_feature, axis=1).astype(np.float32)
    return out

def split_train_validation(windows, val_split=0.1, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(windows))
    rng.shuffle(indices)
    n_val = int(len(indices) * val_split)
    return windows[indices[n_val:]], windows[indices[:n_val]]

def encode_windows(model, windows, batch_size=64):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size]).float().to(device)
            embeddings.append(model(batch).cpu().numpy())
    return np.concatenate(embeddings, axis=0).astype(np.float32)

def run_channel(chan, std_floor=0.01, mad_floor=0.20):
    train_df = pd.read_csv(dataset_dir / f"SMD_{chan}_train.csv")
    test_df = pd.read_csv(dataset_dir / f"SMD_{chan}_test.csv")
    numeric_cols = [c for c in train_df.columns if c not in ["timestamp", "is_anomaly"]]
    test_labels = test_df["is_anomaly"].to_numpy()
    
    train_mean = train_df[numeric_cols].mean().to_numpy()
    train_std = train_df[numeric_cols].std().to_numpy()
    scale = np.maximum(train_std, std_floor)
    
    train_scaled = (train_df[numeric_cols].to_numpy() - train_mean) / scale
    test_scaled = (test_df[numeric_cols].to_numpy() - train_mean) / scale
    
    train_windows = DataLoader.create_windows(train_scaled, window_size, step=6) # subsample for speed
    test_windows = DataLoader.create_windows(test_scaled, window_size, step=1)
    
    training_data, val_data = split_train_validation(train_windows, val_split=0.1, seed=42)
    
    model = SelectiveSSMContextEncoder(
        input_dim=len(numeric_cols), latent_dim=latent_dim, hidden_dim=64, layers=2, dropout=0.25
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    injector = ContextualAnomalyInjector(AnomalyInjectionConfig(injection_ratio=injection_ratio), seed=42)
    
    for epoch in range(1, 2): # 1 epoch is fast
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
            
    train_contexts = train_windows[:, :context_size]
    train_successors = train_windows[:, context_size:]
    train_context_embeddings = encode_windows(model, train_contexts)
    train_successor_embeddings = encode_windows(model, train_successors)
    
    memory = CounterfactualSuccessorMemory(
        SuccessorMemoryConfig(n_neighbors=8, max_memory_windows=3000, context_percentile=99.0, seed=42)
    )
    memory.fit(train_context_embeddings, train_successor_embeddings)
    
    train_local_scores = local_deviation_scores_custom(train_windows, context_size, mad_floor=mad_floor)
    calibration_local_scores = train_local_scores[memory.sample_indices]
    successor_stats = robust_stats(memory.calibration_successor_scores)
    local_stats = robust_stats(calibration_local_scores)
    
    test_contexts = test_windows[:, :context_size]
    test_successors = test_windows[:, context_size:]
    test_context_embeddings = encode_windows(model, test_contexts)
    test_successor_embeddings = encode_windows(model, test_successors)
    
    query = memory.query(test_context_embeddings, test_successor_embeddings)
    test_local_scores = local_deviation_scores_custom(test_windows, context_size, mad_floor=mad_floor)
    
    successor_z = positive_robust_z(query.successor_scores, successor_stats)
    local_z = positive_robust_z(test_local_scores, local_stats)
    
    if float(memory.context_threshold) <= 1e-6:
        context_ratio = np.ones_like(query.context_distances, dtype=np.float32)
    else:
        context_ratio = query.context_distances / float(memory.context_threshold)
    context_ratio = np.minimum(context_ratio, 3.0).astype(np.float32)
    
    window_scores = fuse_evidence_scores(successor_z=successor_z, local_z=local_z, context_ratio=context_ratio)
    point_scores, valid_mask = aggregate_window_scores(
        window_scores, n_points=len(test_df), context_size=context_size, suspect_size=suspect_size, step=1, reducer="mean", mapping_method="middle"
    )
    test_scores = moving_average(point_scores, 12)
    
    best_pa_f1 = 0.0
    candidates = np.linspace(np.percentile(test_scores[valid_mask], 50.0), np.percentile(test_scores[valid_mask], 99.9), 100)
    for th in candidates:
        preds = event_level_filter(test_scores, th, valid_mask, min_run=2, extreme_factor=1.75) * valid_mask.astype(np.float32)
        m_pa = compute_metrics(test_labels, preds, valid_mask=valid_mask, use_pa=True)
        if m_pa.get('f1', 0.0) > best_pa_f1:
            best_pa_f1 = m_pa['f1']
            
    print(f"Channel: {chan} | std_floor: {std_floor} | Oracle PA F1: {best_pa_f1:.4f}", flush=True)

print("--- Testing machine-2-9 ---")
run_channel("machine-2-9", std_floor=0.01, mad_floor=0.20)
run_channel("machine-2-9", std_floor=0.05, mad_floor=0.20)

print("\n--- Testing machine-1-3 ---")
run_channel("machine-1-3", std_floor=0.01, mad_floor=0.20)
run_channel("machine-1-3", std_floor=0.05, mad_floor=0.20)
