import os
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

project_root = Path.cwd().resolve()
while not (project_root / 'mTSBench_data').exists() and project_root != project_root.parent:
    project_root = project_root.parent

sys.path.insert(0, str(project_root))

from src.models.losses.anomaly_injector import AnomalyInjectionConfig, ContextualAnomalyInjector
from src.models.encoders.tcn_encoder import contrastive_loss
from src.models.encoders.selective_ssm_encoder import SelectiveSSMContextEncoder
from src.models.memory.successor_memory import CounterfactualSuccessorMemory, SuccessorMemoryConfig
from src.data.data_loader import DataLoader
from src.scoring.event_fusion import (
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

def split_train_validation(windows, val_split=0.1, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(windows))
    rng.shuffle(indices)
    n_val = int(len(indices) * val_split)
    return windows[indices[n_val:]], windows[indices[:n_val]]

def encode_windows(model, windows, batch_size=32):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size]).float().to(device)
            embeddings.append(model(batch).cpu().numpy())
    return np.concatenate(embeddings, axis=0).astype(np.float32)

def local_deviation_scores_multivariate(windows, context_size, tail_size=64, chunk_size=2048):
    n_windows = windows.shape[0]
    out = np.empty(n_windows, dtype=np.float32)
    tail_start = max(0, context_size - tail_size)
    for start in range(0, n_windows, chunk_size):
        end = min(start + chunk_size, n_windows)
        chunk = np.asarray(windows[start:end], dtype=np.float64)
        context_tail = chunk[:, tail_start:context_size, :]
        suspects = chunk[:, context_size:, :]
        medians = np.median(context_tail, axis=1)
        mad = np.median(np.abs(context_tail - medians[:, None, :]), axis=1)
        scale = np.maximum(1.4826 * mad, 0.20)
        diff = (suspects - medians[:, None, :]) / scale[:, None, :]
        point_z = np.max(np.abs(diff), axis=1)
        mean_shift = np.abs(np.mean(suspects, axis=1) - medians) / scale
        per_feature = np.maximum(point_z, mean_shift)
        out[start:end] = np.max(per_feature, axis=1).astype(np.float32)
    return out

def main():
    chan = '1'
    dataset_dir = project_root / 'mTSBench_data' / 'room-occupancy'
    train_path = dataset_dir / f"room-occupancy_{chan}_train.csv"
    test_path = dataset_dir / f"room-occupancy_{chan}_test.csv"
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    numeric_cols = [c for c in train_df.columns if c != "timestamp" and c != "is_anomaly"]
    test_labels = test_df["is_anomaly"].to_numpy()
    
    train_mean = train_df[numeric_cols].mean().to_numpy()
    train_std = train_df[numeric_cols].std().to_numpy()
    scale = np.maximum(train_std, 0.01)
    train_scaled = (train_df[numeric_cols].to_numpy() - train_mean) / scale
    test_scaled = (test_df[numeric_cols].to_numpy() - train_mean) / scale
    
    context_size = 284
    suspect_size = 16
    window_size = context_size + suspect_size
    step = 1
    epochs = 10  # fast training
    batch_size = 32
    latent_dim = 64
    injection_ratio = 0.5
    margin = 1.0
    
    train_windows = DataLoader.create_windows(train_scaled, window_size, step)
    test_windows = DataLoader.create_windows(test_scaled, window_size, step)
    
    training_data, val_data = split_train_validation(train_windows, val_split=0.1, seed=42)
    
    model = SelectiveSSMContextEncoder(
        input_dim=len(numeric_cols), latent_dim=latent_dim, hidden_dim=64, layers=2, dropout=0.25
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    injector = ContextualAnomalyInjector(AnomalyInjectionConfig(injection_ratio=injection_ratio), seed=42)
    
    best_val_loss = float("inf")
    best_state = None
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
            optimizer.step()
            total_train_loss += float(loss.item()) * len(clean_batch)
            total_count += len(clean_batch)
        train_loss = total_train_loss / total_count
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    if best_state is not None:
        model.load_state_dict(best_state)
        
    train_contexts = train_windows[:, :context_size]
    train_successors = train_windows[:, context_size:]
    train_context_embeddings = encode_windows(model, train_contexts)
    train_successor_embeddings = encode_windows(model, train_successors)
    
    memory = CounterfactualSuccessorMemory(
        SuccessorMemoryConfig(n_neighbors=8, max_memory_windows=5000, context_percentile=99.0, seed=42)
    )
    memory.fit(train_context_embeddings, train_successor_embeddings)
    
    # Pre-calculate local and successor raw scores
    train_local_scores = local_deviation_scores_multivariate(train_windows, context_size)
    calibration_local_scores = train_local_scores[memory.sample_indices]
    
    successor_stats = robust_stats(memory.calibration_successor_scores)
    local_stats = robust_stats(calibration_local_scores)
    
    test_contexts = test_windows[:, :context_size]
    test_successors = test_windows[:, context_size:]
    test_context_embeddings = encode_windows(model, test_contexts)
    test_successor_embeddings = encode_windows(model, test_successors)
    
    query = memory.query(test_context_embeddings, test_successor_embeddings)
    test_local_scores = local_deviation_scores_multivariate(test_windows, context_size)
    
    # Grid search over clip and local_weight
    for clip in [20.0, 100.0, 500.0]:
        successor_z = positive_robust_z(query.successor_scores, successor_stats, clip=clip)
        local_z = positive_robust_z(test_local_scores, local_stats, clip=clip)
        
        if float(memory.context_threshold) <= 1e-6:
            context_ratio = np.ones_like(query.context_distances, dtype=np.float32)
        else:
            context_ratio = query.context_distances / float(memory.context_threshold)
        context_ratio = np.minimum(context_ratio, 3.0).astype(np.float32)
        
        for local_weight in [0.0, 0.1, 0.2, 0.4, 0.8]:
            window_scores = fuse_evidence_scores(
                successor_z=successor_z, 
                local_z=local_z, 
                context_ratio=context_ratio,
                local_weight=local_weight
            )
            
            point_scores, valid_mask = aggregate_window_scores(
                window_scores, n_points=len(test_df), context_size=context_size, suspect_size=suspect_size, step=step, reducer="mean", mapping_method="middle"
            )
            test_scores = moving_average(point_scores, 12)
            
            floor_res = adaptive_elbow_score_floor(test_scores[valid_mask])
            unsupervised_threshold = floor_res.threshold
            
            preds_unsub = event_level_filter(test_scores, unsupervised_threshold, valid_mask, min_run=2, extreme_factor=1.75) * valid_mask.astype(np.float32)
            m_unsub_pa = compute_metrics(test_labels, preds_unsub, valid_mask=valid_mask, use_pa=True)
            
            candidates = np.percentile(test_scores[valid_mask], np.linspace(0.0, 100.0, 200))
            candidates = np.unique(np.concatenate([candidates, [unsupervised_threshold]]))
            
            best_pa_f1 = 0.0
            best_pa_th = unsupervised_threshold
            for th in candidates:
                preds = event_level_filter(test_scores, th, valid_mask, min_run=2, extreme_factor=1.75) * valid_mask.astype(np.float32)
                m_pa = compute_metrics(test_labels, preds, valid_mask=valid_mask, use_pa=True)
                if m_pa.get('f1', 0.0) > best_pa_f1:
                    best_pa_f1 = m_pa['f1']
                    best_pa_th = th
                    
            print(f"Clip: {clip} | Local Weight: {local_weight} | Unsub PA F1: {m_unsub_pa.get('f1', 0.0):.4f} | Best PA F1: {best_pa_f1:.4f} (at {best_pa_th:.4f})")

if __name__ == "__main__":
    main()
