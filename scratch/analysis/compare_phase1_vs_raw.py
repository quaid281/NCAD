import sys, os, time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.losses.anomaly_injector import AnomalyInjectionConfig, ContextualAnomalyInjector
from src.models.encoders.tcn_encoder import contrastive_loss
from src.models.encoders.selective_ssm_encoder import SelectiveSSMContextEncoder
from src.features.features import FeatureConfig, NCADFeatureExtractor
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
print(f"Running on device: {device}")

# Test on SMAP A-1
train_df = pd.read_csv(project_root / "mTSBench_data" / "SMAP" / "SMAP_A-1_train.csv")
test_df = pd.read_csv(project_root / "mTSBench_data" / "SMAP" / "SMAP_A-1_test.csv")
test_labels = test_df["is_anomaly"].to_numpy()

train_raw = train_df["feature_1"].to_numpy()
test_raw = test_df["feature_1"].to_numpy()

mean_val = float(np.mean(train_raw))
std_val = float(np.std(train_raw)) if np.std(train_raw) >= 1e-8 else 1.0

train_normalized = (train_raw - mean_val) / std_val
test_normalized = (test_raw - mean_val) / std_val

feature_cfg = FeatureConfig(max_features=64)
fe = NCADFeatureExtractor(feature_cfg)
train_features = fe.fit_transform(train_normalized)
test_features = fe.transform(test_normalized)

context_size = 284
suspect_size = 16
window_size = context_size + suspect_size
step = 1

train_windows = DataLoader.create_windows(train_features, window_size, step)
test_windows = DataLoader.create_windows(test_features, window_size, step)

def split_train_validation(windows, val_split=0.1, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(windows))
    rng.shuffle(indices)
    n_val = int(len(indices) * val_split)
    return windows[indices[n_val:]], windows[indices[:n_val]]

training_data, val_data = split_train_validation(train_windows, val_split=0.1, seed=42)

model = SelectiveSSMContextEncoder(
    input_dim=train_features.shape[1],
    latent_dim=64,
    hidden_dim=64,
    layers=2,
    dropout=0.25
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
injector = ContextualAnomalyInjector(AnomalyInjectionConfig(injection_ratio=0.5), seed=42)

for epoch in range(1, 6):
    model.train()
    epoch_indices = np.random.permutation(len(training_data))
    for batch_start in range(0, len(epoch_indices), 32):
        batch_indices = epoch_indices[batch_start : batch_start + 32]
        clean_batch = training_data[batch_indices]
        modified_batch, labels = injector.inject_batch(clean_batch, context_size)
        full_tensor = torch.from_numpy(modified_batch).float().to(device)
        context_tensor = torch.from_numpy(clean_batch[:, :context_size]).float().to(device)
        label_tensor = torch.from_numpy(labels).float().to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = contrastive_loss(model(full_tensor), model(context_tensor), label_tensor, margin=1.0)
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

test_contexts = test_windows[:, :context_size]
test_successors = test_windows[:, context_size:]
test_context_embeddings = encode_windows(model, test_contexts)

# Evaluate both Raw successor space vs Phase 1 Latent successor space
for mode in ["raw", "latent"]:
    print(f"\n=================== EVALUATING MODE: {mode.upper()} ===================")
    if mode == "latent":
        train_succ_data = encode_windows(model, train_successors)
        test_succ_data = encode_windows(model, test_successors)
    else:
        train_succ_data = train_successors
        test_succ_data = test_successors

    memory = CounterfactualSuccessorMemory(
        SuccessorMemoryConfig(n_neighbors=8, max_memory_windows=5000, context_percentile=99.0, seed=42)
    )
    memory.fit(train_context_embeddings, train_succ_data)

    query = memory.query(test_context_embeddings, test_succ_data)
    
    print(f"Successor scores summary: min={np.min(query.successor_scores):.4f}, mean={np.mean(query.successor_scores):.4f}, max={np.max(query.successor_scores):.4f}, std={np.std(query.successor_scores):.4f}")
    
    successor_stats = robust_stats(memory.calibration_successor_scores)
    successor_z = positive_robust_z(query.successor_scores, successor_stats)
    
    # Simple percentile threshold sweep for comparison
    best_f1_std = 0.0
    best_f1_pa = 0.0
    best_p_std = 0.0
    best_r_std = 0.0

    point_scores, valid_mask = aggregate_window_scores(
        successor_z,
        n_points=len(test_df),
        context_size=context_size,
        suspect_size=suspect_size,
        step=step,
        reducer="mean",
        mapping_method="middle"
    )
    smoothed_scores = moving_average(point_scores, 12)

    candidates = np.percentile(smoothed_scores[valid_mask], np.linspace(80, 99.9, 100))
    for th in candidates:
        preds = (smoothed_scores >= th).astype(np.float32) * valid_mask.astype(np.float32)
        m_std = compute_metrics(test_labels, preds, valid_mask=valid_mask, use_pa=False)
        m_pa = compute_metrics(test_labels, preds, valid_mask=valid_mask, use_pa=True)
        if m_std.get("f1", 0.0) > best_f1_std:
            best_f1_std = m_std["f1"]
            best_p_std = m_std.get("precision", 0.0)
            best_r_std = m_std.get("recall", 0.0)
        if m_pa.get("f1", 0.0) > best_f1_pa:
            best_f1_pa = m_pa["f1"]

    print(f"Peak Standard Metrics: Precision={best_p_std:.4f}, Recall={best_r_std:.4f}, F1={best_f1_std:.4f}")
    print(f"Peak Point-Adjusted F1: {best_f1_pa:.4f}")
