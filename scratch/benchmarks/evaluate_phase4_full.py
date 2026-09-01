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
from src.models.encoders.tcn_encoder import HybridTCNEncoder, contrastive_loss
from src.features.features import FeatureConfig, NCADFeatureExtractor
from src.models.memory.successor_memory import CounterfactualSuccessorMemory, SuccessorMemoryConfig
from src.models.memory.sindy_scorer import SINDyConfig, SINDyDynamicsScorer
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
print(f"Device: {device}")

calit2_dir = project_root / "mTSBench_data" / "CalIt2"
train_df = pd.read_csv(calit2_dir / "CalIt2_traffic_train.csv")
test_df = pd.read_csv(calit2_dir / "CalIt2_traffic_test.csv")
test_labels = test_df["is_anomaly"].to_numpy()

feature_cols = ["in_count", "out_count"]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_df[feature_cols])
train_scaled = scaler.transform(train_df[feature_cols])
test_scaled = scaler.transform(test_df[feature_cols])

feature_cfg = FeatureConfig(max_features=64, use_delay_embedding=True, delay_embedding_dim=5, delay_lag=4)
fe_in = NCADFeatureExtractor(feature_cfg)
fe_out = NCADFeatureExtractor(feature_cfg)

train_feat_in = fe_in.fit_transform(train_scaled[:, 0])
train_feat_out = fe_out.fit_transform(train_scaled[:, 1])
train_features = np.concatenate([train_feat_in, train_feat_out], axis=1)

test_feat_in = fe_in.transform(test_scaled[:, 0])
test_feat_out = fe_out.transform(test_scaled[:, 1])
test_features = np.concatenate([test_feat_in, test_feat_out], axis=1)

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

model = HybridTCNEncoder(input_dim=train_features.shape[1], latent_dim=16, filters=64, tcn_layers=4, kernel_size=5, dropout=0.20).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
injector = ContextualAnomalyInjector(AnomalyInjectionConfig(injection_ratio=0.70), seed=42)

print("\nTraining HybridTCN Encoder (15 epochs)...")
for epoch in range(1, 16):
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
train_succ_data = encode_windows(model, train_successors)

test_contexts = test_windows[:, :context_size]
test_successors = test_windows[:, context_size:]
test_context_embeddings = encode_windows(model, test_contexts)
test_succ_data = encode_windows(model, test_successors)

memory = CounterfactualSuccessorMemory(
    SuccessorMemoryConfig(n_neighbors=8, max_memory_windows=5000, context_percentile=99.0, seed=42, use_rpca_sanitization=False)
)
memory.fit(train_context_embeddings, train_succ_data)

query = memory.query(test_context_embeddings, test_succ_data)

def local_deviation_scores_multivariate(windows, context_size, tail_size=64):
    scores = []
    for idx in range(windows.shape[2]):
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

train_local_scores = local_deviation_scores_multivariate(train_windows, context_size)
calibration_local_scores = train_local_scores[memory.sample_indices]

successor_stats = robust_stats(memory.calibration_successor_scores)
local_stats = robust_stats(calibration_local_scores)

test_local_scores = local_deviation_scores_multivariate(test_windows, context_size)

successor_z = positive_robust_z(query.successor_scores, successor_stats)
local_z = positive_robust_z(test_local_scores, local_stats)

context_ratio = query.context_distances / max(float(memory.context_threshold), 1e-6)
context_ratio = np.minimum(context_ratio, 3.0).astype(np.float32)

# Fit SINDy Scorer on normal training latent embeddings
sindy = SINDyDynamicsScorer(SINDyConfig(poly_degree=2, threshold=0.05))
sindy.fit(train_context_embeddings)

calibration_sindy_scores = sindy.score(train_context_embeddings[memory.sample_indices])
sindy_stats = robust_stats(calibration_sindy_scores)

test_sindy_scores = sindy.score(test_context_embeddings)
sindy_z = positive_robust_z(test_sindy_scores, sindy_stats)

for use_sindy in [False, True]:
    print(f"\n=================== EVALUATING SINDY DYNAMICS = {use_sindy} ===================")
    if use_sindy:
        window_scores = fuse_evidence_scores(
            successor_z=successor_z,
            local_z=local_z,
            context_ratio=context_ratio,
            sindy_z=sindy_z,
            sindy_weight=0.50,
            normalize_components=True
        )
    else:
        window_scores = fuse_evidence_scores(
            successor_z=successor_z,
            local_z=local_z,
            context_ratio=context_ratio,
            normalize_components=True
        )

    point_scores, valid_mask = aggregate_window_scores(
        window_scores, n_points=len(test_df), context_size=context_size, suspect_size=suspect_size, step=step, reducer="mean", mapping_method="middle"
    )
    test_scores = moving_average(point_scores, 12)

    floor_res = adaptive_elbow_score_floor(test_scores[valid_mask])
    threshold = floor_res.threshold

    preds = event_level_filter(test_scores, threshold, valid_mask, min_run=2, extreme_factor=1.75)
    preds = preds * valid_mask.astype(np.float32)

    m_std = compute_metrics(test_labels, preds, valid_mask=valid_mask, use_pa=False)
    m_pa = compute_metrics(test_labels, preds, valid_mask=valid_mask, use_pa=True)

    print(f"Elbow Threshold: {threshold:.5f}")
    print("Standard Metrics (No PA):", m_std)
    print("Point-Adjusted Metrics (PA):", m_pa)
