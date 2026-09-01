import time
import sys
from pathlib import Path
sys.path.append('c:/Users/andre/OneDrive/Desktop/NCAD_CS')
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from src.models.encoders.selective_ssm_encoder import SelectiveSSMContextEncoder
from src.models.losses.anomaly_injector import ContextualAnomalyInjector, AnomalyInjectionConfig

device = torch.device("cpu")
print(f"Device: {device}")

# Dummy training data matching CalIt2 shapes
# CalIt2 training data: 577 windows of shape (577, 300, 2)
n_windows = 577
context_size = 284
suspect_size = 16
window_size = context_size + suspect_size
n_features = 2

windows = np.random.randn(n_windows, window_size, n_features).astype(np.float32)

model = SelectiveSSMContextEncoder(
    input_dim=n_features,
    latent_dim=16,
    hidden_dim=64,
    layers=4,
    dropout=0.10
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
injector = ContextualAnomalyInjector(AnomalyInjectionConfig(injection_ratio=0.70), seed=42)

def contrastive_loss(z_full, z_clean, labels, margin=1.0):
    dist = torch.norm(z_full - z_clean, p=2, dim=1)
    loss = (1.0 - labels) * (dist ** 2) + labels * torch.clamp(margin - dist, min=0.0) ** 2
    return loss.mean()

print("Timing 1 epoch...")
t0 = time.time()
model.train()
epoch_indices = np.random.permutation(len(windows))
batch_size = 32
total_loss = 0.0

for batch_start in range(0, len(epoch_indices), batch_size):
    batch_indices = epoch_indices[batch_start : batch_start + batch_size]
    clean_batch = windows[batch_indices]
    modified_batch, labels = injector.inject_batch(clean_batch, context_size)
    
    full_tensor = torch.from_numpy(modified_batch).float().to(device)
    context_tensor = torch.from_numpy(clean_batch[:, :context_size]).float().to(device)
    label_tensor = torch.from_numpy(labels).float().to(device)
    
    optimizer.zero_grad(set_to_none=True)
    loss = contrastive_loss(model(full_tensor), model(context_tensor), label_tensor)
    loss.backward()
    optimizer.step()
    
    total_loss += float(loss.item())

t1 = time.time()
print(f"1 epoch took: {t1 - t0:.2f} seconds")
