import sys, os, time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.train_model import CSMConfig, run_experiment

print("=================== RUNNING FULL END-TO-END NCAD-CS BENCHMARK ===================")

# 1. Run Phase 1 (Latent Successor Space)
print("\n--- Running Phase 1 (Latent Successor Space) ---")
config_latent = CSMConfig(
    data_dir=str(project_root / "mTSBench_data"),
    epochs=20,
    batch_size=32,
    successor_space="latent",
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_plots=False
)

# 2. Run Raw Baseline (Raw Successor Space)
print("\n--- Running Baseline (Raw Successor Space) ---")
config_raw = CSMConfig(
    data_dir=str(project_root / "mTSBench_data"),
    epochs=20,
    batch_size=32,
    successor_space="raw",
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_plots=False
)
