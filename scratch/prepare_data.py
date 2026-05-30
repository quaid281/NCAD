import os
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "raw" / "train"
TEST_DIR = DATA_DIR / "raw" / "test"
MTSBENCH_DIR = PROJECT_ROOT / "mTSBench_data"

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

# Load labeled anomalies to know which channels to process
labeled_csv = DATA_DIR / "processed" / "labeled_anomalies.csv"
if not labeled_csv.exists():
    print(f"Error: labeled_anomalies.csv not found at {labeled_csv}")
    exit(1)

df_labels = pd.read_csv(labeled_csv)
print(f"Loaded {len(df_labels)} channels from {labeled_csv}")

converted_count = 0

for _, row in df_labels.iterrows():
    chan_id = str(row["chan_id"])
    spacecraft = str(row["spacecraft"]) # SMAP or MSL
    
    # In mTSBench, MSL contains MSL_* files and SMAP contains SMAP_* files
    # E.g. mTSBench_data/SMAP/SMAP_A-1_train.csv
    src_train_csv = MTSBENCH_DIR / spacecraft / f"{spacecraft}_{chan_id}_train.csv"
    src_test_csv = MTSBENCH_DIR / spacecraft / f"{spacecraft}_{chan_id}_test.csv"
    
    dest_train_npy = TRAIN_DIR / f"{chan_id}.npy"
    dest_test_npy = TEST_DIR / f"{chan_id}.npy"
    
    if src_train_csv.exists() and src_test_csv.exists():
        # Load train
        train_df = pd.read_csv(src_train_csv)
        if "timestamp" in train_df.columns:
            train_df = train_df.drop(columns=["timestamp"])
        train_arr = train_df.to_numpy(dtype=np.float32)
        np.save(dest_train_npy, train_arr)
        
        # Load test
        test_df = pd.read_csv(src_test_csv)
        if "timestamp" in test_df.columns:
            test_df = test_df.drop(columns=["timestamp"])
        test_arr = test_df.to_numpy(dtype=np.float32)
        np.save(dest_test_npy, test_arr)
        
        converted_count += 1
    else:
        print(f"Warning: CSV files not found for {chan_id} ({spacecraft})")
        print(f"  Checked: {src_train_csv}")

print(f"Successfully converted {converted_count} channels to npy.")
