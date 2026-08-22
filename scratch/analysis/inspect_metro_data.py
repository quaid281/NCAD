import pandas as pd
import numpy as np

train_df = pd.read_csv("mTSBench_data/metro/metro_traffic-volume_train.csv")
test_df = pd.read_csv("mTSBench_data/metro/metro_traffic-volume_test.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Columns:", list(test_df.columns))

if "is_anomaly" in test_df.columns:
    labels = test_df["is_anomaly"].to_numpy()
    print("Test labels sum:", labels.sum())
    print("Test anomaly ratio:", labels.mean() * 100)
    
    # Locate where anomalies are in the test set
    anomaly_indices = np.where(labels == 1)[0]
    print("Anomaly indices:", anomaly_indices[:20])
    
    # Check if there are contiguous segments
    diffs = np.diff(anomaly_indices)
    gaps = np.where(diffs > 1)[0]
    print(f"Number of anomaly segments: {len(gaps) + 1}")
    for idx in range(min(len(gaps) + 1, 5)):
        start_idx = 0 if idx == 0 else gaps[idx-1] + 1
        end_idx = len(anomaly_indices) if idx == len(gaps) else gaps[idx] + 1
        segment = anomaly_indices[start_idx:end_idx]
        print(f"Segment {idx+1}: length {len(segment)}, range {segment[0]} to {segment[-1]}")
