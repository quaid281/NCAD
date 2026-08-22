import pandas as pd
import numpy as np
import glob

files = sorted(glob.glob("mTSBench_data/room-occupancy/*_test.csv"))
print(f"=== room-occupancy ({len(files)} channels) ===")
all_lengths = []
for f in files:
    df = pd.read_csv(f)
    if "is_anomaly" in df.columns:
        labels = df["is_anomaly"].to_numpy()
        anomaly_indices = np.where(labels == 1)[0]
        if len(anomaly_indices) > 0:
            diffs = np.diff(anomaly_indices)
            gaps = np.where(diffs > 1)[0]
            for idx in range(len(gaps) + 1):
                start_idx = 0 if idx == 0 else gaps[idx-1] + 1
                end_idx = len(anomaly_indices) if idx == len(gaps) else gaps[idx] + 1
                all_lengths.append(len(anomaly_indices[start_idx:end_idx]))
if all_lengths:
    print("Segment lengths: min={}, max={}, median={}, mean={:.2f}".format(
        np.min(all_lengths), np.max(all_lengths), np.median(all_lengths), np.mean(all_lengths)
    ))
    print("Length distribution:", sorted(all_lengths))
else:
    print("No anomalies found!")
