import pandas as pd
import numpy as np

def inspect_dataset(name, filepath):
    df = pd.read_csv(filepath)
    print(f"=== {name} ===")
    print("Shape:", df.shape)
    if "is_anomaly" in df.columns:
        labels = df["is_anomaly"].to_numpy()
        print("Anomaly sum:", labels.sum())
        print("Anomaly ratio: {:.4f}%".format(labels.mean() * 100))
        
        # Locate segments
        anomaly_indices = np.where(labels == 1)[0]
        if len(anomaly_indices) > 0:
            diffs = np.diff(anomaly_indices)
            gaps = np.where(diffs > 1)[0]
            print("Number of anomaly segments:", len(gaps) + 1)
            lengths = []
            for idx in range(len(gaps) + 1):
                start_idx = 0 if idx == 0 else gaps[idx-1] + 1
                end_idx = len(anomaly_indices) if idx == len(gaps) else gaps[idx] + 1
                lengths.append(len(anomaly_indices[start_idx:end_idx]))
            print("Segment lengths: min={}, max={}, median={}, mean={:.2f}".format(
                np.min(lengths), np.max(lengths), np.median(lengths), np.mean(lengths)
            ))
            print("Length distribution:", sorted(lengths))
        else:
            print("No anomalies found!")
    print("\n" + "="*40 + "\n")

inspect_dataset("GECCO", "mTSBench_data/GECCO/GECCO_water_quality_test.csv")
inspect_dataset("Genesis", "mTSBench_data/Genesis/Genesis_test.csv")
