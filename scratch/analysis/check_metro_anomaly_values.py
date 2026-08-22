import pandas as pd
import numpy as np

test_df = pd.read_csv("mTSBench_data/metro/metro_traffic-volume_test.csv")
normal = test_df[test_df["is_anomaly"] == 0]
anomalous = test_df[test_df["is_anomaly"] == 1]

features = ['traffic_volume', 'temp', 'rain_1h', 'snow_1h', 'clouds_all']

print("=== Normal Mean & Std ===")
for f in features:
    print("{:<15} mean={:<10.2f} std={:<10.2f} min={:<10.2f} max={:<10.2f}".format(
        f, normal[f].mean(), normal[f].std(), normal[f].min(), normal[f].max()
    ))

print("\n=== Anomalous Values ===")
for f in features:
    print("{:<15} mean={:<10.2f} std={:<10.2f} min={:<10.2f} max={:<10.2f}".format(
        f, anomalous[f].mean(), anomalous[f].std(), anomalous[f].min(), anomalous[f].max()
    ))

# Check some individual anomalies
print("\n=== Sample Anomaly Points ===")
cols_to_print = ['timestamp'] + features + ['is_anomaly']
print(test_df[test_df["is_anomaly"] == 1][cols_to_print].head(10).to_string())
