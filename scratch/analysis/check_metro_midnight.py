import pandas as pd

test_df = pd.read_csv("mTSBench_data/metro/metro_traffic-volume_test.csv")

# Extract hour from timestamp
test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
test_df["hour"] = test_df["timestamp"].dt.hour

# Check midnight statistics
midnight_normal = test_df[(test_df["hour"] == 0) & (test_df["is_anomaly"] == 0)]
midnight_anom = test_df[(test_df["hour"] == 0) & (test_df["is_anomaly"] == 1)]

print("=== Normal Midnight Traffic Volume ===")
print(midnight_normal["traffic_volume"].describe())

print("\n=== Anomalous Midnight Traffic Volume ===")
print(midnight_anom["traffic_volume"].describe())

# Check how many normal midnights there are
print("\nNumber of normal midnights:", len(midnight_normal))
print("Number of anomalous midnights:", len(midnight_anom))
