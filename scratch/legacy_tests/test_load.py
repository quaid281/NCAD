from pathlib import Path
import pandas as pd

# Define paths to CalIt2 data files
project_root = Path(r'c:\Users\andre\OneDrive\Desktop\NCAD_CS')
calit2_dir = project_root / 'mTSBench_data' / 'CalIt2'
train_path = calit2_dir / 'CalIt2_traffic_train.csv'
val_path = calit2_dir / 'CalIt2_traffic_val.csv'
test_path = calit2_dir / 'CalIt2_traffic_test.csv'

# Load dataframes
train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

print(f"Loaded datasets successfully.")
print(f"Train Shape: {train_df.shape} (Anomalies: {train_df['is_anomaly'].sum()})")
print(f"Val Shape:   {val_df.shape} (Anomalies: {val_df['is_anomaly'].sum()})")
print(f"Test Shape:  {test_df.shape} (Anomalies: {test_df['is_anomaly'].sum()})")

print("\nSample training rows:")
print(train_df.head())
