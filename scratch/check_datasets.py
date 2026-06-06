import pandas as pd
import numpy as np

datasets = [
    ('CalIt2', 'CalIt2/CalIt2_traffic'),
    ('CICIDS', 'cicids/cicids_0'),
    ('Creditcard', 'creditcard/creditcard'),
]

for name, p in datasets:
    train = pd.read_csv(f'mTSBench_data/{p}_train.csv')
    val = pd.read_csv(f'mTSBench_data/{p}_val.csv')
    test = pd.read_csv(f'mTSBench_data/{p}_test.csv')
    
    numeric_cols = train.select_dtypes(include='number').columns.tolist()
    if 'is_anomaly' in numeric_cols:
        numeric_cols.remove('is_anomaly')
    
    print(f'\n=== {name} ===')
    print(f'Features: {len(numeric_cols)}')
    print(f'Train: {train.shape}, anomalies: {int(train["is_anomaly"].sum())} ({train["is_anomaly"].mean():.4f})')
    print(f'Val:   {val.shape}, anomalies: {int(val["is_anomaly"].sum())} ({val["is_anomaly"].mean():.4f})')
    print(f'Test:  {test.shape}, anomalies: {int(test["is_anomaly"].sum())} ({test["is_anomaly"].mean():.4f})')
    print(f'Feature columns: {numeric_cols[:10]}...' if len(numeric_cols) > 10 else f'Feature columns: {numeric_cols}')
