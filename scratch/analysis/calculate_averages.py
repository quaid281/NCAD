import pandas as pd
from pathlib import Path

results_dir = Path("results/notebook_runs")
csv_files = sorted(results_dir.glob("*.csv"))

print("## Macro-Averaged Performance Metrics (Current Run)\n")
print("| Dataset | Channels | Avg Unsub Std F1 | Avg Unsub PA F1 | Avg Oracle Std F1 | Avg Oracle PA F1 |")
print("|---|---:|---:|---:|---:|---:|")

for csv in csv_files:
    df = pd.read_csv(csv)
    name = csv.stem.replace('_evaluation', '').upper()
    n_chans = len(df)
    
    unsub_std = df['unsub_std_f1'].mean() if 'unsub_std_f1' in df.columns else 0.0
    unsub_pa = df['unsub_pa_f1'].mean() if 'unsub_pa_f1' in df.columns else 0.0
    oracle_std = df['oracle_std_f1'].mean() if 'oracle_std_f1' in df.columns else 0.0
    oracle_pa = df['oracle_pa_f1'].mean() if 'oracle_pa_f1' in df.columns else 0.0
    
    print(f"| **{name}** | {n_chans} | {unsub_std:.4f} | {unsub_pa:.4f} | {oracle_std:.4f} | {oracle_pa:.4f} |")
