import pandas as pd
from pathlib import Path

results_dir = Path("results/notebook_runs")
csv_files = sorted(results_dir.glob("*.csv"))

for csv in csv_files:
    df = pd.read_csv(csv)
    print(f"=== {csv.name} ===")
    print(df.to_string())
    print("\n" + "="*40 + "\n")
