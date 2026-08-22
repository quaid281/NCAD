import pandas as pd
import numpy as np
import json
from pathlib import Path

nb_path = Path("notebooks_v4/metro/metro_SSM_Anomaly_Detection.ipynb")
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

# Find the df_summary or evaluation output if it ran
# Metro evaluation.csv is saved under results/notebook_runs/metro_evaluation.csv
eval_csv = Path("results/notebook_runs/metro_evaluation.csv")
if eval_csv.exists():
    print("=== Metro Evaluation CSV ===")
    print(pd.read_csv(eval_csv).to_string())
