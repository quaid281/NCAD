import json

nb_path = "notebooks_v4/creditcard/Creditcard_SSM_Anomaly_Detection.ipynb"
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    
    src = ''.join(cell['source'])
    if "creditcard_dir = project_root" in src and "train_df = pd.read_csv" not in src:
        print("Fixing data loading in creditcard notebook...")
        cell['source'] = [
            "# Define paths to Creditcard data files\n",
            "creditcard_dir = project_root / 'mTSBench_data' / 'creditcard'\n",
            "train_path = creditcard_dir / 'creditcard_train.csv'\n",
            "val_path = creditcard_dir / 'creditcard_val.csv'\n",
            "test_path = creditcard_dir / 'creditcard_test.csv'\n",
            "\n",
            "# Load dataframes\n",
            "train_df = pd.read_csv(train_path)\n",
            "val_df = pd.read_csv(val_path)\n",
            "test_df = pd.read_csv(test_path)\n",
            "\n",
            "print(f\"Loaded datasets successfully.\")\n",
            "print(f\"Train Shape: {train_df.shape} (Anomalies: {train_df['is_anomaly'].sum()})\")\n",
            "print(f\"Val Shape:   {val_df.shape} (Anomalies: {val_df['is_anomaly'].sum()})\")\n",
            "print(f\"Test Shape:  {test_df.shape} (Anomalies: {test_df['is_anomaly'].sum()})\")\n"
        ]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Creditcard notebook loading fixed.")
