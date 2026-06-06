import json

nb_path = "notebooks_v4/creditcard/Creditcard_SSM_Anomaly_Detection.ipynb"
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        # Replace occurrences of CICIDS/cicids in markdown cells
        cell['source'] = [line.replace("CICIDS", "Creditcard").replace("cicids", "creditcard") for line in cell['source']]
        continue
    
    if cell['cell_type'] != 'code':
        continue
    
    src = ''.join(cell['source'])
    
    # 1. Modify data path cell (Cell 1)
    if "cicids_dir = project_root / 'mTSBench_data' / 'cicids'" in src:
        print("Modifying data paths cell...")
        cell['source'] = [
            "# Define paths to Creditcard data files\n",
            "creditcard_dir = project_root / 'mTSBench_data' / 'creditcard'\n",
            "train_path = creditcard_dir / 'creditcard_train.csv'\n",
            "val_path = creditcard_dir / 'creditcard_val.csv'\n",
            "test_path = creditcard_dir / 'creditcard_test.csv'\n"
        ]
        
    # 2. General text replacements
    new_source = []
    for line in cell['source']:
        line = line.replace("CICIDS", "Creditcard")
        line = line.replace("cicids", "creditcard")
        # Let's ensure epochs = 3 as in the updated CICIDS notebook
        new_source.append(line)
    cell['source'] = new_source

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Creditcard notebook updated successfully.")
