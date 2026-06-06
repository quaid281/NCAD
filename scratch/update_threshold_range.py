import json

notebooks = [
    "notebooks_v4/CalIt2/CalIt2_SSM_Anomaly_Detection.ipynb",
    "notebooks_v4/cicids/CICIDS_SSM_Anomaly_Detection.ipynb"
]

for nb_path in notebooks:
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        
        src = ''.join(cell['source'])
        if "candidates = np.linspace(np.percentile(val_valid_scores, 50.0)" in src:
            print(f"Modifying threshold candidate range in {nb_path}...")
            # We'll replace 50.0 with 1.0, and 99.9 with 99.99, and the number of steps to 300
            new_source = []
            for line in cell['source']:
                if "candidates = np.linspace(np.percentile(val_valid_scores, 50.0)" in line:
                    line = line.replace("50.0", "1.0").replace("99.9", "99.99")
                    if ", 200)" in line:
                        line = line.replace(", 200)", ", 300)")
                new_source.append(line)
            cell['source'] = new_source
            modified = True
            
    if modified:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Updated {nb_path} successfully.")
    else:
        print(f"No changes made to {nb_path}.")
