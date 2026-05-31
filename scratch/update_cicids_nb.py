import json

nb_path = "notebooks_v4/cicids/CICIDS_SSM_Anomaly_Detection.ipynb"
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

modified = False
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    
    src = ''.join(cell['source'])
    
    if "window_scores = fuse_evidence_scores(successor_z, local_z, context_ratio)" in src:
        print("Modifying scoring logic in CICIDS notebook...")
        new_src = src.replace(
            "window_scores = fuse_evidence_scores(successor_z, local_z, context_ratio)",
            "# Unified Pipeline: Use Successor Z-score only\n    window_scores = successor_z"
        )
        cell['source'] = [line + '\n' for line in new_src.split('\n')][:-1]
        modified = True

if modified:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("CICIDS notebook updated successfully.")
else:
    print("No changes made to CICIDS notebook.")
