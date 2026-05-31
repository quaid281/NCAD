import json

nb_path = "notebooks_v4/CalIt2/CalIt2_SSM_Anomaly_Detection.ipynb"
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

modified = False
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    
    src = ''.join(cell['source'])
    
    # 1. Replace Cell 3 (Feature Extraction)
    if "fe_in = NCADFeatureExtractor" in src:
        print("Modifying Feature Extraction cell...")
        cell['source'] = [
            "# Unified Pipeline: Bypass feature extractor, use raw standard-scaled features directly\n",
            "train_features = train_scaled\n",
            "val_features = val_scaled\n",
            "test_features = test_scaled\n",
            "\n",
            "print(f\"Train features shape: {train_features.shape}\")\n",
            "print(f\"Val features shape:   {val_features.shape}\")\n",
            "print(f\"Test features shape:  {test_features.shape}\")\n"
        ]
        modified = True
        
    # 2. Replace scoring logic in compute_anomaly_scores
    if "window_scores = fuse_evidence_scores(successor_z, local_z, context_ratio)" in src:
        print("Modifying scoring logic cell...")
        new_src = src.replace(
            "window_scores = fuse_evidence_scores(successor_z, local_z, context_ratio)",
            "# Unified Pipeline: Use Successor Z-score only\n    window_scores = successor_z"
        )
        cell['source'] = [line + '\n' for line in new_src.split('\n')][:-1]
        modified = True

if modified:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("CalIt2 notebook updated successfully.")
else:
    print("No changes made to CalIt2 notebook.")
