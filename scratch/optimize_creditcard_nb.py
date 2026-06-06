import json

nb_path = "notebooks_v4/creditcard/Creditcard_SSM_Anomaly_Detection.ipynb"
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

modified = False
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    
    src = ''.join(cell['source'])
    
    # 1. Update window parameters
    if "context_size = 284" in src:
        print("Modifying window parameters...")
        cell['source'] = [
            "context_size = 9\n",
            "suspect_size = 1\n",
            "window_size = context_size + suspect_size\n",
            "step = 1\n",
            "\n",
            "# Create sliding windows\n",
            "train_windows = DataLoader.create_windows(train_scaled, window_size, step)\n",
            "val_windows = DataLoader.create_windows(val_scaled, window_size, step)\n",
            "test_windows = DataLoader.create_windows(test_scaled, window_size, step)\n",
            "\n",
            "print(f\"Train windows shape: {train_windows.shape}\")\n",
            "print(f\"Val windows shape:   {val_windows.shape}\")\n",
            "print(f\"Test windows shape:  {test_windows.shape}\")\n"
        ]
        modified = True
        
    # 2. Update model initialization (layers = 2)
    if "layers=4," in src and "SelectiveSSMContextEncoder" in src:
        print("Modifying layers in model init...")
        cell['source'] = [line.replace("layers=4,", "layers=2,") for line in cell['source']]
        modified = True
        
    # 3. Update smoothing window
    if "smoothing_window = 12" in src:
        print("Modifying smoothing window...")
        cell['source'] = [line.replace("smoothing_window = 12", "smoothing_window = 1") for line in cell['source']]
        modified = True
        
    # 4. Update event-level filter parameters (min_run = 1)
    if "min_run=2" in src:
        print("Modifying min_run to 1...")
        cell['source'] = [line.replace("min_run=2", "min_run=1") for line in cell['source']]
        modified = True

if modified:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Creditcard notebook optimized successfully.")
else:
    print("No changes made to Creditcard notebook.")
