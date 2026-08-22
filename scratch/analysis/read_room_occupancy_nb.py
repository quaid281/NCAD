import json
from pathlib import Path

nb_path = Path("notebooks_v4/room-occupancy/room-occupancy_SSM_Anomaly_Detection.ipynb")
if nb_path.exists():
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if "positive_robust_z" in source or "clip" in source:
                print(f"--- Cell {i} ---")
                print(source)
else:
    print("Notebook does not exist.")
