import json
from pathlib import Path

nb_path = Path("notebooks_v4/SMD/SMD_SSM_Anomaly_Detection.ipynb")
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print("".join(cells[1]['source']))
