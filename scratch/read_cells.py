import json, sys

nb_path = sys.argv[1]
cell_indices = [int(x) for x in sys.argv[2:]]

with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
for i in cell_indices:
    src = ''.join(cells[i]['source'])
    print(f"\n=== Cell {i} (FULL) ===")
    print(src)
