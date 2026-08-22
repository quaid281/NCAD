import json, sys

nb_path = sys.argv[1]
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print(f"Total code cells: {len(cells)}")
for i, c in enumerate(cells):
    src = ''.join(c['source'])
    print(f"\n=== Cell {i} ===")
    # Print first 600 chars
    print(src[:600])
    if len(src) > 600:
        print("... [truncated]")
