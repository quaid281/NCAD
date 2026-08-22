"""Extract test evaluation outputs from an executed notebook."""
import json
import sys

nb_path = sys.argv[1]
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
for i, cell in enumerate(code_cells):
    src = ''.join(cell['source'])
    outputs = cell.get('outputs', [])
    
    # Check if this cell has test evaluation results
    has_results = 'Test Set Evaluation' in src or any(
        'Test Set Evaluation' in ''.join(o.get('text', [])) 
        for o in outputs if o.get('output_type') == 'stream'
    )
    
    if has_results and outputs:
        print(f"\n=== Code Cell {i} Outputs ===")
        for o in outputs:
            if o.get('output_type') == 'stream':
                print(''.join(o.get('text', [])), end='')
            elif o.get('output_type') == 'execute_result':
                print(''.join(o.get('data', {}).get('text/plain', [])), end='')
    
    # Also print threshold / scoring outputs
    has_threshold = 'Threshold' in src and ('Adaptive Elbow' in src or 'Optimized' in src)
    if has_threshold and outputs:
        print(f"\n=== Code Cell {i} Outputs (Threshold) ===")
        for o in outputs:
            if o.get('output_type') == 'stream':
                print(''.join(o.get('text', [])), end='')

# Also print any cell with "Fitted Successor Memory" or scoring output
for i, cell in enumerate(code_cells):
    src = ''.join(cell['source'])
    outputs = cell.get('outputs', [])
    if 'Fitted Successor Memory' in src and outputs:
        print(f"\n=== Code Cell {i} Outputs (Memory) ===")
        for o in outputs:
            if o.get('output_type') == 'stream':
                print(''.join(o.get('text', [])), end='')
