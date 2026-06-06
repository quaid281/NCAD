import json

nb_path = "notebooks_v4/cicids/CICIDS_SSM_Anomaly_Detection.ipynb"
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

modified = False
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    
    src = ''.join(cell['source'])
    if "epochs = 3" in src:
        print("Modifying epochs to 2 in CICIDS notebook...")
        cell['source'] = [line.replace("epochs = 3", "epochs = 2") for line in cell['source']]
        modified = True

if modified:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("CICIDS notebook epochs reduced to 2.")
else:
    print("No changes made to CICIDS notebook.")
