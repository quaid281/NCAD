import json
from pathlib import Path

# Identify project root
project_root = Path.cwd().resolve()
while not (project_root / 'mTSBench_data').exists() and project_root != project_root.parent:
    project_root = project_root.parent

notebooks_dir = project_root / 'notebooks_v4'
notebook_files = [
    notebooks_dir / 'SMD' / 'SMD_SSM_Anomaly_Detection.ipynb',
    notebooks_dir / 'MSL' / 'MSL_SSM_Anomaly_Detection.ipynb',
    notebooks_dir / 'SMAP' / 'SMAP_SSM_Anomaly_Detection.ipynb'
]

# We want to replace scale = np.maximum(train_std, 0.10) with scale = np.maximum(train_std, 0.01)
target_scale = "scale = np.maximum(train_std, 0.10)"
replacement_scale = "scale = np.maximum(train_std, 0.01)"

def update_notebook(nb_path):
    print(f"Updating notebook: {nb_path.name}")
    if not nb_path.exists():
        print(f"Error: {nb_path} does not exist.")
        return False
        
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
        
    updated = False
    for cell in nb_data.get("cells", []):
        if cell.get("cell_type") == "code":
            source_lines = cell.get("source", [])
            source_text = "".join(source_lines)
            
            if target_scale in source_text:
                source_text = source_text.replace(target_scale, replacement_scale)
                
                # Split back into lines
                lines = []
                current_line = []
                for char in source_text:
                    current_line.append(char)
                    if char == "\n":
                        lines.append("".join(current_line))
                        current_line = []
                if current_line:
                    lines.append("".join(current_line))
                cell["source"] = lines
                updated = True
                print("  -> Updated StandardScaler variance floor to 0.01.")
                
    if updated:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb_data, f, indent=1, ensure_ascii=False)
        print(f"Successfully updated and saved {nb_path.name}\n")
        return True
    else:
        print(f"Already updated or target not found in {nb_path.name}\n")
        return False

for nb in notebook_files:
    update_notebook(nb)
