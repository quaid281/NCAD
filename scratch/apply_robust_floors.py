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

target_local_dev = "scale = np.maximum(1.4826 * mad, 1e-4)"
replacement_local_dev = "scale = np.maximum(1.4826 * mad, 0.20)"

target_scaler = """    # Standardize features
    scaler = StandardScaler()
    scaler.fit(train_df[numeric_cols])
    train_scaled = scaler.transform(train_df[numeric_cols])
    test_scaled = scaler.transform(test_df[numeric_cols])"""

replacement_scaler = """    # Standardize features with variance floor to avoid OOD explosion on near-constant channels
    train_mean = train_df[numeric_cols].mean().to_numpy()
    train_std = train_df[numeric_cols].std().to_numpy()
    scale = np.maximum(train_std, 0.10)
    train_scaled = (train_df[numeric_cols].to_numpy() - train_mean) / scale
    test_scaled = (test_df[numeric_cols].to_numpy() - train_mean) / scale"""

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
            # Combine lines of the cell source to search for target content
            source_lines = cell.get("source", [])
            source_text = "".join(source_lines)
            
            cell_updated = False
            # Check local deviation replacement
            if target_local_dev in source_text:
                source_text = source_text.replace(target_local_dev, replacement_local_dev)
                cell_updated = True
                print("  -> Updated local deviation MAD floor.")
                
            # Check scaler replacement (handle exact newlines and whitespace)
            if target_scaler in source_text:
                source_text = source_text.replace(target_scaler, replacement_scaler)
                cell_updated = True
                print("  -> Updated StandardScaler variance floor.")
            elif target_scaler.replace("\r\n", "\n") in source_text.replace("\r\n", "\n"):
                # Fallback to normalized newlines
                source_text = source_text.replace(target_scaler.replace("\r\n", "\n"), replacement_scaler)
                cell_updated = True
                print("  -> Updated StandardScaler variance floor (normalized newlines).")
                
            if cell_updated:
                # Convert back to list of lines (keep trailing newlines for readability)
                # Split by newline but keep the separator
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
                
    if updated:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb_data, f, indent=1, ensure_ascii=False)
        print(f"Successfully updated and saved {nb_path.name}\n")
        return True
    else:
        print(f"No targets found or already updated in {nb_path.name}\n")
        return False

for nb in notebook_files:
    update_notebook(nb)
