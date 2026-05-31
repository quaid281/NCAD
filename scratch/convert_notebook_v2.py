import json
from pathlib import Path

# Paths
notebook_dir = Path(r'c:\Users\andre\OneDrive\Desktop\NCAD_CS\notebooks_v4\cicids')
src_path = notebook_dir / 'CICIDS_Anomaly_Detection.ipynb'
dst_path = notebook_dir / 'CICIDS_SSM_Anomaly_Detection.ipynb'

# Load the notebook
with open(src_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Process cells
for cell in nb.get('cells', []):
    # Join source lines into a single string for robust matching
    source_lines = cell.get('source', [])
    source_str = "".join(source_lines)
    
    # Perform general replacements
    source_str = source_str.replace("TCN Encoder", "SSM Encoder")
    source_str = source_str.replace("TCN embeddings", "SSM embeddings")
    source_str = source_str.replace("NCAD-CS v4: Anomaly Detection Protocol on CICIDS Intrusion Detection Dataset", "NCAD-CS v4: Anomaly Detection Protocol on CICIDS Intrusion Detection Dataset (SSM Encoder)")
    
    # Imports replacement
    source_str = source_str.replace("from src.models.tcn_encoder import HybridTCNEncoder, contrastive_loss", 
                      "from src.models.tcn_encoder import contrastive_loss\nfrom src.models.selective_ssm_encoder import SelectiveSSMContextEncoder")
    
    # Model training markdown replacement
    source_str = source_str.replace("### Step 3: Contrastive Training of TCN Encoder", "### Step 3: Contrastive Training of Selective SSM Encoder")
    source_str = source_str.replace("We train the `HybridTCNEncoder` model to map features", "We train the `SelectiveSSMContextEncoder` model to map features")
    
    # Model instantiation code replacement (flexible check)
    old_instantiation = """# Instantiate Encoder model
input_dim = len(feature_cols)
model = HybridTCNEncoder(
    input_dim=input_dim,
    latent_dim=16,
    filters=64,
    tcn_layers=4,
    kernel_size=5,
    dropout=0.20
)"""
    new_instantiation = """# Instantiate Selective SSM Encoder model
input_dim = len(feature_cols)
model = SelectiveSSMContextEncoder(
    input_dim=input_dim,
    latent_dim=16,
    hidden_dim=64,
    layers=4,
    dropout=0.10
)"""
    source_str = source_str.replace(old_instantiation, new_instantiation)
    
    # Split the joined string back into lines (keeping newlines)
    lines = []
    current_line = []
    for char in source_str:
        current_line.append(char)
        if char == '\n':
            lines.append("".join(current_line))
            current_line = []
    if current_line:
        lines.append("".join(current_line))
    
    cell['source'] = lines

# Save new notebook
with open(dst_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Conversion complete: Created CICIDS_SSM_Anomaly_Detection.ipynb successfully!")
