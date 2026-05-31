import json
from pathlib import Path

# Paths
notebook_dir = Path(r'c:\Users\andre\OneDrive\Desktop\NCAD_CS\notebooks_v4\cicids')
src_path = notebook_dir / 'CICIDS_Anomaly_Detection.ipynb'
dst_path = notebook_dir / 'CICIDS_SSM_Anomaly_Detection.ipynb'

# Load the notebook
with open(src_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Replacements function
def replace_text(obj):
    if isinstance(obj, dict):
        return {k: replace_text(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_text(item) for item in obj]
    elif isinstance(obj, str):
        # Perform replacements
        res = obj
        res = res.replace("TCN Encoder", "SSM Encoder")
        res = res.replace("TCN embeddings", "SSM embeddings")
        res = res.replace("NCAD-CS v4: Anomaly Detection Protocol on CICIDS Intrusion Detection Dataset", "NCAD-CS v4: Anomaly Detection Protocol on CICIDS Intrusion Detection Dataset (SSM Encoder)")
        
        # Imports replacement
        res = res.replace("from src.models.tcn_encoder import HybridTCNEncoder, contrastive_loss", 
                          "from src.models.tcn_encoder import contrastive_loss\nfrom src.models.selective_ssm_encoder import SelectiveSSMContextEncoder")
        
        # Model training markdown replacement
        res = res.replace("### Step 3: Contrastive Training of TCN Encoder", "### Step 3: Contrastive Training of Selective SSM Encoder")
        res = res.replace("We train the `HybridTCNEncoder` model to map features", "We train the `SelectiveSSMContextEncoder` model to map features")
        
        # Model instantiation code replacement
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
        res = res.replace(old_instantiation, new_instantiation)
        
        return res
    else:
        return obj

# Modify the notebook cells
nb_new = replace_text(nb)

# Write output
with open(dst_path, 'w', encoding='utf-8') as f:
    json.dump(nb_new, f, indent=1)

print("Conversion complete: Created CICIDS_SSM_Anomaly_Detection.ipynb successfully!")
