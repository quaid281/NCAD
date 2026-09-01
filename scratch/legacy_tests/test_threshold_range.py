import json
import numpy as np
from sklearn.metrics import f1_score
import sys
from pathlib import Path

# Resolve paths
project_root = Path.cwd().resolve()
sys.path.insert(0, str(project_root))

from src.scoring.event_fusion import event_level_filter, compute_metrics, moving_average, aggregate_window_scores

def evaluate_new_search(nb_path, dataset_name, step, context_size, suspect_size):
    print(f"\n=== Evaluating threshold search on {dataset_name} ===")
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
    
    # We need to extract the variables: val_scores, val_mask, test_scores, test_mask, val_df, test_df
    # Since we executed the notebooks inplace, we can extract them by running python code that loads the data
    # and runs the evaluation cells. Or we can just re-execute the evaluation section using the saved scores
    # actually, wait! The scores themselves are not saved to disk by default, but we can write a quick script
    # to run the notebook's cells or run a standalone script that runs the evaluation.
    # Let's write a standalone script that uses the trained models or saved data.
    # Wait, instead of retraining, let's write a python script that replicates the final evaluation of the notebooks
    # but with a wider threshold search range!

# Let's write a standalone script to load datasets, compute scores using the trained notebooks models
# Wait, did the notebooks save the trained models?
# Let's check if the notebooks saved the trained models.
# Let's list the directory contents of notebooks_v4/CalIt2 and notebooks_v4/cicids.
print("Checking model files...")
