import json
from pathlib import Path

def update_notebook(nb_path: Path):
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
    
    scoring_cell_updated = False
    eval_cell_updated = False
    
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        
        src = ''.join(cell['source'])
        
        # 1. Update the scoring calibration
        if 'memory.successor_windows  # expected from self-query' in src:
            new_src = src.replace(
                'memory.successor_windows  # expected from self-query',
                'memory.calibration_expected_successors'
            )
            cell['source'] = [line + '\n' for line in new_src.rstrip('\n').split('\n')]
            scoring_cell_updated = True
            
        # 2. Update the evaluation cell
        if "test_labels = test_df['is_anomaly'].to_numpy()" in src and "Validation Set Optimized" in src:
            if "val_pct = np.mean(val_valid_scores <= best_threshold)" not in src:
                # We need to preserve min_run (1 for creditcard, 2 for others)
                target_str = "for name, threshold in [(\"Unsupervised Adaptive Elbow\", unsupervised_threshold), (\"Validation Set Optimized\", best_threshold)]:"
                replacement_str = (
                    "val_pct = np.mean(val_valid_scores <= best_threshold) * 100.0\n"
                    "test_pct_threshold = np.percentile(test_scores[test_mask], val_pct)\n"
                    "print(f\"Validation-optimized threshold {best_threshold:.5f} mapped to test threshold {test_pct_threshold:.5f} (percentile {val_pct:.4f}%)\")\n\n"
                    "for name, threshold in [(\"Unsupervised Adaptive Elbow\", unsupervised_threshold), (\"Validation Set Optimized\", test_pct_threshold)]:"
                )
                new_src = src.replace(target_str, replacement_str)
                cell['source'] = [line + '\n' for line in new_src.rstrip('\n').split('\n')]
                eval_cell_updated = True
                
    if scoring_cell_updated or eval_cell_updated:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Updated {nb_path.name}: scoring={scoring_cell_updated}, eval={eval_cell_updated}")
    else:
        print(f"No updates needed for {nb_path.name}")

if __name__ == '__main__':
    project = Path('c:/Users/andre/OneDrive/Desktop/NCAD_CS')
    notebooks = [
        project / 'notebooks_v4' / 'CalIt2' / 'CalIt2_SSM_Anomaly_Detection.ipynb',
        project / 'notebooks_v4' / 'cicids' / 'CICIDS_SSM_Anomaly_Detection.ipynb',
        project / 'notebooks_v4' / 'creditcard' / 'Creditcard_SSM_Anomaly_Detection.ipynb',
    ]
    for nb in notebooks:
        if nb.exists():
            update_notebook(nb)
