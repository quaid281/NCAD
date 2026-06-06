"""Fix the import statement in all three SSM notebooks."""
import json
from pathlib import Path

def fix_imports(nb_path: Path):
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        if 'from src.utils.event_fusion import' in src:
            # Fix the broken import - rebuild the import statement properly
            new_lines = []
            in_event_fusion_import = False
            import_done = False
            
            for line in cell['source']:
                if 'from src.utils.event_fusion import' in line and not import_done:
                    # Start rebuilding
                    in_event_fusion_import = True
                    new_lines.append('from src.utils.event_fusion import (\n')
                    new_lines.append('    adaptive_elbow_score_floor,\n')
                    new_lines.append('    aggregate_window_scores,\n')
                    new_lines.append('    compute_metrics,\n')
                    new_lines.append('    event_level_filter,\n')
                    new_lines.append('    fuse_evidence_scores,\n')
                    new_lines.append('    local_deviation_scores,\n')
                    new_lines.append('    moving_average,\n')
                    new_lines.append('    percentile_score_floor,\n')
                    new_lines.append('    positive_robust_z,\n')
                    new_lines.append('    reconstruction_deviation_scores,\n')
                    new_lines.append('    robust_dispersion_floor,\n')
                    new_lines.append('    robust_stats,\n')
                    new_lines.append('    dispersion_confidence,\n')
                    new_lines.append('    successor_manifold_uncertainty_scores,\n')
                    new_lines.append(')\n')
                    continue
                
                if in_event_fusion_import:
                    # Skip old import lines until we find the closing paren
                    stripped = line.strip()
                    if stripped == ')' or stripped.endswith(')'):
                        in_event_fusion_import = False
                        import_done = True
                        continue
                    # Skip intermediate import lines
                    continue
                
                new_lines.append(line)
            
            cell['source'] = new_lines
            break
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Fixed imports in {nb_path.name}")

project = Path('c:/Users/andre/OneDrive/Desktop/NCAD_CS')
notebooks = [
    project / 'notebooks_v4' / 'CalIt2' / 'CalIt2_SSM_Anomaly_Detection.ipynb',
    project / 'notebooks_v4' / 'cicids' / 'CICIDS_SSM_Anomaly_Detection.ipynb',
    project / 'notebooks_v4' / 'creditcard' / 'Creditcard_SSM_Anomaly_Detection.ipynb',
]

for nb_path in notebooks:
    fix_imports(nb_path)

print("Done!")
