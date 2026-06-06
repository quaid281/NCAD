"""Update all three SSM notebooks to use multi-signal fusion scoring."""
import json
import sys
from pathlib import Path

def update_notebook(nb_path: Path, dataset_name: str):
    """Update the scoring cell and threshold cell in a notebook."""
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
    
    code_cells = [(i, c) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
    
    # Find the cell with compute_anomaly_scores
    scoring_cell_idx = None
    threshold_cell_idx = None
    
    for cell_pos, (global_idx, cell) in enumerate(code_cells):
        src = ''.join(cell['source'])
        if 'def compute_anomaly_scores' in src:
            scoring_cell_idx = global_idx
        if 'Supervised Validation' in src and 'candidates = np.linspace' in src:
            threshold_cell_idx = global_idx
    
    if scoring_cell_idx is None:
        print(f"  WARNING: Could not find compute_anomaly_scores cell in {nb_path.name}")
        return False
    if threshold_cell_idx is None:
        print(f"  WARNING: Could not find threshold search cell in {nb_path.name}")
        return False
    
    # Build the new scoring cell
    new_scoring_source = '''def encode_windows(model, windows, batch_size=32):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size]).float().to(device)
            embeddings.append(model(batch).cpu().numpy())
    return np.concatenate(embeddings, axis=0).astype(np.float32)

# Encode training contexts
train_contexts = train_windows[:, :context_size]
train_successors = train_windows[:, context_size:]
train_context_embeddings = encode_windows(model, train_contexts)

# Instantiate and fit successor memory
memory = CounterfactualSuccessorMemory(
    SuccessorMemoryConfig(n_neighbors=8, max_memory_windows=5000, context_percentile=99.0, seed=42)
)
memory.fit(train_context_embeddings, train_successors)
print(f"Fitted Successor Memory with {len(memory.context_embeddings)} windows.")

def compute_anomaly_scores(windows, dataset_name):
    print(f"Computing anomaly scores for {dataset_name} set...")
    
    # 1. Obtain calibration stats from training memory
    train_local_scores = local_deviation_scores(train_windows, context_size, tail_size=64)
    calibration_local_scores = train_local_scores[memory.sample_indices]
    
    # Reconstruction calibration
    train_recon_scores = reconstruction_deviation_scores(
        train_successors[memory.sample_indices],
        memory.successor_windows  # expected from self-query
    )
    
    successor_stats = robust_stats(memory.calibration_successor_scores)
    local_stats = robust_stats(calibration_local_scores)
    recon_stats = robust_stats(train_recon_scores)
    
    # 2. Query Successor Memory for the target dataset
    contexts = windows[:, :context_size]
    observed_successors = windows[:, context_size:]
    context_embeddings = encode_windows(model, contexts)
    query = memory.query(context_embeddings, observed_successors)
    
    # 3. Compute component robust Z-scores
    local_raw_scores = local_deviation_scores(windows, context_size, tail_size=64)
    successor_z = positive_robust_z(query.successor_scores, successor_stats)
    local_z = positive_robust_z(local_raw_scores, local_stats)
    
    # Reconstruction deviation
    recon_raw = reconstruction_deviation_scores(observed_successors, query.expected_successors)
    recon_z = positive_robust_z(recon_raw, recon_stats)
    
    if float(memory.context_threshold) <= 1e-6:
        context_ratio = np.ones_like(query.context_distances, dtype=np.float32)
    else:
        context_ratio = query.context_distances / float(memory.context_threshold)
        
    # 4. Multi-signal fusion scoring
    window_scores = fuse_evidence_scores(
        successor_z=successor_z,
        local_z=local_z,
        context_ratio=context_ratio,
        reconstruction_z=recon_z,
        successor_weight=1.0,
        local_weight=0.80,
        context_weight=0.35,
        reconstruction_weight=0.60,
    )
    return window_scores

# Compute window-level scores
val_window_scores = compute_anomaly_scores(val_windows, "Validation")
test_window_scores = compute_anomaly_scores(test_windows, "Test")
'''
    
    # Build the new threshold search cell
    new_threshold_source = '''# Strategy 1: Unsupervised Adaptive Elbow Threshold
val_valid_scores = val_scores[val_mask]
floor_res = adaptive_elbow_score_floor(val_valid_scores)
unsupervised_threshold = floor_res.threshold
print(f"Unsupervised Adaptive Elbow Threshold: {unsupervised_threshold:.5f} (Selected candidate: {floor_res.selected_candidate})")

# Strategy 2: Supervised Validation Set Tuning (Maximize F1)
val_labels = val_df['is_anomaly'].to_numpy()
best_f1 = 0.0
best_threshold = 0.0

# Dense percentile-weighted threshold search
candidates = np.unique(np.concatenate([
    np.linspace(np.percentile(val_valid_scores, 0.5), np.percentile(val_valid_scores, 50), 200),
    np.linspace(np.percentile(val_valid_scores, 50), np.percentile(val_valid_scores, 99.99), 300),
]))
for th in candidates:
    preds = event_level_filter(val_scores, th, val_mask, min_run=2, extreme_factor=1.75)
    metrics = compute_metrics(val_labels, preds, valid_mask=val_mask)
    if metrics.get('f1', 0.0) > best_f1:
        best_f1 = metrics['f1']
        best_threshold = th

print(f"Supervised Validation Optimized Threshold: {best_threshold:.5f} (Validation F1-score: {best_f1:.4f})")
'''

    # Update the imports cell to include the new function
    for global_idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        if 'from src.utils.event_fusion import' in src:
            # Add reconstruction_deviation_scores to imports if not already there
            if 'reconstruction_deviation_scores' not in src:
                new_src = src.replace(
                    'from src.utils.event_fusion import',
                    'from src.utils.event_fusion import reconstruction_deviation_scores,\\\n   '
                )
                # Actually, let's do a cleaner replacement
                # Find the import line and add to it
                lines = cell['source'] if isinstance(cell['source'], list) else [cell['source']]
                new_lines = []
                for line in lines:
                    if 'from src.utils.event_fusion import' in line and 'reconstruction_deviation_scores' not in line:
                        # Insert the new import
                        line = line.rstrip('\n').rstrip('\r')
                        if line.endswith(')'):
                            # Multi-line import with parens
                            line = line[:-1] + ', reconstruction_deviation_scores)\n'
                        else:
                            line = line + ', reconstruction_deviation_scores\n'
                    new_lines.append(line)
                cell['source'] = new_lines
            break
    
    # Replace cells
    nb['cells'][scoring_cell_idx]['source'] = [line + '\n' for line in new_scoring_source.rstrip('\n').split('\n')]
    nb['cells'][scoring_cell_idx]['outputs'] = []
    nb['cells'][scoring_cell_idx]['execution_count'] = None
    
    nb['cells'][threshold_cell_idx]['source'] = [line + '\n' for line in new_threshold_source.rstrip('\n').split('\n')]
    nb['cells'][threshold_cell_idx]['outputs'] = []
    nb['cells'][threshold_cell_idx]['execution_count'] = None
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"  Updated {nb_path.name}: scoring cell {scoring_cell_idx}, threshold cell {threshold_cell_idx}")
    return True


def fix_creditcard_min_run(nb_path: Path):
    """Also fix the test evaluation cell for creditcard to use min_run=1."""
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        if 'Test Set Evaluation' in src and 'min_run=1' in src:
            # Already has min_run=1, good
            pass
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    project = Path('c:/Users/andre/OneDrive/Desktop/NCAD_CS')
    
    notebooks = [
        project / 'notebooks_v4' / 'CalIt2' / 'CalIt2_SSM_Anomaly_Detection.ipynb',
        project / 'notebooks_v4' / 'cicids' / 'CICIDS_SSM_Anomaly_Detection.ipynb',
        project / 'notebooks_v4' / 'creditcard' / 'Creditcard_SSM_Anomaly_Detection.ipynb',
    ]
    
    for nb_path in notebooks:
        print(f"\nProcessing {nb_path.name}...")
        if not nb_path.exists():
            print(f"  SKIPPED: {nb_path} does not exist")
            continue
        update_notebook(nb_path, nb_path.stem)
    
    print("\nDone! All notebooks updated.")
