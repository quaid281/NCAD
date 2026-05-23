from pathlib import Path
import pandas as pd

runs = {
    'uncertainty': Path('notebooks_v4/results/all_channels_e15_manifold_uncertainty/summary.csv'),
    'adaptive': Path('notebooks_v4/results/all_channels_e15_adaptive_elbow/summary.csv'),
}

def load_stats(path):
    df = pd.read_csv(path)
    valid = df[df['f1'].notna()].copy()
    for col in ['tp', 'tn', 'fp', 'fn']:
        valid[col] = valid[col].fillna(0)
    tp, tn, fp, fn = [valid[col].sum() for col in ['tp', 'tn', 'fp', 'fn']]
    precision = tp / (tp + fp) if tp + fp else float('nan')
    recall = tp / (tp + fn) if tp + fn else float('nan')
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float('nan')
    stats = {
        'valid_channels': len(valid),
        'nan_channels': int(df['f1'].isna().sum()),
        'macro_f1': valid['f1'].mean(),
        'median_f1': valid['f1'].median(),
        'micro_precision': precision,
        'micro_recall': recall,
        'micro_f1': f1,
        'f1_ge_0_8': int((valid['f1'] >= 0.8).sum()),
        'f1_ge_0_5': int((valid['f1'] >= 0.5).sum()),
        'zero_f1': int((valid['f1'] == 0).sum()),
        'elapsed_seconds': valid['elapsed_seconds'].sum(),
    }
    return valid, stats

loaded = {name: load_stats(path) for name, path in runs.items()}
for name, (_, stats) in loaded.items():
    print(f'[{name}]')
    for k, v in stats.items():
        print(f'{k}: {v}')
    print()

unc = loaded['uncertainty'][0].set_index('channel')
adp = loaded['adaptive'][0].set_index('channel')
common = unc.index.intersection(adp.index)
compare = pd.DataFrame({
    'f1_uncertainty': unc.loc[common, 'f1'],
    'f1_adaptive': adp.loc[common, 'f1'],
    'precision_uncertainty': unc.loc[common, 'precision'],
    'precision_adaptive': adp.loc[common, 'precision'],
    'recall_uncertainty': unc.loc[common, 'recall'],
    'recall_adaptive': adp.loc[common, 'recall'],
})
compare['f1_delta'] = compare['f1_uncertainty'] - compare['f1_adaptive']
compare['precision_delta'] = compare['precision_uncertainty'] - compare['precision_adaptive']
compare['recall_delta'] = compare['recall_uncertainty'] - compare['recall_adaptive']
print('better_channels:', int((compare['f1_delta'] > 1e-12).sum()))
print('worse_channels:', int((compare['f1_delta'] < -1e-12).sum()))
print('same_channels:', int((compare['f1_delta'].abs() <= 1e-12).sum()))
print('mean_f1_delta:', compare['f1_delta'].mean())
print('median_f1_delta:', compare['f1_delta'].median())
print('\nTop improvements by F1 delta')
print(compare.sort_values('f1_delta', ascending=False).head(12).to_string())
print('\nWorst regressions by F1 delta')
print(compare.sort_values('f1_delta', ascending=True).head(15).to_string())
