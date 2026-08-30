import os
from pathlib import Path

root = Path("mTSBench_data")
print("=" * 60)
print("mTSBench_data DATASETS & CHANNEL COUNTS:")
print("=" * 60)
total_mTS = 0
for p in sorted(root.iterdir()):
    if p.is_dir():
        train_files = sorted(list(p.glob("*_train.csv")))
        total_mTS += len(train_files)
        chan_names = [f.name for f in train_files]
        print(f"{p.name:20s}: {len(train_files):2d} channels")
        for c in chan_names[:10]:
            print(f"    - {c}")
        if len(chan_names) > 10:
            print(f"    ... and {len(chan_names) - 10} more")

print(f"\nTotal channels in mTSBench_data: {total_mTS}")

raw_dir = Path("data/raw/train")
if raw_dir.exists():
    npy_files = list(raw_dir.glob("*.npy"))
    print(f"data/raw/train (NASA SMAP/MSL .npy files): {len(npy_files)} channels")
