#!/usr/bin/env python3
"""Train Soliton-Reynolds JEPA on the 19 primary streams x 3 seeds and cache whitened residuals.

Usage: python -m scripts.circuit_breaker_19stream.build_cache
"""

from __future__ import annotations

import torch

from scripts.circuit_breaker_19stream.common import CACHE_DIR, SEEDS, cache_path, load_streams_19, train_and_cache


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ds, ch in load_streams_19():
        for seed in SEEDS:
            status = "cached" if cache_path(ds, ch, seed).exists() else "training"
            print(f"{ds}:{ch} seed={seed} ({status})", flush=True)
            train_and_cache(ds, ch, seed, device)
    print(f"Cache ready in {CACHE_DIR}")


if __name__ == "__main__":
    main()
