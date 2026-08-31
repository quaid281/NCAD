"""NCAD-CS training and evaluation legacy shim.

This module re-exports all configuration, training, evaluation, and CLI components
from the new modular engine (`src.config`, `src.engine`, `src.cli`) to ensure
100% backward compatibility with all existing benchmark scripts and test suites.
"""

from __future__ import annotations

# 5. CLI
from src.cli import main, parse_args

# 1. Configuration
from src.config import CSMConfig

# 3. Engine Evaluator
from src.engine.evaluator import (
    build_successor_memory,
    calibrate_event_threshold,
    score_windows,
)

# 4. Engine Orchestrator
from src.engine.orchestrator import (
    default_output_dir,
    run_channel,
    run_experiment,
    save_channel_outputs,
)

# 2. Engine Trainer
from src.engine.trainer import (
    EncoderModel,
    build_encoder,
    encode_windows,
    evaluate_contrastive_loss,
    limit_windows,
    resolve_device,
    set_seed,
    split_train_validation,
    train_encoder,
)

__all__ = [
    # Config
    "CSMConfig",
    # Trainer
    "EncoderModel",
    "set_seed",
    "resolve_device",
    "build_encoder",
    "limit_windows",
    "split_train_validation",
    "encode_windows",
    "train_encoder",
    "evaluate_contrastive_loss",
    # Evaluator
    "build_successor_memory",
    "score_windows",
    "calibrate_event_threshold",
    # Orchestrator
    "default_output_dir",
    "run_channel",
    "save_channel_outputs",
    "run_experiment",
    # CLI
    "parse_args",
    "main",
]

if __name__ == "__main__":
    main()
