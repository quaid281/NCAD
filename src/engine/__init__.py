"""Engine subsystem for NCAD-CS training, evaluation, and pipeline orchestration."""

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
from src.engine.evaluator import (
    build_successor_memory,
    calibrate_event_threshold,
    score_windows,
)
from src.engine.orchestrator import (
    default_output_dir,
    run_channel,
    run_experiment,
    save_channel_outputs,
)

__all__ = [
    # Trainer
    "EncoderModel",
    "build_encoder",
    "encode_windows",
    "evaluate_contrastive_loss",
    "limit_windows",
    "resolve_device",
    "set_seed",
    "split_train_validation",
    "train_encoder",
    # Evaluator
    "build_successor_memory",
    "calibrate_event_threshold",
    "score_windows",
    # Orchestrator
    "default_output_dir",
    "run_channel",
    "run_experiment",
    "save_channel_outputs",
]
