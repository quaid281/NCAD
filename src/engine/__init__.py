"""Engine subsystem for NCAD-CS training, evaluation, and pipeline orchestration.

Orchestrator re-exports are deferred so that importing trainer/evaluator
utilities does not pull in the full orchestration (and plotting) stack.
"""

from src.engine.evaluator import (
    build_successor_memory,
    calibrate_event_threshold,
    calibrate_jepa_threshold,
    compute_jepa_discrepancy,
    score_windows,
)
from src.engine.trainer import (
    EncoderModel,
    build_encoder,
    build_ts_jepa_model,
    encode_windows,
    evaluate_contrastive_loss,
    evaluate_ts_jepa_loss,
    limit_windows,
    resolve_device,
    set_seed,
    split_train_validation,
    train_encoder,
    train_ts_jepa,
)

__all__ = [
    # Trainer
    "EncoderModel",
    "build_encoder",
    "build_ts_jepa_model",
    "encode_windows",
    "evaluate_contrastive_loss",
    "evaluate_ts_jepa_loss",
    "limit_windows",
    "resolve_device",
    "set_seed",
    "split_train_validation",
    "train_encoder",
    "train_ts_jepa",
    # Evaluator
    "build_successor_memory",
    "calibrate_event_threshold",
    "calibrate_jepa_threshold",
    "compute_jepa_discrepancy",
    "score_windows",
    # Orchestrator (lazy)
    "default_output_dir",
    "run_channel",
    "run_experiment",
    "save_channel_outputs",
]

_LAZY_ORCHESTRATOR = {
    "default_output_dir",
    "run_channel",
    "run_experiment",
    "save_channel_outputs",
}


def __getattr__(name: str):
    """Lazy import orchestrator symbols to avoid eager import side effects."""
    if name in _LAZY_ORCHESTRATOR:
        from src.engine import orchestrator as _orch

        return getattr(_orch, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
