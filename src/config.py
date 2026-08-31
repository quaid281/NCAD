"""Configuration dataclasses and settings for NCAD-CS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CSMConfig:
    data_dir: Optional[str] = None
    output_dir: Optional[str] = None
    context_size: int = 284
    suspect_size: int = 16
    step: int = 1
    feature_dim: int = 64
    encoder_architecture: str = "hybrid_tcn"
    model_type: str = "ts_jepa"  # Options: "ts_jepa", "patch_ts_jepa", "gat_jepa", "ncad"
    patch_size: int = 16
    use_mahalanobis: bool = True
    vicreg_cov_weight: float = 0.5
    vicreg_var_weight: float = 1.0
    vicreg_sim_weight: float = 1.0
    latent_dim: int = 32
    filters: int = 48
    tcn_layers: int = 6
    kernel_size: int = 5
    dropout: float = 0.20
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    margin: float = 1.0
    val_split: float = 0.10
    patience: int = 8
    injection_ratio: float = 0.70
    successor_neighbors: int = 8
    context_percentile: float = 99.0
    event_threshold_percentile: float = 99.0
    threshold_method: str = "evt"
    evt_risk_level: float = 1e-3
    evt_init_percentile: float = 98.0
    score_floor_percentile: Optional[float] = None
    degenerate_score_epsilon: float = 1e-6
    local_tail_size: int = 64
    manifold_uncertainty: bool = False
    manifold_dispersion_floor_percentile: float = 50.0
    manifold_min_confidence: float = 0.35
    smoothing_window: int = 12
    min_event_run: int = 2
    extreme_event_factor: float = 1.75
    max_train_windows: Optional[int] = None
    max_test_windows: Optional[int] = None
    max_memory_windows: Optional[int] = 5000
    save_plots: bool = True
    seed: int = 42
    device: str = "auto"
    mapping_method: str = "trailing"  # Causal trailing window alignment (zero lookahead)
    use_pa: bool = True

    _VALID_MODEL_TYPES = None  # populated lazily from the registry
    _VALID_ENCODERS = {"hybrid_tcn", "multi_scale_tcn", "relational_gat", "selective_ssm", "ssm"}
    _VALID_THRESHOLD_METHODS = {"evt", "adaptive_elbow", "percentile"}
    _VALID_MAPPING_METHODS = {"smear", "last", "trailing", "first", "leading", "middle", "center", "suspect_trailing"}

    def __post_init__(self) -> None:
        """Validate interdependent configuration options at construction time."""
        from src.models.registry import requires_patch_division, valid_model_types

        valid_models = valid_model_types()
        if self.model_type not in valid_models:
            raise ValueError(f"Unknown model_type: {self.model_type!r}. Valid: {sorted(valid_models)}")
        if self.encoder_architecture not in self._VALID_ENCODERS:
            raise ValueError(f"Unknown encoder_architecture: {self.encoder_architecture!r}. Valid: {sorted(self._VALID_ENCODERS)}")
        if self.threshold_method not in self._VALID_THRESHOLD_METHODS:
            raise ValueError(f"Unknown threshold_method: {self.threshold_method!r}. Valid: {sorted(self._VALID_THRESHOLD_METHODS)}")
        if self.mapping_method not in self._VALID_MAPPING_METHODS:
            raise ValueError(f"Unknown mapping_method: {self.mapping_method!r}. Valid: {sorted(self._VALID_MAPPING_METHODS)}")
        if self.context_size <= 0:
            raise ValueError(f"context_size must be positive, got {self.context_size}")
        if self.suspect_size <= 0:
            raise ValueError(f"suspect_size must be positive, got {self.suspect_size}")
        if self.step <= 0:
            raise ValueError(f"step must be positive, got {self.step}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if not 0.0 < self.val_split < 1.0:
            raise ValueError(f"val_split must be in (0, 1), got {self.val_split}")
        if not 0.0 < self.event_threshold_percentile <= 100.0:
            raise ValueError(f"event_threshold_percentile must be in (0, 100], got {self.event_threshold_percentile}")
        if not 0.0 < self.evt_init_percentile <= 100.0:
            raise ValueError(f"evt_init_percentile must be in (0, 100], got {self.evt_init_percentile}")
        if requires_patch_division(self.model_type):
            if self.context_size % self.patch_size != 0:
                raise ValueError(
                    f"context_size ({self.context_size}) must be divisible by patch_size ({self.patch_size}) "
                    f"for model_type {self.model_type!r}"
                )
            if self.suspect_size % self.patch_size != 0:
                raise ValueError(
                    f"suspect_size ({self.suspect_size}) must be divisible by patch_size ({self.patch_size}) "
                    f"for model_type {self.model_type!r}"
                )

    @property
    def full_window_size(self) -> int:
        return self.context_size + self.suspect_size
