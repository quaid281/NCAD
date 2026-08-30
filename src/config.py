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

    @property
    def full_window_size(self) -> int:
        return self.context_size + self.suspect_size
