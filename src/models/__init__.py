"""Models package for NCAD-CS."""

from src.models.anomaly_injector import AnomalyInjectionConfig, ContextualAnomalyInjector
from src.models.fei_sigreg import FrequencyMasker, sigreg_loss
from src.models.gat_jepa import RelationalGAT_JEPAModel
from src.models.multi_scale_tcn_encoder import MultiScaleTCNEncoder
from src.models.ncad_jepa import NCADJEPAModel
from src.models.relational_gat_encoder import RelationalGATEncoder
from src.models.selective_ssm_encoder import SelectiveSSMContextEncoder
from src.models.sindy_scorer import SINDyConfig, SINDyDynamicsScorer
from src.models.successor_memory import CounterfactualSuccessorMemory, SuccessorMemoryConfig
from src.models.tcn_encoder import HybridTCNEncoder, contrastive_loss
from src.models.ts_jepa import TSJEPAModel, jepa_vicreg_loss

__all__ = [
    # Encoders
    "HybridTCNEncoder",
    "MultiScaleTCNEncoder",
    "RelationalGATEncoder",
    "SelectiveSSMContextEncoder",
    # Memory & Dynamics
    "CounterfactualSuccessorMemory",
    "SuccessorMemoryConfig",
    "SINDyDynamicsScorer",
    "SINDyConfig",
    # JEPA & Invariance Architectures
    "TSJEPAModel",
    "NCADJEPAModel",
    "RelationalGAT_JEPAModel",
    "FrequencyMasker",
    # Anomaly Injections & Losses
    "ContextualAnomalyInjector",
    "AnomalyInjectionConfig",
    "contrastive_loss",
    "jepa_vicreg_loss",
    "sigreg_loss",
]

