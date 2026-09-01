"""Models package for NCAD-CS.

Organized into subpackages:
- ``encoders``: temporal sequence encoder backbones (TCN, GAT, SSM)
- ``jepa``: JEPA-family predictive architectures
- ``memory``: counterfactual successor memory and dynamics scorers
- ``losses``: anomaly injection and loss utilities
- ``baselines``: SOTA baseline models for comparison
- ``legacy``: backward-compatibility shims
"""

from src.models.baselines import (
    AnomalyTransformer,
    DCdetector,
    TimesNet,
    TranAD,
)
from src.models.encoders import (
    HybridTCNEncoder,
    MultiScaleTCNEncoder,
    RelationalGATEncoder,
    SelectiveSSMContextEncoder,
    contrastive_loss,
)
from src.models.jepa import (
    FlowLatentPredictor,
    FlowTSJEPA,
    FlowTSJEPAModel,
    LatentPredictor,
    MultiScaleTSJEPA,
    NCADJEPAModel,
    PatchFlowJEPA,
    PatchFlowPredictor,
    PatchTSJEPA,
    RelationalGAT_JEPAModel,
    TSJEPAModel,
    flow_matching_vicreg_loss,
    jepa_vicreg_loss,
    von_neumann_operator_entropy_loss,
)
from src.models.losses import (
    AnomalyInjectionConfig,
    ContextualAnomalyInjector,
    FrequencyMasker,
    sigreg_loss,
)
from src.models.memory import (
    CounterfactualSuccessorMemory,
    SINDyConfig,
    SINDyDynamicsScorer,
    SuccessorMemoryConfig,
)

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
    "PatchTSJEPA",
    "MultiScaleTSJEPA",
    "LatentPredictor",
    "NCADJEPAModel",
    "RelationalGAT_JEPAModel",
    "FrequencyMasker",
    # Flow-JEPA (Conditional Flow Matching)
    "FlowTSJEPA",
    "FlowTSJEPAModel",
    "FlowLatentPredictor",
    "flow_matching_vicreg_loss",
    "von_neumann_operator_entropy_loss",
    "PatchFlowJEPA",
    "PatchFlowPredictor",
    # SOTA Baselines
    "AnomalyTransformer",
    "DCdetector",
    "TimesNet",
    "TranAD",
    # Anomaly Injections & Losses
    "ContextualAnomalyInjector",
    "AnomalyInjectionConfig",
    "contrastive_loss",
    "jepa_vicreg_loss",
    "sigreg_loss",
]
