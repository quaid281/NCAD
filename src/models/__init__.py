"""Models package for Laminar-JEPA-Lite / NCAD-CS.

Organized into subpackages:
- ``encoders``: temporal sequence encoder backbones (TCN, Multi-Scale TCN, GAT, SSM)
- ``jepa``: JEPA-family predictive architectures
- ``geometric_layers``: spectral, manifold, fluid-mechanics, and saliency layers
- ``memory``: counterfactual successor memory for the NCAD baseline
- ``losses``: anomaly injection and regularization utilities
- ``baselines``: external SOTA baseline models (TranAD, TimesNet, Anomaly Transformer, DCdetector)
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
from src.models.geometric_layers import (
    AdmissibleStressConeProjection,
    CausalChannelSaliencyGate,
    CayleyOrthogonalGate,
    CohnElkiesFilter,
    CoordinateSaliencyGate,
    GTInterlacingLayer,
    HankelMomentFilter,
    HankelPolynomialFilter,
    LittlewoodPaleyDyadicBlock,
    MovingSubspaceProjector,
    MovingTangentProjection,
    RadialHeatExteriorBoundary,
    ResolventPurification,
    ResolventPurificationBottleneck,
    ShearingWaveletPulseBlock,
    StokesStreamCurlFilter,
    SymplecticLeapfrogBlock,
)
from src.models.jepa import (
    AlienDualTowerJEPA,
    AlienDualTowerJEPAModel,
    AnchoredStokesJEPA,
    AnchoredStokesJEPAModel,
    DeltaJEPA,
    DeltaJEPAModel,
    EddyDeltaJEPA,
    EddyDeltaJEPAModel,
    HelmholtzSolitonJEPA,
    HelmholtzSolitonJEPAModel,
    LatentPredictor,
    MultiScaleTSJEPA,
    SolitonReynoldsJEPA,
    SolitonReynoldsJEPAModel,
    SpeculativeJEPA,
    SpeculativeJEPAModel,
    TSJEPAModel,
    jepa_vicreg_loss,
)
from src.models.laminar_lite import LaminarJEPALiteModel, LaminarLiteEncoder
from src.models.losses import (
    AnomalyInjectionConfig,
    ContextualAnomalyInjector,
    FrequencyMasker,
    sigreg_loss,
)
from src.models.memory import (
    CounterfactualSuccessorMemory,
    SuccessorMemoryConfig,
)

__all__ = [
    # Primary Laminar-JEPA-Lite
    "LaminarJEPALiteModel",
    "LaminarLiteEncoder",
    # Encoders
    "HybridTCNEncoder",
    "MultiScaleTCNEncoder",
    "RelationalGATEncoder",
    "SelectiveSSMContextEncoder",
    # Memory & Dynamics
    "CounterfactualSuccessorMemory",
    "SuccessorMemoryConfig",
    # JEPA Architectures
    "TSJEPAModel",
    "MultiScaleTSJEPA",
    "LatentPredictor",
    "DeltaJEPA",
    "DeltaJEPAModel",
    "SpeculativeJEPA",
    "SpeculativeJEPAModel",
    "AlienDualTowerJEPA",
    "AlienDualTowerJEPAModel",
    "EddyDeltaJEPA",
    "EddyDeltaJEPAModel",
    "AnchoredStokesJEPA",
    "AnchoredStokesJEPAModel",
    "SolitonReynoldsJEPA",
    "SolitonReynoldsJEPAModel",
    "HelmholtzSolitonJEPA",
    "HelmholtzSolitonJEPAModel",
    # Geometric Invariant Layers
    "AdmissibleStressConeProjection",
    "CausalChannelSaliencyGate",
    "CayleyOrthogonalGate",
    "CohnElkiesFilter",
    "CoordinateSaliencyGate",
    "GTInterlacingLayer",
    "HankelMomentFilter",
    "HankelPolynomialFilter",
    "LittlewoodPaleyDyadicBlock",
    "MovingSubspaceProjector",
    "MovingTangentProjection",
    "RadialHeatExteriorBoundary",
    "ResolventPurification",
    "ResolventPurificationBottleneck",
    "ShearingWaveletPulseBlock",
    "StokesStreamCurlFilter",
    "SymplecticLeapfrogBlock",
    # SOTA Baselines
    "AnomalyTransformer",
    "DCdetector",
    "TimesNet",
    "TranAD",
    # Anomaly Injections & Losses
    "ContextualAnomalyInjector",
    "AnomalyInjectionConfig",
    "FrequencyMasker",
    "contrastive_loss",
    "jepa_vicreg_loss",
    "sigreg_loss",
]
