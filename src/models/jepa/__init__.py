"""JEPA-family models: joint-embedding predictive architectures for anomaly detection."""

from src.models.jepa.flow_ts_jepa import (
    FlowLatentPredictor,
    FlowTSJEPA,
    FlowTSJEPAModel,
    flow_matching_vicreg_loss,
    von_neumann_operator_entropy_loss,
)
from src.models.jepa.gat_jepa import RelationalGAT_JEPAModel
from src.models.jepa.multiscale_ts_jepa import MultiScaleTSJEPA
from src.models.jepa.ncad_jepa import NCADJEPAModel
from src.models.jepa.patch_flow_jepa import PatchFlowJEPA, PatchFlowPredictor
from src.models.jepa.patch_ts_jepa import PatchTSJEPA
from src.models.jepa.ts_jepa import LatentPredictor, TSJEPAModel, jepa_vicreg_loss

__all__ = [
    "TSJEPAModel",
    "PatchTSJEPA",
    "MultiScaleTSJEPA",
    "LatentPredictor",
    "NCADJEPAModel",
    "RelationalGAT_JEPAModel",
    "FlowTSJEPA",
    "FlowTSJEPAModel",
    "FlowLatentPredictor",
    "flow_matching_vicreg_loss",
    "von_neumann_operator_entropy_loss",
    "PatchFlowJEPA",
    "PatchFlowPredictor",
    "jepa_vicreg_loss",
]
