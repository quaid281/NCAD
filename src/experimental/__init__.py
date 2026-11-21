"""Auxiliary diagnostic and ablation utilities for multi-horizon JEPA studies.

Contains Weight-Space Programming (WSP) surgical false-positive quenching and
Legendre phase-embedding synthesis used in ablation analyses of the 224-dial
Soliton-Reynolds JEPA tier.
"""

from .weight_space_programming import (
    ORTHOGONAL_LEGENDRE_4BASIS,
    BlindCoordinateDialProfile,
    BlindWSPReport,
    CoordinateWSPProfile,
    LegendreHorizonBasisParametrization,
    PhaseSynthesisResult,
    WSPConfig,
    WSPDiagnosticReport,
    WSPSurgicalResult,
    apply_blind_wsp_preconditioning,
    apply_surgical_wsp_quench,
    apply_unified_wsp_soliton_program,
    bake_structural_legendre_phase_basis,
    bind_structural_legendre_phase_basis,
    blind_reverse_engineer_weight_space,
    evaluate_blind_dial_reconstruction,
    inspect_soliton_reynolds_weight_space,
    synthesize_subblock_phase_embeddings,
)

__all__ = [
    "ORTHOGONAL_LEGENDRE_4BASIS",
    "BlindCoordinateDialProfile",
    "BlindWSPReport",
    "CoordinateWSPProfile",
    "LegendreHorizonBasisParametrization",
    "PhaseSynthesisResult",
    "WSPConfig",
    "WSPDiagnosticReport",
    "WSPSurgicalResult",
    "apply_blind_wsp_preconditioning",
    "apply_surgical_wsp_quench",
    "apply_unified_wsp_soliton_program",
    "bake_structural_legendre_phase_basis",
    "bind_structural_legendre_phase_basis",
    "blind_reverse_engineer_weight_space",
    "evaluate_blind_dial_reconstruction",
    "inspect_soliton_reynolds_weight_space",
    "synthesize_subblock_phase_embeddings",
]
