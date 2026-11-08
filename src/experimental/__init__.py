"""Isolated experimental modules for post-submission novelty exploration.

This package houses self-contained prototypes for next-generation TS-JEPA
extensions without modifying the frozen core manuscript pipeline:
  1. Continuous Hopfield (modern Ramsauer-style associative memory) context banks.
  2. Causal counterfactual context substitution (Pearl do-calculus residual scoring).
  3. Selective state-space (Mamba-style input-dependent gating) context encoders.
  4. Memoryless context substitution (learned decontamination prior).
  5. Weight-Space Programming (WSP) for SolitonReynoldsJEPAModel & 224-dial circuit breakers
     (Experiment 1: Surgical Zero-Gradient False-Positive Quenching;
      Experiment 2: Blind Weight-Space Reverse Engineering & Zero-Data Auditing).
"""

from .causal_counterfactual import (
    CounterfactualConfig,
    CounterfactualContextResult,
    CounterfactualContextSubstitutor,
)
from .hopfield_context import (
    HopfieldContextConfig,
    HopfieldContextMemory,
    HopfieldRetrieval,
)
from .mamba_ssm_encoder import (
    ExperimentalSSMContextEncoder,
    SelectiveStateSpaceBlock,
)
from .memoryless_context import (
    MemorylessContextConfig,
    MemorylessContextResult,
    MemorylessContextSubstitutor,
)
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
    "CounterfactualConfig",
    "CounterfactualContextResult",
    "CounterfactualContextSubstitutor",
    "ExperimentalSSMContextEncoder",
    "HopfieldContextConfig",
    "HopfieldContextMemory",
    "HopfieldRetrieval",
    "LegendreHorizonBasisParametrization",
    "MemorylessContextConfig",
    "MemorylessContextResult",
    "MemorylessContextSubstitutor",
    "PhaseSynthesisResult",
    "SelectiveStateSpaceBlock",
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
