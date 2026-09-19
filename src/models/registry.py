"""Central registry of supported model types and their metadata.

This is the single source of truth for which model types are supported by the
CLI, configuration validation, and the model builder. Keeping the choices in
one place prevents drift between ``src/cli.py``, ``src/config.py``, and
``src/engine/trainer.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Optional, Tuple


@dataclass(frozen=True)
class ModelSpec:
    """Metadata for a supported model variant."""

    canonical_name: str
    aliases: Tuple[str, ...]
    is_jepa: bool
    requires_patch_division: bool
    description: str


# Canonical model registry. The first entry for each canonical name is the
# primary spec; aliases are accepted by the builder and config validation.
_MODEL_SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        canonical_name="ts_jepa",
        aliases=(),
        is_jepa=True,
        requires_patch_division=False,
        description="TS-JEPA with VICReg-style self-supervised predictive coding.",
    ),
    ModelSpec(
        canonical_name="patch_ts_jepa",
        aliases=("patch_jepa",),
        is_jepa=True,
        requires_patch_division=True,
        description="Patch-tokenized Transformer JEPA variant.",
    ),
    ModelSpec(
        canonical_name="gat_jepa",
        aliases=("relational_gat_jepa",),
        is_jepa=True,
        requires_patch_division=False,
        description="Relational graph-attention JEPA variant.",
    ),
    ModelSpec(
        canonical_name="ncad",
        aliases=(),
        is_jepa=False,
        requires_patch_division=False,
        description="Legacy NCAD contrastive encoder (not a JEPA variant).",
    ),
    ModelSpec(
        canonical_name="anomaly_transformer",
        aliases=(),
        is_jepa=False,
        requires_patch_division=False,
        description="Anomaly Transformer (ICLR 2022) baseline.",
    ),
    ModelSpec(
        canonical_name="timesnet",
        aliases=(),
        is_jepa=False,
        requires_patch_division=False,
        description="TimesNet (ICLR 2023) baseline.",
    ),
    ModelSpec(
        canonical_name="dcdetector",
        aliases=(),
        is_jepa=False,
        requires_patch_division=False,
        description="DCdetector (KDD 2023) baseline.",
    ),
    ModelSpec(
        canonical_name="tranad",
        aliases=(),
        is_jepa=False,
        requires_patch_division=False,
        description="TranAD (VLDB 2022) baseline.",
    ),
    ModelSpec(
        canonical_name="ncad_jepa",
        aliases=("ncad_jepa_v1",),
        is_jepa=True,
        requires_patch_division=False,
        description="Fused NCAD + TS-JEPA with VICReg and contextual anomaly injection.",
    ),
    ModelSpec(
        canonical_name="ncad_flow_jepa",
        aliases=("ncad_flow_jepa_v1",),
        is_jepa=True,
        requires_patch_division=False,
        description="Fused NCAD + Flow Matching TS-JEPA with OT-CFM velocity field and contextual anomaly injection.",
    ),
    ModelSpec(
        canonical_name="flow_jepa",
        aliases=("ts_jepa_flow",),
        is_jepa=True,
        requires_patch_division=False,
        description="Conditional Flow Matching TS-JEPA with OT-CFM velocity field predictor.",
    ),
    ModelSpec(
        canonical_name="patch_flow_jepa",
        aliases=("ts_jepa_patch_flow",),
        is_jepa=True,
        requires_patch_division=True,
        description="Patch-tokenized Flow Matching JEPA with cross-attention velocity predictor.",
    ),
    ModelSpec(
        canonical_name="multiscale_ts_jepa",
        aliases=("multiscale_jepa",),
        is_jepa=True,
        requires_patch_division=False,
        description="Multi-horizon hierarchical TS-JEPA with parallel prediction heads.",
    ),
    ModelSpec(
        canonical_name="causal_ssm_flow_jepa",
        aliases=("causal_flow_jepa", "causal_jepa"),
        is_jepa=True,
        requires_patch_division=False,
        description="Causal State-Space Flow-Matching JEPA with Relational GAT and Selective SSM.",
    ),
    ModelSpec(
        canonical_name="potential_flow_jepa",
        aliases=("pf_jepa", "ts_jepa_potential_flow"),
        is_jepa=True,
        requires_patch_division=False,
        description="Potential-Flow JEPA with Conservative Potential Field, Grassmannian Regimes, and Laplacian Curvature.",
    ),
    ModelSpec(
        canonical_name="harmonic_spring_jepa",
        aliases=("spring_jepa", "ts_jepa_spring", "harmonic_spring"),
        is_jepa=True,
        requires_patch_division=False,
        description="Harmonic Spring JEPA with closed-form Riemannian stiffness metric and exact O(1) Laplacian curvature.",
    ),
    ModelSpec(
        canonical_name="cosine_jepa",
        aliases=("cos_jepa", "ts_jepa_cosine", "hyperspherical_jepa"),
        is_jepa=True,
        requires_patch_division=False,
        description="Cosine JEPA with hyperspherical unit-vector projection and angular discrepancy in [0, 2].",
    ),
    ModelSpec(
        canonical_name="cycle_jepa",
        aliases=("bidirectional_jepa", "ts_jepa_cycle"),
        is_jepa=True,
        requires_patch_division=False,
        description="Cycle JEPA with bidirectional forward-backward causal round-trip consistency.",
    ),
    ModelSpec(
        canonical_name="cosine_cycle_jepa",
        aliases=("cos_cycle_jepa", "ts_jepa_cosine_cycle"),
        is_jepa=True,
        requires_patch_division=False,
        description="Cosine-Cycle JEPA combining hyperspherical angle-invariance with bidirectional causal cycle consistency.",
    ),
    ModelSpec(
        canonical_name="koopman_jepa",
        aliases=("koopman", "koopman_spectral_jepa", "ts_jepa_koopman"),
        is_jepa=True,
        requires_patch_division=False,
        description="Koopman-Spectral JEPA with exact linear latent operator, eigenvalue stability spectrum, and Lyapunov drift scoring.",
    ),
    ModelSpec(
        canonical_name="recurrent_koopman_jepa",
        aliases=("recurrent_koopman", "ssm_koopman_jepa", "ssm_koopman", "ts_jepa_ssm_koopman"),
        is_jepa=True,
        requires_patch_division=False,
        description="Recurrent State-Space Koopman JEPA with block-diagonal harmonic 2x2 rotation-dissipation blocks and autonomous trajectory rollout.",
    ),
    ModelSpec(
        canonical_name="tangent_normal_jepa",
        aliases=("tangent_jepa", "normal_jepa", "subspace_jepa", "orthogonal_jepa"),
        is_jepa=True,
        requires_patch_division=False,
        description="Tangent-Normal Subspace Decomposition JEPA with orthogonal projection decoupling point vs contextual anomalies.",
    ),
    ModelSpec(
        canonical_name="prototype_graph_jepa",
        aliases=("prototype_jepa", "anchor_jepa", "graph_jepa", "markov_jepa"),
        is_jepa=True,
        requires_patch_division=False,
        description="Prototype-Graph Transition JEPA with learnable operating regime anchors and Markov transition surprise scoring.",
    ),
    ModelSpec(
        canonical_name="transfer_function_jepa",
        aliases=("transfer_jepa", "spectral_jepa", "bode_jepa", "frequency_jepa"),
        is_jepa=True,
        requires_patch_division=False,
        description="Transfer-Function Spectral Coherence JEPA decoupling magnitude gain spikes from phase desynchronization.",
    ),
    ModelSpec(
        canonical_name="mdl_jepa",
        aliases=("compression_jepa", "entropy_jepa", "kolmogorov_jepa", "bitrate_jepa"),
        is_jepa=True,
        requires_patch_division=False,
        description="Minimum Description Length (MDL) Neural Entropy Compressor JEPA scoring exact description length in bits.",
    ),
    ModelSpec(
        canonical_name="granger_jepa",
        aliases=("causal_jepa", "coupling_jepa", "granger_causal_jepa"),
        is_jepa=True,
        requires_patch_division=False,
        description="Granger Causal Cross-Channel JEPA detecting broken inter-sensor physical couplings via causal attention.",
    ),
    ModelSpec(
        canonical_name="hamiltonian_jepa",
        aliases=("symplectic_jepa", "energy_jepa", "hamiltonian_symplectic_jepa"),
        is_jepa=True,
        requires_patch_division=False,
        description="Hamiltonian Symplectic JEPA preserving phase space volume and detecting energy conservation violations.",
    ),
    ModelSpec(
        canonical_name="latent_world_jepa",
        aliases=("world_jepa", "latent_recurrent_jepa", "latent_dynamics_jepa", "ts_latent_world_jepa"),
        is_jepa=True,
        requires_patch_division=False,
        description="Latent-Space World Model JEPA placing 100% of temporal recurrence and autonomous trajectory rollouts inside latent space.",
    ),
    ModelSpec(
        canonical_name="multiscale_latent_world_jepa",
        aliases=("multiscale_world_jepa", "dilated_latent_jepa", "pyramid_latent_jepa"),
        is_jepa=True,
        requires_patch_division=False,
        description="Multi-Scale Dilated Latent World Model JEPA with 3 parallel dilated temporal tracks (d=1, 4, 16).",
    ),
    ModelSpec(
        canonical_name="selective_latent_world_jepa",
        aliases=("selective_world_jepa", "mamba_latent_jepa", "gated_latent_jepa"),
        is_jepa=True,
        requires_patch_division=False,
        description="Selective Gated Memory Latent World Model JEPA with input-dependent timescale gating and memory retention.",
    ),
    ModelSpec(
        canonical_name="memory_bank_latent_world_jepa",
        aliases=("memory_bank_jepa", "cross_attention_latent_jepa", "episodic_latent_jepa"),
        is_jepa=True,
        requires_patch_division=False,
        description="Episodic Memory Bank Cross-Attention Latent World Model JEPA with lossless keyframe retrieval.",
    ),
    ModelSpec(
        canonical_name="dual_timescale_latent_world_jepa",
        aliases=("dual_timescale_jepa", "cerebellar_latent_jepa", "fast_slow_latent_jepa"),
        is_jepa=True,
        requires_patch_division=False,
        description="Dual-Timescale Cerebellar-Cortical Latent World Model JEPA separating fast reflexes from slow mode tracking.",
    ),
    ModelSpec(
        canonical_name="kinematic_adaptive_latent_jepa",
        aliases=("kinematic_latent_jepa", "ahead_jepa", "phase_space_latent_jepa"),
        is_jepa=True,
        requires_patch_division=False,
        description="Kinematic-Adaptive Latent World Model JEPA (CMU AHEAD 2026) with phase-space state and uncertainty dispersion.",
    ),
    ModelSpec(
        canonical_name="latent_deliberation_jepa",
        aliases=("deliberation_jepa", "monet_jepa", "dmlr_jepa", "thinking_latent_jepa"),
        is_jepa=True,
        requires_patch_division=False,
        description="Latent Deliberation World Model JEPA (DMLR & Monet CVPR 2026) with test-time thinking tokens and cognitive turbulence scoring.",
    ),
    ModelSpec(
        canonical_name="spectral_physics_latent_jepa",
        aliases=("spectral_latent_jepa", "physics_latent_jepa", "psd_latent_jepa"),
        is_jepa=True,
        requires_patch_division=False,
        description="Spectral-Preserving Latent Physics World Model JEPA (Polymathic AI NeurIPS 2025) with dual-stream dynamics and latent PSD invariance.",
    ),
    ModelSpec(
        canonical_name="multiagent_consensus_latent_jepa",
        aliases=("multiagent_latent_jepa", "interlat_jepa", "consensus_latent_jepa"),
        is_jepa=True,
        requires_patch_division=False,
        description="Multi-Agent Consensus Latent World Model JEPA (Interlat ACL 2026) with continuous inter-sensor communication and consensus dissonance scoring.",
    ),
    ModelSpec(
        canonical_name="tangent_harmonic_jepa",
        aliases=("harmonic_subspace_jepa", "tangent_jepa_v2", "tangent_harmonic"),
        is_jepa=True,
        requires_patch_division=False,
        description="Tangent-Harmonic Subspace Projection JEPA (Chapter 2 of Ten Advances) with moving orthogonal projection and Gegenbauer harmonic discrepancy.",
    ),
    ModelSpec(
        canonical_name="reynolds_stress_jepa",
        aliases=("reynolds_jepa", "navier_stokes_jepa", "ns_jepa"),
        is_jepa=True,
        requires_patch_division=False,
        description="Reynolds-Stress Navier-Stokes JEPA with Admissible Cone Constraint S_+.",
    ),
    ModelSpec(
        canonical_name="operator_entropy_jepa",
        aliases=("operator_jepa", "connes_jepa", "vn_entropy_jepa"),
        is_jepa=True,
        requires_patch_division=False,
        description="Von Neumann Operator Entropy & Spectral Rigidity JEPA (Connes Rigidity & Quantum Repetition).",
    ),
)












def _build_alias_index() -> Dict[str, ModelSpec]:
    index: Dict[str, ModelSpec] = {}
    for spec in _MODEL_SPECS:
        index[spec.canonical_name] = spec
        for alias in spec.aliases:
            index[alias] = spec
    return index


_ALIAS_INDEX: Dict[str, ModelSpec] = _build_alias_index()


def canonical_model_type(model_type: str) -> str:
    """Return the canonical name for a model type or alias.

    Raises ``ValueError`` for unknown model types.
    """
    spec = _ALIAS_INDEX.get(model_type)
    if spec is None:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. Valid: {sorted(valid_model_types())}"
        )
    return spec.canonical_name


def valid_model_types() -> FrozenSet[str]:
    """Return the set of all accepted model type strings (canonical + aliases)."""
    return frozenset(_ALIAS_INDEX.keys())


def canonical_model_choices() -> Tuple[str, ...]:
    """Return canonical model names in registry order, for CLI ``choices``."""
    return tuple(spec.canonical_name for spec in _MODEL_SPECS)


def is_jepa_model(model_type: str) -> bool:
    """Return whether the resolved model type is a JEPA variant."""
    spec = _ALIAS_INDEX.get(model_type)
    if spec is None:
        return False
    return spec.is_jepa


def requires_patch_division(model_type: str) -> bool:
    """Return whether the resolved model type requires patch-size divisibility."""
    spec = _ALIAS_INDEX.get(model_type)
    if spec is None:
        raise ValueError(f"Unknown model_type: {model_type!r}")
    return spec.requires_patch_division


def model_spec(model_type: str) -> Optional[ModelSpec]:
    """Return the spec for a model type or alias, or ``None`` if unknown."""
    return _ALIAS_INDEX.get(model_type)
