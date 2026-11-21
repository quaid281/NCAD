"""Central registry of supported model types and their metadata.

This is the single source of truth for which model types are supported by the
CLI, configuration validation, and the model builder. Keeping the choices in
one place prevents drift between ``src/cli.py``, ``src/config.py``, and
``src/engine/builder.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Tuple


@dataclass(frozen=True)
class ModelSpec:
    """Metadata for a supported model variant."""

    canonical_name: str
    aliases: Tuple[str, ...]
    is_jepa: bool
    requires_patch_division: bool
    description: str
    category: str = "ablation"


_MODEL_SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        canonical_name="laminar_jepa_lite",
        aliases=("laminar_jepa", "laminar_lite", "reynolds_lite"),
        is_jepa=True,
        requires_patch_division=False,
        description=(
            "Primary streamlined 3-horizon spherical JEPA with depthwise K>=25 dyadic shells, "
            "4-block TCN, hook-free native tap/barycenter readout, and 128-dial P+D panel."
        ),
        category="primary",
    ),
    ModelSpec(
        canonical_name="laminar_jepa_224",
        aliases=("laminar_224",),
        is_jepa=True,
        requires_patch_division=False,
        description="Streamlined 4-horizon (224-dial) Laminar-JEPA diagnostic tier sharing the Lite encoder.",
        category="ablation",
    ),
    ModelSpec(
        canonical_name="ts_jepa",
        aliases=(),
        is_jepa=True,
        requires_patch_division=False,
        description="TS-JEPA with VICReg-style self-supervised predictive coding.",
        category="primary",
    ),
    ModelSpec(
        canonical_name="multiscale_ts_jepa",
        aliases=("multiscale_jepa",),
        is_jepa=True,
        requires_patch_division=False,
        description="Multi-horizon hierarchical TS-JEPA with parallel prediction heads.",
        category="ablation",
    ),
    ModelSpec(
        canonical_name="ncad",
        aliases=(),
        is_jepa=False,
        requires_patch_division=False,
        description="NCAD contrastive encoder baseline.",
        category="baseline",
    ),
    ModelSpec(
        canonical_name="anomaly_transformer",
        aliases=(),
        is_jepa=False,
        requires_patch_division=False,
        description="Anomaly Transformer (ICLR 2022) baseline.",
        category="baseline",
    ),
    ModelSpec(
        canonical_name="timesnet",
        aliases=(),
        is_jepa=False,
        requires_patch_division=False,
        description="TimesNet (ICLR 2023) baseline.",
        category="baseline",
    ),
    ModelSpec(
        canonical_name="dcdetector",
        aliases=(),
        is_jepa=False,
        requires_patch_division=False,
        description="DCdetector (KDD 2023) baseline.",
        category="baseline",
    ),
    ModelSpec(
        canonical_name="tranad",
        aliases=(),
        is_jepa=False,
        requires_patch_division=False,
        description="TranAD (VLDB 2022) baseline.",
        category="baseline",
    ),
    ModelSpec(
        canonical_name="soliton_reynolds_jepa",
        aliases=("soliton_reynolds", "soliton_jepa", "lieb_thirring_jepa"),
        is_jepa=True,
        requires_patch_division=False,
        description="224-dial Soliton-Reynolds JEPA (6-block HybridTCN + sequential predictor).",
        category="ablation",
    ),
    ModelSpec(
        canonical_name="delta_jepa",
        aliases=("delta", "fast_weight_delta_jepa"),
        is_jepa=True,
        requires_patch_division=False,
        description="Delta-JEPA ablation with fast-weight matrix updates.",
        category="ablation",
    ),
    ModelSpec(
        canonical_name="speculative_jepa",
        aliases=("speculative", "speculative_alien_jepa", "spec_jepa"),
        is_jepa=True,
        requires_patch_division=False,
        description="Speculative-JEPA ablation with multi-depth hierarchy probing.",
        category="ablation",
    ),
    ModelSpec(
        canonical_name="alien_dual_tower_jepa",
        aliases=("alien_dual_tower", "dual_tower_jepa"),
        is_jepa=True,
        requires_patch_division=False,
        description="Dual-Tower JEPA ablation with dynamical coherence tower.",
        category="ablation",
    ),
    ModelSpec(
        canonical_name="eddy_delta_jepa",
        aliases=("eddy_delta", "solenoidal_eddy_jepa", "smagorinsky_eddy_jepa"),
        is_jepa=True,
        requires_patch_division=False,
        description="Eddy-Delta JEPA ablation with solenoidal skew-symmetric vorticity.",
        category="ablation",
    ),
    ModelSpec(
        canonical_name="anchored_stokes_jepa",
        aliases=("anchored_stokes", "stokes_jepa", "halpern_stokes_jepa"),
        is_jepa=True,
        requires_patch_division=False,
        description="Anchored Stokes JEPA ablation with Halpern-anchored rollout.",
        category="ablation",
    ),
    ModelSpec(
        canonical_name="helmholtz_soliton_jepa",
        aliases=("helmholtz_soliton", "helmholtz_jepa", "soliton_eddy_jepa"),
        is_jepa=True,
        requires_patch_division=False,
        description="Helmholtz-Soliton JEPA ablation.",
        category="ablation",
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


def primary_model_choices() -> Tuple[str, ...]:
    """Return canonical model names designated as primary paper models."""
    return tuple(spec.canonical_name for spec in _MODEL_SPECS if spec.category == "primary")


def sota_baseline_choices() -> Tuple[str, ...]:
    """Return canonical model names designated as external SOTA baselines."""
    return tuple(spec.canonical_name for spec in _MODEL_SPECS if spec.category == "baseline")


def ablation_model_choices() -> Tuple[str, ...]:
    """Return canonical model names designated as secondary ablation variants."""
    return tuple(spec.canonical_name for spec in _MODEL_SPECS if spec.category == "ablation")
