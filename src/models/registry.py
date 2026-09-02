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
        raise ValueError(f"Unknown model_type: {model_type!r}")
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
