"""Isolated novelty experiments for NCAD-CS.

These modules are research branches and alternative mechanisms (associative Hopfield
context memory, causal counterfactual substitution, memoryless context discrimination,
and selective SSM state encoding).
"""

from src.experimental.causal_counterfactual import (
    CounterfactualConfig,
    CounterfactualContextResult,
    CounterfactualContextSubstitutor,
)
from src.experimental.hopfield_context import (
    HopfieldContextConfig,
    HopfieldContextMemory,
    HopfieldRetrieval,
)
from src.experimental.memoryless_context import (
    MemorylessContextConfig,
    MemorylessContextResult,
    MemorylessContextSubstitutor,
)
from src.experimental.selective_ssm_encoder import (
    SelectiveSSMContextEncoder,
    SelectiveStateSpaceBlock,
)

__all__ = [
    "CounterfactualConfig",
    "CounterfactualContextResult",
    "CounterfactualContextSubstitutor",
    "HopfieldContextConfig",
    "HopfieldContextMemory",
    "HopfieldRetrieval",
    "MemorylessContextConfig",
    "MemorylessContextResult",
    "MemorylessContextSubstitutor",
    "SelectiveSSMContextEncoder",
    "SelectiveStateSpaceBlock",
]