"""Isolated novelty experiments for NCAD-CS v3.

These modules are intentionally outside the default training path. Importing this
package does not alter the production NCAD-CS implementation in ``train.py``.
"""

from experimental.causal_counterfactual import CounterfactualContextSubstitutor
from experimental.hopfield_context import HopfieldContextMemory
from experimental.memoryless_context import MemorylessContextSubstitutor
from experimental.selective_ssm_encoder import SelectiveSSMContextEncoder

__all__ = [
    "CounterfactualContextSubstitutor",
    "HopfieldContextMemory",
    "MemorylessContextSubstitutor",
    "SelectiveSSMContextEncoder",
]