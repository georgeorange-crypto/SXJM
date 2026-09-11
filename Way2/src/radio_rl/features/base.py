"""FeatureBuilder abstract interface.

Kept torch-free at import so that importing the interface never pulls the tensor
stack. Concrete builders (which do import torch) live in sibling modules and
register themselves in :data:`radio_rl.core.registry.FEATURE_BUILDERS`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..candidates.generator import CandidateSet
from ..geometry.belief import BeliefState


class FeatureBuilder(ABC):
    """Turns (belief, candidates) into a ``FeatureBundle`` of tensors.

    The return type is intentionally annotated loosely (``object``) here so this
    module needn't import the torch-backed :class:`FeatureBundle`; concrete
    builders return a real bundle.
    """

    @abstractmethod
    def build(self, belief: BeliefState, candidates: CandidateSet) -> object:
        """Read the belief and candidate set; return a ``FeatureBundle``.

        Must not mutate ``belief`` and must not add/drop candidates — it only
        tensorizes what the generator produced (the iron rule, builder side).
        """
        ...
