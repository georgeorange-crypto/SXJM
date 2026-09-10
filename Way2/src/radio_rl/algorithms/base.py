"""Agent interface.

An agent maps the current step to a *choice among candidates* — an index into
the :class:`CandidateSet`. It never fabricates coordinates (principle C). The
math baseline reads the belief and the candidates' analytical annotations; the
learnable agents (PPO, later Dreamer) read the tensor FeatureBundle instead, but
they still only emit an index. Keeping the return type an index is what lets us
swap a hand-written policy for a neural one by changing a single config line.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..candidates.generator import CandidateSet
from ..geometry.belief import BeliefState


class Agent(ABC):
    """Selects one candidate per step."""

    #: whether this agent needs the tensor FeatureBundle (RL) or not (math).
    needs_features: bool = False

    def reset(self) -> None:
        """Reset any per-episode internal state (memory, RNG)."""

    @abstractmethod
    def select(self, candidates: CandidateSet, belief: BeliefState,
               features: Optional[object] = None) -> int:
        """Return the index of the chosen candidate in ``candidates.candidates``."""
        raise NotImplementedError
