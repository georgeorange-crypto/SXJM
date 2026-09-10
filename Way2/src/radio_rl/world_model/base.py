"""World-model interface.

A world model annotates a :class:`CandidateAction` with analytical predictions
(time cost, expected information gain, clear probability) that become features
for the agent and drive the heuristic. The default is a deterministic analytical
model; a learned :class:`ResidualWorldModel` and a Dreamer latent model plug in
here later without touching the pipeline. Torch-free at import time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.datatypes import CandidateAction
from ..geometry.belief import BeliefState


class WorldModel(ABC):
    """Predicts the consequences of a candidate action from the belief."""

    @abstractmethod
    def predict_time(self, belief: BeliefState, action_type: int, channel: int,
                     tx: float, ty: float, clear_prob: float = 0.0) -> float:
        ...

    @abstractmethod
    def predict_clear_probability(self, belief: BeliefState, channel: int,
                                  tx: float, ty: float) -> float:
        ...

    @abstractmethod
    def predict_region_reduction(self, belief: BeliefState, channel: int,
                                 tx: float, ty: float) -> float:
        ...

    def annotate(self, belief: BeliefState, cand: CandidateAction) -> CandidateAction:
        """Fill the predicted_* / heuristic_* fields of ``cand`` in place."""
        return cand
