"""Greedy mathematical baseline — the default algorithm.

Pure, torch-free, and deterministic. It argmaxes the world model's analytical
value of each candidate minus a small time penalty:

    value(CLEAR)              = 1e4 * P(success)        (localized => P ~ 1)
    value(LOCALIZE/FOLLOWUP)  = expected region shrink (metres)
    value(SEARCH)             = novelty of the probe point
    value(EXIT)               = 0

so it clears whenever a source is localized, otherwise triangulates the most
informative detected source, otherwise searches, and only EXITs once no probe
can add information (all remaining search points are already excluded). This is
the reference policy every learned agent must beat, and a safe fallback when a
plugin fails to load.
"""

from __future__ import annotations

from typing import Optional

from ..core.datatypes import ActionType
from ..core.registry import ALGORITHMS
from ..candidates.generator import CandidateSet
from ..geometry.belief import BeliefState
from .base import Agent


@ALGORITHMS.register("greedy_math")
class GreedyMathAgent(Agent):
    needs_features = False

    def __init__(self, time_penalty: float = 0.02, **_: object) -> None:
        self.time_penalty = float(time_penalty)

    def _value(self, cand) -> float:
        if cand.action_type == int(ActionType.EXIT):
            return 0.0
        return cand.heuristic_information_gain - self.time_penalty * cand.expected_time

    def select(self, candidates: CandidateSet, belief: BeliefState,
               features: Optional[object] = None) -> int:
        reals = candidates.real()
        if not reals:
            return 0
        exit_i: Optional[int] = None
        best_i, best_v = None, float("-inf")
        for i, c in enumerate(reals):
            if c.action_type == int(ActionType.EXIT):
                exit_i = i
                continue
            v = self._value(c)
            if v > best_v:
                best_v, best_i = v, i
        # Only act when some candidate adds real value; otherwise leave cleanly.
        if best_i is not None and best_v > 1e-6:
            return best_i
        if exit_i is not None:
            return exit_i
        return best_i if best_i is not None else 0
