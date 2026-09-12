"""Small independent tabular learner for exported Way4 SMDP transitions.

This module is intentionally separate from the residual planner.  It consumes
``OptionTransition`` records and learns option values with elapsed-time
discounting; it does not emit coordinates or bypass the safety stack.
"""

from dataclasses import dataclass
from math import exp
from typing import Dict, Iterable, Mapping, Optional


@dataclass(frozen=True)
class SMDPTrainConfig:
    time_scale_s: float = 1.0
    terminal_reward: float = 1.0
    full_clear_bonus: float = 1.0


class TabularSMDPLearner:
    """Average-return option learner suitable for smoke/offline checks."""

    def __init__(self, config: Optional[SMDPTrainConfig] = None) -> None:
        self.config = config or SMDPTrainConfig()
        if self.config.time_scale_s <= 0:
            raise ValueError("time_scale_s must be positive")
        self.values: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def _key(self, transition: Mapping[str, object]) -> str:
        option = transition.get("option_type")
        if not option:
            raise ValueError("transition missing option_type")
        return str(option)

    def target(self, transition: Mapping[str, object], next_value: float = 0.0) -> float:
        elapsed = float(transition.get("elapsed_time_s", 0.0))
        if elapsed < 0:
            raise ValueError("elapsed_time_s must be non-negative")
        reward = -elapsed
        if bool(transition.get("full_clear")):
            reward += self.config.full_clear_bonus
        if bool(transition.get("terminal")):
            reward += self.config.terminal_reward
            return reward
        discount = exp(-elapsed / self.config.time_scale_s)
        return reward + discount * float(next_value)

    def update(self, transition: Mapping[str, object], next_value: float = 0.0) -> float:
        key = self._key(transition)
        target = self.target(transition, next_value)
        n = self.counts.get(key, 0) + 1
        self.counts[key] = n
        self.values[key] = self.values.get(key, 0.0) + (target - self.values.get(key, 0.0)) / n
        return target

    def fit(self, transitions: Iterable[Mapping[str, object]]) -> Dict[str, float]:
        for transition in transitions:
            self.update(transition, self.values.get(self._key(transition), 0.0))
        return dict(self.values)

    def value(self, option_type: str, default: float = 0.0) -> float:
        return self.values.get(str(option_type), default)
