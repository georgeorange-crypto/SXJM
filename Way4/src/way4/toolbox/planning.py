"""Bounded online-planning and safe learning utilities."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

def beam_search(initial, expand: Callable, score: Callable, depth: int, width: int = 8):
    beam = [initial]
    for _ in range(max(0, int(depth))):
        children = [child for state in beam for child in expand(state)]
        beam = sorted(children, key=score)[:max(1, int(width))]
        if not beam: break
    return min(beam, key=score) if beam else initial

def mpc(initial, propose: Callable, execute: Callable, terminal: Callable,
        horizon: int = 5, max_steps: int = 100):
    """Receding-horizon loop: plan a horizon, execute one action, replan."""
    state = initial; actions = []
    for _ in range(max_steps):
        if terminal(state): break
        plan = list(propose(state, horizon))
        if not plan: break
        action = plan[0]; actions.append(action); state = execute(state, action)
    return state, actions

@dataclass
class HyperHeuristic:
    planners: dict
    selector: Callable[[object, Sequence[str]], str] | None = None
    def choose(self, state):
        names = list(self.planners)
        name = self.selector(state, names) if self.selector else names[0]
        return name, self.planners[name](state)

def shield(action, safe: Callable[[object], bool], fallback):
    """Shielded-RL primitive: unsafe learned actions are replaced safely."""
    return action if safe(action) else fallback

def bounded_residual(base_score: float, residual: float, delta: float) -> float:
    return float(base_score) + max(-abs(delta), min(abs(delta), float(residual)))
