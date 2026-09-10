"""Safety shield — the final, non-learnable gate before the environment.

It converts the agent's chosen :class:`CandidateAction` into a concrete, legal
:class:`Action`. Guarantees enforced here (never delegated to the policy):

* the target lies within the arena disc (clamped onto the boundary, or the
  candidate rejected in favour of a safe fallback);
* coordinates stay within the engine's absolute bound;
* the channel is a valid 1..20 integer;
* when the real-time budget is nearly exhausted, the action is overridden with
  EXIT so we always leave cleanly.

Because it sits outside the agent, a mis-trained or adversarial policy can never
drive the robot out of bounds or past the deadline.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from ..core.constants import CONSTANTS
from ..core.datatypes import Action, ActionType, CandidateAction


def _get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    try:
        v = cfg.get(key) if hasattr(cfg, "get") else cfg[key]
    except Exception:
        return default
    return default if v is None else v


class SafetyShield:
    """Validates and, if needed, repairs the agent's chosen action."""

    def __init__(self, cfg: Any = None) -> None:
        self.enforce_arena = bool(_get(cfg, "enforce_arena", True))
        self.arena_slack_m = float(_get(cfg, "arena_slack_m", 0.0))
        self.clamp_out_of_bounds = bool(_get(cfg, "clamp_out_of_bounds", True))
        self.emergency_margin_s = float(
            _get(cfg, "emergency_margin_s", CONSTANTS.emergency_margin_s)
        )
        self.arena_radius = CONSTANTS.region_radius
        self.coord_abs_max = CONSTANTS.coord_abs_max
        self.n_overrides = 0
        self.n_clamps = 0

    # -- geometry ----------------------------------------------------------
    def _clamp_arena(self, x: float, y: float) -> tuple[float, float]:
        limit = self.arena_radius + self.arena_slack_m
        r = math.hypot(x, y)
        if r <= limit or r < 1e-9:
            return x, y
        self.n_clamps += 1
        s = limit / r
        return x * s, y * s

    def _clamp_abs(self, v: float) -> float:
        m = self.coord_abs_max
        return max(-m, min(m, v))

    def is_legal(self, x: float, y: float) -> bool:
        return math.hypot(x, y) <= self.arena_radius + self.arena_slack_m + 1e-6

    # -- main entry --------------------------------------------------------
    def apply(self, cand: CandidateAction, *,
              remaining_real_s: Optional[float] = None) -> Action:
        """Return the safe :class:`Action` to execute for this candidate."""
        # Deadline guard: leave cleanly before the program clock runs out.
        if remaining_real_s is not None and remaining_real_s <= self.emergency_margin_s:
            self.n_overrides += 1
            return Action(ActionType.EXIT, cand.channel,
                          cand.target_x, cand.target_y, cand.candidate_id)

        action_type = ActionType(cand.action_type)
        if action_type == ActionType.EXIT:
            return cand.to_action()

        x, y = float(cand.target_x), float(cand.target_y)
        if self.enforce_arena:
            if self.clamp_out_of_bounds:
                x, y = self._clamp_arena(x, y)
            elif not self.is_legal(x, y):
                # reject: stay in place instead of leaving the arena
                self.n_overrides += 1
                x, y = self._clamp_arena(x, y)
        x, y = self._clamp_abs(x), self._clamp_abs(y)

        ch = int(cand.channel)
        ch = max(CONSTANTS.channel_min, min(CONSTANTS.channel_max, ch))

        return Action(action_type, ch, x, y, cand.candidate_id)
