"""Analytical cost model + robot state (DESIGN.md §1.1, §7, §10).

The single source of timing truth for the planner. It reproduces the environment's
arithmetic *exactly* (verified against the §1.1 golden sample and cross-checked
against the authoritative engine): each component is rounded to whole microseconds
independently and then summed, so the accumulated virtual clock never drifts.

The two rules the golden sample exists to pin down (DESIGN.md §1.1):
  * ``/measure`` pays a 1 s switch **only** when the channel changes, and it *sets*
    the current measuring channel.
  * ``/clear`` names the source channel but does **not** switch and does **not**
    change the measuring channel — so a ``/measure`` right after a ``/clear`` on the
    still-current channel pays zero switch (step 5: switch = 0).

This is a *predictor*, not the simulator (禁止1): the environment remains
authoritative for the realised clock; the executor only uses these numbers to plan
and to pre-check the time budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Tuple

from sxjm_core.geometry import Point


def _us(seconds: float) -> int:
    """Whole microseconds, matching the engine's per-component rounding."""
    return int(round(seconds * 1_000_000))


@dataclass(frozen=True)
class RobotState:
    """Robot pose + measuring channel + accumulated virtual time.

    ``vt_us`` is integer microseconds (authoritative, drift-free); ``channel`` is
    the current measuring channel (``/measure`` sets it, ``/clear`` leaves it)."""

    x: float = 0.0
    y: float = 0.0
    channel: int = 1
    vt_us: int = 0

    @property
    def pos(self) -> Point:
        return (self.x, self.y)

    @property
    def virtual_time_s(self) -> float:
        return self.vt_us / 1_000_000.0

    def at_time_s(self, seconds: float) -> "RobotState":
        """Same pose/channel, clock set from an authoritative env ``virtual_time_s``."""
        return replace(self, vt_us=_us(seconds))


class AnalyticalCostModel:
    """Closed-form timing consistent with the engine (DESIGN.md §1.1)."""

    def __init__(
        self,
        speed: float = 5.0,
        switch_s: float = 1.0,
        measure_s: float = 5.0,
        clear_hit_s: float = 5.0,
        clear_miss_s: float = 3.0,
    ) -> None:
        self.speed = float(speed)
        self.switch_s = float(switch_s)
        self.measure_s = float(measure_s)
        self.clear_hit_s = float(clear_hit_s)
        self.clear_miss_s = float(clear_miss_s)

    # -- component costs (microseconds) ------------------------------------

    def move_us(self, a: Point, b: Point) -> int:
        return _us(math.hypot(b[0] - a[0], b[1] - a[1]) / self.speed)

    def measure_us(self, state: RobotState, target: Point, channel: int) -> int:
        switch = self.switch_s if int(channel) != int(state.channel) else 0.0
        return self.move_us(state.pos, target) + _us(switch) + _us(self.measure_s)

    def clear_us(self, state: RobotState, target: Point, hit: bool) -> int:
        act = self.clear_hit_s if hit else self.clear_miss_s
        return self.move_us(state.pos, target) + _us(act)

    # -- seconds convenience (planner-facing) ------------------------------

    def measure_time_s(self, state: RobotState, target: Point, channel: int) -> float:
        return self.measure_us(state, target, channel) / 1_000_000.0

    def clear_time_s(self, state: RobotState, target: Point, hit: bool) -> float:
        return self.clear_us(state, target, hit) / 1_000_000.0

    # -- state transitions (mirror the engine exactly) --------------------

    def apply_measure(self, state: RobotState, target: Point, channel: int) -> Tuple[RobotState, float]:
        """Predict the state after ``/measure channel @ target``: pose moves, the
        measuring channel becomes ``channel``, the clock advances."""
        cost = self.measure_us(state, target, channel)
        nxt = RobotState(float(target[0]), float(target[1]), int(channel), state.vt_us + cost)
        return nxt, cost / 1_000_000.0

    def apply_clear(self, state: RobotState, target: Point, hit: bool) -> Tuple[RobotState, float]:
        """Predict the state after ``/clear @ target``: pose moves, the measuring
        channel is UNCHANGED (§1.1 crux), the clock advances by hit/miss."""
        cost = self.clear_us(state, target, hit)
        nxt = RobotState(float(target[0]), float(target[1]), state.channel, state.vt_us + cost)
        return nxt, cost / 1_000_000.0

    # -- batch scan (§7): several /measure at one waypoint ----------------

    def batch_scan_us(self, state: RobotState, target: Point, channels) -> int:
        """Total cost of measuring ``channels`` in order at one waypoint. Only the
        first measure pays the move; each subsequent one pays switch (if the channel
        changed) + detect. Accumulated in microseconds to avoid float re-rounding."""
        total = 0
        s = state
        for c in channels:
            u = self.measure_us(s, target, c)
            total += u
            s = RobotState(float(target[0]), float(target[1]), int(c), s.vt_us + u)
        return total

    def batch_scan_time_s(self, state: RobotState, target: Point, channels) -> float:
        return self.batch_scan_us(state, target, channels) / 1_000_000.0
