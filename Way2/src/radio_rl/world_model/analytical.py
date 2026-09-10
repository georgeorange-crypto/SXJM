"""Analytical world model — deterministic physics predictions.

No learning: it reproduces the environment's own timing arithmetic (microsecond
rounding per component, then summed) and reads the mathematical belief to predict
information gain and clear success. This is the default world model and the
reference the learned :class:`ResidualWorldModel` will correct against.
"""

from __future__ import annotations

import math

from ..core.constants import CONSTANTS
from ..core.datatypes import ActionType, CandidateAction, CandidateSource
from ..core.registry import WORLD_MODELS
from ..geometry.belief import BeliefState
from ..geometry.information_gain import expected_region_shrink
from .base import WorldModel

_SPEED = CONSTANTS.robot_speed
_SWITCH_S = CONSTANTS.channel_switch_time
_MEASURE_S = CONSTANTS.detection_time
_CLEAR_HIT_S = CONSTANTS.clear_hit_time
_CLEAR_MISS_S = CONSTANTS.clear_miss_time
_CLEAR_R = CONSTANTS.clear_radius
_NEAR_R = CONSTANTS.too_strong_radius
_RANGE_MIN = CONSTANTS.receive_radius_min


def _us(seconds: float) -> int:
    return int(round(seconds * 1_000_000))


@WORLD_MODELS.register("analytical")
class AnalyticalWorldModel(WorldModel):
    """Closed-form predictions consistent with :class:`LocalEnv`."""

    def __init__(self, bearing_delta_deg: float = CONSTANTS.bearing_error_deg,
                 search_value: float = 1500.0) -> None:
        self.bearing_delta_deg = float(bearing_delta_deg)
        self.search_value = float(search_value)

    # -- timing (matches the env's per-component microsecond rounding) -----
    def predict_time(self, belief: BeliefState, action_type: int, channel: int,
                     tx: float, ty: float, clear_prob: float = 0.0) -> float:
        move_s = math.hypot(tx - belief.pose_x, ty - belief.pose_y) / _SPEED
        total_us = _us(move_s)
        if action_type == int(ActionType.SCAN):
            if int(channel) != int(belief.current_channel):
                total_us += _us(_SWITCH_S)
            total_us += _us(_MEASURE_S)
        elif action_type == int(ActionType.CLEAR):
            # Expected clear duration blends hit/miss by predicted probability.
            act_s = clear_prob * _CLEAR_HIT_S + (1.0 - clear_prob) * _CLEAR_MISS_S
            total_us += _us(act_s)
        # EXIT: only the move (usually zero).
        return total_us / 1_000_000.0

    # -- clear success ------------------------------------------------------
    def predict_clear_probability(self, belief: BeliefState, channel: int,
                                  tx: float, ty: float) -> float:
        cb = belief.channels.get(int(channel))
        if cb is None or cb.is_cleared or cb.estimate is None:
            return 0.0
        ex, ey = cb.estimate
        dist_to_est = math.hypot(tx - ex, ty - ey)
        # worst-case reachable slack: clear succeeds for sure when the whole MEC
        # lies within the clear radius of the target.
        reach = _CLEAR_R - dist_to_est
        r = max(cb.mec_radius, 1e-6)
        if reach >= r:
            return 1.0
        if reach <= -r:
            return 0.0
        return max(0.0, min(1.0, (reach + r) / (2.0 * r)))

    # -- information gain ---------------------------------------------------
    def predict_region_reduction(self, belief: BeliefState, channel: int,
                                 tx: float, ty: float) -> float:
        cb = belief.channels.get(int(channel))
        if cb is None or cb.region.is_empty():
            return 0.0
        return expected_region_shrink(cb.region, tx, ty, self.bearing_delta_deg)

    def _search_value(self, belief: BeliefState, channel: int,
                      tx: float, ty: float) -> float:
        """Novelty of probing (tx, ty) for an undetected channel: 0 if already
        excluded, otherwise scaled by distance to the nearest prior exclusion."""
        if belief.excluded(tx, ty, channel=int(channel)):
            return 0.0
        nearest = float("inf")
        for e in belief.exclusions:
            if e.channel != int(channel):
                continue
            nearest = min(nearest, math.hypot(tx - e.x, ty - e.y))
        if nearest == float("inf"):
            return self.search_value
        return self.search_value * min(1.0, nearest / _RANGE_MIN)

    # -- annotate a candidate in place -------------------------------------
    def annotate(self, belief: BeliefState, cand: CandidateAction) -> CandidateAction:
        at = cand.action_type
        if at == int(ActionType.CLEAR):
            p = self.predict_clear_probability(belief, cand.channel,
                                               cand.target_x, cand.target_y)
            cand.predicted_clear_probability = p
            cand.expected_time = self.predict_time(belief, at, cand.channel,
                                                   cand.target_x, cand.target_y, p)
            cand.predicted_region_reduction = 0.0
            # A confident clear is the most valuable move there is.
            cand.heuristic_information_gain = 10_000.0 * p
        elif at == int(ActionType.SCAN):
            cand.expected_time = self.predict_time(belief, at, cand.channel,
                                                   cand.target_x, cand.target_y)
            if cand.source == int(CandidateSource.SEARCH):
                cand.predicted_region_reduction = 0.0
                cand.heuristic_information_gain = self._search_value(
                    belief, cand.channel, cand.target_x, cand.target_y)
            else:
                red = self.predict_region_reduction(belief, cand.channel,
                                                    cand.target_x, cand.target_y)
                cand.predicted_region_reduction = red
                cand.heuristic_information_gain = red
        else:  # EXIT / safety
            cand.expected_time = self.predict_time(belief, at, cand.channel,
                                                   cand.target_x, cand.target_y)
            cand.predicted_region_reduction = 0.0
            cand.heuristic_information_gain = 0.0
        return cand
