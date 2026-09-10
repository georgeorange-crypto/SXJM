"""Self-contained local simulator — the physics/timing oracle ported verbatim.

Timing fidelity is the whole point of this file: each component is rounded to
microseconds *separately* and then summed (``_us(move) + _us(switch) + _us(act)``),
exactly as the official engine does, so accumulated task time matches to the
microsecond. The reference vectors 105 / 111 / 194 / 199 (see
``tests/test_timing.py``) pin this down.

Rules implemented (Problem B, appendix 2 §2/§4):
- move time = straight-line distance / 5 m/s from the last legal-action pose;
- /measure total = move + channel-switch (1 s iff channel changes) + 5 s detect;
  afterwards the current channel becomes the measured channel;
- /clear total = move + (5 s on a hit / 3 s on a miss); channel unchanged;
- a bearing is returned only when distance <= R_eff AND the point is inside the
  source's coverage arc; <= 5 m in coverage returns "near" (too strong, no bearing);
- a clear hits iff distance to the (uncleared) source <= 20 m (heading-independent);
- neither /enter nor /exit advances the virtual clock.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional

from ..core.constants import CONSTANTS
from ..core.datatypes import (
    Action,
    ActionType,
    DetectionObservation,
    ObservationType,
)
from .base import RadioEnv
from .error_field import make_error_field
from .generators import (
    DIRECTIONAL_HALF_ANGLE_DEG,
    Case,
    Jammer,
    ang_diff,
    generate_case,
    norm_deg,
)

# Timing constants sourced from the single frozen table (they match the oracle).
_SPEED = CONSTANTS.robot_speed
_SWITCH_S = CONSTANTS.channel_switch_time
_MEASURE_S = CONSTANTS.detection_time
_CLEAR_HIT_S = CONSTANTS.clear_hit_time
_CLEAR_MISS_S = CONSTANTS.clear_miss_time
_NEAR_M = CONSTANTS.too_strong_radius
_CLEAR_R = CONSTANTS.clear_radius


def _us(seconds: float) -> int:
    """Seconds -> microseconds (round-half-to-even via round()), for exact
    integer accumulation. Must match the oracle bit-for-bit."""
    return int(round(seconds * 1_000_000))


class LocalEnv(RadioEnv):
    """A local, network-free environment that is behaviourally indistinguishable
    from the official one at the observation/timing level."""

    def __init__(
        self,
        problem: int = 3,
        *,
        error_field_kind: str = "smooth",
        error_field_params: Optional[dict] = None,
        generator: Optional[dict] = None,
        mode: str = "practice",
        max_virtual_duration_s: float = CONSTANTS.max_virtual_duration_s,
        max_real_duration_s: float = CONSTANTS.program_time_limit_s,
        real_clock: Callable[[], float] = time.monotonic,
        case: Optional[Case] = None,
    ) -> None:
        self.problem = int(problem)
        self.error_field_kind = error_field_kind
        self.error_field_params = dict(error_field_params or {})
        self.generator_cfg = dict(generator or {})
        self.mode = mode
        self.max_virtual_duration_s = float(max_virtual_duration_s)
        self.max_real_duration_s = float(max_real_duration_s)
        self._real_clock = real_clock
        self._fixed_case = case      # if given, reset reuses it (reproducible)

        self.case: Optional[Case] = None
        self._pos_x = CONSTANTS.initial_x
        self._pos_y = CONSTANTS.initial_y
        self._channel = CONSTANTS.initial_channel
        self._vt_us = 0
        self._finished = False
        self._finish_reason: Optional[str] = None
        self._enter_ts: Optional[float] = None

    # -- lifecycle ----------------------------------------------------------
    def reset(self, seed: int | None = None) -> DetectionObservation:
        if self._fixed_case is not None:
            self.case = self._fixed_case
        else:
            self.case = generate_case(
                seed=seed,
                problem=self.problem,
                n_jammers=self.generator_cfg.get("n_jammers"),
                n_directional=self.generator_cfg.get("n_directional"),
                mode=self.mode,
                margin_m=self.generator_cfg.get("margin_m", 30.0),
                field_kind=self.error_field_kind,
                field_params=self.error_field_params or None,
            )
        # Keep the caps consistent with what enter() would advertise.
        self.case.max_virtual_duration_s = self.max_virtual_duration_s
        self.case.max_real_duration_s = int(self.max_real_duration_s)

        self._pos_x = CONSTANTS.initial_x
        self._pos_y = CONSTANTS.initial_y
        self._channel = CONSTANTS.initial_channel
        self._vt_us = 0
        self._finished = False
        self._finish_reason = None
        self._enter_ts = self._real_clock()      # /enter does not advance vt
        return self._reset_observation()

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def finish_reason(self) -> Optional[str]:
        return self._finish_reason

    @property
    def virtual_time_s(self) -> float:
        return self._vt_us / 1_000_000.0

    def remaining_real_duration_s(self) -> float:
        if self._enter_ts is None:
            return self.max_real_duration_s
        elapsed = self._real_clock() - self._enter_ts
        return max(0.0, self.max_real_duration_s - elapsed)

    def reveal(self) -> dict:
        """Ground truth for practice-mode evaluation only."""
        assert self.case is not None
        return self.case.reveal()

    # -- deadlines ----------------------------------------------------------
    def _deadline_reason(self) -> Optional[str]:
        if self.virtual_time_s >= self.max_virtual_duration_s:
            return "timeout_virtual"
        if self._enter_ts is not None:
            elapsed = self._real_clock() - self._enter_ts
            if elapsed >= self.max_real_duration_s:
                return "timeout_real"
        return None

    # -- physics helpers (ported verbatim) ----------------------------------
    @staticmethod
    def _distance(x: float, y: float, j: Jammer) -> float:
        return math.hypot(x - j.x, y - j.y)

    @staticmethod
    def _in_coverage(x: float, y: float, j: Jammer) -> bool:
        if j.kind == "omni":
            return True
        bearing_src_to_pt = norm_deg(math.degrees(math.atan2(y - j.y, x - j.x)))
        return ang_diff(bearing_src_to_pt, j.direction_deg) <= DIRECTIONAL_HALF_ANGLE_DEG + 1e-9

    @staticmethod
    def _true_bearing_pt_to_src(x: float, y: float, j: Jammer) -> float:
        return norm_deg(math.degrees(math.atan2(j.y - y, j.x - x)))

    # -- action dispatch ----------------------------------------------------
    def execute(self, action: Action) -> DetectionObservation:
        assert self.case is not None, "call reset() before execute()"
        if self._finished:
            return self._noop_observation(action)

        if action.action_type == ActionType.EXIT:
            self._finish("user_exit")
            return DetectionObservation(
                result_type=ObservationType.NO_SIGNAL,
                channel=self._channel,
                position_x=self._pos_x,
                position_y=self._pos_y,
                virtual_time_delta=0.0,
            )

        reason = self._deadline_reason()
        if reason is not None:
            self._finish(reason)
            return self._noop_observation(action)

        if action.action_type == ActionType.SCAN:
            return self._measure(action.target_x, action.target_y, action.channel)
        if action.action_type == ActionType.CLEAR:
            return self._clear(action.target_x, action.target_y, action.channel)
        raise ValueError(f"unknown action type {action.action_type!r}")

    def _measure(self, x: float, y: float, channel: int) -> DetectionObservation:
        move_s = math.hypot(x - self._pos_x, y - self._pos_y) / _SPEED
        switch_s = _SWITCH_S if channel != self._channel else 0.0
        delta_us = _us(move_s) + _us(switch_s) + _us(_MEASURE_S)
        self._vt_us += delta_us
        self._pos_x, self._pos_y = x, y
        self._channel = channel

        result_type, bearing = self._eval_measure(x, y, channel)
        return DetectionObservation(
            result_type=result_type,
            channel=channel,
            position_x=x,
            position_y=y,
            virtual_time_delta=delta_us / 1_000_000.0,
            bearing_deg=bearing,
        )

    def _eval_measure(
        self, x: float, y: float, channel: int
    ) -> tuple[ObservationType, Optional[float]]:
        assert self.case is not None
        j = self.case.jammer_on_channel(channel)
        if j is None or j.cleared:
            return ObservationType.NO_SIGNAL, None

        dist = self._distance(x, y, j)
        in_cov = self._in_coverage(x, y, j)

        if in_cov and dist <= _NEAR_M:
            return ObservationType.TOO_STRONG, None

        if dist <= j.r_eff and in_cov:
            true_bearing = self._true_bearing_pt_to_src(x, y, j)
            err = self.case.field.error_deg(x, y)   # deterministic, [-1, 1] deg
            svd = round(norm_deg(true_bearing + err), 2)
            svd = norm_deg(svd)
            return ObservationType.SIGNAL, svd

        return ObservationType.NO_SIGNAL, None

    def _clear(self, x: float, y: float, channel: int) -> DetectionObservation:
        assert self.case is not None
        move_s = math.hypot(x - self._pos_x, y - self._pos_y) / _SPEED
        j = self.case.jammer_on_channel(channel)
        hit = (j is not None) and (not j.cleared) and (self._distance(x, y, j) <= _CLEAR_R)

        action_s = _CLEAR_HIT_S if hit else _CLEAR_MISS_S
        delta_us = _us(move_s) + _us(action_s)
        self._vt_us += delta_us
        self._pos_x, self._pos_y = x, y     # channel unchanged on clear

        if hit:
            j.cleared = True

        return DetectionObservation(
            result_type=ObservationType.CLEAR_SUCCESS if hit else ObservationType.CLEAR_FAILURE,
            channel=channel,
            position_x=x,
            position_y=y,
            virtual_time_delta=delta_us / 1_000_000.0,
            clear_success=bool(hit),
        )

    # -- misc ---------------------------------------------------------------
    def _finish(self, reason: str) -> None:
        self._finished = True
        self._finish_reason = reason

    def _noop_observation(self, action: Action) -> DetectionObservation:
        return DetectionObservation(
            result_type=ObservationType.NO_SIGNAL,
            channel=action.channel,
            position_x=self._pos_x,
            position_y=self._pos_y,
            virtual_time_delta=0.0,
        )
