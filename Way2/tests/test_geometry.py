"""Geometry / estimator layer.

Two things must hold: (1) soundness — the true source is never clipped out of a
channel's feasible region; (2) triangulation from spread-out scan points shrinks
the region until the channel localizes, and the MEC centre is then within clear
range of the true source.
"""

from __future__ import annotations

import math

import pytest

from radio_rl.core.datatypes import Action, ActionType, ObservationType
from radio_rl.env.error_field import SmoothErrorField
from radio_rl.env.generators import Case, Jammer
from radio_rl.env.local_env import LocalEnv
from radio_rl.geometry import BeliefState, EstimatorConfig, Region


def _point_in_region(region: Region, x: float, y: float, eps: float = 1e-3) -> bool:
    """True if (x, y) is inside the CCW convex polygon (with tolerance)."""
    poly = region.poly
    if len(poly) < 3:
        return False
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        # left-of test for CCW edge; allow small slack scaled by edge length
        crs = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
        edge = math.hypot(bx - ax, by - ay)
        if crs < -eps * max(1.0, edge):
            return False
    return True


def _scan(ch, x, y):
    return Action(ActionType.SCAN, ch, x, y)


def _feed(belief: BeliefState, env: LocalEnv, action: Action) -> None:
    belief.update(env.execute(action))


def test_single_bearing_contains_source():
    src = Jammer(4, 800.0, 600.0, 1500.0, "omni")
    env = LocalEnv(case=Case([src], SmoothErrorField(seed=5), seed=1))
    env.reset()
    belief = BeliefState(problem=3)
    for (x, y) in [(0, 0), (100, -100), (-200, 50)]:
        _feed(belief, env, _scan(4, x, y))
        cb = belief.belief(4)
        assert cb.is_detected
        assert not cb.region.is_empty()
        assert _point_in_region(cb.region, src.x, src.y), "true source clipped out!"


def test_triangulation_localizes_and_mec_covers():
    src = Jammer(7, 600.0, -300.0, 1500.0, "omni")
    env = LocalEnv(case=Case([src], SmoothErrorField(seed=11), seed=1))
    env.reset()
    belief = BeliefState(problem=3)
    # scan from well-separated points to get crossing bearings
    for (x, y) in [(0, 0), (0, 400), (-300, 0), (200, -600), (500, 200)]:
        _feed(belief, env, _scan(7, x, y))
        assert _point_in_region(belief.belief(7).region, src.x, src.y)
    cb = belief.belief(7)
    assert cb.is_localized, f"MEC radius still {cb.mec_radius:.1f} m"
    ex, ey = cb.estimate
    # localized MEC radius <= threshold guarantees the estimate is within clear range
    assert math.hypot(ex - src.x, ey - src.y) <= 20.0


def test_near_reading_localizes_at_pose():
    src = Jammer(2, 100.0, 0.0, 1500.0, "omni")
    env = LocalEnv(case=Case([src], SmoothErrorField(seed=1), seed=1))
    env.reset()
    belief = BeliefState(problem=3)
    _feed(belief, env, _scan(2, 100.0, 3.0))  # within 5 m -> TOO_STRONG
    cb = belief.belief(2)
    assert cb.is_localized
    ex, ey = cb.estimate
    assert math.hypot(ex - src.x, ey - src.y) <= 20.0


def test_no_signal_records_exclusion_p3_only():
    src = Jammer(9, 1600.0, 0.0, 1500.0, "omni")  # far: (0,0) is out of range
    env = LocalEnv(case=Case([src], SmoothErrorField(seed=1), seed=1))
    env.reset()
    b3 = BeliefState(problem=3)
    _feed(b3, env, _scan(9, 0.0, 0.0))
    assert b3.excluded(0.0, 0.0, channel=9)          # d>range_min -> forbidden
    assert not b3.excluded(1600.0, 0.0, channel=9)   # true source not excluded

    env2 = LocalEnv(case=Case([src], SmoothErrorField(seed=1), seed=1))
    env2.reset()
    b4 = BeliefState(problem=4)                       # dir-possible: no exclusion
    _feed(b4, env2, _scan(9, 0.0, 0.0))
    assert not b4.excluded(0.0, 0.0, channel=9)


def test_clear_success_marks_and_failure_excludes():
    src = Jammer(3, 50.0, 0.0, 1500.0, "omni")
    env = LocalEnv(case=Case([src], SmoothErrorField(seed=1), seed=1))
    env.reset()
    belief = BeliefState(problem=3)
    # miss far away
    belief.update(env.execute(Action(ActionType.CLEAR, 3, 500.0, 0.0)))
    assert belief.excluded(500.0, 0.0, channel=3)
    assert not belief.belief(3).is_cleared
    # hit
    belief.update(env.execute(Action(ActionType.CLEAR, 3, 50.0, 0.0)))
    assert belief.belief(3).is_cleared
    assert 3 in belief.cleared_channels()
