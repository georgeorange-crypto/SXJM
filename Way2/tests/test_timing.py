"""LocalEnv must reproduce the official timing oracle to the microsecond.

The canonical vector from appendix 2 §10 / appendix 1 §3:
enter -> 0; measure ch1 @(300,400) -> 105; measure ch2 in place -> 111;
clear ch3 @(300,0) (miss) -> 194; measure ch2 in place -> 199.
"""

from __future__ import annotations

import math

import pytest

from radio_rl.core.datatypes import Action, ActionType, ObservationType
from radio_rl.env.error_field import SmoothErrorField, make_error_field
from radio_rl.env.generators import Case, Jammer, norm_deg
from radio_rl.env.local_env import LocalEnv


def _scan(ch, x, y):
    return Action(ActionType.SCAN, ch, x, y)


def _clear(ch, x, y):
    return Action(ActionType.CLEAR, ch, x, y)


def _timing_case() -> Case:
    jammers = [
        Jammer(1, 1000.0, 0.0, 1200.0, "omni"),
        Jammer(2, -1000.0, 0.0, 1200.0, "omni"),
        Jammer(3, -1500.0, 100.0, 1200.0, "omni"),
    ]
    return Case(jammers, SmoothErrorField(seed=123), seed=1)


def _env(case: Case) -> LocalEnv:
    env = LocalEnv(case=case)
    env.reset()
    return env


def test_timing_vector_105_111_194_199():
    env = _env(_timing_case())
    assert env.virtual_time_s == 0.0

    env.execute(_scan(1, 300, 400))
    assert env.virtual_time_s == pytest.approx(105.0, abs=1e-6)

    env.execute(_scan(2, 300, 400))
    assert env.virtual_time_s == pytest.approx(111.0, abs=1e-6)

    env.execute(_clear(3, 300, 0))
    assert env.virtual_time_s == pytest.approx(194.0, abs=1e-6)

    env.execute(_scan(2, 300, 0))
    assert env.virtual_time_s == pytest.approx(199.0, abs=1e-6)


def test_clear_hit_timing_and_once():
    j = Jammer(5, 100.0, 0.0, 1200.0, "omni")
    env = _env(Case([j], SmoothErrorField(1), 1))
    # move 100/5 = 20 s + hit 5 s = 25
    obs = env.execute(_clear(5, 100, 0))
    assert env.virtual_time_s == pytest.approx(25.0, abs=1e-6)
    assert obs.result_type == ObservationType.CLEAR_SUCCESS
    obs2 = env.execute(_clear(5, 100, 0))
    assert obs2.result_type == ObservationType.CLEAR_FAILURE


def test_near_direction_nosignal():
    j = Jammer(7, 0.0, 0.0, 1200.0, "omni")
    env = _env(Case([j], SmoothErrorField(1), 1))
    assert env.execute(_scan(7, 3, 0)).result_type == ObservationType.TOO_STRONG
    assert env.execute(_scan(7, 500, 0)).result_type == ObservationType.SIGNAL
    assert env.execute(_scan(7, 1300, 0)).result_type == ObservationType.NO_SIGNAL
    assert env.execute(_scan(9, 10, 0)).result_type == ObservationType.NO_SIGNAL


def test_directional_coverage():
    j = Jammer(8, 0.0, 0.0, 1200.0, "dir", direction_deg=0.0)
    env = _env(Case([j], SmoothErrorField(1), 1))
    assert env.execute(_scan(8, 500, 0)).result_type == ObservationType.SIGNAL
    assert env.execute(_scan(8, -500, 0)).result_type == ObservationType.NO_SIGNAL
    assert env.execute(_scan(8, 0, 500)).result_type == ObservationType.SIGNAL  # boundary
    assert env.execute(_clear(8, -10, 0)).result_type == ObservationType.CLEAR_SUCCESS


def test_boundary_near_and_clear_radius():
    eps = 1e-6
    for d, expect in [(5.0, ObservationType.TOO_STRONG),
                      (5.0 + eps, ObservationType.SIGNAL),
                      (5.0 - eps, ObservationType.TOO_STRONG)]:
        env = _env(Case([Jammer(3, 0.0, 0.0, 1200.0, "omni")], SmoothErrorField(1), 1))
        assert env.execute(_scan(3, d, 0)).result_type == expect
    for d, expect in [(20.0, ObservationType.CLEAR_SUCCESS),
                      (20.0 + eps, ObservationType.CLEAR_FAILURE),
                      (20.0 - eps, ObservationType.CLEAR_SUCCESS)]:
        env = _env(Case([Jammer(3, 0.0, 0.0, 1200.0, "omni")], SmoothErrorField(1), 1))
        assert env.execute(_clear(3, d, 0)).result_type == expect


def test_svd_error_bounded_and_repeatable():
    j = Jammer(9, 800.0, 600.0, 1400.0, "omni")
    env = _env(Case([j], SmoothErrorField(seed=777), 1))
    max_err = 0.0
    for (x, y) in [(0, 0), (100, 50), (-200, 300), (500, -400), (700, 100)]:
        obs = env.execute(_scan(9, x, y))
        if obs.result_type == ObservationType.SIGNAL:
            true_b = norm_deg(math.degrees(math.atan2(j.y - y, j.x - x)))
            d = abs(((obs.bearing_deg - true_b + 180) % 360) - 180)
            max_err = max(max_err, d)
            assert 0.0 <= obs.bearing_deg < 360.0
    assert max_err <= 1.0 + 5e-3

    env2 = _env(Case([j], SmoothErrorField(seed=777), 1))
    o1 = env2.execute(_scan(9, 123.0, -45.0))
    o2 = env2.execute(_scan(9, 123.0, -45.0))
    assert o1.bearing_deg == pytest.approx(o2.bearing_deg, abs=1e-9)


def test_channel_state_update_and_switch_cost():
    j1 = Jammer(1, 500, 0, 1200, "omni")
    j4 = Jammer(4, 0, 500, 1200, "omni")
    env = _env(Case([j1, j4], SmoothErrorField(1), 1))
    env.execute(_scan(4, 10, 0))         # switch 1->4
    env.execute(_clear(1, 10, 0))        # clear must not change current channel
    before = env.virtual_time_s
    env.execute(_scan(4, 10, 0))         # same channel, in place -> +5 only
    assert env.virtual_time_s - before == pytest.approx(5.0, abs=1e-6)


def test_svd_wraparound_stays_in_range():
    for tb_target in (359.996, 359.999, 0.001, 359.5):
        ang = math.radians(tb_target)
        j = Jammer(5, 500 * math.cos(ang), 500 * math.sin(ang), 1200.0, "omni")
        fld = make_error_field("adversarial", seed=1,
                               params={"mode": "constant", "magnitude": 1.0})
        env = _env(Case([j], fld, 1))
        obs = env.execute(_scan(5, 0, 0))
        if obs.result_type == ObservationType.SIGNAL:
            assert 0.0 <= obs.bearing_deg < 360.0
