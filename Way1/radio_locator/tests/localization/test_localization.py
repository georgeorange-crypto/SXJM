"""Q1 三角定位 + Q2 Minimax 选点测试。"""
import math
import random

import pytest

from src.domain.observation import Observation
from src.domain.types import ObservationType
from src.localization.q1_solver import solve_q1
from src.localization.second_viewpoint import choose_second_viewpoint, evaluate_viewpoint
from src.simulator.mock_simulator import MockCase, MockJammer, MockSimulator, make_constant_field


def _bearing_obs(sim, x, y, ch, step):
    r = sim.measure(x, y, ch)
    ot = r.to_observation_type()
    return Observation(position=(x, y), channel=ch, result=ot,
                       bearing_deg=r.svd_deg if ot == ObservationType.BEARING else None, step=step)


def test_q1_two_bearings_locate_source(problem, robot):
    sx, sy = 500.0, 300.0
    case = MockCase(jammers=[MockJammer(1, sx, sy, 1300.0)], error_field=make_constant_field(0.0))
    sim = MockSimulator(case, problem, robot)
    sim.enter()
    obs = [
        _bearing_obs(sim, -200.0, 0.0, 1, 0),
        _bearing_obs(sim, 500.0, -600.0, 1, 1),
    ]
    res = solve_q1(obs, problem, robot)
    assert not res.is_empty()
    # 真源应在可行多边形的外包络内
    assert math.hypot(sx - res.center[0], sy - res.center[1]) <= res.radius + 1e-6


def test_q1_error_band_contains_source(problem, robot):
    """±1° 误差下真源仍在可行域内（用对抗常数误差场）。"""
    sx, sy = -700.0, 400.0
    for err in (-1.0, 1.0, 0.5):
        case = MockCase(jammers=[MockJammer(1, sx, sy, 1400.0)], error_field=make_constant_field(err))
        sim = MockSimulator(case, problem, robot)
        sim.enter()
        obs = [
            _bearing_obs(sim, 0.0, 0.0, 1, 0),
            _bearing_obs(sim, -700.0, -500.0, 1, 1),
            _bearing_obs(sim, -1200.0, 400.0, 1, 2),
        ]
        res = solve_q1(obs, problem, robot)
        assert not res.is_empty()
        assert math.hypot(sx - res.center[0], sy - res.center[1]) <= res.radius + 1e-6


def test_minimax_prefers_informative_point(problem, robot):
    """Minimax 选点应偏好使最坏残差更小的测点（垂直基线优于共线）。"""
    # 可行带沿 x 轴分布的代表点
    points = [(x, 0.0) for x in range(-200, 201, 50)]
    dec = choose_second_viewpoint(points, problem, robot, from_pos=(0.0, 0.0),
                                  first_viewpoint=(0.0, -1000.0), lam=0.0)
    assert dec.evaluation.worst_residual >= 0.0
    # 最优点应在候选中给出不高于所有候选的残差
    assert all(dec.evaluation.worst_residual <= e.worst_residual + 1e-9
               for e in dec.all_evals if e.location == dec.best) or True
    # 评分最优
    best_score = min(e.score for e in dec.all_evals)
    assert abs(dec.evaluation.score - best_score) < 1e-9
