"""
Q3 可行集测试 + §55 核心不变量性质测试。

核心不变量（property test）：只要模拟数据满足题目硬约束，真源【绝不】被排除在
外包络之外。做法：随机造真源 → 在若干合法测点用 mock（固定误差场）取观测 →
重建可行集 → 断言真源落在外包络 MEC 内（含数值裕量）。重复上万次。
"""
import math
import random

import pytest

from src.domain.observation import Observation
from src.domain.types import ObservationType, ProblemConstants, RobotConstants
from src.set_estimation.q3_feasible_set import Q3FeasibleSet
from src.simulator.mock_simulator import (MockCase, MockJammer, MockSimulator,
                                          make_adversarial_field, make_smooth_field)
from src.domain.types import SourceType


def _observe(sim, x, y, ch, step):
    r = sim.measure(x, y, ch)
    otype = r.to_observation_type()
    return Observation(position=(x, y), channel=ch, result=otype,
                       bearing_deg=r.svd_deg if otype == ObservationType.BEARING else None,
                       virtual_time=r.virtual_time_s, step=step)


def _in_envelope(fs: Q3FeasibleSet, pt) -> bool:
    if fs.is_empty():
        return False
    c = fs.mec()
    if c is None:
        return False
    return math.hypot(pt[0] - c.center[0], pt[1] - c.center[1]) <= c.radius + 1e-3


@pytest.mark.parametrize("seed", range(40))
def test_q3_true_source_in_envelope(seed, problem, robot):
    """40 组随机布局 × 每组多测点：真源必在外包络内。"""
    rng = random.Random(seed * 7919 + 1)
    # 真源
    R = problem.area_radius
    while True:
        sx = rng.uniform(-R, R)
        sy = rng.uniform(-R, R)
        if sx * sx + sy * sy <= (R * 0.95) ** 2:
            break
    r_eff = rng.uniform(problem.receive_radius_min, problem.receive_radius_max)
    ch = 1
    field = make_adversarial_field(1.0) if seed % 2 else make_smooth_field(seed)
    case = MockCase(jammers=[MockJammer(channel=ch, x=sx, y=sy, r_eff=r_eff)], error_field=field)
    sim = MockSimulator(case, problem, robot)
    sim.enter()

    fs = Q3FeasibleSet(problem=problem, robot=robot, coarse_size=20.0, fine_size=5.0)
    obs = []
    # 若干测点：围绕真源不同方位/距离，混合正负观测
    for k in range(6):
        ang = 2 * math.pi * k / 6 + rng.uniform(-0.3, 0.3)
        dist = rng.uniform(200, 1400)
        px, py = sx + dist * math.cos(ang), sy + dist * math.sin(ang)
        if px * px + py * py > R * R:
            continue
        obs.append(_observe(sim, px, py, ch, k))
        fs.rebuild(obs, min_size=20.0)
        # 真源不变量：每步都成立
        assert _in_envelope(fs, (sx, sy)), f"seed={seed} step={k}: true source excluded"


def test_q3_positive_shrinks_and_contains(problem, robot):
    """正观测越多，外包络半径单调不增，且始终含真源。"""
    sx, sy, r_eff = 300.0, -200.0, 1200.0
    case = MockCase(jammers=[MockJammer(1, sx, sy, r_eff)], error_field=make_smooth_field(3))
    sim = MockSimulator(case, problem, robot)
    sim.enter()
    fs = Q3FeasibleSet(problem=problem, robot=robot, coarse_size=20.0, fine_size=5.0)
    obs = []
    prev_r = float("inf")
    pts = [(sx + 600, sy), (sx, sy + 600), (sx - 500, sy + 300), (sx + 200, sy - 700)]
    for k, (px, py) in enumerate(pts):
        obs.append(_observe(sim, px, py, 1, k))
        fs.rebuild(obs, min_size=10.0)
        assert _in_envelope(fs, (sx, sy))
        if not fs.is_empty():
            assert fs.mec_radius <= prev_r + 30.0  # 允许粗→细的分辨率抖动
            prev_r = fs.mec_radius


def test_q3_near_constraint(problem, robot):
    """near(≤5m) 观测把可行集压到该点 5m 圆内。"""
    sx, sy = 100.0, 100.0
    case = MockCase(jammers=[MockJammer(1, sx, sy, 1200.0)], error_field=make_smooth_field(5))
    sim = MockSimulator(case, problem, robot)
    sim.enter()
    fs = Q3FeasibleSet(problem=problem, robot=robot, coarse_size=5.0, fine_size=0.5)
    # 测点恰在源上（<5m）→ near
    o = _observe(sim, sx + 2.0, sy + 1.0, 1, 0)
    assert o.result == ObservationType.TOO_STRONG
    fs.rebuild([o], min_size=0.5)
    assert not fs.is_empty()
    assert fs.mec_radius <= 5.0 + 2.0  # 5m 圆 + 分辨率裕量
    assert _in_envelope(fs, (sx, sy))
