"""
§57 Q4 定向 + 组合可行集测试。

不变量：无论真源是全向还是定向，其真实位置都落在【组合可行集】外包络内。
组合集 = omni 假设 ∪ directional 假设；真源类型对应的那个假设集必含真源。
"""
import math
import random

import pytest

from src.domain.observation import Observation
from src.domain.types import ObservationType, SourceType
from src.set_estimation.q4_directional_feasible import Q4DirectionalFeasibleSet
from src.set_estimation.q4_omni_feasible import Q4ChannelFeasible
from src.simulator.mock_simulator import (MockCase, MockJammer, MockSimulator,
                                          make_adversarial_field, make_smooth_field)


def _observe(sim, x, y, ch, step):
    r = sim.measure(x, y, ch)
    otype = r.to_observation_type()
    return Observation(position=(x, y), channel=ch, result=otype,
                       bearing_deg=r.svd_deg if otype == ObservationType.BEARING else None,
                       virtual_time=r.virtual_time_s, step=step)


def _in_combined(fs: Q4ChannelFeasible, pt) -> bool:
    c = fs.mec()
    if c is None:
        return False
    return math.hypot(pt[0] - c.center[0], pt[1] - c.center[1]) <= c.radius + 1e-3


@pytest.mark.parametrize("seed", range(30))
def test_q4_directional_true_source_in_envelope(seed, problem, robot):
    rng = random.Random(seed * 104729 + 3)
    R = problem.area_radius
    while True:
        sx = rng.uniform(-R, R); sy = rng.uniform(-R, R)
        if sx * sx + sy * sy <= (R * 0.9) ** 2:
            break
    r_eff = rng.uniform(problem.receive_radius_min, problem.receive_radius_max)
    is_dir = (seed % 2 == 0)
    direction = rng.uniform(0, 360)
    kind = SourceType.DIRECTIONAL if is_dir else SourceType.OMNI
    field = make_adversarial_field(1.0) if seed % 3 == 0 else make_smooth_field(seed)
    case = MockCase(jammers=[MockJammer(1, sx, sy, r_eff, kind=kind, direction_deg=direction)],
                    error_field=field)
    sim = MockSimulator(case, problem, robot)
    sim.enter()

    fs = Q4ChannelFeasible(problem=problem, robot=robot, coarse_size=25.0, fine_size=10.0)
    obs = []
    for k in range(8):
        ang = 2 * math.pi * k / 8 + rng.uniform(-0.2, 0.2)
        dist = rng.uniform(150, 1400)
        px, py = sx + dist * math.cos(ang), sy + dist * math.sin(ang)
        if px * px + py * py > R * R:
            continue
        obs.append(_observe(sim, px, py, 1, k))
        fs.rebuild(obs, min_size=25.0)
        assert _in_combined(fs, (sx, sy)), (
            f"seed={seed} kind={kind} step={k}: true source excluded from combined envelope")


def test_q4_directional_hypothesis_holds_for_directional(problem, robot):
    """真源为定向：定向假设集必含真源。"""
    sx, sy, direction = -400.0, 250.0, 30.0
    case = MockCase(jammers=[MockJammer(1, sx, sy, 1300.0, kind=SourceType.DIRECTIONAL,
                                        direction_deg=direction)],
                    error_field=make_smooth_field(9))
    sim = MockSimulator(case, problem, robot)
    sim.enter()
    ds = Q4DirectionalFeasibleSet(problem=problem, robot=robot, coarse_size=25.0, fine_size=10.0)
    obs = []
    # 从锥内方向布点（保证收到）
    for k in range(6):
        ang = math.radians(direction) + rng_off(k)
        dist = 800.0
        px, py = sx + dist * math.cos(ang), sy + dist * math.sin(ang)
        if px * px + py * py > problem.area_radius ** 2:
            continue
        obs.append(_observe(sim, px, py, 1, k))
        ds.rebuild(obs, min_size=25.0)
        if not ds.is_empty():
            c = ds.mec()
            assert math.hypot(sx - c.center[0], sy - c.center[1]) <= c.radius + 1e-3


def rng_off(k):
    return [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5][k % 6]
