"""
端到端集成测试：controller 在 mock 上跑完整任务。

验证：
  - 时间账本一致（本地虚拟钟 == 复算总时间）。
  - 所有 CLEARED 频道确实命中真源（success），无误清。
  - 不变量守卫无 I3（证书）/I5（时间）违规。
  - 简单单源/双源场景能完成清除。
"""
import math

import pytest

from src.coverage.q3_hex_cover import hex_cover
from src.domain.world_state import WorldState
from src.runtime.controller import Controller, ControllerConfig
from src.analysis.metrics import compute_metrics
from src.simulator.mock_simulator import (MockCase, MockJammer, MockSimulator,
                                          make_smooth_field)


def _run(jammers, problem, robot, cover_r_safe=980.0, max_steps=800, coarse=40.0, fine=15.0):
    case = MockCase(jammers=jammers, error_field=make_smooth_field(42))
    sim = MockSimulator(case, problem, robot)
    world = WorldState(problem=problem, robot=robot)
    plan = hex_cover(problem.area_radius, r_safe=cover_r_safe, m=6)
    cfg = ControllerConfig(problem_mode="q3", coarse_size=coarse, fine_size=fine,
                           max_steps=max_steps, lam=1.0)
    ctrl = Controller(sim, world, plan.points, cfg)
    ctrl.run()
    return ctrl, world, case


def test_single_source_cleared(problem, robot):
    ctrl, world, case = _run([MockJammer(1, 250.0, -150.0, 1200.0)], problem, robot)
    # 该频道应被清除
    from src.domain.types import ChannelStatus
    assert world.channels[1].status == ChannelStatus.CLEARED
    # 真源确实已被 mock 标记清除（命中 20m 内）
    assert case.jammers[0].cleared
    # 无证书/时间违规
    codes = [v.code for v in ctrl.monitor.violations]
    assert "I3_CLEAR_CERT" not in codes
    assert "I5_TIME" not in codes


def test_time_ledger_consistent(problem, robot):
    ctrl, world, case = _run([MockJammer(3, -400.0, 500.0, 1100.0)], problem, robot)
    metrics = compute_metrics(ctrl.log, world, robot)
    assert metrics.ledger_consistent, (
        f"ledger drift: recomputed {metrics.total_time:.4f} vs world {world.virtual_time:.4f}")


def test_no_false_clear(problem, robot):
    """清除成功的频道必对应真实存在且被命中的源。"""
    jammers = [MockJammer(2, 600.0, 100.0, 1300.0), MockJammer(5, -300.0, -600.0, 1000.0)]
    ctrl, world, case = _run(jammers, problem, robot, max_steps=1500)
    from src.domain.types import ChannelStatus
    for c, cs in world.channels.items():
        if cs.status == ChannelStatus.CLEARED:
            j = case.jammer_on(c)
            assert j is not None and j.cleared, f"ch{c} marked CLEARED but no real hit"
