"""计时分解的一致性：InstrumentedWorld 累加 == 引擎虚拟钟；指标派生正确。"""

import math

import pytest

from eval_suite import _paths  # noqa
from eval_suite.interface_shim import LocalWorld
from eval_suite.metrics import (
    ORACLE_SCAN_S,
    EpisodeMetrics,
    InstrumentedWorld,
    TimeBreakdown,
    attach_certificate,
    oracle_time,
)
from eval_suite import lowerbound as lb
from jammerhunt import environment as env
from jammerhunt.agent import Hunter


@pytest.mark.parametrize("seed,problem", [(1000, 3), (1001, 3), (2000, 4), (2001, 4)])
def test_timing_decomposition_matches_engine(seed, problem):
    case = env.generate_case(seed=seed, problem=problem)
    engine = env.Engine(case)
    world = InstrumentedWorld(LocalWorld(engine))
    Hunter(problem=problem).run(world)
    # 分解累加必须等于引擎虚拟钟（微秒级；容差 1ms）
    assert world.bd.total_s == pytest.approx(engine.virtual_time_s, abs=1e-3)


def test_breakdown_components_nonneg():
    case = env.generate_case(seed=1234, problem=3)
    engine = env.Engine(case)
    world = InstrumentedWorld(LocalWorld(engine))
    Hunter(problem=3).run(world)
    bd = world.bd
    assert bd.move_s >= 0 and bd.detect_s >= 0 and bd.switch_s >= 0 and bd.clear_s >= 0
    assert bd.detect_s == pytest.approx(5.0 * bd.n_measure)
    assert bd.switch_s == pytest.approx(1.0 * bd.n_switch)
    assert bd.clear_s == pytest.approx(5.0 * bd.n_clear_hit + 3.0 * bd.n_clear_miss)
    assert bd.n_switch <= bd.n_measure


def test_oracle_scan_constant():
    assert ORACLE_SCAN_S == pytest.approx(119.0)


def test_metric_ratios():
    srcs = [(500.0, 0.0), (0.0, 800.0), (-600.0, -300.0)]
    cert = lb.certificate(srcs)
    m = EpisodeMetrics(seed=0, problem=3, method="x", total=3, cleared=3, success=True)
    attach_certificate(m, cert)
    m.T = 2.0 * m.T_abs_lb
    assert m.R_LB == pytest.approx(2.0)
    # oracle_time = 119 + L_UB/5 + 5*3
    assert m.T_oracle == pytest.approx(oracle_time(cert))


def test_dt_info_matches_ideal_when_oracle_scan():
    # 若某方法只做 20 次扫描各切频一次：detect=100, switch=19 → dT_info=0
    m = EpisodeMetrics(seed=0, problem=3, method="x", total=1, cleared=1, success=True)
    m.bd = TimeBreakdown(detect_s=100.0, switch_s=19.0)
    assert m.dT_info == pytest.approx(0.0)
