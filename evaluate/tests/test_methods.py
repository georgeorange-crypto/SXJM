"""方法层的行为契约：oracle 全清且贴近下界；基线/消融可运行；证书序在真实案例成立。"""

import math

import pytest

from eval_suite import runner


@pytest.mark.parametrize("seed", [1000, 1001, 1002])
def test_oracle_clears_all_and_near_lb(seed):
    m = runner.run_episode("oracle", seed=seed, problem=3)
    assert m.success and m.cleared == m.total
    # Oracle 应非常接近其定义时间（119 + L_UB/5 + 5m）；R_oracle ~ 1
    assert m.R_oracle == pytest.approx(1.0, abs=0.05)
    # 且必不快于绝对下界
    assert m.R_LB >= 1.0 - 1e-6


def test_lb_pseudo_episode():
    m = runner.run_episode("lb", seed=1000, problem=3)
    assert m.method == "lb" and m.success
    assert m.T == pytest.approx(m.T_abs_lb)
    assert m.R_LB == pytest.approx(1.0)
    # 分解累加应等于总时间
    assert m.bd.total_s == pytest.approx(m.T, abs=1e-6)


@pytest.mark.parametrize("method", ["reactive", "greedy_scan", "ours",
                                    "no_global_scan", "no_global_route",
                                    "no_active_sensing", "no_replanning"])
def test_methods_run_without_error(method):
    m = runner.run_episode(method, seed=1000, problem=3)
    assert m.error is None, m.error
    assert 0 <= m.cleared <= m.total
    # 计时分解一致（无 timing mismatch 被写进 error）
    assert m.bd.total_s == pytest.approx(m.T, abs=1e-3)


def test_ours_not_slower_than_reactive_typical():
    # 全局信息 + 规划应显著优于 reactive（同一 seed）。
    o = runner.run_episode("ours", seed=1000, problem=3)
    r = runner.run_episode("reactive", seed=1000, problem=3)
    if o.success and r.success:
        assert o.T <= r.T


def test_no_global_scan_equals_reactive_behaviorally():
    # w/o Global Scan 的骨架就是 reactive；同 seed 时间应一致。
    a = runner.run_episode("no_global_scan", seed=1000, problem=3)
    b = runner.run_episode("reactive", seed=1000, problem=3)
    assert a.T == pytest.approx(b.T, abs=1e-6)


def test_certificate_ordering_on_real_case():
    m = runner.run_episode("ours", seed=1000, problem=3)
    assert m.L_LB <= m.L_UB + 1e-6
    assert m.T_abs_lb <= m.T_oracle + 1e-6      # 下界 ≤ oracle
