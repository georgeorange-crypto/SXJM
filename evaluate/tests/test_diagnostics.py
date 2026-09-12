"""逐动作诊断的正确性：与既有 metrics 归因严格对齐；地板/超支恒等式成立。"""

import math

import pytest

from eval_suite import diagnostics as dg
from eval_suite import runner
from eval_suite.metrics import ORACLE_SCAN_S


def _profiles(problem=3, n=6):
    summaries = runner.run_all(
        methods=["lb", "oracle", "reactive", "ours"], n=n, problem=problem)
    return summaries


def test_action_excess_matches_metrics_attribution():
    """Δ_move == ΔT_move，且 Δ_detect+Δ_switch == ΔT_info（逐动作诊断是既有归因的细化）。"""
    summaries = _profiles()
    for method in ("ours", "reactive"):
        succ = [r for r in summaries[method].results if r.success]
        if not succ:
            continue
        p = dg.build_profile(summaries[method])
        # 用同一批成功局的均值核对
        import statistics
        dT_move = statistics.fmean([r.dT_move for r in succ])
        dT_info = statistics.fmean([r.dT_info for r in succ])
        assert p.get("move").excess == pytest.approx(dT_move, abs=1e-6)
        assert (p.get("detect").excess + p.get("switch").excess
                == pytest.approx(dT_info, abs=1e-6))


def test_floor_sum_and_total_excess_identity():
    """T_floor = L_LB/5 + 119 + 5m；Σ动作超支 == T - T_floor。"""
    summaries = _profiles()
    p = dg.build_profile(summaries["ours"])
    assert p.n_success > 0
    # detect+switch 地板 == 119
    assert p.get("detect").necessary + p.get("switch").necessary == pytest.approx(ORACLE_SCAN_S)
    sum_excess = sum(a.excess for a in p.actions)
    # 容差同 test_metrics：T 用引擎微秒钟，动作分量是重算 float，二者有 μs 量级量化差。
    assert sum_excess == pytest.approx(p.total_excess_floor, abs=1e-2)
    assert p.T_floor == pytest.approx(sum(a.necessary for a in p.actions), abs=1e-9)


def test_lb_has_zero_move_and_clear_excess():
    """下界伪局：移动=地板、清除=5m、检测/切频=0，故 Δmove=Δclear=0。"""
    summaries = runner.run_all(methods=["lb"], n=3, problem=3)
    p = dg.build_profile(summaries["lb"])
    assert p.get("move").excess == pytest.approx(0.0, abs=1e-6)
    assert p.get("clear").excess == pytest.approx(0.0, abs=1e-6)


def test_oracle_move_bounds_are_ordered():
    """Oracle 移动(L_UB/5) 应 ≥ 必要移动地板(L_LB/5)：邻域松弛非负。"""
    summaries = _profiles()
    p = dg.build_profile(summaries["ours"])
    assert p.oracle_move >= p.get("move").necessary - 1e-6


def test_tables_and_diagnosis_render():
    summaries = _profiles()
    for fn in (dg.action_time_table, dg.headroom_table, dg.action_count_table):
        txt = fn(summaries)
        assert "method" in txt and "ours" in txt
    diag = dg.format_diagnosis(summaries, problem=3)
    assert "逐动作诊断" in diag and "最大可优化头" in diag


def test_reactive_move_excess_exceeds_ours():
    """现实结论：reactive 的移动超支应远大于 ours（缺全局规划 → 大量绕路）。"""
    summaries = _profiles(n=8)
    po = dg.build_profile(summaries["ours"])
    pr = dg.build_profile(summaries["reactive"])
    if po.n_success and pr.n_success:
        assert pr.get("move").excess > po.get("move").excess
