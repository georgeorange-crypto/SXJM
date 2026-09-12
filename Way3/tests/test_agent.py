"""策略端到端测试（本包 environment + LocalWorld）：核心指标是“全清成功率 = 100%”。

覆盖：第三问/第四问随机案例扫描（smooth 与最坏 constant ±1° 场），已修复缺陷的回归用例，
以及手工构造的确定性案例（含定向源），验证“判定无源”的频道确实被正确标记。
correctness（全清）优先于速度——这些用例只断言全清与无异常，不对时间设阈。
"""

import math

import pytest

from jammerhunt import environment as env
from jammerhunt.agent import Hunter
from jammerhunt.interface import LocalWorld
from jammerhunt.mc import run_local_episode


# --------------------------------------------------------------------------- #
# 随机案例：全清成功率必须 100%
# --------------------------------------------------------------------------- #
P3_SEEDS = list(range(1000, 1010))
P4_SEEDS = list(range(2000, 2008))


@pytest.mark.parametrize("seed", P3_SEEDS)
def test_problem3_smooth_full_clear(seed):
    r = run_local_episode(seed, problem=3, field_kind="smooth")
    assert r.error is None, f"策略异常: {r.error}"
    assert r.success, f"P3 seed={seed} 仅清 {r.cleared}/{r.total}"


@pytest.mark.parametrize("seed", P3_SEEDS[:6])
def test_problem3_constant_worstcase_full_clear(seed):
    # 常数 +1° 场：全场最坏偏置，检验 ±1° 误差下定位鲁棒性上界
    r = run_local_episode(seed, problem=3, field_kind="constant")
    assert r.error is None, f"策略异常: {r.error}"
    assert r.success, f"P3(constant) seed={seed} 仅清 {r.cleared}/{r.total}"


@pytest.mark.parametrize("seed", P4_SEEDS)
def test_problem4_smooth_full_clear(seed):
    r = run_local_episode(seed, problem=4, field_kind="smooth")
    assert r.error is None, f"策略异常: {r.error}"
    assert r.success, f"P4 seed={seed} 仅清 {r.cleared}/{r.total} (dir={r.n_dir})"


@pytest.mark.parametrize("seed", P4_SEEDS[:6])
def test_problem4_constant_worstcase_full_clear(seed):
    r = run_local_episode(seed, problem=4, field_kind="constant")
    assert r.error is None, f"策略异常: {r.error}"
    assert r.success, f"P4(constant) seed={seed} 仅清 {r.cleared}/{r.total} (dir={r.n_dir})"


# --------------------------------------------------------------------------- #
# 回归：曾经漏清的定向源（过冲缺陷）——现须全清
# --------------------------------------------------------------------------- #
def test_regression_p4_overshoot_seed4068():
    """曾因‘沿方位前进 420 m 越过 357 m 处的源’落入盲区导致漏清；
    改为‘垂直优先’建基线后应稳定全清。"""
    r = run_local_episode(4068, problem=4, field_kind="smooth")
    assert r.error is None
    assert r.success, f"回归失败 seed=4068 仅清 {r.cleared}/{r.total}"


# --------------------------------------------------------------------------- #
# 手工确定性案例：验证清除 + “判无源”标记
# --------------------------------------------------------------------------- #
def _run_case(case: env.Case, problem: int):
    eng = env.Engine(case)
    hunter = Hunter(problem=problem)
    hunter.run(LocalWorld(eng))
    return hunter, eng


def test_handcrafted_two_omni_cleared_and_absence_flagged():
    case = env.Case(
        [env.Jammer(5, 400.0, 300.0, 1200.0, "omni"),
         env.Jammer(12, -600.0, -200.0, 1100.0, "omni")],
        env.ConstantField(1.0),                       # 最坏偏置
    )
    hunter, _ = _run_case(case, problem=3)
    assert case.cleared_count == 2
    assert hunter.chans[5].cleared and hunter.chans[12].cleared
    # 其余 18 个频道无源 → 应全部判定为 absent，且无一误报“有源未清”
    for c in range(1, 21):
        if c in (5, 12):
            continue
        assert hunter.chans[c].absent, f"ch{c} 应判无源"
        assert not hunter.chans[c].detected
    present = [c for c, s in hunter.chans.items() if s.detected and not s.cleared]
    assert present == []


def test_handcrafted_directional_cleared():
    # 定向源：朝向 225°（西南），置于第二象限内侧，3-覆盖点集足以从其覆盖半平面探测
    case = env.Case(
        [env.Jammer(7, 300.0, 400.0, 1300.0, "dir", direction_deg=225.0),
         env.Jammer(3, -500.0, 200.0, 1200.0, "omni")],
        env.ErrorField(seed=11),
    )
    hunter, _ = _run_case(case, problem=4)
    assert case.cleared_count == 2, "定向 + 全向均应清除"
    assert hunter.chans[7].cleared and hunter.chans[3].cleared


def test_near_source_cleared_in_place():
    """扫描点恰在源 5 m 内 → near → 就地清除（无需归航）。"""
    # 全向 1-覆盖含原点扫描点；把源放在原点近旁
    case = env.Case([env.Jammer(9, 2.0, 0.0, 1200.0, "omni")], env.ConstantField(0.0))
    hunter, _ = _run_case(case, problem=3)
    assert hunter.chans[9].cleared
    assert case.cleared_count == 1


def test_empty_case_all_absent_no_false_positive():
    """一个源都没有 → 全部频道判无源，绝不虚报。"""
    case = env.Case([], env.ErrorField(seed=1))
    hunter, _ = _run_case(case, problem=3)
    assert all(hunter.chans[c].absent for c in range(1, 21))
    assert hunter.cleared_count == 0


def test_early_stop_at_16_sources():
    """已知一局至多 16 源：清满 16 即应停止（不再空扫）。"""
    # 构造 16 个全向源（占满 16 个频道），验证全清且计数达上限
    jammers = []
    positions = [(r * math.cos(math.radians(a)), r * math.sin(math.radians(a)))
                 for r, a in zip(range(200, 1400, 75), range(0, 360, 22))]
    for ch, (x, y) in zip(range(1, 17), positions):
        jammers.append(env.Jammer(ch, x, y, 1300.0, "omni"))
    case = env.Case(jammers, env.ErrorField(seed=5))
    hunter, _ = _run_case(case, problem=3)
    assert case.cleared_count == 16
    assert hunter.cleared_count == 16
