"""虚拟世界（environment.Engine）测试。

【头号用例】精确复现 附件2 §10 计时示例：四步动作后虚拟时间 = [105, 111, 194, 199] s。
其余用例校验物理规则：全向/定向覆盖、有效接收半径、近距离 near、清除半径、
示向度 ±1° 误差与两位小数归一，以及 generate_case 的硬约束（第三问全全向；第四问含定向）。
"""

import math

import pytest

from jammerhunt import environment as env
from jammerhunt.environment import (
    Case, ConstantField, Engine, ErrorField, Jammer, generate_case,
    ARENA_RADIUS_M, N_MIN, N_MAX, R_EFF_MIN, R_EFF_MAX,
)


# --------------------------------------------------------------------------- #
# 附件2 §10 计时示例（本项目忠实度的“金标准”）
# --------------------------------------------------------------------------- #
def _timing_case() -> Case:
    """与 §10 示例同构：三只全向源，位置使 clear(300,0,ch3) 必然未命中（源在远处）。

    §10 明确“耗时只取决于移动/切换/动作，与检测结果无关”，故源的具体布局不影响计时序列，
    只需保证 (300,0) 处 ch3 无可清目标（>20 m）→ clear 记未命中(3 s)。
    """
    jammers = [
        Jammer(1, 1000.0, 0.0, 1200.0, "omni"),
        Jammer(2, -1000.0, 0.0, 1200.0, "omni"),
        Jammer(3, -1500.0, 100.0, 1200.0, "omni"),
    ]
    return Case(jammers, ErrorField(seed=123), seed=1)


def test_section10_timing_vector():
    eng = Engine(_timing_case())
    eng.enter()
    assert eng.virtual_time_s == pytest.approx(0.0, abs=1e-6)          # /enter 不推进钟

    eng.measure(300.0, 400.0, 1)      # 移动 500/5=100，切频 0（初始即 ch1），检测 5
    assert eng.virtual_time_s == pytest.approx(105.0, abs=1e-6)

    eng.measure(300.0, 400.0, 2)      # 移动 0，切频 1，检测 5
    assert eng.virtual_time_s == pytest.approx(111.0, abs=1e-6)

    eng.clear(300.0, 0.0, 3)          # 移动 400/5=80，未命中 3（/clear 不切频）
    assert eng.virtual_time_s == pytest.approx(194.0, abs=1e-6)

    eng.measure(300.0, 0.0, 2)        # 移动 0，切频 0（仍 ch2），检测 5
    assert eng.virtual_time_s == pytest.approx(199.0, abs=1e-6)


def test_timing_microsecond_accumulation_no_drift():
    """大量小步移动用微秒整数累计，不应产生浮点漂移。"""
    eng = Engine(_timing_case())
    eng.enter()
    # 反复在两点间往返：每次移动 5 m → 1 s，外加检测 5 s、可能切频。
    x = 0.0
    for _ in range(1000):
        x = 5.0 if x == 0.0 else 0.0
        eng.measure(x, 0.0, 1)        # ch 不变，无切频；移动 5/5=1 + 检测 5 = 6 s
    assert eng.virtual_time_s == pytest.approx(6.0 * 1000, abs=1e-6)


# --------------------------------------------------------------------------- #
# /enter 语义
# --------------------------------------------------------------------------- #
def test_enter_resets_state():
    eng = Engine(_timing_case())
    eng.measure(10.0, 10.0, 5)        # 未 enter 也能推进，先弄脏状态
    eng.enter()
    assert (eng.x, eng.y) == (0.0, 0.0)
    assert eng.channel == 1            # 测向机初始频道 1（附件1 §1）
    assert eng.virtual_time_s == pytest.approx(0.0)
    assert eng.entered and not eng.finished


# --------------------------------------------------------------------------- #
# 全向物理：量程 / near / 示向度误差
# --------------------------------------------------------------------------- #
def _one_omni(x, y, r_eff=1200.0, ch=1, field=None):
    return Engine(Case([Jammer(ch, x, y, r_eff, "omni")],
                       field if field is not None else ConstantField(0.0)))


def test_omni_direction_within_reff():
    eng = _one_omni(1000.0, 0.0, r_eff=1200.0)
    eng.enter()
    _, out = eng.measure(0.0, 0.0, 1)     # 距源 1000 ≤ 1200 → direction
    assert out.result == "direction"
    assert out.svd_deg == pytest.approx(0.0, abs=1e-9)     # 零误差场：正东


def test_omni_no_signal_beyond_reff():
    eng = _one_omni(1000.0, 0.0, r_eff=1000.0)
    eng.enter()
    _, out = eng.measure(-500.0, 0.0, 1)  # 距源 1500 > 1000 → no_signal
    assert out.result == "no_signal"


def test_near_within_5m_no_svd():
    eng = _one_omni(100.0, 100.0)
    eng.enter()
    _, out = eng.measure(102.0, 100.0, 1)  # 距源 2 ≤ 5 → near
    assert out.result == "near"
    assert out.svd_deg is None


def test_svd_error_bounded_and_two_decimals():
    field = ErrorField(seed=7)
    eng = Engine(Case([Jammer(1, 900.0, 300.0, 1400.0, "omni")], field))
    eng.enter()
    for px, py in [(0, 0), (100, -200), (-300, 400), (500, 500), (200, 50)]:
        _, out = eng.measure(px, py, 1)
        if out.result != "direction":
            continue
        true_b = env._norm_deg(math.degrees(math.atan2(300.0 - py, 900.0 - px)))
        # |误差| ≤ 1°（含两位小数舍入余量）
        assert env._ang_diff(out.svd_deg, true_b) <= 1.0 + 1e-6
        assert 0.0 <= out.svd_deg < 360.0
        assert out.svd_deg == pytest.approx(round(out.svd_deg, 2), abs=1e-9)


def test_error_field_magnitude_bounded():
    f = ErrorField(seed=999)
    for x in range(-1800, 1801, 200):
        for y in range(-1800, 1801, 200):
            assert abs(f.error_deg(x, y)) <= 1.0 + 1e-12


def test_error_field_is_deterministic_by_location():
    f = ErrorField(seed=3)
    assert f.error_deg(123.4, -56.7) == f.error_deg(123.4, -56.7)   # 同点必同（§2.3）


def test_constant_field_is_worst_case_bias():
    f = ConstantField(1.0)
    assert f.error_deg(0, 0) == 1.0 and f.error_deg(500, -800) == 1.0


# --------------------------------------------------------------------------- #
# 定向物理：仅覆盖半平面内可测
# --------------------------------------------------------------------------- #
def test_directional_only_in_coverage_halfplane():
    # 源在原点，指向 0°（东）→ 覆盖“从源看方位 ∈ [-90,90]”即东半平面
    eng = Engine(Case([Jammer(1, 0.0, 0.0, 1400.0, "dir", direction_deg=0.0)],
                      ConstantField(0.0)))
    eng.enter()
    _, front = eng.measure(500.0, 0.0, 1)      # 东侧（覆盖内）
    assert front.result == "direction"
    _, back = eng.measure(-500.0, 0.0, 1)      # 西侧（覆盖外）
    assert back.result == "no_signal"


def test_directional_coverage_boundary_included():
    # 边界（正好 ±90°）按“含边界”应可测
    eng = Engine(Case([Jammer(1, 0.0, 0.0, 1400.0, "dir", direction_deg=0.0)],
                      ConstantField(0.0)))
    eng.enter()
    _, side = eng.measure(0.0, 500.0, 1)       # 从源看方位 90°，边界
    assert side.result == "direction"


# --------------------------------------------------------------------------- #
# 清除半径
# --------------------------------------------------------------------------- #
def test_clear_hit_within_20m():
    eng = _one_omni(100.0, 100.0)
    eng.enter()
    _, out = eng.clear(115.0, 105.0, 1)         # 距源 ~15.8 ≤ 20 → 命中
    assert out.result == "success"
    assert eng.case.jammer_on_channel(1).cleared


def test_clear_miss_beyond_20m():
    eng = _one_omni(100.0, 100.0)
    eng.enter()
    _, out = eng.clear(130.0, 130.0, 1)         # 距源 ~42 > 20 → 未命中
    assert out.result == "no_target_in_range"
    assert not eng.case.jammer_on_channel(1).cleared


def test_clear_success_independent_of_heading():
    # 清除成功与朝向无关：定向源被清亦然（从其“背面”清也算命中，只要 ≤20 m）
    eng = Engine(Case([Jammer(1, 0.0, 0.0, 1400.0, "dir", direction_deg=0.0)],
                      ConstantField(0.0)))
    eng.enter()
    _, out = eng.clear(-10.0, 0.0, 1)           # 背面 10 m
    assert out.result == "success"


def test_cleared_source_gives_no_signal():
    eng = _one_omni(100.0, 0.0)
    eng.enter()
    eng.clear(100.0, 0.0, 1)
    _, out = eng.measure(0.0, 0.0, 1)
    assert out.result == "no_signal"


# --------------------------------------------------------------------------- #
# 案例生成的硬约束（附件2 §1.3 / §2）
# --------------------------------------------------------------------------- #
def test_generate_case_problem3_all_omni():
    for seed in range(20):
        case = generate_case(seed=seed, problem=3)
        assert N_MIN <= case.total <= N_MAX
        assert case.n_dir == 0 and case.n_omni == case.total
        chans = [j.channel for j in case.jammers]
        assert len(set(chans)) == len(chans)                 # 每频道至多一个源
        assert all(1 <= c <= 20 for c in chans)


def test_generate_case_problem4_has_directional_and_omni():
    for seed in range(20):
        case = generate_case(seed=seed, problem=4)
        assert N_MIN <= case.total <= N_MAX
        assert case.n_dir >= 1 and case.n_omni >= 1          # 混合
        for j in case.jammers:
            assert R_EFF_MIN <= j.r_eff <= R_EFF_MAX
            if j.kind == "dir":
                assert j.direction_deg is not None


def test_generate_case_within_arena():
    for seed in range(15):
        for problem in (3, 4):
            case = generate_case(seed=seed, problem=problem)
            for j in case.jammers:
                assert math.hypot(j.x, j.y) <= ARENA_RADIUS_M + 1e-6


def test_generate_case_reproducible():
    a = generate_case(seed=42, problem=4)
    b = generate_case(seed=42, problem=4)
    assert [(j.channel, j.x, j.y, j.kind, j.direction_deg) for j in a.jammers] == \
           [(j.channel, j.x, j.y, j.kind, j.direction_deg) for j in b.jammers]
