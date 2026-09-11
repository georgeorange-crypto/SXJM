"""扫描点覆盖保证测试——本题“确保清除所有源（个数未知）”的理论支柱。

第三问（全向）：1-覆盖——区域内任意点到最近扫描点 ≤ R_eff 最坏值 1000 m。
第四问（定向）：3-覆盖——区域内任意点，其 1000 m 内扫描点“从该点看”的最大方位缺口 < 180°，
              于是任何 180° 盲区都挡不住全部扫描点 → 任意朝向的定向源必被至少一个点收到。
"""

import math

import pytest

from jammerhunt.coverage import (
    omni_scan_points, directional_scan_points,
    verify_one_cover, verify_three_cover, max_angular_gap,
    ARENA_RADIUS_M, WORST_REFF_M,
)


# --------------------------------------------------------------------------- #
# max_angular_gap 工具
# --------------------------------------------------------------------------- #
def test_max_angular_gap_uniform():
    # 4 个均布方位 → 最大缺口恰 90°
    assert max_angular_gap([0.0, 90.0, 180.0, 270.0]) == pytest.approx(90.0)


def test_max_angular_gap_wraps_around():
    # 仅 10° 与 350°：最大缺口是跨 0° 的 340° 那侧？不——环绕缺口 = 20°，另一侧 340°
    assert max_angular_gap([10.0, 350.0]) == pytest.approx(340.0)


def test_max_angular_gap_degenerate():
    assert max_angular_gap([]) == 360.0
    assert max_angular_gap([42.0]) == 360.0


def test_max_angular_gap_clustered_leaves_big_gap():
    # 全挤在东侧一小簇 → 存在接近 360° 的大缺口
    assert max_angular_gap([0.0, 5.0, 10.0, 15.0]) > 180.0


# --------------------------------------------------------------------------- #
# 全向 1-覆盖（第三问）
# --------------------------------------------------------------------------- #
def test_omni_one_cover_holds():
    pts = omni_scan_points()
    ok, worst = verify_one_cover(pts)
    assert ok
    assert worst <= WORST_REFF_M + 1e-9        # 最坏“到最近扫描点”距离 ≤ 1000
    # 设计留了余量，实际应明显小于 1000
    assert worst < WORST_REFF_M


def test_omni_points_are_few_and_inside_arena():
    pts = omni_scan_points()
    assert 5 <= len(pts) <= 12                 # 圆心+单环，点数应很少
    for x, y in pts:
        assert math.hypot(x, y) <= ARENA_RADIUS_M + 1e-6


def test_omni_cover_fails_if_points_removed():
    """去掉环上的点只留圆心 → 边缘必然覆盖不到，验证 verify_one_cover 有鉴别力。"""
    ok, worst = verify_one_cover([(0.0, 0.0)])
    assert not ok
    assert worst == pytest.approx(ARENA_RADIUS_M, abs=1.0)


# --------------------------------------------------------------------------- #
# 定向 3-覆盖（第四问）
# --------------------------------------------------------------------------- #
def test_directional_three_cover_holds_full_disk():
    pts = directional_scan_points(src_radius=ARENA_RADIUS_M)
    ok, worst_gap = verify_three_cover(pts, R=ARENA_RADIUS_M, gap_limit_deg=180.0)
    assert ok
    assert worst_gap < 180.0                   # 任意 180° 盲区都挡不住全部扫描点


def test_directional_gap_has_safety_margin():
    pts = directional_scan_points(src_radius=ARENA_RADIUS_M)
    _, worst_gap = verify_three_cover(pts, R=ARENA_RADIUS_M, gap_limit_deg=180.0)
    assert worst_gap <= 174.0                  # 设计含 ~6° 缺口余量


def test_directional_three_cover_fails_if_too_sparse():
    """极稀疏点集（仅原点+一圈 3 点）不可能 3-覆盖全盘，验证 verify_three_cover 有鉴别力。"""
    sparse = [(0.0, 0.0), (500.0, 0.0), (-250.0, 433.0), (-250.0, -433.0)]
    ok, _ = verify_three_cover(sparse, R=ARENA_RADIUS_M, gap_limit_deg=180.0)
    assert not ok


def test_directional_points_reasonable_count():
    pts = directional_scan_points(src_radius=ARENA_RADIUS_M)
    assert 15 <= len(pts) <= 60                # 三角格裁剪，几十个点量级


def test_directional_smaller_src_radius_needs_fewer_points():
    """假定源半径更小（可容忍近边界外向源不可探测）→ 点更省。"""
    full = directional_scan_points(src_radius=ARENA_RADIUS_M)
    small = directional_scan_points(src_radius=1200.0)
    assert len(small) <= len(full)
