"""几何核心单元测试：角度、交会定位、定位区域(±1°楔形交)最小包围圆、路径排序。

重点验证策略赖以“确保命中”的性质：真源必落在定位区域最小包围圆内，
且该圆半径随【交会角变小 / 距离变远】而变大——这正是“抵近+两点交会”缩圆的依据。
"""

import math

import pytest

from jammerhunt import geometry as geo


# --------------------------------------------------------------------------- #
# 角度工具
# --------------------------------------------------------------------------- #
def test_norm_deg_range():
    for a in (-720.0, -1.0, 0.0, 359.9, 360.0, 720.5):
        n = geo.norm_deg(a)
        assert 0.0 <= n < 360.0
    assert geo.norm_deg(-1.0) == pytest.approx(359.0)
    assert geo.norm_deg(360.0) == pytest.approx(0.0)


def test_ang_diff_symmetric_and_bounded():
    assert geo.ang_diff(10.0, 350.0) == pytest.approx(20.0)     # 环绕取小角
    assert geo.ang_diff(0.0, 180.0) == pytest.approx(180.0)
    assert geo.ang_diff(90.0, 270.0) == pytest.approx(180.0)
    for a, b in [(0, 45), (200, 5), (359, 1)]:
        assert 0.0 <= geo.ang_diff(a, b) <= 180.0
        assert geo.ang_diff(a, b) == pytest.approx(geo.ang_diff(b, a))


def test_signed_ang_diff():
    assert geo.signed_ang_diff(10.0, 350.0) == pytest.approx(20.0)    # 逆时针为正
    assert geo.signed_ang_diff(350.0, 10.0) == pytest.approx(-20.0)
    assert -180.0 < geo.signed_ang_diff(0.0, 179.0) <= 180.0


def test_bearing_to_cardinal():
    assert geo.bearing_to((0, 0), (1, 0)) == pytest.approx(0.0)       # 东
    assert geo.bearing_to((0, 0), (0, 1)) == pytest.approx(90.0)      # 北
    assert geo.bearing_to((0, 0), (-1, 0)) == pytest.approx(180.0)    # 西
    assert geo.bearing_to((0, 0), (0, -1)) == pytest.approx(270.0)    # 南


def test_unit_vector():
    ux, uy = geo.unit(0.0)
    assert (ux, uy) == pytest.approx((1.0, 0.0))
    ux, uy = geo.unit(90.0)
    assert (ux, uy) == pytest.approx((0.0, 1.0))


# --------------------------------------------------------------------------- #
# 交会定位
# --------------------------------------------------------------------------- #
def test_ray_intersect_known_point():
    # 直线 y=x（过原点，45°）与直线 y=100-x（过 (100,0)，135°）交于 (50,50)
    pt = geo.ray_intersect((0.0, 0.0), 45.0, (100.0, 0.0), 135.0)
    assert pt is not None
    assert pt == pytest.approx((50.0, 50.0))


def test_ray_intersect_parallel_returns_none():
    assert geo.ray_intersect((0.0, 0.0), 30.0, (10.0, 0.0), 30.0) is None
    assert geo.ray_intersect((0.0, 0.0), 30.0, (10.0, 0.0), 210.0) is None   # 反平行


def test_crossing_angle():
    assert geo.crossing_angle((0, 0), 0.0, (0, 0), 90.0) == pytest.approx(90.0)
    assert geo.crossing_angle((0, 0), 0.0, (0, 0), 30.0) == pytest.approx(30.0)
    # 交会角定义域 (0,90]：150° 夹角等价于 30° 交会
    assert geo.crossing_angle((0, 0), 0.0, (0, 0), 150.0) == pytest.approx(30.0)


def _exact_obs(src, pts):
    """从若干观测点看 src 的【无误差】示向观测。"""
    return [(p, geo.bearing_to(p, src)) for p in pts]


def test_least_squares_recovers_source():
    src = (300.0, 400.0)
    obs = _exact_obs(src, [(0.0, 0.0), (600.0, 0.0), (0.0, 800.0)])
    fix = geo.least_squares_fix(obs)
    assert fix is not None
    assert fix == pytest.approx(src, abs=1e-6)


def test_best_pair_fix_recovers_source():
    src = (-500.0, 250.0)
    obs = _exact_obs(src, [(0.0, 0.0), (-900.0, -100.0), (200.0, 900.0)])
    fix = geo.best_pair_fix(obs)
    assert fix is not None
    assert fix == pytest.approx(src, abs=1e-6)


def test_least_squares_needs_two():
    assert geo.least_squares_fix([((0.0, 0.0), 10.0)]) is None


# --------------------------------------------------------------------------- #
# 定位区域（±1° 楔形交）与最小包围圆
# --------------------------------------------------------------------------- #
def test_min_enclosing_circle_two_points():
    c, r = geo.min_enclosing_circle([(0.0, 0.0), (10.0, 0.0)])
    assert c == pytest.approx((5.0, 0.0))
    assert r == pytest.approx(5.0)


def test_min_enclosing_circle_covers_all():
    pts = [(0.0, 0.0), (10.0, 0.0), (5.0, 8.0), (3.0, 2.0)]
    c, r = geo.min_enclosing_circle(pts)
    for p in pts:
        assert math.hypot(p[0] - c[0], p[1] - c[1]) <= r + 1e-6


def test_region_contains_true_source():
    """核心保证：真源（用无误差示向构造的交点）必在 ±1° 定位区域最小包围圆内。"""
    src = (900.0, 0.0)
    p1, p2 = (0.0, 0.0), (0.0, -400.0)
    b1, b2 = geo.bearing_to(p1, src), geo.bearing_to(p2, src)
    cr = geo.region_center_radius(p1, b1, p2, b2)
    assert cr is not None
    c, r = cr
    assert math.hypot(src[0] - c[0], src[1] - c[1]) <= r + 1e-6
    assert r > 0.0


def test_region_radius_grows_with_distance():
    """同一交会角下，观测点离源越远 → ±1° 楔形张开越大 → 定位区域越大。"""
    def radius_at(scale):
        src = (600.0 * scale, 0.0)
        p1 = (0.0, 0.0)
        p2 = (0.0, -600.0 * scale)          # 保持约 45° 交会角的相似几何
        b1, b2 = geo.bearing_to(p1, src), geo.bearing_to(p2, src)
        return geo.region_center_radius(p1, b1, p2, b2)[1]

    assert radius_at(2.0) > radius_at(1.0) > 0.0


def test_region_radius_grows_as_crossing_angle_shrinks():
    """近平行（交会角小）→ 定位区域急剧变大（策略据此判定‘不可靠’）。"""
    src = (800.0, 0.0)
    p1 = (0.0, 0.0)
    # 好几何：p2 在源正下方 → 交会角约 90°
    good = geo.region_center_radius(p1, geo.bearing_to(p1, src),
                                    (800.0, -600.0), geo.bearing_to((800.0, -600.0), src))
    # 差几何：p2 与 p1 近乎共线看源 → 交会角很小
    bad = geo.region_center_radius(p1, geo.bearing_to(p1, src),
                                   (-200.0, 5.0), geo.bearing_to((-200.0, 5.0), src))
    assert good is not None and bad is not None
    assert bad[1] > good[1]


def test_best_region_picks_tightest():
    src = (700.0, 100.0)
    pts = [(0.0, 0.0), (700.0, -600.0), (-300.0, 20.0), (720.0, 700.0)]
    obs = _exact_obs(src, pts)
    br = geo.best_region(obs)
    assert br is not None
    c, r = br
    assert math.hypot(src[0] - c[0], src[1] - c[1]) <= r + 1e-6


# --------------------------------------------------------------------------- #
# 路径排序
# --------------------------------------------------------------------------- #
def test_ordered_tour_visits_all_once():
    pts = [(100.0, 0.0), (0.0, 100.0), (-100.0, 0.0), (0.0, -100.0), (50.0, 50.0)]
    order = geo.ordered_tour((0.0, 0.0), pts)
    assert sorted(order) == list(range(len(pts)))


def test_two_opt_not_worse_than_nearest_neighbor():
    pts = [(100.0, 0.0), (0.0, 100.0), (-100.0, 0.0), (0.0, -100.0),
           (300.0, 300.0), (-300.0, -300.0), (250.0, -50.0)]
    start = (0.0, 0.0)
    nn = geo.nearest_neighbor_order(start, pts)
    opt = geo.ordered_tour(start, pts)
    assert geo.path_length(opt, pts, start) <= geo.path_length(nn, pts, start) + 1e-6


def test_ordered_tour_empty():
    assert geo.ordered_tour((0.0, 0.0), []) == []
