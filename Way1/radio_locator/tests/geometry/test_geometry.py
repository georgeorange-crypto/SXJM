"""几何基元测试：角度环绕、半平面楔形、MEC 包含性、box 距离/方位界、圆周区间。"""
import math

import pytest

from src.geometry.angle import (angle_in_arc, norm_deg, wrap_2pi, wrap_pi)
from src.geometry.box_distance import Box, bearing_interval, distance_max, distance_min
from src.geometry.circular_interval import CircularIntervalSet
from src.geometry.halfplane import wedge_halfplanes
from src.geometry.mec import min_enclosing_circle
from src.geometry.wedge import Wedge, box_wedge_relation


# ---------- 角度 ----------
def test_wrap_pi_range():
    for d in range(-720, 721, 7):
        t = wrap_pi(math.radians(d))
        assert -math.pi - 1e-9 < t <= math.pi + 1e-9


def test_wrap_2pi_range():
    for d in range(-720, 721, 7):
        t = wrap_2pi(math.radians(d))
        assert 0.0 <= t < 2 * math.pi + 1e-12


def test_angle_in_arc_wrap_zero():
    # 350°..20° 跨零弧
    lo = math.radians(350)
    hi = math.radians(20)
    assert angle_in_arc(math.radians(0), lo, hi)
    assert angle_in_arc(math.radians(10), lo, hi)
    assert angle_in_arc(math.radians(355), lo, hi)
    assert not angle_in_arc(math.radians(180), lo, hi)


# ---------- 楔形 ----------
def test_wedge_contains_center_direction():
    w = Wedge((0.0, 0.0), 45.0, 1.0)
    # 45° 方向上的点应在楔内
    p = (100.0 * math.cos(math.radians(45)), 100.0 * math.sin(math.radians(45)))
    assert w.contains_point(p)
    # 偏出 2° 的点应在楔外
    q = (100.0 * math.cos(math.radians(48)), 100.0 * math.sin(math.radians(48)))
    assert not w.contains_point(q)


def test_wedge_wrap_zero_degree():
    # 中线 0°，±1°；359° 与 1° 都应在楔内
    w = Wedge((0.0, 0.0), 0.0, 1.0)
    for ang in (0.0, 0.9, 359.2):
        p = (100.0 * math.cos(math.radians(ang)), 100.0 * math.sin(math.radians(ang)))
        assert w.contains_point(p), ang


def test_wedge_halfplanes_intersection():
    hlo, hhi = wedge_halfplanes((0.0, 0.0), 90.0, 1.0)
    p_in = (0.0, 100.0)  # 正北，90°
    assert hlo.contains(p_in) and hhi.contains(p_in)


# ---------- MEC ----------
def test_mec_contains_all_points():
    import random
    rng = random.Random(1)
    for _ in range(200):
        pts = [(rng.uniform(-1000, 1000), rng.uniform(-1000, 1000))
               for _ in range(rng.randint(1, 30))]
        c = min_enclosing_circle(pts)
        for p in pts:
            assert math.hypot(p[0] - c.center[0], p[1] - c.center[1]) <= c.radius + 1e-6


def test_mec_two_points_diameter():
    c = min_enclosing_circle([(0.0, 0.0), (10.0, 0.0)])
    assert abs(c.radius - 5.0) < 1e-9
    assert abs(c.center[0] - 5.0) < 1e-9 and abs(c.center[1]) < 1e-9


# ---------- box 距离/方位 ----------
def test_box_distance_bounds():
    b = Box(10.0, 10.0, 20.0, 20.0)
    s = (0.0, 0.0)
    dmin = distance_min(s, b)
    dmax = distance_max(s, b)
    assert abs(dmin - math.hypot(10, 10)) < 1e-9
    assert abs(dmax - math.hypot(20, 20)) < 1e-9
    # 内部点距离 0
    assert distance_min((15.0, 15.0), b) == 0.0


def test_bearing_interval_covers_corners():
    b = Box(10.0, 10.0, 20.0, 20.0)
    s = (0.0, 0.0)
    lo, hi, contains = bearing_interval(s, b)
    assert not contains
    # 四角方位都应落在 [lo,hi] 弧内
    for c in b.corners():
        ang = wrap_2pi(math.atan2(c[1] - s[1], c[0] - s[0]))
        rel = wrap_2pi(ang - lo)
        span = wrap_2pi(hi - lo)
        assert rel <= span + 1e-9


def test_box_wedge_outside_inside():
    # 楔形指向 +x，box 在 +y 远处 → OUTSIDE
    w = Wedge((0.0, 0.0), 0.0, 1.0)
    b_far = Box(-10.0, 1000.0, 10.0, 1010.0)
    assert box_wedge_relation(w, b_far) == "OUTSIDE"
    # box 恰在 +x 轴远处小盒 → INSIDE
    b_on = Box(1000.0, -1.0, 1010.0, 1.0)
    rel = box_wedge_relation(w, b_on)
    assert rel in ("INSIDE", "UNKNOWN")  # 取决于 ±1° 宽度


# ---------- 圆周区间 ----------
def test_circular_interval_from_center_half_180():
    s = CircularIntervalSet.from_center_half(math.radians(90), math.pi / 2)
    # 180° 弧覆盖 0°..180°
    assert s.contains(math.radians(0))
    assert s.contains(math.radians(90))
    assert s.contains(math.radians(180))
    assert not s.contains(math.radians(270))


def test_circular_interval_intersection_difference():
    a = CircularIntervalSet.from_center_half(0.0, math.pi / 2)       # -90..90
    b = CircularIntervalSet.from_center_half(math.pi / 2, math.pi / 2)  # 0..180
    inter = a.intersection(b)
    assert inter.contains(math.radians(45))
    assert not inter.contains(math.radians(135))
    assert not inter.contains(math.radians(-45))
    diff = a.difference(b)
    assert diff.contains(math.radians(-45))
    assert not diff.contains(math.radians(45))


def test_circular_interval_full_empty():
    assert CircularIntervalSet.full().is_full()
    assert CircularIntervalSet.empty().is_empty()
    assert CircularIntervalSet.full().difference(CircularIntervalSet.full()).is_empty()
    assert abs(CircularIntervalSet.full().total_length() - 2 * math.pi) < 1e-9


def test_circular_interval_wrap_zero():
    # 中心 0，半宽 30° → 330°..30°
    s = CircularIntervalSet.from_center_half(0.0, math.radians(30))
    assert s.contains(math.radians(350))
    assert s.contains(math.radians(10))
    assert not s.contains(math.radians(90))
