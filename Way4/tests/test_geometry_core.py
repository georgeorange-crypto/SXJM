"""sxjm_core.geometry unit + property tests (Way4/DESIGN.md M1, §15).

The headline property (DESIGN.md §15: "真源永不被 sound update 排除") is tested
here at the geometry level: a channel's feasible polygon built from ``±(1°+ε)``
bearing wedges always contains the true source, for any legal measurement error.
"""

import math
import random

from sxjm_core.geometry import (
    Halfplane,
    angle_sep_deg,
    arena_polygon,
    bearing_deg,
    circle_circle_intersections,
    circumcircle,
    clip_polygon,
    convex_hull,
    dist,
    halfplane_intersection,
    in_arena,
    min_enclosing_circle,
    norm_deg,
    point_in_wedge,
    polygon_area,
    polygon_diameter,
    wedge_halfplanes,
)

ARENA = 1800.0
EPS_NUM_DEG = 1e-6


# --- angles ------------------------------------------------------------------


def test_norm_deg_range():
    for d in (-720.0, -1.0, 0.0, 359.9, 360.0, 361.0, 1080.0):
        n = norm_deg(d)
        assert 0.0 <= n < 360.0
    assert norm_deg(-1.0) == 359.0
    assert norm_deg(360.0) == 0.0


def test_angle_sep_wraparound():
    assert math.isclose(angle_sep_deg(350.0, 10.0), 20.0)
    assert math.isclose(angle_sep_deg(10.0, 350.0), 20.0)
    assert math.isclose(angle_sep_deg(0.0, 180.0), 180.0)
    assert math.isclose(angle_sep_deg(90.0, 90.0), 0.0)


def test_bearing_deg_cardinals():
    o = (0.0, 0.0)
    assert math.isclose(bearing_deg(o, (1.0, 0.0)), 0.0)
    assert math.isclose(bearing_deg(o, (0.0, 1.0)), 90.0)
    assert math.isclose(bearing_deg(o, (-1.0, 0.0)), 180.0)
    assert math.isclose(bearing_deg(o, (0.0, -1.0)), 270.0)


# --- wedge -------------------------------------------------------------------


def test_point_in_wedge_boundary_inclusive():
    apex = (0.0, 0.0)
    # bearing 45 exactly on the +1 deg edge of a wedge centred at 44
    q = (math.cos(math.radians(45.0)), math.sin(math.radians(45.0)))
    assert point_in_wedge(apex, 44.0, 1.0, q) is True  # boundary inclusive
    assert point_in_wedge(apex, 43.9, 1.0, q) is False  # just outside


def test_wedge_halfplanes_match_membership():
    rng = random.Random(3)
    apex = (120.0, -60.0)
    center = 200.0
    half = 1.0
    hps = wedge_halfplanes(apex, center, half)
    for _ in range(4000):
        q = (rng.uniform(-2000, 2000), rng.uniform(-2000, 2000))
        in_hp = all(hp.contains(q, eps=1e-6) for hp in hps)
        in_wedge = point_in_wedge(apex, center, half, q, eps_deg=1e-6)
        # The two characterisations must agree away from the razor-thin boundary.
        b = bearing_deg(apex, q)
        if abs(angle_sep_deg(b, center) - half) > 1e-3:
            assert in_hp == in_wedge, (q, b, in_hp, in_wedge)


# --- arena polygon superset --------------------------------------------------


def test_arena_polygon_is_superset_of_disc():
    poly = arena_polygon(ARENA, n_sides=96)
    # Every disc-boundary point must be inside the polygon (superset).
    hps_ok = True
    for k in range(720):
        ang = 2.0 * math.pi * k / 720
        p = (ARENA * math.cos(ang), ARENA * math.sin(ang))
        # point-in-convex-polygon via winding sign against every edge
        inside = _in_convex_polygon(p, poly, eps=1e-6)
        hps_ok = hps_ok and inside
    assert hps_ok
    # ...but not absurdly loose: circumscribed 96-gon overshoots area < 0.2%.
    assert polygon_area(poly) < math.pi * ARENA * ARENA * 1.002


def _in_convex_polygon(p, poly, eps=1e-9):
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        cr = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if cr > eps:
            s = 1
        elif cr < -eps:
            s = -1
        else:
            s = 0
        if s != 0:
            if sign == 0:
                sign = s
            elif s != sign:
                return False
    return True


# --- feasible-set construction & THE property --------------------------------


def _feasible_polygon(source, obs_points, errs):
    """Build F_c from bearing wedges at obs_points with the given signed errors."""
    hps = []
    for s, e in zip(obs_points, errs):
        true_b = bearing_deg(s, source)
        svd = norm_deg(true_b + e)  # measured within +-1 deg of the truth
        hps.extend(wedge_halfplanes(s, svd, 1.0 + EPS_NUM_DEG))
    return halfplane_intersection(hps, arena_polygon(ARENA))


def test_true_source_never_excluded_property():
    """For random legal sources and any +-1 deg errors, the feasible polygon
    built with the +(1+eps) margin always contains the true source."""
    rng = random.Random(2024)
    for _ in range(3000):
        ang = rng.uniform(0, 2 * math.pi)
        rad = ARENA * math.sqrt(rng.random())
        source = (rad * math.cos(ang), rad * math.sin(ang))
        k = rng.randint(2, 4)
        obs = []
        while len(obs) < k:
            q = (rng.uniform(-2200, 2200), rng.uniform(-2200, 2200))
            if dist(q, source) > 50.0:  # avoid degenerate near-apex bearings
                obs.append(q)
        errs = [rng.uniform(-1.0, 1.0) for _ in range(k)]
        fc = _feasible_polygon(source, obs, errs)
        assert fc, "feasible polygon unexpectedly empty"
        assert _in_convex_polygon(source, fc, eps=1e-4), (
            f"TRUE SOURCE EXCLUDED: {source}"
        )


def test_two_bearings_bound_the_region():
    # Two well-separated bearings must yield a bounded (finite-diameter) F_c.
    source = (300.0, 400.0)
    fc = _feasible_polygon(source, [(0.0, 0.0), (700.0, -200.0)], [0.3, -0.4])
    assert fc
    assert polygon_diameter(fc) < 2.0 * ARENA * 1.01  # bounded by the arena


# --- convex hull -------------------------------------------------------------


def test_convex_hull_contains_all_and_is_convex():
    rng = random.Random(7)
    pts = [(rng.uniform(-100, 100), rng.uniform(-100, 100)) for _ in range(200)]
    hull = convex_hull(pts)
    assert len(hull) >= 3
    for p in pts:
        assert _in_convex_polygon(p, hull, eps=1e-6)
    # CCW convexity: every consecutive turn is a left turn.
    n = len(hull)
    for i in range(n):
        a, b, c = hull[i], hull[(i + 1) % n], hull[(i + 2) % n]
        cr = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        assert cr >= -1e-6


# --- MEC vs brute-force oracle ----------------------------------------------


def _brute_mec(points, eps=1e-6):
    pts = list({(float(x), float(y)) for x, y in points})
    n = len(pts)
    if n == 0:
        return ((0.0, 0.0), 0.0)
    if n == 1:
        return (pts[0], 0.0)

    def covers(center, radius):
        return all(dist(center, p) <= radius + eps for p in pts)

    best = None
    for i in range(n):
        for j in range(i + 1, n):
            c = (0.5 * (pts[i][0] + pts[j][0]), 0.5 * (pts[i][1] + pts[j][1]))
            r = 0.5 * dist(pts[i], pts[j])
            if covers(c, r) and (best is None or r < best[1]):
                best = (c, r)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                cc = circumcircle(pts[i], pts[j], pts[k])
                if cc and covers(cc[0], cc[1]) and (best is None or cc[1] < best[1]):
                    best = cc
    return best


def test_mec_matches_brute_force():
    rng = random.Random(11)
    for _ in range(300):
        n = rng.randint(1, 10)
        pts = [(rng.uniform(-500, 500), rng.uniform(-500, 500)) for _ in range(n)]
        (_, r_welzl) = min_enclosing_circle(pts)
        (_, r_brute) = _brute_mec(pts)
        assert math.isclose(r_welzl, r_brute, rel_tol=1e-4, abs_tol=1e-3), (
            r_welzl, r_brute, pts,
        )


def test_mec_covers_all_points():
    rng = random.Random(13)
    for _ in range(200):
        pts = [(rng.uniform(-1800, 1800), rng.uniform(-1800, 1800)) for _ in range(rng.randint(2, 40))]
        c, r = min_enclosing_circle(pts)
        for p in pts:
            assert dist(c, p) <= r + 1e-6


# --- circle-circle intersection ----------------------------------------------


def test_circle_circle_intersections_cases():
    # Two unit circles centred at (-1,0) and (1,0): cross at (0, +-sqrt(... )) -> actually
    # distance 2, r=1.5 each -> two points on x=0.
    pts = circle_circle_intersections((-1.0, 0.0), 1.5, (1.0, 0.0), 1.5)
    assert len(pts) == 2
    for p in pts:
        assert math.isclose(dist(p, (-1.0, 0.0)), 1.5, abs_tol=1e-9)
        assert math.isclose(dist(p, (1.0, 0.0)), 1.5, abs_tol=1e-9)
    # Disjoint
    assert circle_circle_intersections((0.0, 0.0), 1.0, (10.0, 0.0), 1.0) == []
    # One inside the other
    assert circle_circle_intersections((0.0, 0.0), 5.0, (0.0, 0.0), 1.0) == []
    # Tangent (external): distance 2, radii 1 and 1 -> single point at origin
    tang = circle_circle_intersections((-1.0, 0.0), 1.0, (1.0, 0.0), 1.0)
    assert len(tang) == 1
    assert math.isclose(tang[0][0], 0.0, abs_tol=1e-6)


def test_clip_polygon_removes_outside():
    square = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    # keep x <= 0
    hp = Halfplane(1.0, 0.0, 0.0)
    clipped = clip_polygon(square, hp)
    assert clipped
    for p in clipped:
        assert p[0] <= 1e-9
    assert math.isclose(polygon_area(clipped), 2.0, abs_tol=1e-6)
