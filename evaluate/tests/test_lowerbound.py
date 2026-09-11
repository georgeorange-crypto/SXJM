"""下界证书的正确性：Held-Karp 精确性、证书区间序、退化情形。"""

import math
import random

import pytest

from eval_suite import lowerbound as lb


def _brute_open_path(w0, W):
    """暴力枚举所有排列的开放式最短路（校验 Held-Karp）。"""
    import itertools
    m = len(w0)
    best = math.inf
    for perm in itertools.permutations(range(m)):
        tot = w0[perm[0]]
        for a, b in zip(perm, perm[1:]):
            tot += W[a][b]
        best = min(best, tot)
    return best


def test_disc_matrix_basic():
    w0, W = lb.disc_distance_matrix([(100.0, 0.0), (300.0, 0.0)])
    assert w0[0] == pytest.approx(80.0)      # 100-20
    assert w0[1] == pytest.approx(280.0)     # 300-20
    assert W[0][1] == pytest.approx(160.0)   # 200-40
    assert W[0][0] == 0.0


def test_disc_matrix_overlap_clamped_zero():
    # 两源相距 30 < 40 → 圆盘重叠 → 距离 0；原点在圆盘内 → w0=0
    w0, W = lb.disc_distance_matrix([(10.0, 0.0), (30.0, 0.0)])
    assert w0[0] == 0.0
    assert W[0][1] == 0.0


@pytest.mark.parametrize("m", [1, 2, 3, 5, 7])
def test_held_karp_matches_brute(m):
    rng = random.Random(100 + m)
    srcs = [(rng.uniform(-1000, 1000), rng.uniform(-1000, 1000)) for _ in range(m)]
    w0, W = lb.disc_distance_matrix(srcs)
    Wl = W.tolist()
    val, order = lb.held_karp_open_path(w0, W)
    assert sorted(order) == list(range(m))          # 是一个排列
    assert val == pytest.approx(_brute_open_path(w0.tolist(), Wl), rel=1e-9)


def test_certificate_ordering():
    rng = random.Random(7)
    for _ in range(20):
        m = rng.randint(10, 16)
        srcs = [(rng.uniform(-1700, 1700), rng.uniform(-1700, 1700)) for _ in range(m)]
        c = lb.certificate(srcs)
        assert c.L_LB <= c.L_UB + 1e-6
        assert c.L_UB <= c.L_center + 1e-6
        assert 0.0 <= c.gap_rel < 1.0


def test_lb_is_valid_lower_bound_vs_center():
    # LB 必 ≤ 任何可行路线；中心 TSP 是可行路线 → LB ≤ center。
    rng = random.Random(11)
    srcs = [(rng.uniform(-1500, 1500), rng.uniform(-1500, 1500)) for _ in range(12)]
    c = lb.certificate(srcs)
    assert c.L_LB <= c.L_center + 1e-6


def test_empty_and_single():
    c0 = lb.certificate([])
    assert c0.m == 0 and c0.L_LB == 0.0 and c0.T_abs_lb() == 0.0
    c1 = lb.certificate([(500.0, 0.0)])
    assert c1.L_LB == pytest.approx(480.0)          # 500-20
    # 单源 T_abs_lb = 480/5 + 5*1
    assert c1.T_abs_lb() == pytest.approx(480.0 / 5 + 5.0)


def test_convex_placement_never_below_lb():
    rng = random.Random(3)
    srcs = [(rng.uniform(-1500, 1500), rng.uniform(-1500, 1500)) for _ in range(10)]
    w0, W = lb.disc_distance_matrix(srcs)
    L_LB, order = lb.held_karp_open_path(w0, W)
    L_UB = lb.convex_placement_length(order, srcs)
    assert L_UB >= L_LB - 1e-6
