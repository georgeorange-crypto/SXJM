"""路径优化测试：Held-Karp 最优性、MST 下界、2-opt 不劣于最近邻。"""
import math
import random

import pytest

from src.optimization.held_karp import held_karp_open, path_length
from src.optimization.mst_bound import mst_weight, open_tsp_lower_bound
from src.optimization.two_opt import nearest_neighbor_order, solve_route, two_opt


def _brute_open(nodes, start=0):
    from itertools import permutations
    others = [i for i in range(len(nodes)) if i != start]
    best = float("inf"); bo = None
    for perm in permutations(others):
        order = [start] + list(perm)
        L = path_length(nodes, order)
        if L < best:
            best = L; bo = order
    return best, bo


def test_held_karp_matches_bruteforce():
    rng = random.Random(7)
    for _ in range(20):
        n = rng.randint(2, 7)
        nodes = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]
        hk, _ = held_karp_open(nodes, 0)
        bf, _ = _brute_open(nodes, 0)
        assert abs(hk - bf) < 1e-6, (n, hk, bf)


def test_mst_is_lower_bound():
    rng = random.Random(11)
    for _ in range(30):
        n = rng.randint(2, 9)
        nodes = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]
        hk, _ = held_karp_open(nodes, 0)
        lb = open_tsp_lower_bound(nodes)
        assert lb <= hk + 1e-6


def test_two_opt_not_worse_than_nn():
    rng = random.Random(13)
    for _ in range(20):
        n = rng.randint(5, 20)
        nodes = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(n)]
        nn = nearest_neighbor_order(nodes, 0)
        nn_len = path_length(nodes, nn)
        opt_len, _ = two_opt(nodes, list(nn), 0)
        assert opt_len <= nn_len + 1e-6


def test_solve_route_dispatch():
    rng = random.Random(17)
    nodes = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(6)]
    L, order = solve_route(nodes, 0, held_karp_max=16)
    assert order[0] == 0
    assert len(order) == 6
    assert L >= 0.0
