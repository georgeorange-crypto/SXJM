"""覆盖布点测试：Q3 全向覆盖半径 <= Rmin；Q4 三角网可检测任意定向源。"""
import math

import pytest

from src.coverage.coverage_verify import (coverage_radius, verify_directional_cover,
                                          verify_omni_cover)
from src.coverage.q3_hex_cover import hex_cover, ring_radius_for
from src.coverage.q4_triangular_mesh import q4_mesh_cover, triangular_mesh


ARENA = 1800.0


def test_q3_hex_cover_covers_arena():
    plan = hex_cover(ARENA, r_safe=980.0, m=6)
    assert plan.covers_arena, f"hex cover failed, worst radius exceeds r_safe"
    # 覆盖半径应 <= 全向接收半径 1000
    assert verify_omni_cover(plan.points, ARENA, r_recv=1000.0)


def test_ring_radius_positive():
    rho = ring_radius_for(ARENA, 980.0, 6)
    assert rho > 0.0
    assert rho < ARENA


def test_q4_triangular_mesh_edge():
    mesh = triangular_mesh(ARENA, edge=950.0)
    assert len(mesh) > 0
    # 相邻顶点最小间距 ~edge，且 <= Rmin
    assert all(x * x + y * y <= ARENA * ARENA + 1.0 for (x, y) in mesh)


def test_q4_mesh_detects_directional_sources():
    """
    定向源覆盖：三角网 + 边界环保证【核心区】内任意朝向的定向源都可探测。

    几何事实：竞技场是凸域，边界上朝径向外的定向源，其 180° 半平面只切于边界，
    任何内部点都收不到 → 有限点无法覆盖到精确边界（100%R）。N=24 边界环把保证覆盖
    推到 arena·cos(π/24)≈0.991R；只有最外 ~15m 薄环不可覆盖，属固有几何极限。

    故断言：核心区（<=0.985R）100% 覆盖；失败仅出现在此核心之外。
    """
    plan = q4_mesh_cover(ARENA, edge=950.0)
    # 只在核心区采样：核心内必须全部可探测。
    ok, core_frac = verify_directional_cover(plan.points, ARENA * 0.985, r_recv=1000.0,
                                             pos_samples=300, dir_samples=48)
    assert ok, f"directional coverage failed inside guaranteed core at fraction {core_frac:.3f}"


def test_q4_directional_boundary_limit_is_geometric():
    """全域采样时，任何失败样本都落在最外薄环（>0.985R）——即固有几何极限处。"""
    plan = q4_mesh_cover(ARENA, edge=950.0)
    full, min_fail = verify_directional_cover(plan.points, ARENA, r_recv=1000.0,
                                              pos_samples=400, dir_samples=48)
    # 若存在失败，其最内失败半径必在核心之外（证明失败只因边界几何，非布点不足）。
    assert full or min_fail >= 0.985, f"unexpected interior coverage gap at {min_fail:.3f}R"


def test_q4_mesh_also_covers_omni():
    plan = q4_mesh_cover(ARENA, edge=950.0)
    # 三角网边长 950 → 覆盖半径 950/√3 ≈ 548 < 1000，全向也被覆盖
    assert plan.omni_covers
