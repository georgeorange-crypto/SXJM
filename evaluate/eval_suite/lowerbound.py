"""
严格物理下界 / TSPN 上下界证书（全知情形，用真值 p_1..p_m + 原点）。

四个量（单位：米，移动距离；时间在 metrics.py 里 /5 并加 5m）：
  L_LB     绝对物理下界：圆盘间最短距离矩阵上的开放式最短 Hamilton 路（固定起点原点，不返回）。
           w_0i = max(0, |p_i|-20)，w_ij = max(0, |p_i-p_j|-40)。用【精确】Held-Karp 求解。
           因真实 TSPN 从一个 20m 圆盘走到另一个，段长 ≥ 两圆盘最小间距，故 L_LB ≤ L_TSPN*，可证。
  L_center 简单可行上界：强制走到每个源【中心】的开放式 TSP（NN+2opt 近似其最优；仍是可行上界）。
  L_UB     收紧可行上界：取 Held-Karp 给出的【圆盘访问顺序】，在该固定顺序下对 x_i∈D_i
           最小化路径长（凸问题，SLSQP 求全局）。任一可行路线都是 L_TSPN* 的上界。
  证书区间 L_LB ≤ L_TSPN* ≤ L_UB(≤ L_center)。

Held-Karp 精确性对“可证下界”是必须的：近似解只会给上界，不能当下界。m≤16（附件2 §1.3），
2^16·16 DP，numpy 加速，单例毫秒~百毫秒级。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

Point = tuple[float, float]

CLEAR_RADIUS_M = 20.0          # 附件2 §2.4：清除半径（进入即可清）
SPEED_MPS = 5.0
PER_SOURCE_S = 5.0             # 3s 光学定位 + 2s 激光清除（每源必付，与策略无关）


# --------------------------------------------------------------------------- #
# 距离
# --------------------------------------------------------------------------- #
def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def disc_distance_matrix(sources: Sequence[Point], radius: float = CLEAR_RADIUS_M
                         ) -> tuple[np.ndarray, np.ndarray]:
    """
    返回 (w0, W)：
      w0[i] = max(0, |p_i| - radius)          原点(0,0)到圆盘 i 的最短移动
      W[i,j]= max(0, |p_i-p_j| - 2*radius)     圆盘 i 到圆盘 j 的最短移动
    """
    m = len(sources)
    P = np.asarray(sources, dtype=float).reshape(m, 2)
    w0 = np.maximum(0.0, np.hypot(P[:, 0], P[:, 1]) - radius)
    diff = P[:, None, :] - P[None, :, :]
    D = np.hypot(diff[:, :, 0], diff[:, :, 1])
    W = np.maximum(0.0, D - 2.0 * radius)
    np.fill_diagonal(W, 0.0)
    return w0, W


# --------------------------------------------------------------------------- #
# 精确开放式最短 Hamilton 路（固定起点，不返回）—— Held-Karp
# --------------------------------------------------------------------------- #
def held_karp_open_path(w0: np.ndarray, W: np.ndarray
                        ) -> tuple[float, list[int]]:
    """
    dp[mask][j] = 从起点出发、恰好访问集合 mask、终于 j 的最短路。
    base dp[{j}][j]=w0[j]；转移 dp[mask|k][k]=min_j dp[mask][j]+W[j][k]。
    返回 (最短总长, 访问顺序 list[int])。m=0 → (0, [])。
    """
    m = len(w0)
    if m == 0:
        return 0.0, []
    if m == 1:
        return float(w0[0]), [0]

    INF = math.inf
    full = (1 << m) - 1
    dp = np.full((1 << m, m), INF, dtype=float)
    parent = np.full((1 << m, m), -1, dtype=np.int32)
    for j in range(m):
        dp[1 << j, j] = w0[j]

    for mask in range(1, 1 << m):
        row = dp[mask]
        # 该 mask 下从内部到达各终点 j 的成本（用于扩展到 mask|k）
        finite = np.isfinite(row)
        if not finite.any():
            continue
        # cand[k] = min_j (row[j] + W[j,k])，只在 j∈mask 上取 min
        base = np.where(finite, row, INF)
        cost_to_k = base[:, None] + W          # (m from-j, m to-k)
        cand = cost_to_k.min(axis=0)           # (m,)
        arg = cost_to_k.argmin(axis=0)
        for k in range(m):
            if mask & (1 << k):
                continue
            nm = mask | (1 << k)
            if cand[k] < dp[nm, k]:
                dp[nm, k] = cand[k]
                parent[nm, k] = arg[k]

    end = int(np.argmin(dp[full]))
    best = float(dp[full, end])

    order: list[int] = []
    mask, j = full, end
    while j != -1:
        order.append(j)
        pj = int(parent[mask, j])
        mask ^= (1 << j)
        j = pj
    order.reverse()
    return best, order


# --------------------------------------------------------------------------- #
# 中心 TSP 可行上界（NN + 2opt）
# --------------------------------------------------------------------------- #
def _open_tour_nn_2opt(start: Point, pts: Sequence[Point]) -> list[int]:
    n = len(pts)
    if n == 0:
        return []
    remaining = list(range(n))
    order: list[int] = []
    cur = start
    while remaining:
        j = min(remaining, key=lambda i: _dist(cur, pts[i]))
        order.append(j)
        cur = pts[j]
        remaining.remove(j)

    def plen(o: list[int]) -> float:
        tot, cur = 0.0, start
        for i in o:
            tot += _dist(cur, pts[i])
            cur = pts[i]
        return tot

    best = order[:]
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for k in range(i + 1, len(best)):
                cand = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
                if plen(cand) + 1e-9 < plen(best):
                    best, improved = cand, True
    return best


def center_tour_length(sources: Sequence[Point], start: Point = (0.0, 0.0)) -> float:
    """访问每个源中心的开放式 TSP 长（NN+2opt 近似）。走到中心必进 20m → 可行上界。"""
    order = _open_tour_nn_2opt(start, sources)
    tot, cur = 0.0, start
    for i in order:
        tot += _dist(cur, sources[i])
        cur = sources[i]
    return tot


# --------------------------------------------------------------------------- #
# 固定顺序下的凸放置上界 L_UB（x_i ∈ D_i）
# --------------------------------------------------------------------------- #
def convex_placement_length(order: Sequence[int], sources: Sequence[Point],
                            start: Point = (0.0, 0.0),
                            radius: float = CLEAR_RADIUS_M) -> float:
    """
    固定圆盘访问顺序 order，最小化 |x_{o0}-start| + Σ|x_{o(k+1)}-x_{ok}|，约束 |x_i-p_i|≤radius。
    凸问题；优先用 scipy SLSQP（全局最优），不可用则退回中心值（=走中心，仍是可行上界）。
    返回该顺序下的最优可行路径长（L_TSPN* 的上界）。
    """
    pts = [sources[i] for i in order]
    n = len(pts)
    if n == 0:
        return 0.0

    try:
        from scipy.optimize import minimize
    except Exception:
        # 退回：走中心
        tot, cur = 0.0, start
        for p in pts:
            tot += _dist(cur, p)
            cur = p
        return tot

    P = np.asarray(pts, dtype=float)          # (n,2) 顺序后的中心
    s = np.asarray(start, dtype=float)

    def unpack(v):
        return v.reshape(n, 2)

    def obj(v):
        X = unpack(v)
        seg = np.vstack([X[0] - s, np.diff(X, axis=0)])
        return float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))

    def obj_grad(v):
        X = unpack(v)
        pts_seq = np.vstack([s, X])            # (n+1,2)
        d = np.diff(pts_seq, axis=0)           # (n,2) 段向量 X_k - prev
        L = np.hypot(d[:, 0], d[:, 1])
        L = np.where(L < 1e-12, 1e-12, L)
        u = d / L[:, None]                     # 单位段向量
        g = np.zeros((n, 2))
        # X_k 出现在段 k（+u_k）和段 k+1（-u_{k+1}）
        g += u
        g[:-1] -= u[1:]
        return g.reshape(-1)

    # 圆盘约束：radius^2 - |X_i - p_i|^2 >= 0（凸可行域，用平方形式利于 SLSQP）
    cons = []
    for i in range(n):
        def mk(i):
            def c(v):
                X = unpack(v)
                dx = X[i] - P[i]
                return radius * radius - float(dx @ dx)

            def cj(v):
                X = unpack(v)
                g = np.zeros((n, 2))
                g[i] = -2.0 * (X[i] - P[i])
                return g.reshape(-1)
            return {"type": "ineq", "fun": c, "jac": cj}
        cons.append(mk(i))

    x0 = P.reshape(-1).copy()
    res = minimize(obj, x0, jac=obj_grad, constraints=cons, method="SLSQP",
                   options={"maxiter": 400, "ftol": 1e-7})
    val = obj(res.x)
    center_val = obj(x0)
    # SLSQP 是可行上界求解；若数值异常给出比中心还差的值，则回退中心（仍可行）
    return float(min(val, center_val)) if math.isfinite(val) else float(center_val)


# --------------------------------------------------------------------------- #
# 汇总证书
# --------------------------------------------------------------------------- #
@dataclass
class TSPNCertificate:
    m: int
    L_LB: float                    # 严格下界（Held-Karp on disc matrix）
    L_UB: float                    # 收紧可行上界（凸放置）
    L_center: float                # 简单可行上界（走中心）
    lb_order: list[int]            # LB 的圆盘访问顺序（下标）
    gap_abs: float                 # L_UB - L_LB
    gap_rel: float                 # (L_UB - L_LB) / L_UB

    @property
    def T_move_LB(self) -> float:
        return self.L_LB / SPEED_MPS

    def T_abs_lb(self) -> float:
        """绝对物理时间下界 = L_LB/5 + 5m。"""
        return self.L_LB / SPEED_MPS + PER_SOURCE_S * self.m

    def T_move_UB(self) -> float:
        return self.L_UB / SPEED_MPS


def certificate(sources: Sequence[Point], start: Point = (0.0, 0.0),
                radius: float = CLEAR_RADIUS_M) -> TSPNCertificate:
    """给定真源位置，算出 L_LB / L_UB / L_center 及证书区间。"""
    m = len(sources)
    if m == 0:
        return TSPNCertificate(0, 0.0, 0.0, 0.0, [], 0.0, 0.0)
    w0, W = disc_distance_matrix(sources, radius)
    L_LB, order = held_karp_open_path(w0, W)
    L_UB = convex_placement_length(order, sources, start, radius)
    L_center = center_tour_length(sources, start)
    L_UB = min(L_UB, L_center)                 # 两个可行上界取更紧者
    L_UB = max(L_UB, L_LB)                      # 数值保护：上界不得低于下界
    gap_abs = L_UB - L_LB
    gap_rel = gap_abs / L_UB if L_UB > 1e-9 else 0.0
    return TSPNCertificate(m, L_LB, L_UB, L_center, order, gap_abs, gap_rel)
