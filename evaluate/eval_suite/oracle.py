"""
One-shot Oracle 策略（O1）：理想化探测器基准。

设定（比真实系统强，故作 benchmark 而非严格下界）：
- 在原点把 20 个频道各扫一次（初始频道 1）。每次扫描赋予“超能力”：不仅知道该频道有无源，
  若有则【立刻获得精确坐标】。这对应 100s 检测 + 19s 切频 = 119s（无移动，全在原点）。
- 之后已知全部源坐标，按最优 TSPN 顺序（Held-Karp 圆盘序 + 凸放置点）走到每个 20m 清除点清除。

Oracle 需要真值，只能在离线已知真值的引擎上运行。为让 InstrumentedWorld 的计时分解自然
等于 119 + L_UB/5 + 5m，本策略【真的】发出 20 次原点 measure 与 m 次带移动的 clear，
但清除点直接用真值圆盘的凸放置点（不做定位）。这样它同时是“策略”又能被同一套计时器度量。
"""

from __future__ import annotations

import numpy as np

from .interface_shim import World
from .lowerbound import (
    CLEAR_RADIUS_M,
    convex_placement_length,
    disc_distance_matrix,
    held_karp_open_path,
)

CH_MIN, CH_MAX = 1, 20


def _placement_points(order, sources, start=(0.0, 0.0), radius=CLEAR_RADIUS_M):
    """复算凸放置的实际落点（与 convex_placement_length 同问题，返回各点坐标）。"""
    pts = [sources[i] for i in order]
    n = len(pts)
    if n == 0:
        return []
    try:
        from scipy.optimize import minimize
    except Exception:
        return list(pts)                       # 退回走中心

    P = np.asarray(pts, dtype=float)
    s = np.asarray(start, dtype=float)

    def unpack(v):
        return v.reshape(n, 2)

    def obj(v):
        X = unpack(v)
        seg = np.vstack([X[0] - s, np.diff(X, axis=0)])
        return float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))

    cons = []
    for i in range(n):
        def mk(i):
            def c(v):
                X = unpack(v)
                dx = X[i] - P[i]
                return radius * radius - float(dx @ dx)
            return {"type": "ineq", "fun": c}
        cons.append(mk(i))

    res = minimize(obj, P.reshape(-1).copy(), constraints=cons, method="SLSQP",
                   options={"maxiter": 400, "ftol": 1e-7})
    X = unpack(res.x)
    # 数值保护：凸最优常落在圆盘【边界】(=radius)，引擎命中判据是 dist≤20，浮点会把
    # 边界点推到 20.0000001 → 漏清。故把每个点收进 safe=radius-2m（=18，同 agent 盲清裕量），
    # 保证必命中；对路径长影响可忽略（≤2m/点）。
    safe = max(0.0, radius - 2.0)
    out = []
    for i in range(n):
        d = X[i] - P[i]
        nrm = float(np.hypot(d[0], d[1]))
        if nrm > safe:
            X[i] = P[i] + d / nrm * safe
        out.append((float(X[i][0]), float(X[i][1])))
    return out


class OracleHunter:
    """
    One-shot Oracle 策略。需要能拿到真值的引擎（LocalWorld 包 environment.Engine）。
    reveal: dict from case.reveal()，含 jammers[{channel,x,y,...}]。
    """

    def __init__(self, reveal: dict):
        self.reveal = reveal

    def run(self, world: World) -> None:
        if not world.enter():
            return
        # 1) 原点各频道扫一次（初始频道 1 → 20 次 measure，19 次切频，全在原点无移动）
        for ch in range(CH_MIN, CH_MAX + 1):
            if world.finished:
                return
            world.measure(0.0, 0.0, ch)

        # 2) 已知真值：取有源频道坐标，Held-Karp 圆盘序 + 凸放置点，依次走到并清除
        jam = [(j["channel"], (float(j["x"]), float(j["y"]))) for j in self.reveal["jammers"]]
        if jam:
            sources = [p for _, p in jam]
            chans = [c for c, _ in jam]
            w0, W = disc_distance_matrix(sources)
            _, order = held_karp_open_path(w0, W)
            pts = _placement_points(order, sources)
            for oi, pt in zip(order, pts):
                if world.finished:
                    return
                world.clear(pt[0], pt[1], chans[oi])
        if not world.finished:
            world.exit()
