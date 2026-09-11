"""
圆周区间集合（CircularIntervalSet）—— Q4 方向可行集的基础设施。

方向 φ 是圆周量（mod 2π），普通区间无法表示跨零点（如 350°~20°）。
本类将方向可行集表示为 [0,2π) 上若干【不相交、按序】的闭弧之并，
支持 union / intersection / difference / is_empty / contains。

约定：
- 每个弧 (a, b) 表示从 a 逆时针到 b 的闭弧，弧长 b-a ∈ [0, 2π]。
- 内部规范化：所有弧的端点在 [0,2π)，整圆用特殊标记 full。
- 半圆及跨零点是 Q4 最易出 bug 处，务必配合 test_circular_interval 使用。

§43 强调：这是 Q4 最容易藏 bug 的地方，单测必须充分。
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

TWO_PI = 2.0 * math.pi
_EPS = 1e-9


def _wrap(a: float) -> float:
    x = math.fmod(a, TWO_PI)
    if x < 0.0:
        x += TWO_PI
    if x >= TWO_PI:
        x -= TWO_PI
    return x


class CircularIntervalSet:
    """[0,2π) 上闭弧之并。内部用规范化的 (start, end) 列表，start<=end，均在 [0,2π]。"""

    def __init__(self, arcs: Optional[List[Tuple[float, float]]] = None, full: bool = False, empty: bool = False):
        self._full = full
        self._arcs: List[Tuple[float, float]] = []
        if full:
            return
        if empty or arcs is None:
            return
        raw = []
        for (a, b) in arcs:
            raw.extend(self._split_wrap(a, b))
        self._arcs = self._merge(raw)
        # 合并后若覆盖整圈，标记 full
        if self._covers_full(self._arcs):
            self._full = True
            self._arcs = []

    # ---------- 构造器 ----------
    @classmethod
    def empty(cls) -> "CircularIntervalSet":
        return cls(empty=True)

    @classmethod
    def full(cls) -> "CircularIntervalSet":
        return cls(full=True)

    @classmethod
    def from_center_half(cls, center: float, half_width: float) -> "CircularIntervalSet":
        """
        以 center 为中心、半宽 half_width（弧度）的闭弧，弧长 2*half_width。
        half_width>=π 即覆盖整圆；half_width<=0 为空。
        定向源覆盖角 half_width=π/2（→ 180° 弧）是主要用例。
        """
        if half_width <= 0.0:
            return cls.empty()
        if half_width >= math.pi - _EPS:
            return cls.full()
        return cls([(center - half_width, center + half_width)])

    # ---------- 内部工具 ----------
    @staticmethod
    def _split_wrap(a: float, b: float) -> List[Tuple[float, float]]:
        """把任意 (a,b)（弧长 b-a）拆成不跨零的规范弧。"""
        length = b - a
        if length <= 0.0:
            return []
        if length >= TWO_PI - _EPS:
            return [(0.0, TWO_PI)]
        a2 = _wrap(a)
        end = a2 + length
        if end <= TWO_PI + _EPS:
            return [(a2, min(end, TWO_PI))]
        # 跨零点：拆两段
        return [(a2, TWO_PI), (0.0, _wrap(end))]

    @staticmethod
    def _covers_full(arcs: List[Tuple[float, float]]) -> bool:
        if not arcs:
            return False
        total = sum(b - a for a, b in arcs)
        return total >= TWO_PI - 1e-6

    @staticmethod
    def _merge(arcs: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """合并重叠/相邻弧（不跨零的规范弧列表）。"""
        segs = [s for s in arcs if s[1] - s[0] > _EPS]
        if not segs:
            return []
        segs.sort()
        merged = [list(segs[0])]
        for a, b in segs[1:]:
            if a <= merged[-1][1] + _EPS:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        # 处理 0 与 2π 端点接合（跨零合并）
        if len(merged) >= 2 and merged[0][0] <= _EPS and merged[-1][1] >= TWO_PI - _EPS:
            # 首段起于0、尾段止于2π → 它们在圆周上相连；用旋转表示仍分开存储，
            # 但 contains 判定按圆周处理，这里保留分段。
            pass
        return [(a, b) for a, b in merged]

    # ---------- 查询 ----------
    def is_empty(self) -> bool:
        return (not self._full) and (len(self._arcs) == 0)

    def is_full(self) -> bool:
        return self._full

    def contains(self, theta: float) -> bool:
        if self._full:
            return True
        t = _wrap(theta)
        for a, b in self._arcs:
            if a - _EPS <= t <= b + _EPS:
                return True
            # 跨零：尾弧止于2π、首弧起于0 时，t 接近 0/2π 都算
        # 额外检查 t 以 2π 形式落入止于2π的弧
        for a, b in self._arcs:
            if b >= TWO_PI - _EPS and abs(t) <= _EPS:
                return True
        return False

    def total_length(self) -> float:
        if self._full:
            return TWO_PI
        return sum(b - a for a, b in self._arcs)

    def arcs(self) -> List[Tuple[float, float]]:
        """返回规范弧列表（只读快照）。full 返回 [(0,2π)]。"""
        if self._full:
            return [(0.0, TWO_PI)]
        return list(self._arcs)

    # ---------- 集合运算 ----------
    def union(self, other: "CircularIntervalSet") -> "CircularIntervalSet":
        if self._full or other._full:
            return CircularIntervalSet.full()
        return CircularIntervalSet(self._arcs + other._arcs)

    def intersection(self, other: "CircularIntervalSet") -> "CircularIntervalSet":
        if self._full:
            return CircularIntervalSet(other._arcs, full=other._full)
        if other._full:
            return CircularIntervalSet(self._arcs, full=self._full)
        out: List[Tuple[float, float]] = []
        for a1, b1 in self._arcs:
            for a2, b2 in other._arcs:
                lo = max(a1, a2)
                hi = min(b1, b2)
                if hi - lo > _EPS:
                    out.append((lo, hi))
        return CircularIntervalSet(out)

    def difference(self, other: "CircularIntervalSet") -> "CircularIntervalSet":
        """self \\ other。"""
        if other._full:
            return CircularIntervalSet.empty()
        if self._full:
            base = [(0.0, TWO_PI)]
        else:
            base = list(self._arcs)
        if not other._arcs:
            return CircularIntervalSet(base, full=self._full)
        result = base
        for a2, b2 in other._arcs:
            nxt: List[Tuple[float, float]] = []
            for a1, b1 in result:
                # a1..b1 减去 a2..b2
                if b2 <= a1 + _EPS or a2 >= b1 - _EPS:
                    nxt.append((a1, b1))  # 不相交
                    continue
                if a2 > a1 + _EPS:
                    nxt.append((a1, min(a2, b1)))
                if b2 < b1 - _EPS:
                    nxt.append((max(b2, a1), b1))
            result = [s for s in nxt if s[1] - s[0] > _EPS]
        return CircularIntervalSet(result)

    def __repr__(self) -> str:
        if self._full:
            return "CircularIntervalSet(FULL)"
        if not self._arcs:
            return "CircularIntervalSet(EMPTY)"
        segs = ", ".join(f"[{math.degrees(a):.2f},{math.degrees(b):.2f}]" for a, b in self._arcs)
        return f"CircularIntervalSet({segs})"
