"""
Q4 定向假设可行集：解析半径消元 (R=L(x)) + 圆周方向区间 Φ(x)。

物理（定向源，覆盖角 180°，半张角 90°）：
  收到信号 ⟺ dist(x, 接收点) <= R_eff 且 接收点在源的锥内
            ⟺ R_eff >= d 且 φ ∈ I⁺ = { φ : |∠(接收点−源) − φ| <= 90° }（一段 180° 弧）。
  no_signal ⟺ dist > R_eff  或  接收点在锥外（φ ∉ I⁺）。   ← 与全向的关键区别

解析消元（消去 R_eff∈[Rmin,Rmax]）：
  L(x)=max(Rmin, max_{i∈P} ‖x−xᵢ‖)。取 R_eff=L(x)（最小可行半径 → 对负观测最宽松）。
  位置 x 定向可行 ⟺ L(x) <= Rmax 且 方向可行集 Φ(x) ≠ ∅，其中
    Φ(x) = (∩_{i∈P} Iᵢ⁺(x)) \ (∪_{j∈N: dⱼ(x)<=L(x)} Iⱼ⁺(x))。
  （dⱼ<=L(x) 的负观测无法用“超距”解释，只能靠“出锥”，故必须避开 Iⱼ⁺。）

box 层保守判定（真源永不误删）——用【方位角区间】把 Iᵢ⁺(x) 随 x 变化的部分包住：
  记从 box 内点 x 看接收点 q 的方向 ∠(q−x) 在 B 上的区间为 [α₀,α₀+w]。
  Iⁿ_super(B)=∪_{x∈B} arc(∠(q−x),90°)  —— 超集（用于 OUTSIDE 判定的正锥交、INSIDE 的负锥并）。
  Iⁿ_sub(B)  =∩_{x∈B} arc(∠(q−x),90°)  —— 子集（用于 OUTSIDE 的负锥并、INSIDE 的正锥交）。
  Φ_super(B) ⊇ ∪_{x∈B}Φ(x)：为空 ⟹ 整盒不可行 ⟹ OUTSIDE。
  Φ_sub(B)   ⊆ ∩_{x∈B}Φ(x)：非空且位置/半径全可行 ⟹ 整盒可行 ⟹ INSIDE。

位置楔形约束（示向度指向源，定向源同样成立）与 near 5m 圆盘、竞技场约束，做法同 Q3。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..domain.observation import Observation
from ..domain.types import ObservationType, ProblemConstants, RobotConstants, Vec2
from ..geometry.box_distance import Box, bearing_interval, distance_max, distance_min
from ..geometry.circular_interval import TWO_PI, CircularIntervalSet
from ..geometry.mec import Circle
from ..geometry.rotating_calipers import diameter as poly_diameter
from ..geometry.wedge import Wedge, box_wedge_relation
from .quadtree import INSIDE, OUTSIDE, UNKNOWN, QuadTree, pad_box

_EPS = 1e-9
_HALF_CONE = math.pi / 2.0   # 定向源半张角 90°


# ---------- 方向锥的 box 级 超集/子集 ----------
def _dir_arc_params(b: Box, q: Vec2) -> Tuple[float, float, bool]:
    """
    从 box 内点 x 看固定点 q 的方向 ∠(q−x)：返回 (base, width, contains_q)。
    方向区间为 CCW 弧 [base, base+width]。contains_q 时 q 在盒内，方向覆盖整圆。
    依据 ∠(q−x) = ∠(x−q) + π，∠(x−q) 的区间由 bearing_interval(q, b) 给出。
    """
    lo, hi, contains = bearing_interval(q, b)
    if contains:
        return 0.0, TWO_PI, True
    w = (hi - lo) % TWO_PI
    base = lo + math.pi
    return base, w, False


def _cone_super(b: Box, q: Vec2, half: float = _HALF_CONE) -> CircularIntervalSet:
    """∪_{x∈B} arc(center=∠(q−x), half)：把中心的移动 + ±half 都并进来。"""
    base, w, contains = _dir_arc_params(b, q)
    if contains:
        return CircularIntervalSet.full()
    length = w + 2.0 * half
    if length >= TWO_PI - _EPS:
        return CircularIntervalSet.full()
    a = base - half
    return CircularIntervalSet([(a, a + length)])


def _cone_sub(b: Box, q: Vec2, half: float = _HALF_CONE) -> CircularIntervalSet:
    """∩_{x∈B} arc(center=∠(q−x), half)：两端点弧之交（滑动 180° 弧的公共部分）。"""
    base, w, contains = _dir_arc_params(b, q)
    if contains:
        return CircularIntervalSet.empty()
    if w >= 2.0 * half - _EPS:
        return CircularIntervalSet.empty()
    arc_lo = CircularIntervalSet.from_center_half(base, half)
    arc_hi = CircularIntervalSet.from_center_half(base + w, half)
    return arc_lo.intersection(arc_hi)


@dataclass
class Q4DirectionalFeasibleSet:
    """单个频道“定向源假设”下的位置可行集。接口与 Q3FeasibleSet 平行。"""
    problem: ProblemConstants
    robot: RobotConstants
    half_deg: float = 1.0
    coarse_size: float = 10.0
    fine_size: float = 0.5
    max_depth: int = 16
    delete_margin: float = 1e-3

    tree: Optional[QuadTree] = None
    mec_center: Optional[Vec2] = None
    mec_radius: float = float("inf")
    diameter: float = float("inf")
    _feasible_empty: bool = False
    # running-min 证书（见 Q3FeasibleSet 同名字段说明）：半径单调不增的合法外包围圆。
    _cert_center: Optional[Vec2] = None
    _cert_radius: float = float("inf")

    # ---------- 观测分类 ----------
    def _positives(self, obs: List[Observation]) -> List[Observation]:
        return [o for o in obs if o.is_positive()]

    def _negatives(self, obs: List[Observation]) -> List[Observation]:
        return [o for o in obs if o.is_negative()]

    def _nears(self, obs: List[Observation]) -> List[Observation]:
        return [o for o in obs if o.result == ObservationType.TOO_STRONG]

    def _wedges(self, obs: List[Observation]) -> List[Wedge]:
        return [Wedge(o.position, o.bearing_deg, self.half_deg)
                for o in obs if o.result == ObservationType.BEARING and o.bearing_deg is not None]

    def _make_classifier(self, obs: List[Observation]):
        pos = self._positives(obs)
        neg = self._negatives(obs)
        nears = self._nears(obs)
        wedges = self._wedges(obs)
        Rmin = self.problem.receive_radius_min
        Rmax = self.problem.receive_radius_max
        R_area = self.problem.area_radius
        near_r = self.robot.too_strong_radius
        dm = self.delete_margin
        origin: Vec2 = (0.0, 0.0)

        def classify(b: Box) -> str:
            # --- 位置硬约束（与 Q3 同构）---
            if distance_min(origin, b) > R_area:
                return OUTSIDE
            for o in nears:
                if distance_min(o.position, b) > near_r + dm:
                    return OUTSIDE
            wedge_all_inside = True
            for w in wedges:
                rel = box_wedge_relation(w, b)
                if rel == OUTSIDE:
                    return OUTSIDE
                if rel != INSIDE:
                    wedge_all_inside = False

            # --- 半径可行性：只有下界 L(x)<=Rmax（定向负观测不给上界）---
            L_min = Rmin
            for o in pos:
                d = distance_min(o.position, b)
                if d > L_min:
                    L_min = d
            if L_min > Rmax + dm:
                return OUTSIDE
            L_max = Rmin
            for o in pos:
                d = distance_max(o.position, b)
                if d > L_max:
                    L_max = d

            # --- 方向可行性：Φ_super(B) 为空 ⟹ OUTSIDE ---
            phi_super = CircularIntervalSet.full()
            for o in pos:
                phi_super = phi_super.intersection(_cone_super(b, o.position))
                if phi_super.is_empty():
                    return OUTSIDE
            for o in neg:
                if distance_max(o.position, b) <= L_min:   # 该负观测对所有 x 都“在射程内” → 必须出锥
                    phi_super = phi_super.difference(_cone_sub(b, o.position))
                    if phi_super.is_empty():
                        return OUTSIDE

            # --- INSIDE（效率优化；判错只会多细分，不丢真源）---
            inside_ok = wedge_all_inside and distance_max(origin, b) <= R_area and L_max <= Rmax
            if inside_ok:
                for o in nears:
                    if distance_max(o.position, b) > near_r:
                        inside_ok = False
                        break
            if inside_ok:
                phi_sub = CircularIntervalSet.full()
                for o in pos:
                    phi_sub = phi_sub.intersection(_cone_sub(b, o.position))
                    if phi_sub.is_empty():
                        inside_ok = False
                        break
                if inside_ok:
                    for o in neg:
                        if distance_min(o.position, b) <= L_max:  # 可能在射程内 → 保守地整并出去
                            phi_sub = phi_sub.difference(_cone_super(b, o.position))
                            if phi_sub.is_empty():
                                inside_ok = False
                                break
                if inside_ok and not phi_sub.is_empty():
                    return INSIDE
            return UNKNOWN

        return classify

    # ---------- 重建 / 派生（与 Q3 同流程）----------
    def rebuild(self, obs: List[Observation], min_size: Optional[float] = None) -> None:
        classify = self._make_classifier(obs)
        target = min_size if min_size is not None else self.coarse_size
        arena = self.problem.area_radius
        root = Box(-arena, -arena, arena, arena)

        coarse = QuadTree(root, classify, min_size=max(self.coarse_size, target),
                          max_depth=self.max_depth)
        if coarse.is_empty():
            self._set_empty()
            return
        bbox = coarse.bounding_box()
        if target < self.coarse_size and bbox is not None:
            sub_root = pad_box(bbox, self.coarse_size)
            tree = QuadTree(sub_root, classify, min_size=target, max_depth=self.max_depth)
            if tree.is_empty():
                tree = coarse
        else:
            tree = coarse
        self.tree = tree
        self._feasible_empty = False
        self._update_derived()

    def _set_empty(self) -> None:
        self.tree = None
        self._feasible_empty = True
        self.mec_center = None
        self.mec_radius = float("inf")
        self.diameter = float("inf")

    def _update_derived(self) -> None:
        assert self.tree is not None
        circle = self.tree.mec()
        if circle is None:
            self._set_empty()
            return
        if circle.radius < self._cert_radius:
            self._cert_radius = circle.radius
            self._cert_center = circle.center
        self.mec_center = self._cert_center
        self.mec_radius = self._cert_radius
        d, _, _ = poly_diameter(self.tree.envelope_points())
        self.diameter = d

    # ---------- 查询 ----------
    def is_empty(self) -> bool:
        return self._feasible_empty or self.tree is None or self.tree.is_empty()

    def mec(self) -> Optional[Circle]:
        return self.tree.mec() if self.tree is not None else None

    def ready_to_clear(self) -> bool:
        return (not self.is_empty()) and self.mec_radius <= self.robot.clear_mec_radius

    def clear_point(self) -> Optional[Vec2]:
        return self.mec_center

    def envelope_points(self) -> List[Vec2]:
        return self.tree.envelope_points() if self.tree is not None else []

    def representative_points(self) -> List[Vec2]:
        return self.tree.representative_points() if self.tree is not None else []

    def leaf_boxes(self) -> List[Box]:
        return self.tree.leaf_boxes() if self.tree is not None else []
