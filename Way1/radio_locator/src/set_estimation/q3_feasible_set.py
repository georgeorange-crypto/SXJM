"""
Q3 全向源可行集：L(x)/U(x) 半径消元 + 楔形 + near 圆盘 + 竞技场约束。

数学（存在性投影，消去未知 R_eff∈[Rmin,Rmax]）：
  正观测 P（收到信号，dist<=R_eff）：R_eff >= ‖x-Sᵢ‖。
  负观测 N（no_signal，全向⟺dist>R_eff）：R_eff <  ‖x-Sⱼ‖。
  ⟹ L(x)=max(Rmin, max_{i∈P}‖x-Sᵢ‖) ；U(x)=min(Rmax, min_{j∈N}‖x-Sⱼ‖)。
  位置 x 可行 ⟺ 存在 R_eff∈[L(x),U(x)) ⟺ L(x) < U(x)。

box 层【保守】消元（真源永不被误删）：
  L_min(B)=max(Rmin, max_{i∈P} dist_min(Sᵢ,B))   —— L 在 B 上的下界
  U_max(B)=min(Rmax, min_{j∈N} dist_max(Sⱼ,B))   —— U 在 B 上的上界
  若 L_min(B) >= U_max(B) + δ_R ⟹ 对所有 x∈B 有 L(x)>=U(x) ⟹ 整盒不可行 ⟹ 删。
  δ_R (radius_delete_margin) 只让删除更保守，永不误删真源盒。

附加硬约束（同为保守 box 判定）：
  bearing：源在楔形内 → box_wedge_relation==OUTSIDE 则删。
  near   ：源在该点 5m 内 → dist_min(nearpt,B)>5 则删。
  arena  ：源在半径 R_area 圆内 → dist_min(O,B)>R_area 则删。

外包络 MEC（feasible∪unknown 角点的最小包围圆）是【超集】，其半径 <=19m
即可作为清除证书：移动到圆心，真源必在 20m 内。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..domain.observation import Observation
from ..domain.types import ObservationType, ProblemConstants, RobotConstants, Vec2
from ..geometry.box_distance import Box, distance_max, distance_min
from ..geometry.mec import Circle
from ..geometry.rotating_calipers import diameter as poly_diameter
from ..geometry.wedge import Wedge, box_wedge_relation
from .quadtree import INSIDE, OUTSIDE, UNKNOWN, QuadTree, bounding_box_of, pad_box


@dataclass
class Q3FeasibleSet:
    """单个全向频道的位置可行集（随观测增量重建）。"""
    problem: ProblemConstants
    robot: RobotConstants
    half_deg: float = 1.0                # 楔形半张角（示向度误差界）
    coarse_size: float = 10.0
    fine_size: float = 0.5
    max_depth: int = 16
    delete_margin: float = 1e-3

    # 派生
    tree: Optional[QuadTree] = None
    mec_center: Optional[Vec2] = None
    mec_radius: float = float("inf")
    diameter: float = float("inf")
    _feasible_empty: bool = False
    # 运行时“最优证书”（running-min）：曾算得的最小外包围圆 (圆心, 半径)。
    #   正确性：真源恒在当前可行集 F(obs_k) 内，而 F 随观测单调收缩
    #   （F(obs_j) ⊇ F(obs_k), j<=k），故任一历史外包围圆 C_j ⊇ F(obs_j) ⊇ F(obs_k) ∋ 真源。
    #   于是保留“见过的最小半径圆”仍然包住当前可行集与真源 —— 既是合法证书，又保证
    #   半径单调不增（消除粗分辨率下 MEC 随盒重分类抖动导致的 I1_MONOTONIC 违规）。
    _cert_center: Optional[Vec2] = None
    _cert_radius: float = float("inf")

    # ---------- 约束提取 ----------
    def _positives(self, obs: List[Observation]) -> List[Observation]:
        return [o for o in obs if o.is_positive()]

    def _negatives(self, obs: List[Observation]) -> List[Observation]:
        return [o for o in obs if o.is_negative()]

    def _nears(self, obs: List[Observation]) -> List[Observation]:
        return [o for o in obs if o.result == ObservationType.TOO_STRONG]

    def _wedges(self, obs: List[Observation]) -> List[Wedge]:
        return [Wedge(o.position, o.bearing_deg, self.half_deg)
                for o in obs if o.result == ObservationType.BEARING and o.bearing_deg is not None]

    # ---------- box 分类器（保守）----------
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
            # arena：整盒在竞技场外 → 删
            if distance_min(origin, b) > R_area:
                return OUTSIDE
            # near：源必在某 near 点的 5m 内
            for o in nears:
                if distance_min(o.position, b) > near_r + dm:
                    return OUTSIDE
            # bearing 楔形：整盒在楔外 → 删
            wedge_all_inside = True
            for w in wedges:
                rel = box_wedge_relation(w, b)
                if rel == OUTSIDE:
                    return OUTSIDE
                if rel != INSIDE:
                    wedge_all_inside = False
            # 半径可行性（L_min vs U_max）
            L_min = Rmin
            for o in pos:
                d = distance_min(o.position, b)
                if d > L_min:
                    L_min = d
            U_max = Rmax
            for o in neg:
                d = distance_max(o.position, b)
                if d < U_max:
                    U_max = d
            if L_min >= U_max + dm:
                return OUTSIDE

            # ---- 尝试判 INSIDE（纯效率优化；判错只会让包络更大，不丢真源）----
            if distance_max(origin, b) > R_area:
                return UNKNOWN
            for o in nears:
                if distance_max(o.position, b) > near_r:
                    return UNKNOWN
            if not wedge_all_inside:
                return UNKNOWN
            L_max = Rmin
            for o in pos:
                d = distance_max(o.position, b)
                if d > L_max:
                    L_max = d
            U_min = Rmax
            for o in neg:
                d = distance_min(o.position, b)
                if d < U_min:
                    U_min = d
            if L_max < U_min:
                return INSIDE
            return UNKNOWN

        return classify

    # ---------- 重建 ----------
    def rebuild(self, obs: List[Observation], min_size: Optional[float] = None) -> None:
        """
        两遍构建：先粗分辨率扫全场得到外包 bbox，再在 bbox+pad 上按目标分辨率精修。
        这样即便 min_size=0.5，代价也只落在真正的可行带上。
        """
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
                tree = coarse  # 精修意外清空则回退（不应发生；保守保留）
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
        # running-min 证书：仅当本次外包围圆更小才替换（半径单调不增；旧圆仍含真源）。
        if circle.radius < self._cert_radius:
            self._cert_radius = circle.radius
            self._cert_center = circle.center
        self.mec_center = self._cert_center
        self.mec_radius = self._cert_radius
        pts = self.tree.envelope_points()
        d, _, _ = poly_diameter(pts)
        self.diameter = d

    # ---------- 查询 ----------
    def is_empty(self) -> bool:
        return self._feasible_empty or self.tree is None or self.tree.is_empty()

    def mec(self) -> Optional[Circle]:
        if self.tree is None:
            return None
        return self.tree.mec()

    def ready_to_clear(self) -> bool:
        """外包络 MEC 半径 <= 清除阈值（默认 19m，留 1m 数值裕量到 20m）。"""
        return (not self.is_empty()) and self.mec_radius <= self.robot.clear_mec_radius

    def clear_point(self) -> Optional[Vec2]:
        """清除时机器人应前往的点：外包络圆心。"""
        return self.mec_center

    def envelope_points(self) -> List[Vec2]:
        return self.tree.envelope_points() if self.tree is not None else []

    def representative_points(self) -> List[Vec2]:
        return self.tree.representative_points() if self.tree is not None else []

    def leaf_boxes(self) -> List[Box]:
        return self.tree.leaf_boxes() if self.tree is not None else []
