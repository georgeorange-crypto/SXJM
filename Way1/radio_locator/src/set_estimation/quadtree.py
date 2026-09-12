"""
自适应四叉树：集员估计的通用空间表示。

给定一个【保守分类器】classify(box) ∈ {"INSIDE","OUTSIDE","UNKNOWN"}：
- OUTSIDE：整盒不含可行点 → 剪掉（绝不保留）。
- INSIDE ：整盒全可行 → 作为 feasible 叶保留，不再细分。
- UNKNOWN：跨边界 → 继续四分，直到 size<=min_size 或 depth>=max_depth。

正确性铁律：分类器必须【保守】—— 只有在确证整盒不可行时才返回 OUTSIDE。
只要真源可能落在盒内，就必须返回 INSIDE 或 UNKNOWN。这样四叉树保留的
feasible ∪ unknown 一定是真可行集的【超集】，真源永不被排除。

外包络（envelope）= feasible ∪ unknown 全部角点的最小包围圆，是清除证书的保守依据。
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from ..geometry.box_distance import Box
from ..geometry.mec import Circle, min_enclosing_circle
from ..geometry.vector import Vec2

Classifier = Callable[[Box], str]

INSIDE = "INSIDE"
OUTSIDE = "OUTSIDE"
UNKNOWN = "UNKNOWN"


class QuadTree:
    """对 root 盒执行自适应四分，收集 feasible / unknown 叶。"""

    def __init__(self, root: Box, classify: Classifier,
                 min_size: float, max_depth: int = 16):
        self.root = root
        self._classify = classify
        self._min_size = min_size
        self._max_depth = max_depth
        self.feasible: List[Box] = []
        self.unknown: List[Box] = []
        self._build(root, 0)

    def _build(self, box: Box, depth: int) -> None:
        c = self._classify(box)
        if c == OUTSIDE:
            return
        if c == INSIDE:
            self.feasible.append(box)
            return
        # UNKNOWN
        if box.size <= self._min_size or depth >= self._max_depth:
            self.unknown.append(box)
            return
        for sub in box.split4():
            self._build(sub, depth + 1)

    # ---------- 查询 ----------
    def is_empty(self) -> bool:
        """无任何 feasible/unknown 叶 → 可行集为空（该假设被证否）。"""
        return not self.feasible and not self.unknown

    def leaf_boxes(self) -> List[Box]:
        return self.feasible + self.unknown

    def envelope_points(self) -> List[Vec2]:
        """feasible ∪ unknown 所有叶盒角点（外包络 MEC 的输入）。"""
        pts: List[Vec2] = []
        for b in self.leaf_boxes():
            pts.extend(b.corners())
        return pts

    def representative_points(self) -> List[Vec2]:
        """各叶盒中心（用于粗略采样，如 minimax 候选场景）。"""
        return [b.center for b in self.leaf_boxes()]

    def mec(self) -> Optional[Circle]:
        """外包络最小包围圆。空集返回 None。"""
        pts = self.envelope_points()
        if not pts:
            return None
        return min_enclosing_circle(pts)

    def total_area_upper(self) -> float:
        """可行集面积上界（feasible 精确 + unknown 全计）。"""
        return sum(b.width * b.height for b in self.leaf_boxes())

    def bounding_box(self) -> Optional[Box]:
        """所有叶盒的外包 AABB（用于二阶段细化时缩小 root）。"""
        return bounding_box_of(self.leaf_boxes())


def bounding_box_of(boxes: List[Box]) -> Optional[Box]:
    if not boxes:
        return None
    xl = min(b.xl for b in boxes)
    yl = min(b.yl for b in boxes)
    xu = max(b.xu for b in boxes)
    yu = max(b.yu for b in boxes)
    return Box(xl, yl, xu, yu)


def pad_box(b: Box, pad: float) -> Box:
    return Box(b.xl - pad, b.yl - pad, b.xu + pad, b.yu + pad)
