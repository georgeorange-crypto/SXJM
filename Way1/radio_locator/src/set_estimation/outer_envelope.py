"""
外包络与清除证书：把 feasible∪unknown 盒集合归约为一个保守的最小包围圆。

清除证书（correctness）：
  设 Ω̂ = feasible∪unknown 的并（真可行集 Ω 的超集，Ω⊆Ω̂）。
  C = MEC(Ω̂ 的角点)。因 Ω̂ 的每点都在其所属盒内、盒角点是该盒的极点，
  故 C 包含整个 Ω̂ ⊇ Ω ∋ 真源。
  若 radius(C) <= 20 - margin，则移动到 C.center，真源必在 20m 内 → /clear 成功。

本模块把该逻辑独立出来，供 Q3/Q4 feasible set 与 runtime 复用，
并给出“证书是否成立”的显式判定，避免各处重复实现。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..geometry.box_distance import Box
from ..geometry.mec import Circle, min_enclosing_circle
from ..geometry.rotating_calipers import diameter as poly_diameter
from ..geometry.vector import Vec2


@dataclass(frozen=True)
class Envelope:
    circle: Optional[Circle]
    diameter: float
    n_points: int

    @property
    def center(self) -> Optional[Vec2]:
        return self.circle.center if self.circle else None

    @property
    def radius(self) -> float:
        return self.circle.radius if self.circle else float("inf")

    def is_empty(self) -> bool:
        return self.circle is None

    def certificate_holds(self, threshold: float) -> bool:
        """外包络半径 <= threshold（如 19m）⟹ 清除证书成立。"""
        return (self.circle is not None) and self.circle.radius <= threshold


def envelope_of_boxes(boxes: List[Box]) -> Envelope:
    """对盒集合角点求外包络 MEC 与直径。空集返回空 Envelope。"""
    pts: List[Vec2] = []
    for b in boxes:
        pts.extend(b.corners())
    return envelope_of_points(pts)


def envelope_of_points(points: List[Vec2]) -> Envelope:
    if not points:
        return Envelope(circle=None, diameter=float("inf"), n_points=0)
    circle = min_enclosing_circle(points)
    d, _, _ = poly_diameter(points)
    return Envelope(circle=circle, diameter=d, n_points=len(points))


def clear_certificate(boxes: List[Box], threshold: float) -> Optional[Vec2]:
    """
    若盒集合外包络半径 <= threshold，返回应前往的清除点（圆心）；否则 None。
    """
    env = envelope_of_boxes(boxes)
    if env.certificate_holds(threshold):
        return env.center
    return None
