"""
Q4 全向假设可行集，以及“同时持有两种源型假设”的组合可行集。

Q4 每个频道源型未知：真源要么全向、要么定向。策略——
  同时维护两个可行集（omni / directional），各自是对应假设下真源位置的【超集】。
  组合可行集 = 两者之并；其外包络 MEC 是【无论何种源型】真源位置的保守超集。
  只要该并集外包络半径 <= 19m，移动到圆心即可清除，无需先判定源型。
  当某一假设集变空 ⟹ 该源型被排除 ⟹ 源型确定（副产品）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..domain.observation import Observation
from ..domain.types import ProblemConstants, RobotConstants, Vec2
from ..geometry.mec import Circle, min_enclosing_circle
from ..geometry.rotating_calipers import diameter as poly_diameter
from .outer_envelope import Envelope, envelope_of_points
from .q3_feasible_set import Q3FeasibleSet
from .q4_directional_feasible import Q4DirectionalFeasibleSet


@dataclass
class Q4OmniFeasibleSet(Q3FeasibleSet):
    """Q4 全向假设：探测物理与 Q3 完全一致。"""
    pass


@dataclass
class Q4ChannelFeasible:
    """
    组合可行集：omni 假设 ⊕ directional 假设。
    对外暴露与单一 feasible set 相同的 is_empty / mec_radius / ready_to_clear / clear_point。
    """
    problem: ProblemConstants
    robot: RobotConstants
    half_deg: float = 1.0
    coarse_size: float = 10.0
    fine_size: float = 0.5
    max_depth: int = 16
    delete_margin: float = 1e-3

    omni: Q4OmniFeasibleSet = field(init=False)
    directional: Q4DirectionalFeasibleSet = field(init=False)

    mec_center: Optional[Vec2] = None
    mec_radius: float = float("inf")
    diameter: float = float("inf")
    # running-min 证书：omni∪directional 的外包围圆随观测单调收缩，保留见过的最小圆。
    _cert_center: Optional[Vec2] = None
    _cert_radius: float = float("inf")

    def __post_init__(self) -> None:
        kw = dict(problem=self.problem, robot=self.robot, half_deg=self.half_deg,
                  coarse_size=self.coarse_size, fine_size=self.fine_size,
                  max_depth=self.max_depth, delete_margin=self.delete_margin)
        self.omni = Q4OmniFeasibleSet(**kw)
        self.directional = Q4DirectionalFeasibleSet(**kw)

    # ---------- 源型可能性 ----------
    @property
    def source_type_possible_omni(self) -> bool:
        return not self.omni.is_empty()

    @property
    def source_type_possible_directional(self) -> bool:
        return not self.directional.is_empty()

    # ---------- 重建 ----------
    def rebuild(self, obs: List[Observation], min_size: Optional[float] = None) -> None:
        self.omni.rebuild(obs, min_size=min_size)
        self.directional.rebuild(obs, min_size=min_size)
        self._update_derived()

    def _update_derived(self) -> None:
        pts: List[Vec2] = []
        if not self.omni.is_empty():
            pts.extend(self.omni.envelope_points())
        if not self.directional.is_empty():
            pts.extend(self.directional.envelope_points())
        if not pts:
            self.mec_center = None
            self.mec_radius = float("inf")
            self.diameter = float("inf")
            return
        circle = min_enclosing_circle(pts)
        if circle.radius < self._cert_radius:
            self._cert_radius = circle.radius
            self._cert_center = circle.center
        self.mec_center = self._cert_center
        self.mec_radius = self._cert_radius
        d, _, _ = poly_diameter(pts)
        self.diameter = d

    # ---------- 查询 ----------
    def is_empty(self) -> bool:
        return self.omni.is_empty() and self.directional.is_empty()

    def mec(self) -> Optional[Circle]:
        pts: List[Vec2] = []
        if not self.omni.is_empty():
            pts.extend(self.omni.envelope_points())
        if not self.directional.is_empty():
            pts.extend(self.directional.envelope_points())
        return min_enclosing_circle(pts) if pts else None

    def envelope(self) -> Envelope:
        pts: List[Vec2] = []
        if not self.omni.is_empty():
            pts.extend(self.omni.envelope_points())
        if not self.directional.is_empty():
            pts.extend(self.directional.envelope_points())
        return envelope_of_points(pts)

    def envelope_points(self) -> List[Vec2]:
        pts: List[Vec2] = []
        if not self.omni.is_empty():
            pts.extend(self.omni.envelope_points())
        if not self.directional.is_empty():
            pts.extend(self.directional.envelope_points())
        return pts

    def representative_points(self) -> List[Vec2]:
        pts: List[Vec2] = []
        if not self.omni.is_empty():
            pts.extend(self.omni.representative_points())
        if not self.directional.is_empty():
            pts.extend(self.directional.representative_points())
        return pts

    def ready_to_clear(self) -> bool:
        return (not self.is_empty()) and self.mec_radius <= self.robot.clear_mec_radius

    def clear_point(self) -> Optional[Vec2]:
        return self.mec_center
