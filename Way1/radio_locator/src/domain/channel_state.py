"""
频道状态：每个频道 1..20 维护其观测历史、可行集与派生量。
feasible_set 由 set_estimation 层注入（鸭子类型），避免循环依赖。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .observation import Observation
from .types import ChannelStatus, Vec2


@dataclass
class ChannelState:
    channel: int
    status: ChannelStatus = ChannelStatus.UNKNOWN
    observations: List[Observation] = field(default_factory=list)

    feasible_set: object = None            # set_estimation.* 实例（有 rebuild/mec 接口）

    mec_center: Optional[Vec2] = None
    mec_radius: float = float("inf")
    diameter: float = float("inf")

    # Q4 源型可能性（随观测收窄）
    source_type_possible_omni: bool = True
    source_type_possible_directional: bool = True

    def add_observation(self, obs: Observation) -> None:
        self.observations.append(obs)

    def positive_obs(self) -> List[Observation]:
        return [o for o in self.observations if o.is_positive()]

    def negative_obs(self) -> List[Observation]:
        return [o for o in self.observations if o.is_negative()]

    def bearing_obs(self) -> List[Observation]:
        from .types import ObservationType
        return [o for o in self.observations if o.result == ObservationType.BEARING]

    def is_settled(self) -> bool:
        """是否已达终态（CLEARED 或 ABSENT）。"""
        return self.status in (ChannelStatus.CLEARED, ChannelStatus.ABSENT)
