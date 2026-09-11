"""
世界状态：机器人位姿、当前频道、虚拟钟、20 个频道状态、基数约束。
这是运行时的单一真相源（single source of truth）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .channel_state import ChannelState
from .types import ChannelStatus, ProblemConstants, RobotConstants, Vec2


@dataclass
class WorldState:
    problem: ProblemConstants
    robot: RobotConstants
    position: Vec2 = (0.0, 0.0)
    current_channel: int = 1
    virtual_time: float = 0.0
    step: int = 0
    channels: Dict[int, ChannelState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.channels:
            self.channels = {c: ChannelState(channel=c) for c in range(1, self.problem.channels + 1)}

    # ---------- 基数约束（§26）----------
    def count_status(self, status: ChannelStatus) -> int:
        return sum(1 for cs in self.channels.values() if cs.status == status)

    def confirmed_present(self) -> int:
        """已确认存在（DETECTED/LOCALIZING/READY/CLEARED）。"""
        present = {ChannelStatus.DETECTED, ChannelStatus.LOCALIZING,
                   ChannelStatus.READY_TO_CLEAR, ChannelStatus.CLEARED}
        return sum(1 for cs in self.channels.values() if cs.status in present)

    def confirmed_absent(self) -> int:
        return self.count_status(ChannelStatus.ABSENT)

    def unknown_channels(self) -> List[int]:
        return [c for c, cs in self.channels.items() if cs.status == ChannelStatus.UNKNOWN]

    def cleared_count(self) -> int:
        return self.count_status(ChannelStatus.CLEARED)

    def source_count_bounds(self) -> Tuple[int, int]:
        """
        未知频道中实际存在源数量的区间：
          N_U ∈ [max(0, Nmin - E), min(U, Nmax - E)]
        其中 E=已确认存在, U=未知数。
        """
        E = self.confirmed_present()
        U = len(self.unknown_channels())
        nmin = self.problem.source_count_min
        nmax = self.problem.source_count_max
        lo = max(0, nmin - E)
        hi = min(U, nmax - E)
        return lo, hi

    def all_settled(self) -> bool:
        """所有频道均达终态（CLEARED/ABSENT）。"""
        return all(cs.is_settled() for cs in self.channels.values())

    def mission_complete(self) -> bool:
        """
        任务完成的严格判据（§37）：
          所有频道 ∈ {CLEARED, ABSENT}  或  已清除数 == 源数上界。
        """
        if self.all_settled():
            return True
        if self.cleared_count() >= self.problem.source_count_max:
            return True
        return False
