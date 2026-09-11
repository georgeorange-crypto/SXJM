"""
正确性不变量（runtime 守卫）：把方案的“铁律”固化为运行时可检查断言。

这些不变量在每步 apply 后被 controller 调用；任一被违反即说明【实现或数据异常】，
应立即停机/进入 recovery，绝不带病继续。它们是把“数学正确性”落到代码的最后一道闸。

核心不变量：
  I1 单调性：任一频道可行集在新增观测后只会收缩，MEC 半径不增（数值裕量内）。
  I2 非空性：只要该频道确有源且观测合法，可行集不得为空（真源不被排除）。
  I3 清除证书：仅当外包络 MEC<=clear_mec_radius 才允许 CLEAR；清除点=圆心。
  I4 基数一致：cleared + present + absent 的推断与 [Nmin,Nmax] 不矛盾。
  I5 时间账本：本地虚拟钟与模拟器返回的 virtual_time 一致（差 <= eps）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..domain.types import ChannelStatus
from ..domain.world_state import WorldState


@dataclass
class InvariantViolation:
    code: str
    message: str


@dataclass
class InvariantMonitor:
    eps_radius: float = 1e-3
    eps_time: float = 1e-3
    _last_mec: dict = field(default_factory=dict)
    violations: List[InvariantViolation] = field(default_factory=list)

    def check_monotonic(self, channel: int, mec_radius: float) -> Optional[InvariantViolation]:
        """
        I1：MEC 半径不随观测增加而增大（允许 eps 数值抖动）。

        约定 mec_radius==inf 表示可行集为【空】（∅）。空集是任意非空集的子集，
        故 finite→inf 是可行集收缩到空（真源该频道不存在，将判 ABSENT），并非“增大”——
        必须排除该假阳性。inf→finite 不会发生（观测只增，空集不复活），亦一并跳过。
        """
        prev = self._last_mec.get(channel)
        self._last_mec[channel] = mec_radius
        if prev is None or prev == float("inf") or mec_radius == float("inf"):
            return None
        if mec_radius > prev + self.eps_radius:
            v = InvariantViolation("I1_MONOTONIC",
                                   f"ch{channel} MEC grew {prev:.4f}→{mec_radius:.4f}")
            self.violations.append(v)
            return v
        return None

    def check_clear_certificate(self, mec_radius: float, threshold: float) -> Optional[InvariantViolation]:
        """I3：CLEAR 前必须证书成立。"""
        if mec_radius > threshold + self.eps_radius:
            v = InvariantViolation("I3_CLEAR_CERT",
                                   f"clear attempted with MEC {mec_radius:.4f} > {threshold}")
            self.violations.append(v)
            return v
        return None

    def check_cardinality(self, world: WorldState) -> Optional[InvariantViolation]:
        """I4：present + cleared 不超过 Nmax；absent 不使可能源数 < Nmin。"""
        present = world.confirmed_present()
        absent = world.confirmed_absent()
        total_ch = world.problem.channels
        nmax = world.problem.source_count_max
        nmin = world.problem.source_count_min
        if present > nmax:
            v = InvariantViolation("I4_CARD_MAX", f"present {present} > Nmax {nmax}")
            self.violations.append(v)
            return v
        # 剩余可能承载源的频道 = 总数 - absent；须 >= Nmin
        possible_max = total_ch - absent
        if possible_max < nmin:
            v = InvariantViolation("I4_CARD_MIN",
                                   f"absent {absent} leaves {possible_max} < Nmin {nmin}")
            self.violations.append(v)
            return v
        return None

    def check_time_ledger(self, local_vt: float, sim_vt: float) -> Optional[InvariantViolation]:
        """I5：本地虚拟钟与模拟器一致。"""
        if abs(local_vt - sim_vt) > self.eps_time:
            v = InvariantViolation("I5_TIME",
                                   f"local vt {local_vt:.6f} vs sim {sim_vt:.6f}")
            self.violations.append(v)
            return v
        return None

    def has_violations(self) -> bool:
        return len(self.violations) > 0
