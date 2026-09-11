"""
动作执行器：把“决策动作”落到模拟器（mock 或真实 client 同接口），
并把返回结果吸收进 WorldState —— 记录观测、重建可行集、推进状态机与虚拟钟。

状态机（单频道）：
  UNKNOWN --measure→ (direction/near)→DETECTED ；(no_signal 且覆盖充分)→ 累积否证
  DETECTED/LOCALIZING --measure→ 缩小可行集；MEC<=阈值 → READY_TO_CLEAR
  READY_TO_CLEAR --clear success→ CLEARED ；miss→ 回 LOCALIZING（证书失败，继续收敛）
  覆盖网测遍仍无信号 → ABSENT（completeness 由 coverage 层保证）

时间账本严格对齐模拟器：measure=移动+切频+5s；clear=移动+(5/3)s。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from ..domain.channel_state import ChannelState
from ..domain.observation import Observation
from ..domain.types import (ChannelStatus, ObservationType, ProblemConstants,
                            RobotConstants, SourceType, Vec2)
from ..domain.world_state import WorldState
from ..set_estimation.q3_feasible_set import Q3FeasibleSet
from ..set_estimation.q4_omni_feasible import Q4ChannelFeasible


@dataclass
class ExecResult:
    action: str
    channel: int
    observation: Optional[Observation]
    success: Optional[bool] = None
    sim_virtual_time: float = 0.0


class Executor:
    """
    problem_mode ∈ {"q3","q4"} 决定用哪种可行集工厂。
    sim 需实现 measure(x,y,ch)->MeasureResponse, clear(x,y,ch)->ClearResponse。
    """

    def __init__(self, sim, world: WorldState, problem_mode: str = "q3",
                 half_deg: Optional[float] = None, coarse_size: float = 10.0,
                 fine_size: float = 0.5, max_depth: int = 16,
                 delete_margin: float = 1e-3):
        self.sim = sim
        self.world = world
        self.problem_mode = problem_mode
        # 楔形半张角缺省取【物理误差界 + 报告量化裕量】，保证真源永不被裁出可行集。
        self.half_deg = half_deg if half_deg is not None else world.problem.wedge_half_deg
        self.coarse_size = coarse_size
        self.fine_size = fine_size
        self.max_depth = max_depth
        self.delete_margin = delete_margin
        # 精修触底分辨率：盒边长小到对 MEC 的贡献（角点到中心 ~size·√2/2）远小于
        # 证书裕量（20−19=1m），保证清除证书可达而不会被盒粒度卡死。
        self._fine_floor = 1.0

    # ---------- 可行集工厂 ----------
    def _new_feasible(self):
        kw = dict(problem=self.world.problem, robot=self.world.robot,
                  half_deg=self.half_deg, coarse_size=self.coarse_size,
                  fine_size=self.fine_size, max_depth=self.max_depth,
                  delete_margin=self.delete_margin)
        if self.problem_mode == "q4":
            return Q4ChannelFeasible(**kw)
        return Q3FeasibleSet(**kw)

    def _ensure_feasible(self, cs: ChannelState):
        if cs.feasible_set is None:
            cs.feasible_set = self._new_feasible()

    # ---------- MEASURE ----------
    def measure(self, x: float, y: float, channel: int, refine: bool = False) -> ExecResult:
        resp = self.sim.measure(x, y, channel)
        if not resp.accepted:
            return ExecResult("measure", channel, None, sim_virtual_time=self.world.virtual_time)

        # 时间账本（本地推进，与 sim 对齐校验交给 controller）
        self._advance_measure(x, y, channel)
        self.world.position = (x, y)
        self.world.current_channel = channel
        self.world.step += 1

        otype = resp.to_observation_type()
        obs = Observation(position=(x, y), channel=channel, result=otype,
                          bearing_deg=resp.svd_deg if otype == ObservationType.BEARING else None,
                          virtual_time=resp.virtual_time_s, step=self.world.step)
        cs = self.world.channels[channel]
        self._ensure_feasible(cs)
        cs.add_observation(obs)

        # 重建可行集：先粗扫全场，再按需逐级精修到清除证书分辨率。
        cs.feasible_set.rebuild(cs.observations, min_size=self.coarse_size)
        if not cs.feasible_set.is_empty():
            self._refine_for_certificate(cs)
        self._sync_channel_derived(cs)
        self._update_status_after_measure(cs, obs)
        return ExecResult("measure", channel, obs, sim_virtual_time=resp.virtual_time_s)

    def _refine_for_certificate(self, cs: ChannelState) -> None:
        """
        逐级加密精修：仅当粗 MEC 已进入“精修有意义”的范围才启动，随后不断减半
        min_size，直到清除证书成立（MEC<=阈值）、或盒粒度已远小于阈值（触底）。
        这样既避免全程精分辨率的开销，又消除“粗 MEC 卡在触发阈值上方永不细化”的死结。
        真源永在超集内——加密只让外包络更紧，绝不排除真源。
        """
        fp = cs.feasible_set
        threshold = self.world.robot.clear_mec_radius
        # 精修启动条件（效率闸）：区域已缩到与粗分辨率同量级，或已接近证书阈值。
        gate = max(4.0 * threshold, 3.0 * self.coarse_size)
        if fp.mec_radius > gate:
            return
        size = self.fine_size
        floor = max(self._fine_floor, threshold / 8.0)
        fp.rebuild(cs.observations, min_size=size)
        # 盒粒度不再是瓶颈前，持续加密（区域本身>阈值时会自然停在触底，等待下次测量）。
        guard = 0
        while (not fp.is_empty()) and fp.mec_radius > threshold and size > floor and guard < 8:
            size = max(floor, size / 2.0)
            fp.rebuild(cs.observations, min_size=size)
            guard += 1

    def _advance_measure(self, x: float, y: float, channel: int) -> None:
        r = self.world.robot
        move = math.hypot(x - self.world.position[0], y - self.world.position[1]) / r.speed
        switch = r.switch_time if channel != self.world.current_channel else 0.0
        self.world.virtual_time += move + switch + r.detection_time

    def _sync_channel_derived(self, cs: ChannelState) -> None:
        fp = cs.feasible_set
        cs.mec_center = fp.mec_center
        cs.mec_radius = fp.mec_radius
        cs.diameter = fp.diameter
        if hasattr(fp, "source_type_possible_omni"):
            cs.source_type_possible_omni = fp.source_type_possible_omni
            cs.source_type_possible_directional = fp.source_type_possible_directional

    def _update_status_after_measure(self, cs: ChannelState, obs: Observation) -> None:
        fp = cs.feasible_set
        if obs.is_positive():
            if fp.is_empty():
                # 不应发生（正观测下真源存在）；保守留在 DETECTED
                cs.status = ChannelStatus.DETECTED
            elif fp.ready_to_clear():
                cs.status = ChannelStatus.READY_TO_CLEAR
            else:
                cs.status = ChannelStatus.LOCALIZING if cs.status in (
                    ChannelStatus.DETECTED, ChannelStatus.LOCALIZING) else ChannelStatus.DETECTED
                if cs.status == ChannelStatus.UNKNOWN:
                    cs.status = ChannelStatus.DETECTED
        else:
            # no_signal：若该频道尚未检测到，保持 UNKNOWN（ABSENT 由覆盖完成后判定）
            if cs.status in (ChannelStatus.DETECTED, ChannelStatus.LOCALIZING):
                # 已检测到过又出现 no_signal：可行集收缩（负观测已并入），可能就绪
                if not fp.is_empty() and fp.ready_to_clear():
                    cs.status = ChannelStatus.READY_TO_CLEAR

    # ---------- CLEAR ----------
    def clear(self, x: float, y: float, channel: int) -> ExecResult:
        resp = self.sim.clear(x, y, channel)
        if not resp.accepted:
            return ExecResult("clear", channel, None, sim_virtual_time=self.world.virtual_time)
        self._advance_clear(x, y, resp.is_success())
        self.world.position = (x, y)
        self.world.step += 1
        cs = self.world.channels[channel]
        if resp.is_success():
            cs.status = ChannelStatus.CLEARED
            return ExecResult("clear", channel, None, success=True, sim_virtual_time=resp.virtual_time_s)
        else:
            # 未命中：证书本应成立却 miss → 记 CLEAR_FAILURE 观测约束（源不在该点 20m 内），继续收敛
            obs = Observation(position=(x, y), channel=channel,
                              result=ObservationType.CLEAR_FAILURE, virtual_time=resp.virtual_time_s,
                              step=self.world.step)
            cs.add_observation(obs)
            cs.status = ChannelStatus.LOCALIZING
            return ExecResult("clear", channel, obs, success=False, sim_virtual_time=resp.virtual_time_s)

    def _advance_clear(self, x: float, y: float, success: bool) -> None:
        r = self.world.robot
        move = math.hypot(x - self.world.position[0], y - self.world.position[1]) / r.speed
        self.world.virtual_time += move + (r.clear_hit_time if success else r.clear_miss_time)

    # ---------- 覆盖完成后判 ABSENT ----------
    def mark_absent_if_covered(self, channel: int, covered: bool) -> None:
        """
        判 ABSENT 的【充分】条件（completeness 由 coverage 层保证）：
          该频道在【整个覆盖网】的每一个覆盖点都实测得 no_signal，且从未有正观测。
        covered 必须表示“该频道 c 已在所有覆盖点被测负”——不是“所有覆盖点已被访问”。
        绝不能对【零观测】或【只测过部分覆盖点】的频道判 ABSENT（否则漏掉真源）。
        """
        cs = self.world.channels[channel]
        if not covered or cs.status != ChannelStatus.UNKNOWN:
            return
        has_positive = any(o.is_positive() for o in cs.observations)
        has_negative = any(o.is_negative() for o in cs.observations)
        # 必须有实测负观测支撑；零观测频道永不判 ABSENT。
        if has_negative and not has_positive:
            cs.status = ChannelStatus.ABSENT
