"""
主控制回路：事件驱动的 receding-horizon 调度。

流程：
  enter → 沿覆盖计划边探索边处理，直到 mission_complete → exit。
  每一步：
    1. 若有 READY_TO_CLEAR 频道 → 立即 CLEAR（消灭确定任务）。
    2. 否则 scheduler.next_task 在“定位 / 探索”间按时间代价择优。
    3. 执行动作（Executor），吸收观测、重建可行集、推进状态机与虚拟钟。
    4. 不变量守卫校验；异常 → recovery。
    5. 覆盖计划走完后，对仍 UNKNOWN 且无正观测的频道判 ABSENT。

正确性由下层保证：本回路只决定顺序，不改变“真源永不被排除 / 仅证书成立才清除”。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..domain.types import ChannelStatus, Vec2
from ..domain.world_state import WorldState
from ..optimization.task_scheduler import Scheduler, Task, TaskKind
from .executor import Executor
from .invariants import InvariantMonitor
from . import recovery


@dataclass
class ControllerConfig:
    problem_mode: str = "q3"
    half_deg: Optional[float] = None       # None → 由 problem.wedge_half_deg 决定
    coarse_size: float = 10.0
    fine_size: float = 0.5
    max_depth: int = 16
    delete_margin: float = 1e-3
    lam: float = 1.0
    max_candidates: int = 48
    max_steps: int = 5000
    held_karp_max: int = 16


@dataclass
class StepLog:
    step: int
    kind: str
    channel: Optional[int]
    location: Optional[Vec2]
    virtual_time: float
    mec_radius: Optional[float] = None
    note: str = ""


class Controller:
    def __init__(self, sim, world: WorldState, cover_points: Sequence[Vec2],
                 config: Optional[ControllerConfig] = None):
        self.sim = sim
        self.world = world
        self.cfg = config or ControllerConfig()
        self.cover_points = list(cover_points)
        # 楔形半张角：cfg 未指定则取物理误差界+量化裕量（真源永不被裁出）。
        half_deg = self.cfg.half_deg if self.cfg.half_deg is not None else world.problem.wedge_half_deg
        self.executor = Executor(sim, world, problem_mode=self.cfg.problem_mode,
                                 half_deg=half_deg, coarse_size=self.cfg.coarse_size,
                                 fine_size=self.cfg.fine_size, max_depth=self.cfg.max_depth,
                                 delete_margin=self.cfg.delete_margin)
        self.scheduler = Scheduler(world.problem, world.robot, lam=self.cfg.lam,
                                   half_deg=half_deg, max_candidates=self.cfg.max_candidates)
        self.monitor = InvariantMonitor()
        self.visited: List[Vec2] = []
        self.log: List[StepLog] = []
        self._clear_attempts: Dict[int, List[Vec2]] = {}
        # 覆盖扫描状态：completeness 的核心账本。
        #   _measured_at[c] = 已在【哪些覆盖点下标】实测过频道 c（任意结果）。
        #   _neg_at[c]      = 频道 c 在【哪些覆盖点下标】得 no_signal。
        # 频道 c 可判 ABSENT 的充分条件：_neg_at[c] 覆盖【全部】覆盖点且无正观测。
        self._measured_at: Dict[int, set] = {}
        self._neg_at: Dict[int, set] = {}

    # ---------- 覆盖顺序 ----------
    def _plan_cover_route(self) -> None:
        if not self.cover_points:
            return
        ordered = self.scheduler.order_cover_route(self.world.position, self.cover_points,
                                                   held_karp_max=self.cfg.held_karp_max)
        self.cover_points = ordered

    def _cover_index_of(self, loc: Vec2, tol: float = 1.0) -> Optional[int]:
        """定位一个坐标对应的覆盖点下标（用于登记 per-channel 扫描）。"""
        for i, p in enumerate(self._original_cover):
            if math.hypot(loc[0] - p[0], loc[1] - p[1]) <= tol:
                return i
        return None

    def _channel_swept(self, channel: int) -> bool:
        """频道 c 是否已在【全部】覆盖点实测得 no_signal（可判 ABSENT 的必要证据）。"""
        need = len(self._original_cover)
        return need > 0 and len(self._neg_at.get(channel, set())) >= need

    def _cover_complete(self) -> bool:
        """
        覆盖完成 = 每个频道要么已确认存在/清除，要么已在全网被测负（可判 ABSENT）。
        这是 completeness 的真正判据；旧版“访问完所有点”不足以对未测频道下结论。
        注：主循环已改用 per-channel 的 _channel_swept 逐频道收尾，本方法仅作诊断/汇总用。
        """
        for c, cs in self.world.channels.items():
            if cs.status in (ChannelStatus.CLEARED, ChannelStatus.ABSENT):
                continue
            if cs.status == ChannelStatus.UNKNOWN and not self._channel_swept(c):
                return False
            # DETECTED/LOCALIZING/READY 仍需后续清除；未 settle 则未完成。
            if cs.status not in (ChannelStatus.CLEARED, ChannelStatus.ABSENT) \
                    and cs.status != ChannelStatus.UNKNOWN:
                return False
        return True

    # ---------- 主循环 ----------
    def run(self) -> WorldState:
        er = self.sim.enter()
        self._original_cover = list(self.cover_points)
        self._plan_cover_route()

        steps = 0
        while steps < self.cfg.max_steps and not self.world.mission_complete():
            steps += 1
            task = self._select_task()
            if task is None:
                # 无定位/清除候选：推进【覆盖扫描】（把未知频道在各覆盖点测负）。
                if not self._sweep_step():
                    # 扫描也无事可做：尝试判 ABSENT 收尾。
                    self._finalize_absent()
                    break
                self._finalize_absent()
                continue
            self._execute(task)
            self._finalize_absent()

        self.sim.exit()
        return self.world

    def _select_task(self) -> Optional[Task]:
        # 优先就绪清除，其次定位（已检测频道收敛）。覆盖扫描作为兜底在 run() 中推进。
        clears = self.scheduler.ready_clears(self.world)
        if clears:
            return min(clears, key=lambda t: t.score)
        locs = self.scheduler.localize_candidates(self.world)
        if locs:
            return min(locs, key=lambda t: t.score)
        return None

    def _sweep_step(self) -> bool:
        """
        覆盖扫描的一步：选一个【未完成扫描】的 (覆盖点, 未知频道) 组合去测量。
        策略（降切频/降移动）：就近选覆盖点，在该点把所有还没测过的未知频道逐个测一遍，
        优先当前频道以省切频。返回 False 表示已无待扫描组合。
        """
        target = self._next_sweep_target()
        if target is None:
            return False
        loc, channel, idx = target
        self._do_measure(loc, channel, cover_index=idx)
        return True

    def _next_sweep_target(self) -> Optional[Tuple[Vec2, int, int]]:
        """挑下一个待测 (覆盖点坐标, 频道, 覆盖点下标)。"""
        unknown = [c for c, cs in self.world.channels.items()
                   if cs.status == ChannelStatus.UNKNOWN and not self._channel_swept(c)]
        if not unknown:
            return None
        pos = self.world.position
        # 候选覆盖点：仍有未知频道未在此点测过的点，按“是否要切频 + 距离”排序。
        best = None
        best_key = None
        for idx, p in enumerate(self._original_cover):
            pend = [c for c in unknown if idx not in self._measured_at.get(c, set())]
            if not pend:
                continue
            move = math.hypot(p[0] - pos[0], p[1] - pos[1])
            # 若当前频道在此点仍待测，优先它（切频成本 0）。
            if self.world.current_channel in pend:
                ch = self.world.current_channel
                switch = 0.0
            else:
                ch = min(pend)
                switch = self.world.robot.switch_time
            key = (self.world.robot.speed * switch + move)  # 统一成移动等价代价
            if best_key is None or key < best_key:
                best_key = key
                best = (p, ch, idx)
        return best

    # ---------- 执行分发 ----------
    def _execute(self, task: Task) -> None:
        if task.kind == TaskKind.CLEAR:
            self._do_clear(task.location, task.channel)
        elif task.kind == TaskKind.LOCALIZE:
            self._do_measure(task.location, task.channel)
        else:  # COVER（保留兼容：按覆盖点扫描登记）
            idx = self._cover_index_of(task.location)
            ch = task.channel if task.channel is not None else self.world.current_channel
            self._do_measure(task.location, ch, cover_index=idx)

    def _do_measure(self, loc: Vec2, channel: int, cover_index: Optional[int] = None) -> None:
        res = self.executor.measure(loc[0], loc[1], channel)
        cs = self.world.channels[channel]
        self.monitor.check_monotonic(channel, cs.mec_radius)
        self.monitor.check_time_ledger(self.world.virtual_time, res.sim_virtual_time)
        # 覆盖扫描登记：在某覆盖点实测过该频道（记结果正负），供 per-channel ABSENT 判据。
        if cover_index is None:
            cover_index = self._cover_index_of(loc)
        if cover_index is not None:
            self._measured_at.setdefault(channel, set()).add(cover_index)
            if res.observation is not None and res.observation.is_negative():
                self._neg_at.setdefault(channel, set()).add(cover_index)
            if loc not in self.visited:
                self.visited.append(loc)
        self.log.append(StepLog(self.world.step, "measure", channel, loc,
                                self.world.virtual_time, cs.mec_radius,
                                note=res.observation.result.value if res.observation else "rejected"))

    def _do_clear(self, loc: Vec2, channel: int) -> None:
        cs = self.world.channels[channel]
        v = self.monitor.check_clear_certificate(cs.mec_radius, self.world.robot.clear_mec_radius)
        if v is not None:
            # 证书不成立却被选中清除：转为继续定位（recovery）
            self.log.append(StepLog(self.world.step, "clear-blocked", channel, loc,
                                    self.world.virtual_time, cs.mec_radius, note=v.code))
            return
        self._clear_attempts.setdefault(channel, [])
        res = self.executor.clear(loc[0], loc[1], channel)
        self._clear_attempts[channel].append(loc)
        self.monitor.check_time_ledger(self.world.virtual_time, res.sim_virtual_time)
        note = "success" if res.success else "miss"
        self.log.append(StepLog(self.world.step, "clear", channel, loc,
                                self.world.virtual_time, cs.mec_radius, note=note))
        if not res.success:
            self._handle_clear_miss(channel)

    def _handle_clear_miss(self, channel: int) -> None:
        cs = self.world.channels[channel]
        fp = cs.feasible_set
        if fp is None or fp.is_empty():
            return
        # 细化后换清除点
        fp.rebuild(cs.observations, min_size=self.cfg.fine_size)
        self.executor._sync_channel_derived(cs)
        adv = recovery.alt_clear_point(fp.envelope_points(), self._clear_attempts[channel])
        if adv.location is not None and not fp.is_empty() and fp.ready_to_clear():
            cs.status = ChannelStatus.READY_TO_CLEAR

    # ---------- ABSENT 收尾 ----------
    def _finalize_absent(self) -> None:
        """
        逐频道判 ABSENT（不用全局闸）：某未知频道只要已在【全部覆盖点】被测负，
        即可独立判 ABSENT，无需等其它频道 settle。这样避免“任一频道卡在 DETECTED
        导致所有已扫描频道都无法收尾”的死锁；正确性由 mark_absent_if_covered 复核
        （必须有负观测、无正观测）保证。
        """
        for c in list(self.world.unknown_channels()):
            if self._channel_swept(c):
                self.executor.mark_absent_if_covered(c, covered=True)

    # ---------- 工具 ----------
    def _is_visited(self, p: Vec2, tol: float = 1.0) -> bool:
        return any(math.hypot(p[0] - v[0], p[1] - v[1]) <= tol for v in self.visited)

    def _matches_cover_point(self, loc: Vec2, tol: float = 1.0) -> bool:
        return any(math.hypot(loc[0] - p[0], loc[1] - p[1]) <= tol for p in self.cover_points)

    # ---------- 报告 ----------
    def summary(self) -> Dict[str, object]:
        return {
            "virtual_time": self.world.virtual_time,
            "steps": self.world.step,
            "cleared": self.world.cleared_count(),
            "absent": self.world.confirmed_absent(),
            "present": self.world.confirmed_present(),
            "violations": [v.code for v in self.monitor.violations],
            "mission_complete": self.world.mission_complete(),
        }
