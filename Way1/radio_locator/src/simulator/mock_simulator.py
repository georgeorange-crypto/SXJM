"""
本地 mock 模拟器：逐条复现 附件2 的物理与计时规则，用于无限次离线自测与调参。

关键点（与 offline_sim/engine.py 对齐）：
- 收到信号 ⟺ 距离 <= R_eff 且在覆盖角内（全向恒真）。
- near ⟺ 距离 <= 5 且在覆盖角内（定向盲区不算 near）。
- 示向度误差 = 每个【地点】固定的系统偏差 e(x,y) ∈ [-1,1]°（同点重复不变）。
- 计时：measure=移动+切频(变则1s)+5s；clear=移动+(成功5s/未发现3s)，不切频。
- /enter、/exit 不推进虚拟钟。

误差场绝不默认 iid：默认用 constant-by-location field，可注入其他场做 adversarial 测试。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..domain.types import ProblemConstants, RobotConstants, SourceType
from .protocol import ClearResponse, EnterResponse, ExitResponse, MeasureResponse

# ---- 误差场：给定 (x,y) 返回固定偏差 ∈ [-1,1]°（度）----
ErrorField = Callable[[float, float], float]


def make_smooth_field(seed: int = 0, amplitude: float = 1.0) -> ErrorField:
    """平滑空间误差场：几个正弦叠加，按地点固定。amplitude<=1 保证 |e|<=1。"""
    rng = random.Random(seed)
    freqs = [(rng.uniform(0.0005, 0.003), rng.uniform(0.0005, 0.003), rng.uniform(0, 2 * math.pi))
             for _ in range(3)]
    w = [rng.uniform(0.3, 1.0) for _ in range(3)]
    wsum = sum(w)

    def field(x: float, y: float) -> float:
        v = 0.0
        for (fx, fy, ph), wi in zip(freqs, w):
            v += wi * math.sin(fx * x + fy * y + ph)
        v /= wsum
        return max(-1.0, min(1.0, amplitude * v))

    return field


def make_constant_field(value: float = 0.0) -> ErrorField:
    """常数误差场（用于确定性单测）。"""
    v = max(-1.0, min(1.0, value))
    return lambda x, y: v


def make_adversarial_field(amplitude: float = 1.0) -> ErrorField:
    """对抗误差场：按象限取 ±amplitude 极值，逼近误差上界。"""
    a = max(-1.0, min(1.0, amplitude))

    def field(x: float, y: float) -> float:
        s = 1.0 if (int(math.floor(x / 137.0)) + int(math.floor(y / 149.0))) % 2 == 0 else -1.0
        return a * s

    return field


@dataclass
class MockJammer:
    channel: int
    x: float
    y: float
    r_eff: float
    kind: SourceType = SourceType.OMNI
    direction_deg: float = 0.0     # 仅定向源有效
    cleared: bool = False


@dataclass
class MockCase:
    """一局：干扰源布置 + 误差场 + 时限。"""
    jammers: List[MockJammer]
    error_field: ErrorField
    max_virtual_duration_s: float = 360000.0
    max_real_duration_s: int = 1200

    def jammer_on(self, channel: int) -> Optional[MockJammer]:
        for j in self.jammers:
            if j.channel == channel:
                return j
        return None


class MockSimulator:
    """
    与 client.py 同接口的本地模拟器（enter/measure/clear/exit）。
    计时用微秒累计避免浮点漂移。真值仅内部持有，绝不通过响应泄露。
    """

    def __init__(self, case: MockCase, problem: ProblemConstants, robot: RobotConstants):
        self.case = case
        self.problem = problem
        self.robot = robot
        self._reset()

    def _reset(self) -> None:
        self.entered = False
        self.finished = False
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.channel = 1
        self.vt_us = 0
        # 记录真值指标
        self.true_source_count = sum(1 for _ in self.case.jammers)

    # ---------- 时间 ----------
    @staticmethod
    def _us(seconds: float) -> int:
        return int(round(seconds * 1_000_000))

    def _vt(self) -> float:
        return self.vt_us / 1_000_000.0

    def _advance(self, seconds: float) -> None:
        self.vt_us += self._us(seconds)

    # ---------- 物理 ----------
    def _distance(self, x: float, y: float, j: MockJammer) -> float:
        return math.hypot(x - j.x, y - j.y)

    def _in_coverage(self, x: float, y: float, j: MockJammer) -> bool:
        if j.kind == SourceType.OMNI:
            return True
        # 源指向检测点的方位 vs 定向方向，夹角 <= 90°
        bearing = math.degrees(math.atan2(y - j.y, x - j.x)) % 360.0
        diff = abs((bearing - j.direction_deg + 180.0) % 360.0 - 180.0)
        return diff <= self.problem.directional_half_angle_deg + 1e-9

    def _true_bearing(self, x: float, y: float, j: MockJammer) -> float:
        return math.degrees(math.atan2(j.y - y, j.x - x)) % 360.0

    # ---------- 动作 ----------
    def enter(self) -> EnterResponse:
        self._reset()
        self.entered = True
        return EnterResponse(
            accepted=True,
            virtual_time_s=0.0,
            max_virtual_duration_s=self.case.max_virtual_duration_s,
            max_real_duration_s=float(self.case.max_real_duration_s),
            remaining_real_duration_s=self.case.max_real_duration_s,
        )

    def measure(self, x: float, y: float, channel: int) -> MeasureResponse:
        if not self.entered or self.finished:
            return MeasureResponse(accepted=False)
        move_s = math.hypot(x - self.pos_x, y - self.pos_y) / self.robot.speed
        switch_s = self.robot.switch_time if channel != self.channel else 0.0
        self._advance(move_s + switch_s + self.robot.detection_time)
        self.pos_x, self.pos_y = x, y
        self.channel = channel

        j = self.case.jammer_on(channel)
        if j is None or j.cleared:
            return MeasureResponse(accepted=True, virtual_time_s=self._vt(), measure_result="no_signal")

        dist = self._distance(x, y, j)
        in_cov = self._in_coverage(x, y, j)
        if in_cov and dist <= self.robot.too_strong_radius:
            return MeasureResponse(accepted=True, virtual_time_s=self._vt(), measure_result="near")
        if dist <= j.r_eff and in_cov:
            true_b = self._true_bearing(x, y, j)
            err = self.case.error_field(x, y)
            svd = round((true_b + err) % 360.0, 2) % 360.0
            return MeasureResponse(accepted=True, virtual_time_s=self._vt(),
                                   measure_result="direction", svd_deg=svd)
        return MeasureResponse(accepted=True, virtual_time_s=self._vt(), measure_result="no_signal")

    def clear(self, x: float, y: float, channel: int) -> ClearResponse:
        if not self.entered or self.finished:
            return ClearResponse(accepted=False)
        move_s = math.hypot(x - self.pos_x, y - self.pos_y) / self.robot.speed
        j = self.case.jammer_on(channel)
        hit = (j is not None) and (not j.cleared) and (self._distance(x, y, j) <= self.robot.optical_radius)
        self._advance(move_s + (self.robot.clear_hit_time if hit else self.robot.clear_miss_time))
        self.pos_x, self.pos_y = x, y
        if hit:
            j.cleared = True
            return ClearResponse(accepted=True, virtual_time_s=self._vt(), clear_result="success")
        return ClearResponse(accepted=True, virtual_time_s=self._vt(), clear_result="no_target_in_range")

    def exit(self) -> ExitResponse:
        self.finished = True
        return ExitResponse(accepted=True, virtual_time_s=self._vt(), exit_reason="user_exit")


# ---- 随机布局生成（仅用于测试/调参，不作为主模型先验）----
def random_case(seed: int = 0, problem: Optional[ProblemConstants] = None,
                q4: bool = False, error_field: Optional[ErrorField] = None,
                n_sources: Optional[int] = None) -> MockCase:
    """
    随机生成一局。注意：这只是【测试数据生成器】，主算法绝不依赖此处的分布假设。
    仅保证满足题目硬约束：源在圆内、R∈[1000,1500]、频道互异、数量 10..16。
    """
    p = problem or ProblemConstants()
    rng = random.Random(seed)
    n = n_sources if n_sources is not None else rng.randint(p.source_count_min, p.source_count_max)
    channels = rng.sample(range(1, p.channels + 1), n)
    jammers: List[MockJammer] = []
    for c in channels:
        # 圆内均匀
        while True:
            r = p.area_radius * math.sqrt(rng.random())
            a = rng.uniform(0, 2 * math.pi)
            x, y = r * math.cos(a), r * math.sin(a)
            if x * x + y * y <= p.area_radius ** 2:
                break
        r_eff = rng.uniform(p.receive_radius_min, p.receive_radius_max)
        if q4 and rng.random() < 0.4:
            kind = SourceType.DIRECTIONAL
            direction = rng.uniform(0, 360)
        else:
            kind = SourceType.OMNI
            direction = 0.0
        jammers.append(MockJammer(channel=c, x=x, y=y, r_eff=r_eff, kind=kind, direction_deg=direction))
    field = error_field or make_smooth_field(seed=seed)
    return MockCase(jammers=jammers, error_field=field)
