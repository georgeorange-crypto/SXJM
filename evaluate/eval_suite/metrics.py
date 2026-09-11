"""
计时分解 + 核心指标。

把一局的虚拟时间【精确】拆成四块，与 environment.Engine 的计时规则一一对应：
    T = T_move + T_detect + T_switch + T_clear
其中
    T_move   = Σ 每次 measure/clear 的移动时间 (距离/5)
    T_detect = 5s × measure 次数                (MEASURE_S)
    T_switch = 1s × measure 中发生的换频次数     (CH_SWITCH_S，仅 /measure)
    T_clear  = 5s × 命中次数 + 3s × 未命中次数    (CLEAR_HIT_S / CLEAR_MISS_S)
InstrumentedWorld 包住任意 World，逐动作累加，最终 total 必等于底层 engine.virtual_time_s
（见 tests 校验）。这让“额外移动 / 额外感知”可被干净地拆出来。

核心指标（针对已知真值的离线案例）：
    R_LB    = T / T_ABS_LB              相对绝对物理下界的倍率（核心统一指标）
    R_oracle= T / T_OneShotOracle       相对“一搜即知”理想探测器的倍率
    ΔT_move = T_move - T_move_LB         额外移动代价（因不知真位置多走的路）
    ΔT_info = T_detect + T_switch - 119  额外感知代价（相对 20 频道各搜一次的理想 119s）
    C_move / C_sense 把非理想损失 (T - T_oracle) 归因到移动 vs 感知。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .interface_shim import ClearObs, MeasureObs, World
from .lowerbound import PER_SOURCE_S, SPEED_MPS, TSPNCertificate

# One-shot Oracle 的理想探测开销：20 频道各测一次 = 100s 检测 + 19s 切频（初始频道 1）。
ORACLE_SCAN_DETECT_S = 20 * 5.0        # 100
ORACLE_SCAN_SWITCH_S = 19 * 1.0        # 19
ORACLE_SCAN_S = ORACLE_SCAN_DETECT_S + ORACLE_SCAN_SWITCH_S   # 119


@dataclass
class TimeBreakdown:
    """一局虚拟时间的精确分解（秒）。"""
    move_s: float = 0.0
    detect_s: float = 0.0
    switch_s: float = 0.0
    clear_s: float = 0.0
    # 计数
    n_measure: int = 0
    n_switch: int = 0
    n_clear: int = 0
    n_clear_hit: int = 0
    n_clear_miss: int = 0
    move_m: float = 0.0                 # 总移动距离（米）

    @property
    def total_s(self) -> float:
        return self.move_s + self.detect_s + self.switch_s + self.clear_s


class InstrumentedWorld(World):
    """包住任意底层 World，按 Engine 计时规则累加计时分解。策略无感知。"""

    def __init__(self, inner: World):
        self.inner = inner
        self.bd = TimeBreakdown()

    # ---- 计时常量（与 environment 对齐；此处独立持有，避免耦合） ----
    MEASURE_S = 5.0
    CH_SWITCH_S = 1.0
    CLEAR_HIT_S = 5.0
    CLEAR_MISS_S = 3.0

    def enter(self) -> bool:
        return self.inner.enter()

    def measure(self, x, y, channel) -> MeasureObs:
        prev = self.inner.position
        prev_ch = self.inner.channel
        obs = self.inner.measure(x, y, channel)
        if not obs.ok:
            return obs
        self.bd.n_measure += 1
        d = math.hypot(x - prev[0], y - prev[1])
        self.bd.move_m += d
        self.bd.move_s += d / SPEED_MPS
        self.bd.detect_s += self.MEASURE_S
        if int(channel) != int(prev_ch):
            self.bd.n_switch += 1
            self.bd.switch_s += self.CH_SWITCH_S
        return obs

    def clear(self, x, y, channel) -> ClearObs:
        prev = self.inner.position
        obs = self.inner.clear(x, y, channel)
        if not obs.ok:
            return obs
        self.bd.n_clear += 1
        d = math.hypot(x - prev[0], y - prev[1])
        self.bd.move_m += d
        self.bd.move_s += d / SPEED_MPS
        if obs.success:
            self.bd.n_clear_hit += 1
            self.bd.clear_s += self.CLEAR_HIT_S
        else:
            self.bd.n_clear_miss += 1
            self.bd.clear_s += self.CLEAR_MISS_S
        return obs

    def exit(self) -> None:
        self.inner.exit()

    @property
    def position(self):
        return self.inner.position

    @property
    def channel(self) -> int:
        return self.inner.channel

    @property
    def virtual_time_s(self) -> float:
        return self.inner.virtual_time_s

    @property
    def finished(self) -> bool:
        return self.inner.finished


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeMetrics:
    """单局的完整指标：清除结果 + 计时分解 + 相对下界/Oracle 的倍率与归因。"""
    seed: int
    problem: int
    method: str
    total: int                     # 真实源数 m
    cleared: int
    success: bool                  # 全清
    n_omni: int = 0
    n_dir: int = 0
    finish_reason: Optional[str] = None
    error: Optional[str] = None

    T: float = 0.0                 # 总虚拟时间
    bd: TimeBreakdown = field(default_factory=TimeBreakdown)

    # 下界 / Oracle（来自真值证书）
    T_abs_lb: float = 0.0
    T_oracle: float = 0.0
    T_move_lb: float = 0.0         # L_LB / 5
    L_LB: float = 0.0
    L_UB: float = 0.0
    cert_gap_rel: float = 0.0

    # 派生倍率
    @property
    def R_LB(self) -> float:
        return self.T / self.T_abs_lb if self.T_abs_lb > 1e-9 else math.inf

    @property
    def R_oracle(self) -> float:
        return self.T / self.T_oracle if self.T_oracle > 1e-9 else math.inf

    @property
    def dT_move(self) -> float:
        """额外移动代价 = 实际移动 - 必要移动 (L_LB/5)。"""
        return self.bd.move_s - self.T_move_lb

    @property
    def dT_info(self) -> float:
        """额外感知代价 = 实际(检测+切频) - 理想 119s。"""
        return self.bd.detect_s + self.bd.switch_s - ORACLE_SCAN_S

    @property
    def gap_to_oracle(self) -> float:
        return self.T - self.T_oracle

    @property
    def C_move(self) -> float:
        """非理想损失里“额外移动”占比（C_move + C_sense ≈ 1，忽略清除未命中）。"""
        g = self.gap_to_oracle
        return self.dT_move / g if abs(g) > 1e-9 else 0.0

    @property
    def C_sense(self) -> float:
        g = self.gap_to_oracle
        return self.dT_info / g if abs(g) > 1e-9 else 0.0


def oracle_time(cert: TSPNCertificate) -> float:
    """One-shot Oracle 总时间 = 119 (理想扫描) + L_TSPN*/5 + 5m。用 L_UB 近似 L_TSPN*（收紧可行）。"""
    return ORACLE_SCAN_S + cert.L_UB / SPEED_MPS + PER_SOURCE_S * cert.m


def attach_certificate(m: EpisodeMetrics, cert: TSPNCertificate) -> EpisodeMetrics:
    """把真值证书写入指标（下界/Oracle/移动下界）。"""
    m.T_abs_lb = cert.T_abs_lb()
    m.T_oracle = oracle_time(cert)
    m.T_move_lb = cert.T_move_LB
    m.L_LB = cert.L_LB
    m.L_UB = cert.L_UB
    m.cert_gap_rel = cert.gap_rel
    return m
