"""
虚拟世界引擎（Engine）：不含 HTTP，只负责物理规则与计时规则的精确落实。

对应文件规则要点（逐条落实，注释标注出处）：

物理（附件2 §2；附件1 §2）：
- §2.1 有效接收半径：距离 > R_eff 收不到该频道信号。
- §2.2 定向：需距离 ≤ R_eff 且检测点在覆盖角（方向两侧各 90°，含边界，共 180°）内。
       全向：仅需距离 ≤ R_eff。信号均匀辐射、场强只随距离衰减、与角度无关 → 不返回场强。
- §2.3 示向度误差 ∈[-1,1]°，保留两位小数，按 [0,360) 归一化；由地点决定（同点重复不变）。
- §2.4 近距离阈值 5 m：距离 ≤ 5 且在覆盖角内 → measure_result="near"，不返回 svd_deg。
       清除半径 20 m：/clear 与目标距离 ≤ 20 → 成功（与定向朝向无关）。

计时（附件2 §4；附件1 §2）：
- §4.1 只有 accepted=true 才推进虚拟钟；/enter、/exit 即使 accepted=true 也不推进。
       virtual_time_s 为 JSON number，可带小数，内部按微秒累计。
- §4.2 移动耗时 = 直线距离 / 5 (m/s)，从“上一次合法动作的位置”起算。
- §4.3 /measure 总耗时 = 移动 + 切换频道(变则 1 s) + 检测 5 s；完成后当前频道更新为本次频道。
- §4.4 /clear 总耗时 = 移动 + (未发现 3 s / 成功 5 s)；不切换频道、不改变当前频道。
- §4.5 现实/虚拟时限；机器狗成功 /enter 后开始计现实时间。

状态（附件1 §1；附件2 §1.4）：
- 初始位置 (0,0)，测向机初始频道 1，/enter 不推进虚拟钟。
- 到达/超过任一时限：结束测试、关闭接口。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from .case import Case, Jammer, norm_deg, ang_diff, DIRECTIONAL_HALF_ANGLE_DEG


# ---- 计时常量（附件1 表1 / 附件2 §4）----------------------------------------
SPEED_MPS = 5.0                 # 机器狗速度 5 m/s
CH_SWITCH_S = 1.0               # 切换频道耗时 1 s
MEASURE_ACTION_S = 5.0          # 检测动作耗时 5 s
CLEAR_MISS_S = 3.0              # /clear 未发现（仅精定位）3 s
CLEAR_HIT_S = 5.0               # /clear 成功（精定位+清除）5 s
NEAR_THRESHOLD_M = 5.0          # 近距离阈值
CLEAR_RADIUS_M = 20.0           # 清除半径
COORD_ABS_MAX = 2_000_000.0     # 坐标分量绝对值上限（附件2 §1.1/§7.4）


@dataclass
class MeasureOutcome:
    """检测结果（业务层）。"""
    result: str                     # "no_signal" | "near" | "direction"
    svd_deg: Optional[float] = None # 仅 direction 时有值，已含误差并保留两位小数


@dataclass
class ClearOutcome:
    """清除结果（业务层）。"""
    result: str                     # "success" | "no_target_in_range"


@dataclass
class EngineState:
    entered: bool = False
    finished: bool = False
    finish_reason: Optional[str] = None     # user_exit / timeout_real / timeout_virtual / aborted
    pos_x: float = 0.0                      # 上一次合法动作位置（附件2 §4.2）
    pos_y: float = 0.0
    channel: int = 1                        # 测向机当前频道（初始 1）
    virtual_time_s: float = 0.0             # 虚拟时钟（秒）
    virtual_time_us: int = 0                # 内部按微秒累计，避免浮点漂移
    enter_real_ts_ms: Optional[float] = None
    remaining_real_duration_s: int = 1200


class Engine:
    """
    纯逻辑引擎。HTTP 层（server.py）在校验通过后调用这里的 enter/measure/clear/exit。
    时间返回统一用微秒累计再换算为秒，最多保留 6 位小数（附件2 §4.1）。
    """

    def __init__(self, case: Case, real_clock=time.time):
        self.case = case
        self.st = EngineState()
        self.st.remaining_real_duration_s = case.max_real_duration_s
        self._real_clock = real_clock       # 便于测试注入假时钟

    # ---------- 时间辅助 ----------
    @staticmethod
    def _us(seconds: float) -> int:
        """秒 → 微秒（四舍五入），内部累计用。"""
        return int(round(seconds * 1_000_000))

    def _vt_seconds(self) -> float:
        """当前虚拟时刻（秒），最多 6 位小数、去除无意义末尾零由 JSON 层处理。"""
        return self.st.virtual_time_us / 1_000_000.0

    def _advance_us(self, micros: int) -> None:
        self.st.virtual_time_us += micros
        self.st.virtual_time_s = self._vt_seconds()

    def _now_ms(self) -> float:
        return self._real_clock() * 1000.0

    # ---------- 时限检查 ----------
    def _virtual_over(self) -> bool:
        return self._vt_seconds() >= self.case.max_virtual_duration_s

    def _real_over(self) -> bool:
        if self.st.enter_real_ts_ms is None:
            return False
        elapsed = (self._now_ms() - self.st.enter_real_ts_ms) / 1000.0
        return elapsed >= self.st.remaining_real_duration_s

    def _check_deadlines_before_action(self) -> Optional[str]:
        """
        动作开始前的时限检查。截止前已登记的动作允许完成，
        在截止时刻或之后到达的请求不再执行（附件2 §4.5）。
        返回结束原因（若已到时限），否则 None。
        """
        if self._virtual_over():
            return "timeout_virtual"
        if self._real_over():
            return "timeout_real"
        return None

    # ---------- 物理判定 ----------
    def _distance(self, x: float, y: float, j: Jammer) -> float:
        return math.hypot(x - j.x, y - j.y)

    def _in_coverage(self, x: float, y: float, j: Jammer) -> bool:
        """
        检测点是否在该源覆盖角内。
        全向：恒 True。定向：检测点相对源方位角与定向方向夹角 ≤ 90°（含边界）。
        注意：定向覆盖判定用的是“源指向检测点”的方向 vs 定向方向。
        """
        if j.kind == "omni":
            return True
        bearing_src_to_pt = norm_deg(math.degrees(math.atan2(y - j.y, x - j.x)))
        return ang_diff(bearing_src_to_pt, j.direction_deg) <= DIRECTIONAL_HALF_ANGLE_DEG + 1e-9

    def _true_bearing_pt_to_src(self, x: float, y: float, j: Jammer) -> float:
        """检测点指向源的真实方位角（度，[0,360)）。"""
        return norm_deg(math.degrees(math.atan2(j.y - y, j.x - x)))

    def _signal_visible(self, x: float, y: float, j: Jammer) -> bool:
        """在 (x,y) 能否收到该（未清除）源信号：距离 ≤ R_eff 且在覆盖角内。"""
        if j.cleared:
            return False
        if self._distance(x, y, j) > j.r_eff:
            return False
        return self._in_coverage(x, y, j)

    # ---------- 动作：enter ----------
    def enter(self) -> dict:
        """
        /enter：进入目标区域。accepted=true 也不推进虚拟钟（附件2 §6.2）。
        返回附加字段 max_virtual_duration_s / max_real_duration_s / remaining_real_duration_s。
        """
        self.st.entered = True
        self.st.finished = False
        self.st.finish_reason = None
        self.st.pos_x, self.st.pos_y = 0.0, 0.0
        self.st.channel = 1
        self.st.virtual_time_us = 0
        self.st.virtual_time_s = 0.0
        self.st.enter_real_ts_ms = self._now_ms()
        return {
            "virtual_time_s": self._vt_seconds(),
            "max_virtual_duration_s": self.case.max_virtual_duration_s,
            "max_real_duration_s": self.case.max_real_duration_s,
            "remaining_real_duration_s": self.st.remaining_real_duration_s,
        }

    # ---------- 动作：measure ----------
    def measure(self, x: float, y: float, channel: int) -> tuple[dict, Optional[MeasureOutcome]]:
        """
        /measure：到 (x,y) 对 channel 检测。返回 (业务响应字段, MeasureOutcome)。
        若时限已到，标记结束并返回 finished 标志（HTTP 层据此关闭连接）。
        """
        reason = self._check_deadlines_before_action()
        if reason is not None:
            self._finish(reason)
            return {"__finished__": True}, None

        # 计时：移动 + 切换频道 + 检测（附件2 §4.3）
        move_s = math.hypot(x - self.st.pos_x, y - self.st.pos_y) / SPEED_MPS
        switch_s = CH_SWITCH_S if channel != self.st.channel else 0.0
        total_us = self._us(move_s) + self._us(switch_s) + self._us(MEASURE_ACTION_S)
        self._advance_us(total_us)

        # 状态推进：位置、当前频道（附件2 §4.3：完成后当前频道更新为本次频道）
        self.st.pos_x, self.st.pos_y = x, y
        self.st.channel = channel

        outcome = self._eval_measure(x, y, channel)
        resp = {"virtual_time_s": self._vt_seconds(), "measure_result": outcome.result}
        if outcome.result == "direction":
            resp["svd_deg"] = outcome.svd_deg
        return resp, outcome

    def _eval_measure(self, x: float, y: float, channel: int) -> MeasureOutcome:
        j = self.case.jammer_on_channel(channel)
        if j is None or j.cleared:
            return MeasureOutcome("no_signal")

        dist = self._distance(x, y, j)
        in_cov = self._in_coverage(x, y, j)

        # 距离过近：≤5 m 且在覆盖角内（附件2 §2.4）。定向在盲区则收不到 → 不算 near。
        if in_cov and dist <= NEAR_THRESHOLD_M:
            return MeasureOutcome("near")

        # 有信号：距离 ≤ R_eff 且在覆盖角内 → 返回含误差示向度
        if dist <= j.r_eff and in_cov:
            true_bearing = self._true_bearing_pt_to_src(x, y, j)
            err = self.case.field.error_deg(x, y)     # ∈[-1,1]°，按位置固定
            # 先加误差归一化，round 两位后【再归一化一次】：
            # 例如 359.996→round→360.00 不属于 [0,360)，必须再 mod 回 0.00。
            svd = round(norm_deg(true_bearing + err), 2)
            svd = norm_deg(svd)
            return MeasureOutcome("direction", svd_deg=svd)

        # 其余：无信号（超距 / 定向盲区 / 已清）
        return MeasureOutcome("no_signal")

    # ---------- 动作：clear ----------
    def clear(self, x: float, y: float, channel: int) -> tuple[dict, Optional[ClearOutcome]]:
        """
        /clear：到 (x,y) 尝试清除 channel 的源。不切换频道、不改变当前频道（附件2 §4.4/§8.2）。
        """
        reason = self._check_deadlines_before_action()
        if reason is not None:
            self._finish(reason)
            return {"__finished__": True}, None

        move_s = math.hypot(x - self.st.pos_x, y - self.st.pos_y) / SPEED_MPS

        j = self.case.jammer_on_channel(channel)
        hit = (j is not None) and (not j.cleared) and (self._distance(x, y, j) <= CLEAR_RADIUS_M)
        # 注：清除与定向朝向无关，只看距离（附件2 §8.2）。

        action_s = CLEAR_HIT_S if hit else CLEAR_MISS_S
        total_us = self._us(move_s) + self._us(action_s)
        self._advance_us(total_us)

        # 位置推进；当前频道不变（/clear 不切频道）
        self.st.pos_x, self.st.pos_y = x, y

        if hit:
            j.cleared = True
            outcome = ClearOutcome("success")
        else:
            outcome = ClearOutcome("no_target_in_range")

        resp = {"virtual_time_s": self._vt_seconds(), "clear_result": outcome.result}
        return resp, outcome

    # ---------- 动作：exit ----------
    def exit(self) -> dict:
        """/exit：主动结束。不推进虚拟钟（附件2 §9.2），exit_reason=user_exit。"""
        self._finish("user_exit")
        return {"virtual_time_s": self._vt_seconds(), "exit_reason": "user_exit"}

    def _finish(self, reason: str) -> None:
        self.st.finished = True
        self.st.finish_reason = reason

    # ---------- 供服务器读取的现实时间戳 ----------
    def real_timestamp_ms(self) -> float:
        return self._now_ms()
