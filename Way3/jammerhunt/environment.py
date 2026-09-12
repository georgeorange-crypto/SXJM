"""
离线虚拟世界：干扰源布局 + 物理规则 + 计时规则的独立实现（Way3 自带，便于蒙特卡洛）。

严格对齐 附件1/附件2：
- 区域：半径 1800 m 圆盘，圆心原点，x 东 y 北（附件2 §1.1）。
- 频道 1..20，每频道至多一个源，一局总数 10..16（附件2 §1.3）。
- 有效接收半径 R_eff ∈ [1000,1500] m，接口不返回（附件2 §2.1）。
- 全向：距离 ≤ R_eff 可测；定向：还需检测点在源覆盖角内（方向两侧各 90°，共 180°，含边界）（附件2 §2.2）。
- 示向度误差 ∈ [-1,1]°，按“地点”固定（同点重复不变），保留两位小数，[0,360) 归一（附件2 §2.3）。
- 近距离 ≤5 m 且在覆盖角内 → near，无示向度；清除半径 20 m，成功与朝向无关（附件2 §2.4/§8.2）。
- 计时：移动=距离/5；切频=1s（仅 /measure）；检测=5s；清除未中=3s、命中=5s（附件2 §4）。
- /enter、/exit 不推进虚拟钟；/measure、/clear 推进（附件2 §4.1）。

计时用微秒整数累计，避免浮点漂移（附件2 §4.1：内部按微秒累计）。
本模块经 test_environment 校验可精确复现 附件2 §10 计时示例 [105,111,194,199]。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field as dc_field
from typing import Optional

# ---- 物理/计时常量 ---------------------------------------------------------- #
ARENA_RADIUS_M = 1800.0
R_EFF_MIN, R_EFF_MAX = 1000.0, 1500.0
DIRECTIONAL_HALF_ANGLE_DEG = 90.0
N_MIN, N_MAX = 10, 16
CH_MIN, CH_MAX = 1, 20

SPEED_MPS = 5.0
CH_SWITCH_S = 1.0
MEASURE_S = 5.0
CLEAR_MISS_S = 3.0
CLEAR_HIT_S = 5.0
NEAR_M = 5.0
CLEAR_RADIUS_M = 20.0

MAX_VIRTUAL_S = 360000.0
MAX_REAL_S = 1200


def _norm_deg(a: float) -> float:
    a = math.fmod(a, 360.0)
    if a < 0.0:
        a += 360.0
    if a >= 360.0:
        a -= 360.0
    return a


def _ang_diff(a: float, b: float) -> float:
    d = abs(_norm_deg(a) - _norm_deg(b))
    return d if d <= 180.0 else 360.0 - d


# --------------------------------------------------------------------------- #
# 示向度误差场：确定性、按位置固定、严格 |e|≤1°
# --------------------------------------------------------------------------- #
class ErrorField:
    """
    默认平滑误差场：若干正弦波叠加，L1 归一保证 |e|≤amp≤1°，且是 (x,y) 的确定函数
    （同一地点重复测量必然给同一误差，符合附件2 §2.3“误差由地点决定”）。
    """

    def __init__(self, seed: int = 0, n_waves: int = 10, length_scale: float = 350.0,
                 amp: float = 1.0):
        rng = random.Random(seed)
        self.amp = min(1.0, amp)
        self._waves = []
        wsum = 0.0
        raw = []
        for _ in range(n_waves):
            ang = rng.uniform(0, 2 * math.pi)
            k = (2 * math.pi / length_scale) * rng.uniform(0.5, 1.5)
            kx, ky = k * math.cos(ang), k * math.sin(ang)
            phase = rng.uniform(0, 2 * math.pi)
            w = rng.uniform(0.3, 1.0)
            raw.append((kx, ky, phase, w))
            wsum += w
        for kx, ky, phase, w in raw:
            self._waves.append((kx, ky, phase, w / wsum))   # Σw=1 → |Σ w sin| ≤ 1

    def error_deg(self, x: float, y: float) -> float:
        e = 0.0
        for kx, ky, phase, w in self._waves:
            e += w * math.sin(kx * x + ky * y + phase)
        return self.amp * e

    def config(self) -> dict:
        return {"kind": "smooth", "amp": self.amp, "n_waves": len(self._waves)}


class ConstantField(ErrorField):
    """常数误差场（最坏情况：全场恒为 +v 或 -v，用于鲁棒性上界测试）。"""

    def __init__(self, value: float = 1.0):
        self.value = max(-1.0, min(1.0, value))

    def error_deg(self, x: float, y: float) -> float:
        return self.value

    def config(self) -> dict:
        return {"kind": "constant", "value": self.value}


# --------------------------------------------------------------------------- #
# 干扰源 / 案例
# --------------------------------------------------------------------------- #
@dataclass
class Jammer:
    channel: int
    x: float
    y: float
    r_eff: float
    kind: str                       # "omni" | "dir"
    direction_deg: Optional[float] = None
    cleared: bool = False

    @property
    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Case:
    jammers: list[Jammer]
    field: ErrorField = dc_field(default_factory=ErrorField)
    seed: int = 0
    problem: int = 3
    mode: str = "practice"
    max_virtual_duration_s: float = MAX_VIRTUAL_S
    max_real_duration_s: int = MAX_REAL_S

    def __post_init__(self):
        self._by_ch = {j.channel: j for j in self.jammers}

    def jammer_on_channel(self, ch: int) -> Optional[Jammer]:
        return self._by_ch.get(ch)

    @property
    def total(self) -> int:
        return len(self.jammers)

    @property
    def n_omni(self) -> int:
        return sum(1 for j in self.jammers if j.kind == "omni")

    @property
    def n_dir(self) -> int:
        return sum(1 for j in self.jammers if j.kind == "dir")

    @property
    def cleared_count(self) -> int:
        return sum(1 for j in self.jammers if j.cleared)

    def reveal(self) -> dict:
        return {
            "total": self.total, "n_omni": self.n_omni, "n_dir": self.n_dir,
            "jammers": [vars(j) for j in self.jammers],
        }


def _sample_in_disk(rng: random.Random, radius: float) -> tuple[float, float]:
    r = radius * math.sqrt(rng.random())
    t = rng.uniform(0.0, 2 * math.pi)
    return r * math.cos(t), r * math.sin(t)


def generate_case(
    seed: Optional[int] = None,
    problem: int = 3,
    n_jammers: Optional[int] = None,
    n_directional: Optional[int] = None,
    mode: str = "practice",
    margin_m: float = 30.0,
    field_seed: Optional[int] = None,
    field_kind: str = "smooth",
) -> Case:
    """
    随机生成一局案例（严格满足硬约束）。
    problem=3 全部全向；problem=4 至少 1 全向 + 至少 1 定向（方向随机、未知）。
    """
    rng = random.Random(seed if seed is not None else random.randrange(1 << 30))
    n = n_jammers if n_jammers is not None else rng.randint(N_MIN, N_MAX)
    n = max(N_MIN, min(N_MAX, n))
    channels = rng.sample(range(CH_MIN, CH_MAX + 1), n)

    if problem == 4:
        nd = n_directional if n_directional is not None else rng.randint(1, n - 1)
        nd = max(1, min(n - 1, nd))
    else:
        nd = 0
    flags = [True] * nd + [False] * (n - nd)
    rng.shuffle(flags)

    jammers = []
    for ch, is_dir in zip(channels, flags):
        x, y = _sample_in_disk(rng, ARENA_RADIUS_M - margin_m)
        r_eff = rng.uniform(R_EFF_MIN, R_EFF_MAX)
        if is_dir:
            jammers.append(Jammer(ch, x, y, r_eff, "dir", rng.uniform(0, 360)))
        else:
            jammers.append(Jammer(ch, x, y, r_eff, "omni", None))

    fs = field_seed if field_seed is not None else rng.randrange(1 << 30)
    field = ConstantField(1.0) if field_kind == "constant" else ErrorField(fs)
    return Case(jammers, field, seed=seed or 0, problem=problem, mode=mode)


# --------------------------------------------------------------------------- #
# 引擎（物理 + 计时）
# --------------------------------------------------------------------------- #
@dataclass
class MeasureOutcome:
    result: str                     # "no_signal" | "near" | "direction"
    svd_deg: Optional[float] = None


@dataclass
class ClearOutcome:
    result: str                     # "success" | "no_target_in_range"


class Engine:
    """纯逻辑虚拟世界；不含 HTTP。时间以微秒整数累计。"""

    def __init__(self, case: Case, real_clock=None):
        import time
        self.case = case
        self._clock = real_clock or time.time
        self.entered = False
        self.finished = False
        self.finish_reason: Optional[str] = None
        self.x = 0.0
        self.y = 0.0
        self.channel = 1
        self._vt_us = 0
        self._enter_real_ms: Optional[float] = None
        self.remaining_real_s = case.max_real_duration_s

    # ---- 时间 ----
    @staticmethod
    def _us(sec: float) -> int:
        return int(round(sec * 1_000_000))

    @property
    def virtual_time_s(self) -> float:
        return self._vt_us / 1_000_000.0

    def _advance(self, micros: int) -> None:
        self._vt_us += micros

    def _now_ms(self) -> float:
        return self._clock() * 1000.0

    def _deadline(self) -> Optional[str]:
        if self.virtual_time_s >= self.case.max_virtual_duration_s:
            return "timeout_virtual"
        if self._enter_real_ms is not None:
            if (self._now_ms() - self._enter_real_ms) / 1000.0 >= self.remaining_real_s:
                return "timeout_real"
        return None

    # ---- 物理 ----
    def _in_coverage(self, x: float, y: float, j: Jammer) -> bool:
        if j.kind == "omni":
            return True
        bearing_src_to_pt = _norm_deg(math.degrees(math.atan2(y - j.y, x - j.x)))
        return _ang_diff(bearing_src_to_pt, j.direction_deg) <= DIRECTIONAL_HALF_ANGLE_DEG + 1e-9

    def _eval_measure(self, x: float, y: float, ch: int) -> MeasureOutcome:
        j = self.case.jammer_on_channel(ch)
        if j is None or j.cleared:
            return MeasureOutcome("no_signal")
        d = math.hypot(x - j.x, y - j.y)
        in_cov = self._in_coverage(x, y, j)
        if in_cov and d <= NEAR_M:
            return MeasureOutcome("near")
        if in_cov and d <= j.r_eff:
            true_b = _norm_deg(math.degrees(math.atan2(j.y - y, j.x - x)))
            err = self.case.field.error_deg(x, y)
            svd = _norm_deg(round(_norm_deg(true_b + err), 2))
            return MeasureOutcome("direction", svd_deg=svd)
        return MeasureOutcome("no_signal")

    # ---- 动作 ----
    def enter(self) -> dict:
        self.entered = True
        self.finished = False
        self.finish_reason = None
        self.x = self.y = 0.0
        self.channel = 1
        self._vt_us = 0
        self._enter_real_ms = self._now_ms()
        return {
            "virtual_time_s": self.virtual_time_s,
            "max_virtual_duration_s": self.case.max_virtual_duration_s,
            "max_real_duration_s": self.case.max_real_duration_s,
            "remaining_real_duration_s": self.remaining_real_s,
        }

    def measure(self, x: float, y: float, ch: int):
        r = self._deadline()
        if r:
            self.finished, self.finish_reason = True, r
            return {"__finished__": True}, None
        move = math.hypot(x - self.x, y - self.y) / SPEED_MPS
        switch = CH_SWITCH_S if ch != self.channel else 0.0
        self._advance(self._us(move) + self._us(switch) + self._us(MEASURE_S))
        self.x, self.y, self.channel = x, y, ch
        out = self._eval_measure(x, y, ch)
        resp = {"virtual_time_s": self.virtual_time_s, "measure_result": out.result}
        if out.result == "direction":
            resp["svd_deg"] = out.svd_deg
        return resp, out

    def clear(self, x: float, y: float, ch: int):
        r = self._deadline()
        if r:
            self.finished, self.finish_reason = True, r
            return {"__finished__": True}, None
        move = math.hypot(x - self.x, y - self.y) / SPEED_MPS
        j = self.case.jammer_on_channel(ch)
        hit = (j is not None) and (not j.cleared) and (math.hypot(x - j.x, y - j.y) <= CLEAR_RADIUS_M)
        self._advance(self._us(move) + self._us(CLEAR_HIT_S if hit else CLEAR_MISS_S))
        self.x, self.y = x, y
        if hit:
            j.cleared = True
        out = ClearOutcome("success" if hit else "no_target_in_range")
        return {"virtual_time_s": self.virtual_time_s, "clear_result": out.result}, out

    def exit(self) -> dict:
        self.finished, self.finish_reason = True, "user_exit"
        return {"virtual_time_s": self.virtual_time_s, "exit_reason": "user_exit"}

    def real_timestamp_ms(self) -> float:
        return self._now_ms()
