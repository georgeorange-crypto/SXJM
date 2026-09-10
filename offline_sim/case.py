"""
案例（Case）与干扰源（Jammer）模型 + 局部示向度误差场（ErrorField）。

严格对应文件规则：
- 附件2 §1.1 目标区域：半径 1800 m 圆形，圆心原点，x 正东、y 正北，单位米；所有干扰源在区域内。
- 附件2 §1.3 频道：整数 1..20，每频道至多一个干扰源，一局总数 10..16（个数不通过接口返回）。
- 附件2 §2.1 有效接收半径 R_eff ∈ [1000,1500] m，各源不同，接口不返回。
- 附件2 §2.2 全向 / 定向：定向有效覆盖角为 180°（定向方向两侧各 90°，含边界），方向未知。
- 附件2 §2.3 示向度误差 ∈ [-1°,+1°]：由“检测点所在地点的电磁环境”决定，
  故【同一地点重复测量误差不变】，不同地点才呈统计规律 —— 用一个“按位置固定”的空间场实现。
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from typing import Optional


# ---- 全局常量（来自文件规则）--------------------------------------------------
ARENA_RADIUS_M = 1800.0          # 目标区域半径（附件2 §1.1）
R_EFF_MIN, R_EFF_MAX = 1000.0, 1500.0   # 有效接收半径区间（附件2 §2.1）
DIRECTIONAL_HALF_ANGLE_DEG = 90.0        # 定向覆盖：方向两侧各 90°（附件2 §2.2）
JAMMER_COUNT_MIN, JAMMER_COUNT_MAX = 10, 16   # 一局干扰源总数区间（附件2 §1.3）
CHANNEL_MIN, CHANNEL_MAX = 1, 20              # 频道范围（附件2 §1.3）


def norm_deg(a: float) -> float:
    """把角度归一化到 [0,360)。"""
    a = math.fmod(a, 360.0)
    if a < 0:
        a += 360.0
    # 处理 -0.0 / 边界
    if a >= 360.0:
        a -= 360.0
    return a


def ang_diff(a: float, b: float) -> float:
    """两个方位角的最小夹角，返回 [0,180]。"""
    d = abs(norm_deg(a) - norm_deg(b))
    return d if d <= 180.0 else 360.0 - d


class ErrorField:
    """
    示向度误差场 e(x,y) ∈ [-1°, +1°]，随位置固定、随地点连续变化。

    实现：若干个正弦分量之和，振幅归一化使总幅度 ≤ 1（严格落在 [-1,1]）。
    - 【按位置确定】：同一 (x,y) 永远返回同一误差 → 复现“同一地点重复测量误差不变”。
    - 波长从数十米到约两千米混合 → 近处相关、远处近似独立，符合“不同地点呈统计规律”。
    """

    def __init__(self, seed: int, n_components: int = 8):
        self.seed = int(seed)
        self.n_components = int(n_components)
        rng = random.Random(self.seed ^ 0x9E3779B9)
        amps = [rng.random() for _ in range(self.n_components)]
        s = sum(amps) or 1.0
        amps = [a / s for a in amps]          # Σ|amp| = 1 → 场值严格 ∈ [-1,1]
        self._comps = []
        for a in amps:
            wavelength = rng.uniform(40.0, 2000.0)   # 米
            k = 2.0 * math.pi / wavelength
            theta = rng.uniform(0.0, 2.0 * math.pi)  # 波矢方向
            phase = rng.uniform(0.0, 2.0 * math.pi)
            self._comps.append((a, k * math.cos(theta), k * math.sin(theta), phase))

    def error_deg(self, x: float, y: float) -> float:
        e = 0.0
        for a, kx, ky, ph in self._comps:
            e += a * math.sin(kx * x + ky * y + ph)
        # 数值上严格在 [-1,1]；再夹紧一次以防浮点越界
        if e > 1.0:
            e = 1.0
        elif e < -1.0:
            e = -1.0
        return e


@dataclass
class Jammer:
    channel: int                 # 频道 1..20，唯一
    x: float                     # 位置 x（米）
    y: float                     # 位置 y（米）
    r_eff: float                 # 有效接收半径（米）∈ [1000,1500]
    kind: str                    # "omni" 全向 / "dir" 定向
    direction_deg: Optional[float] = None   # 定向方向（度），全向为 None
    cleared: bool = False        # 是否已被清除（同一源只能清除一次）

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("cleared", None)
        return d


class Case:
    """一局测试案例：干扰源布局 + 误差场 + 时限 + 模式。"""

    def __init__(
        self,
        jammers: list[Jammer],
        field: ErrorField,
        seed: int,
        mode: str = "practice",              # practice 演练（结束可揭示真值）/ formal 正式
        max_virtual_duration_s: float = 360000.0,   # 附件2 §4.5 虚拟世界限时
        max_real_duration_s: int = 1200,             # 附件2 §4.5 现实限时（20 分钟）
    ):
        self.jammers = jammers
        self.field = field
        self.seed = seed
        self.mode = mode
        self.max_virtual_duration_s = float(max_virtual_duration_s)
        self.max_real_duration_s = int(max_real_duration_s)
        self._by_channel = {j.channel: j for j in jammers}

    # -- 查询 --
    def jammer_on_channel(self, channel: int) -> Optional[Jammer]:
        return self._by_channel.get(channel)

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
        """演练测试结束后可揭示的真值（正式测试不得对外显示）。"""
        return {
            "total": self.total,
            "n_omni": self.n_omni,
            "n_dir": self.n_dir,
            "jammers": [
                {**j.to_dict(), "cleared": j.cleared} for j in self.jammers
            ],
        }

    # -- 持久化（便于复现某个具体场景）--
    def to_json(self) -> str:
        return json.dumps(
            {
                "seed": self.seed,
                "mode": self.mode,
                "max_virtual_duration_s": self.max_virtual_duration_s,
                "max_real_duration_s": self.max_real_duration_s,
                "field_seed": self.field.seed,
                "field_components": self.field.n_components,
                "jammers": [j.to_dict() for j in self.jammers],
            },
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def from_json(text: str) -> "Case":
        d = json.loads(text)
        field = ErrorField(d["field_seed"], d.get("field_components", 8))
        jammers = [
            Jammer(
                channel=int(j["channel"]),
                x=float(j["x"]),
                y=float(j["y"]),
                r_eff=float(j["r_eff"]),
                kind=str(j["kind"]),
                direction_deg=(None if j.get("direction_deg") is None
                               else float(j["direction_deg"])),
            )
            for j in d["jammers"]
        ]
        return Case(
            jammers=jammers,
            field=field,
            seed=int(d.get("seed", 0)),
            mode=str(d.get("mode", "practice")),
            max_virtual_duration_s=float(d.get("max_virtual_duration_s", 360000.0)),
            max_real_duration_s=int(d.get("max_real_duration_s", 1200)),
        )


def _sample_point_in_disk(rng: random.Random, radius: float) -> tuple[float, float]:
    """在半径 radius 的圆盘内均匀采样一点。"""
    r = radius * math.sqrt(rng.random())
    t = rng.uniform(0.0, 2.0 * math.pi)
    return r * math.cos(t), r * math.sin(t)


def generate_case(
    seed: Optional[int] = None,
    problem: int = 3,                      # 3=全为全向；4=全向+定向混合
    n_jammers: Optional[int] = None,       # None → 随机 10..16
    n_directional: Optional[int] = None,   # 仅 problem=4 有意义；None → 随机
    mode: str = "practice",
    margin_m: float = 30.0,                # 干扰源离区域边界的最小内缩，避免贴边
) -> Case:
    """
    随机生成一局案例，严格满足文件约束：
    - 频道互不相同、取自 1..20；总数 ∈ [10,16]。
    - 位置在半径 1800 圆内。
    - R_eff ∈ [1000,1500]。
    - problem=3：全部 omni；problem=4：随机若干个 dir（方向随机、未知）。
    """
    rng = random.Random(seed if seed is not None else random.randrange(1 << 30))

    if n_jammers is None:
        n_jammers = rng.randint(JAMMER_COUNT_MIN, JAMMER_COUNT_MAX)
    n_jammers = max(JAMMER_COUNT_MIN, min(JAMMER_COUNT_MAX, int(n_jammers)))

    channels = rng.sample(range(CHANNEL_MIN, CHANNEL_MAX + 1), n_jammers)

    if problem == 4:
        if n_directional is None:
            n_directional = rng.randint(1, max(1, n_jammers - 1))
        n_directional = max(0, min(n_jammers, int(n_directional)))
    else:
        n_directional = 0

    dir_flags = [True] * n_directional + [False] * (n_jammers - n_directional)
    rng.shuffle(dir_flags)

    jammers: list[Jammer] = []
    for ch, is_dir in zip(channels, dir_flags):
        x, y = _sample_point_in_disk(rng, ARENA_RADIUS_M - margin_m)
        r_eff = rng.uniform(R_EFF_MIN, R_EFF_MAX)
        if is_dir:
            jammers.append(Jammer(ch, x, y, r_eff, "dir",
                                  direction_deg=rng.uniform(0.0, 360.0)))
        else:
            jammers.append(Jammer(ch, x, y, r_eff, "omni", None))

    field = ErrorField(seed=(rng.randrange(1 << 30) if seed is None else (seed * 2654435761 & 0x7FFFFFFF)))
    return Case(jammers=jammers, field=field, seed=(seed or 0), mode=mode)
