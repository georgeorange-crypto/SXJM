"""
案例（Case）与干扰源（Jammer）模型 + 局部示向度误差场（ErrorField）。

严格对应文件规则：
- 附件2 §1.1 目标区域：半径 1800 m 圆形，圆心原点，x 正东、y 正北，单位米；所有干扰源在区域内。
- 附件2 §1.3 频道：整数 1..20，每频道至多一个干扰源，一局总数 10..16（个数不通过接口返回）。
- 附件2 §2.1 有效接收半径 R_eff ∈ [1000,1500] m，接口不返回。
  注意：题面只规定各源 R_eff 落在该区间，【并未规定各源必须互不相同】，故允许重复。
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

from .fields import BaseErrorField, SmoothErrorField, make_error_field, FIELD_KINDS


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


class ErrorField(SmoothErrorField):
    """
    向后兼容别名：默认误差场（空间相关、严格有界、按位置固定）。

    误差场家族已抽象到 fields.py（iid / smooth / biased / adversarial / piecewise），
    详见那里对"题面只约束有界性+地点一致性、未规定分布"的说明。本类等价于
    SmoothErrorField，仅保留旧构造签名 ErrorField(seed, n_components=...) 以兼容既有代码。
    """

    def __init__(self, seed: int, n_components: int = 12, length_scale: float = 300.0,
                 amplitude: float = 1.0):
        super().__init__(seed, length_scale=length_scale,
                         n_components=n_components, amplitude=amplitude)


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
        field: BaseErrorField,
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
                "field": self.field.config(),   # {"kind","seed","params"}
                "jammers": [j.to_dict() for j in self.jammers],
            },
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def from_json(text: str) -> "Case":
        d = json.loads(text)
        if "field" in d:
            fc = d["field"]
            field = make_error_field(kind=fc.get("kind", "smooth"),
                                     seed=fc.get("seed", 0),
                                     params=fc.get("params"))
        else:
            # 旧格式兼容：field_seed / field_components
            field = ErrorField(d.get("field_seed", 0), d.get("field_components", 12))
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


def _build_field(rng: random.Random, seed: Optional[int],
                 field_kind: str, field_params: Optional[dict]) -> BaseErrorField:
    """按 kind 造误差场；种子从主 rng 或 seed 派生（保证可复现）。"""
    fseed = rng.randrange(1 << 30) if seed is None else (seed * 2654435761 & 0x7FFFFFFF)
    return make_error_field(kind=field_kind, seed=fseed, params=field_params)


def generate_case(
    seed: Optional[int] = None,
    problem: int = 3,                      # 3=全为全向；4=全向+定向混合
    n_jammers: Optional[int] = None,       # None → 随机 10..16
    n_directional: Optional[int] = None,   # 仅 problem=4 有意义；None → 随机
    mode: str = "practice",
    hard: bool = False,
    margin_m: float = 30.0,                # 干扰源离区域边界的最小内缩，避免贴边
    field_kind: str = "smooth",            # 误差场类型（见 fields.py）
    field_params: Optional[dict] = None,   # 误差场参数（如 length_scale/bias/…）
) -> Case:
    """
    随机生成一局案例，严格满足文件约束：
    - 频道互不相同、取自 1..20；总数 ∈ [10,16]。
    - 位置在半径 1800 圆内。
    - R_eff ∈ [1000,1500]（允许不同源相同，题面未规定互异）。
    - problem=3：全部 omni；problem=4：至少 1 个 omni 且至少 1 个 dir（方向随机、未知）。
    """
    rng = random.Random(seed if seed is not None else random.randrange(1 << 30))

    if n_jammers is None:
        n_jammers = rng.randint(JAMMER_COUNT_MIN, JAMMER_COUNT_MAX)
    n_jammers = max(JAMMER_COUNT_MIN, min(JAMMER_COUNT_MAX, int(n_jammers)))

    channels = rng.sample(range(CHANNEL_MIN, CHANNEL_MAX + 1), n_jammers)

    if problem == 4 and hard:
        n_directional = n_jammers
    elif problem == 4:
        # 题面：P4 既有全向也有定向 → 强制 1 ≤ n_dir ≤ n-1（两类都至少 1 个）。
        if n_directional is None:
            n_directional = rng.randint(1, n_jammers - 1)
        n_directional = max(1, min(n_jammers - 1, int(n_directional)))
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

    field = _build_field(rng, seed, field_kind, field_params)
    return Case(jammers=jammers, field=field, seed=(seed or 0), mode=mode)


# ---- 压力 / 最坏情况案例生成器 ------------------------------------------------
STRESS_TYPES = (
    "edge_cluster",       # 全部贴近 r=1800 边界
    "min_reff",           # 所有 R_eff=1000（最小接收半径，最难被发现）
    "tiny_cluster",       # 目标挤在一个很小区域
    "collinear",          # 目标沿一条直线排列（两测向线近乎平行 → 三角定位病态）
    "far_pair",           # 成对相距极近（清除/分辨困难）
    "max_count",          # 16 个源
    "min_count",          # 10 个源
    "dir_outward",        # P4：定向源几乎全部背向原点（原点扫描落在盲区）
    "dir_boundary",       # P4：定向源朝向使目标落在覆盖 90° 边界附近
    "dir_evasive",        # P4：定向朝向专门避开给定扫描点集合
)


def generate_stress_case(
    stress_type: str,
    seed: Optional[int] = None,
    problem: int = 3,
    field_kind: str = "adversarial",
    field_params: Optional[dict] = None,
    scan_points: Optional[list[tuple[float, float]]] = None,
    mode: str = "practice",
) -> Case:
    """
    生成"我们最怕"的极端案例，用于最坏情况鲁棒性评估（配合 adversarial 误差场）。
    仍严格满足题面硬约束（频道互异、位置在圆内、R_eff∈[1000,1500]、P4 两类各≥1）。

    dir_evasive 需要 scan_points（机器狗的既定扫描点集），生成器会把每个定向源
    的朝向调成"背对最近扫描点"，使那些点恰好落在盲区。
    """
    if stress_type not in STRESS_TYPES:
        raise ValueError(f"unknown stress_type {stress_type!r}; valid: {STRESS_TYPES}")
    rng = random.Random(seed if seed is not None else random.randrange(1 << 30))
    R = ARENA_RADIUS_M

    # 源个数
    if stress_type == "max_count":
        n = JAMMER_COUNT_MAX
    elif stress_type == "min_count":
        n = JAMMER_COUNT_MIN
    else:
        n = rng.randint(JAMMER_COUNT_MIN, JAMMER_COUNT_MAX)

    channels = rng.sample(range(CHANNEL_MIN, CHANNEL_MAX + 1), n)

    # 定向源个数：P4 强制两类各 ≥1；dir_* 类型即使 problem=3 也升级为混合
    is_dir_type = stress_type.startswith("dir_")
    if problem == 4 or is_dir_type:
        n_dir = max(1, min(n - 1, rng.randint(max(1, n // 2), n - 1)))
    else:
        n_dir = 0

    # ---- 位置布局 ----
    positions: list[tuple[float, float]] = []
    if stress_type == "edge_cluster":
        for _ in range(n):
            t = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(R - 60.0, R - 5.0)   # 贴边（仍在圆内）
            positions.append((r * math.cos(t), r * math.sin(t)))
    elif stress_type == "tiny_cluster":
        cx, cy = _sample_point_in_disk(rng, R - 300.0)
        for _ in range(n):
            positions.append((cx + rng.uniform(-40, 40), cy + rng.uniform(-40, 40)))
    elif stress_type == "collinear":
        # 过原点附近的一条随机直线，源沿线分布
        ang = rng.uniform(0, math.pi)
        ux, uy = math.cos(ang), math.sin(ang)
        for _ in range(n):
            t = rng.uniform(-(R - 100.0), R - 100.0)
            jitter = rng.uniform(-3.0, 3.0)       # 近乎共线（微抖）
            positions.append((ux * t - uy * jitter, uy * t + ux * jitter))
    elif stress_type == "far_pair":
        # 成对：每对两点相距 <10m，对间散开
        k = 0
        while len(positions) < n:
            cx, cy = _sample_point_in_disk(rng, R - 100.0)
            positions.append((cx, cy))
            if len(positions) < n:
                positions.append((cx + rng.uniform(-6, 6), cy + rng.uniform(-6, 6)))
            k += 1
    else:
        # 其余类型位置用普通圆盘均匀
        for _ in range(n):
            positions.append(_sample_point_in_disk(rng, R - 30.0))

    # ---- R_eff ----
    if stress_type == "min_reff":
        reffs = [R_EFF_MIN] * n
    else:
        reffs = [rng.uniform(R_EFF_MIN, R_EFF_MAX) for _ in range(n)]

    # ---- 定向标记 ----
    dir_flags = [True] * n_dir + [False] * (n - n_dir)
    rng.shuffle(dir_flags)

    # ---- 朝向 ----
    def _dir_for(idx: int, x: float, y: float) -> float:
        if stress_type == "dir_outward":
            # 背向原点：朝向 = 从源指向原点相反 = 源的向外方位
            return norm_deg(math.degrees(math.atan2(y, x)))
        if stress_type == "dir_boundary":
            # 使原点落在覆盖 90° 边界附近：源→原点方位 ± 90°
            to_origin = norm_deg(math.degrees(math.atan2(-y, -x)))
            return norm_deg(to_origin + (90.0 if idx % 2 == 0 else -90.0))
        if stress_type == "dir_evasive" and scan_points:
            # 背对最近扫描点：找最近扫描点 p，朝向 = 源→p 的反方向
            best = min(scan_points, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
            to_scan = math.degrees(math.atan2(best[1] - y, best[0] - x))
            return norm_deg(to_scan + 180.0)
        return rng.uniform(0.0, 360.0)

    jammers: list[Jammer] = []
    for i, (ch, (x, y), reff, is_dir) in enumerate(zip(channels, positions, reffs, dir_flags)):
        # 夹回圆内（布局微抖后可能越界）
        d = math.hypot(x, y)
        if d > R - 3.0:
            s = (R - 3.0) / d
            x, y = x * s, y * s
        if is_dir:
            jammers.append(Jammer(ch, x, y, reff, "dir", direction_deg=_dir_for(i, x, y)))
        else:
            jammers.append(Jammer(ch, x, y, reff, "omni", None))

    field = _build_field(rng, seed, field_kind, field_params)
    return Case(jammers=jammers, field=field, seed=(seed or 0), mode=mode)
