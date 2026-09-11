"""
策略核心：覆盖扫描 → 交会定位 → 归航精定位 → 清除（第三/四问统一，参数区分）。

完备性（“确保清除所有源”）的保证：
- 逐个扫描点、对每个尚未定位的频道做 /measure。若某频道在【全部覆盖扫描点】上都是
  no_signal，则由覆盖保证（coverage 1-覆盖 / 3-覆盖）可断言该频道【无源】；否则为有源。
- 有源频道用不同地点的示向度交会定位，再归航把定位区域收缩到清除半径内并清除。
- 已知总数 ∈[10,16]：一旦已清除数达 16 立即停止（不可能更多），省去剩余扫描。

效率：
- 扫描点用最近邻+2-opt 排短巡回；定位后即停止在后续扫描点重复测该频道（剪枝）。
- 归航用“抵近测量”（standoff）：朝估计位置走到留出 standoff 的近处再测，量程骤减→
  ±1°误差楔形变窄→定位区域迅速 <清除半径；配合清除结果自校验，稳健命中。
- 定向源：抵近沿“测得方位”接近，始终处于覆盖半平面内（沿指向源的直线不出覆盖角）。

仅依赖标准库 + 本包 geometry / coverage / interface。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from . import geometry as geo
from .coverage import omni_scan_points, directional_scan_points
from .interface import World

Point = tuple[float, float]

CH_MIN, CH_MAX = 1, 20
MAX_TOTAL_JAMMERS = 16          # 附件2 §1.3 一局总数上限


@dataclass
class ChannelState:
    ch: int
    obs: list[tuple[Point, float]] = field(default_factory=list)   # (测量点, 示向度)
    detected: bool = False       # 是否曾收到该频道信号（direction/near）
    cleared: bool = False
    absent: bool = False         # 全覆盖点均 no_signal → 判定无源
    est: Optional[Point] = None
    region_r: float = math.inf

    def refresh_fix(self) -> None:
        if len(self.obs) >= 2:
            br = geo.best_region(self.obs)
            if br is not None:
                self.est, self.region_r = br
                return
            est = geo.best_pair_fix(self.obs)
            self.est = est if est is not None else geo.least_squares_fix(self.obs)
            self.region_r = math.inf
        elif len(self.obs) == 1:
            self.est, self.region_r = None, math.inf


@dataclass
class Hunter:
    """一局策略。problem=3（全向 1-覆盖）/ 4（定向 3-覆盖）。"""
    problem: int = 3
    clear_margin_m: float = 14.0     # 定位区域 MEC 半径 ≤ 此值即认为可盲清（< 20 清除半径）
    prune_radius_m: float = 30.0     # 频道定位区域 < 此值后，扫描阶段不再重复测该频道
    orbit_radius_m: float = 85.0     # 归航“绕 est 两点交会”的初始半径
    orbit_delta_deg: float = 42.0    # 两交会点相对‘来向’的角偏移（交会角≈2δ）
    reliable_region_m: float = 300.0 # 定位区域 MEC 半径 > 此值视为不可靠（近平行→远处伪交点）
    baseline_fwd_m: float = 420.0    # 单示向建基线：沿方位朝源前进量（减小量程，保证仍在 r_eff 内）
    baseline_perp_m: float = 320.0   # 单示向建基线：垂直偏移量（制造交会角）
    max_homing_iters: int = 8
    src_radius_m: float = 1800.0     # 定向 3-覆盖假定的源半径上限（默认全盘）

    # 运行期状态
    world: Optional[World] = None
    chans: dict[int, ChannelState] = field(default_factory=dict)

    # ---------------------------------------------------------------- #
    def scan_points(self) -> list[Point]:
        if self.problem == 4:
            return directional_scan_points(src_radius=self.src_radius_m)
        return omni_scan_points()

    @property
    def cleared_count(self) -> int:
        return sum(1 for s in self.chans.values() if s.cleared)

    # ---------------------------------------------------------------- #
    def run(self, world: World) -> None:
        self.world = world
        self.chans = {c: ChannelState(c) for c in range(CH_MIN, CH_MAX + 1)}
        if not world.enter():
            return
        pts = self.scan_points()
        order = geo.ordered_tour(world.position, pts)
        # 阶段 A：覆盖扫描 + 收集示向度
        for idx in order:
            if world.finished or self.cleared_count >= MAX_TOTAL_JAMMERS:
                break
            self._scan_at(pts[idx])
        # 阶段 B：对有源频道归航精定位并清除（按短巡回顺序）
        self._clear_all_present()
        # 结束
        if not world.finished:
            world.exit()

    # ---------------------------------------------------------------- #
    def _active_channels_for_scan(self) -> list[int]:
        """扫描阶段仍需测量的频道：未清除、未判无源、且未被剪枝（定位尚不够准）。"""
        out = []
        for c in range(CH_MIN, CH_MAX + 1):
            s = self.chans[c]
            if s.cleared or s.absent:
                continue
            if s.detected and s.region_r <= self.prune_radius_m:
                continue                     # 已定位够准，不必再测
            out.append(c)
        return sorted(out)

    def _scan_at(self, p: Point) -> None:
        w = self.world
        for c in self._active_channels_for_scan():
            if w.finished:
                return
            obs = w.measure(p[0], p[1], c)
            if not obs.ok:
                return
            s = self.chans[c]
            if obs.is_direction:
                s.detected = True
                s.obs.append((p, obs.svd_deg))
                s.refresh_fix()
            elif obs.is_near:
                # 恰好离该源 <5 m：就地清除（在 20 m 内）
                s.detected = True
                co = w.clear(w.position[0], w.position[1], c)
                if co.ok and co.success:
                    s.cleared = True
            # no_signal：该点未收到；留待“全覆盖点皆无 → 无源”判定

    def _finalize_absence(self) -> None:
        """扫描结束后：从未被探测到的频道判为无源。"""
        for s in self.chans.values():
            if not s.detected and not s.cleared:
                s.absent = True

    # ---------------------------------------------------------------- #
    def _clear_all_present(self) -> None:
        self._finalize_absence()
        w = self.world
        # 待清除频道（有探测、未清除）
        pending = [c for c, s in self.chans.items() if s.detected and not s.cleared]
        # 用当前估计位置排短巡回；无估计的排最后
        known = [c for c in pending if self.chans[c].est is not None]
        unknown = [c for c in pending if self.chans[c].est is None]
        if known:
            pts = [self.chans[c].est for c in known]
            order = geo.ordered_tour(w.position, pts)
            known = [known[i] for i in order]
        for c in known + unknown:
            if w.finished or self.cleared_count >= MAX_TOTAL_JAMMERS:
                break
            self._home_and_clear(c)

    def _home_and_clear(self, c: int) -> bool:
        """
        对频道 c 归航精定位并清除：绕估计位置做“两点交会”把定位区域收缩到清除半径内。
        关键：两测量点都【近】源且交会角≈2δ（良好几何），避免与远处扫描点配对时
        ±1°误差被距离放大（远点 900 m 时 ±1°≈±16 m，恰卡在清除半径边界 → 漏清）。
        """
        w = self.world
        s = self.chans[c]
        R = self.orbit_radius_m
        for _ in range(self.max_homing_iters):
            if w.finished:
                return False
            s.refresh_fix()
            reliable = s.est is not None and s.region_r <= self.reliable_region_m
            # 1) 定位够准 → 盲清于区域中心
            if reliable and s.region_r <= self.clear_margin_m:
                if self._try_clear(c, s.est):
                    return True
                # 理论上不该 miss（真源∈区域⊆该圆）；继续收缩再试
            # 2) 无 est 或 est 不可靠（近平行示向→远处伪交点）→ 从单条示向建“前进+垂直”基线
            if not reliable:
                if not s.obs:
                    return False
                if self._establish_baseline(c, s):
                    return True
                s.refresh_fix()
                if not (s.est is not None and s.region_r <= self.reliable_region_m):
                    self._forward_step(c, s)     # 拉近后下一轮重建基线
                continue
            # 3) est 可靠但不够紧 → 绕 est 两点交会，逐轮收半径逼近
            if self._orbit_triangulate(c, s, R):
                return True
            R = max(35.0, R * 0.6)
        # 兜底：仅当存在够紧的估计才清（不拿远处伪交点乱清，免得徒增 miss 用时）
        s.refresh_fix()
        if s.est is not None and s.region_r <= self.prune_radius_m:
            return self._try_clear(c, s.est)
        return False

    def _forward_step(self, c: int, s: ChannelState) -> None:
        """沿最近一条示向朝源前进一步（减小量程），为下一轮建更好基线。"""
        if not s.obs:
            return
        p, b = s.obs[-1]
        ux, uy = geo.unit(b)
        self._measure_into(c, (p[0] + self.baseline_fwd_m * ux, p[1] + self.baseline_fwd_m * uy))

    def _orbit_triangulate(self, c: int, s: ChannelState, R: float) -> bool:
        """
        在 est 周围、朝【已观测到信号的方向】一侧取两点（各偏 ±δ，交会角≈2δ），
        获得两条近距示向，收缩定位区域。返回是否已清除。

        关键（seed815 ch17 教训）：探测点必须落在【覆盖半平面】内。定向源覆盖半平面
        必含“从源指向各观测点”的方向（我在那些点收到过信号），故以【所有观测点相对 est
        方向的圆均值】为 α——它必在覆盖半平面内且居中，最稳。切忌用“机器狗当前位置方向”：
        TSP 巡回后当前位置可能在源【盲区】一侧，那样绕点全落盲区 → 全 no_signal（即便 <20m）。
        """
        w = self.world
        est = s.est
        if s.obs:
            cx = sum(math.cos(math.radians(geo.bearing_to(est, p))) for p, _ in s.obs)
            cy = sum(math.sin(math.radians(geo.bearing_to(est, p))) for p, _ in s.obs)
            alpha = math.degrees(math.atan2(cy, cx)) if (cx or cy) else geo.bearing_to(est, w.position)
        else:
            alpha = geo.bearing_to(est, w.position)
        for delta in (self.orbit_delta_deg, -self.orbit_delta_deg):
            if w.finished:
                return False
            a = math.radians(alpha + delta)
            wp = (est[0] + R * math.cos(a), est[1] + R * math.sin(a))
            r = self._measure_into(c, wp)
            if r == "cleared":
                return True
            s.refresh_fix()
            if s.est is not None and s.region_r <= self.clear_margin_m:
                if self._try_clear(c, s.est):
                    return True
        return False

    def _establish_baseline(self, c: int, s: ChannelState) -> bool:
        """
        仅 1 条示向 (p,b)：源在 p 出发、方位 b 的射线上（±1°），量程 d 未知（≤ r_eff）。
        目标：再取一条【可交会】的示向，使 est 变“可靠”（region ≤ reliable_region），
        精定位交给外层 orbit。返回是否【已就地清除】（仅 near 命中时）。

        为何“垂直优先”：沿方位【前进】若量程 d < 前进量会【越过源】，落到源覆盖半平面
        的【背面】→ no_signal（定向源尤甚：seed4068 ch15 即在此翻车）。而【垂直】于方位
        偏移【绝不越过源】：量程仅由 d 增至 √(d²+L²)，示向角相对‘从源看我’仅偏 atan(L/d)
        ＜90°，两侧至少一侧仍在 180° 覆盖半平面内 → 过冲免疫。仅当 d 逼近 r_eff 边缘、
        纯垂直会跳出量程时，才辅以“前进减小量程再垂直”（此时 d 大，前进不会过冲）。
        """
        p, b = s.obs[-1]
        ux, uy = geo.unit(b)
        nx, ny = -uy, ux                      # 垂直单位向量
        L = self.baseline_perp_m
        # 由几何最安全到次安全：纯垂直（过冲免疫）→ 更小垂直 → 前进+垂直（应对贴 r_eff 边缘）。
        plans = [(0.0, L), (0.0, 0.62 * L), (0.5 * self.baseline_fwd_m, L)]
        for fwd, off in plans:
            cands = [(p[0] + fwd * ux + sg * off * nx, p[1] + fwd * uy + sg * off * ny)
                     for sg in (+1.0, -1.0)]
            cands.sort(key=lambda q: q[0] * q[0] + q[1] * q[1])   # 先测更靠区域中心（更可能在场内且在覆盖内）
            for q in cands:
                if self.world.finished:
                    return False
                r = self._measure_into(c, q)
                if r == "cleared":
                    return True
                if r == "obs":
                    s.refresh_fix()
                    if s.est is not None and s.region_r <= self.clear_margin_m:
                        if self._try_clear(c, s.est):
                            return True
                    if s.est is not None and s.region_r <= self.reliable_region_m:
                        return False          # 已得可靠交会示向，交外层 orbit 精定位
        return False

    def _try_clear(self, c: int, target: Point) -> bool:
        """在 target 清除频道 c；成功则标记并返回 True。"""
        co = self.world.clear(target[0], target[1], c)
        if co.ok and co.success:
            self.chans[c].cleared = True
            return True
        return False

    def _measure_into(self, c: int, pt: Point) -> Optional[str]:
        """
        在 pt 测频道 c：near→就地清除并返回 'cleared'；direction→记录并返回 'obs'；
        no_signal / 时限 → None。
        """
        w = self.world
        mo = w.measure(pt[0], pt[1], c)
        if not mo.ok:
            return None
        s = self.chans[c]
        if mo.is_near:
            s.detected = True
            return "cleared" if self._try_clear(c, w.position) else None
        if mo.is_direction:
            s.detected = True
            s.obs.append((pt, mo.svd_deg))
            s.refresh_fix()
            return "obs"
        return None


# --------------------------------------------------------------------------- #
# 便捷入口
# --------------------------------------------------------------------------- #
def make_hunter(problem: int, **kw) -> Hunter:
    return Hunter(problem=problem, **kw)


def make_runner_policy(problem: int, **kw):
    """
    返回适配 offline_sim.harness 的 policy(runner)：把 EpisodeRunner 包成 RunnerWorld，
    从而复用 offline_sim 的 evaluate / evaluate_stress（含对抗案例）。
    """
    from .interface import RunnerWorld

    def policy(runner):
        Hunter(problem=problem, **kw).run(RunnerWorld(runner))

    return policy
