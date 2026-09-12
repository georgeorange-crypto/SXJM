"""
现实基线策略（只用可观测信息，不看真值）：

B1  Reactive / Sequential —— 最朴素现实基线。
    逐频道处理：对频道 c，在【全向 1-覆盖扫描点】上依次 measure，一旦收到示向就立刻围绕它
    定位并清除，然后再进入下一频道。它【不预先建立全局信息】，走一个清一个，频道间几乎没有
    路径协同 → 大量往返冤枉路。用于给出“信息不足”的上限损失参照。

B2  Global Scan + Greedy Route —— 先建立全局信息，但按最近目标贪心走。
    与 OURS 共用同一套覆盖扫描 + 交会定位（先把所有源找出来、粗定位），但清除阶段用
    纯最近邻贪心序（无 2-opt / 无全局优化 / 无重规划）。用于隔离“全局路径优化”本身的价值：
    B2 与 OURS 的差异 ≈ 路径规划贡献。

两者都复用 Way3 的 geometry / coverage / Hunter 内部机制，尽量只改“决策骨架”，保证可比性。
定位/归航/清除的底层动作沿用 Hunter 的稳健实现（抵近、盲清），使基线不会因“定位太菜”而
虚高损失——我们要比较的是【决策结构】，不是定位工程细节。
"""

from __future__ import annotations

from typing import Optional

from jammerhunt import geometry as geo
from jammerhunt.agent import Hunter, MAX_TOTAL_JAMMERS
from jammerhunt.coverage import omni_scan_points, directional_scan_points

from .interface_shim import World


class ReactiveHunter(Hunter):
    """
    B1：逐频道 reactive。继承 Hunter 复用其归航/清除机制，只重写 run 的决策骨架。
    对每个频道：沿覆盖扫描点找信号 → 找到即 home_and_clear → 下一个频道。
    没有全局扫描、没有跨频道路径协同。
    """

    def run(self, world: World) -> None:
        self.world = world
        from jammerhunt.agent import ChannelState, CH_MIN, CH_MAX
        self.chans = {c: ChannelState(c) for c in range(CH_MIN, CH_MAX + 1)}
        if not world.enter():
            return
        pts = self.scan_points()
        for c in range(CH_MIN, CH_MAX + 1):
            if world.finished or self.cleared_count >= MAX_TOTAL_JAMMERS:
                break
            s = self.chans[c]
            # 从当前位置起，按最近邻顺序逐个扫描点测该频道，收到信号即停止扫描转入定位
            order = geo.ordered_tour(world.position, pts)
            for idx in order:
                if world.finished:
                    break
                mo = world.measure(pts[idx][0], pts[idx][1], c)
                if not mo.ok:
                    return
                if mo.is_near:
                    s.detected = True
                    co = world.clear(world.position[0], world.position[1], c)
                    if co.ok and co.success:
                        s.cleared = True
                    break
                if mo.is_direction:
                    s.detected = True
                    s.obs.append((pts[idx], mo.svd_deg))
                    s.refresh_fix()
                    break
            # 若探测到但未就地清除，围绕它归航清除（复用父类稳健实现）
            if s.detected and not s.cleared and not world.finished:
                self._home_and_clear(c)
        if not world.finished:
            world.exit()


class GreedyScanHunter(Hunter):
    """
    B2：全局覆盖扫描（与 OURS 同）→ 但清除阶段用纯最近邻贪心序，无 2-opt / 无全局重排。
    只重写 _clear_all_present 的排序部分，其余（扫描、定位、归航、盲清）与 OURS 完全一致。
    """

    def _clear_all_present(self) -> None:
        self._finalize_absence()
        w = self.world
        pending = [c for c, s in self.chans.items() if s.detected and not s.cleared]
        known = [c for c in pending if self.chans[c].est is not None]
        unknown = [c for c in pending if self.chans[c].est is None]
        # 纯最近邻贪心（无 2-opt）：每次去当前位置最近的已估计源
        ordered: list[int] = []
        cur = w.position
        rest = known[:]
        while rest:
            c = min(rest, key=lambda cc: geo.dist(cur, self.chans[cc].est))
            ordered.append(c)
            cur = self.chans[c].est
            rest.remove(c)
        for c in ordered + unknown:
            if w.finished or self.cleared_count >= MAX_TOTAL_JAMMERS:
                break
            self._home_and_clear(c)


def make_reactive(problem: int, **kw) -> ReactiveHunter:
    return ReactiveHunter(problem=problem, **kw)


def make_greedy_scan(problem: int, **kw) -> GreedyScanHunter:
    return GreedyScanHunter(problem=problem, **kw)
