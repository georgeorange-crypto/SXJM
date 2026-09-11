"""
消融实验：OURS 的四个假设逐个拔掉，每个消融只与 Full 差【一个】部件。

Full 方法（=Way3 Hunter）由四部分组成：
    多频道全局侦察 + 全局路径规划/重规划 + 主动感知选点 + 在线动态重规划。

AblatableHunter 用四个布尔开关表示这四部件，全开 = Full = OURS：
    global_scan     True: 先在全覆盖扫描点上测【所有】活跃频道，建立全局信息再统一清除。
                    False: reactive——发现一个立刻定位清除，不预先建立全局图。
    global_route    True: 清除顺序用 NN+2-opt 全局优化 (geometry.ordered_tour)。
                    False: 纯最近邻贪心。
    active_sensing  True: 归航第二观测点朝【观测方向圆均值 α】两侧 ±δ 布置（保证落在覆盖半平面
                    内且交会角好）。False: 用【机器狗当前位置方向】布点（代码注释记录的反面教材：
                    TSP 后当前位置可能在源盲区一侧 → 观测落盲区，主动感知失效）。
    replanning      True: 归航中每轮 refresh_fix，用新观测动态收缩并再决策。
                    False: 只用首个可用估计，不再依据新反馈更新（committed）。

每个开关都对应一处真实行为改动（见各 override），因此消融差异可干净归因到单一部件。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from jammerhunt import geometry as geo
from jammerhunt.agent import CH_MAX, CH_MIN, ChannelState, Hunter, MAX_TOTAL_JAMMERS

from .interface_shim import World


@dataclass
class AblatableHunter(Hunter):
    global_scan: bool = True
    global_route: bool = True
    active_sensing: bool = True
    replanning: bool = True

    # ------------------------------------------------------------------ #
    def run(self, world: World) -> None:
        self.world = world
        self.chans = {c: ChannelState(c) for c in range(CH_MIN, CH_MAX + 1)}
        if not world.enter():
            return
        if self.global_scan:
            super().run(world)                 # Full 扫描骨架（含全局扫描 + _clear_all_present）
            return
        # --- w/o Global Scan：reactive 逐频道发现-清除，不建全局图 ---
        pts = self.scan_points()
        for c in range(CH_MIN, CH_MAX + 1):
            if world.finished or self.cleared_count >= MAX_TOTAL_JAMMERS:
                break
            s = self.chans[c]
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
            if s.detected and not s.cleared and not world.finished:
                self._home_and_clear(c)
        if not world.finished:
            world.exit()

    # ------------------------------------------------------------------ #
    def _clear_all_present(self) -> None:
        if self.global_route:
            super()._clear_all_present()
            return
        # --- w/o Global Route：纯最近邻贪心序 ---
        self._finalize_absence()
        w = self.world
        pending = [c for c, s in self.chans.items() if s.detected and not s.cleared]
        known = [c for c in pending if self.chans[c].est is not None]
        unknown = [c for c in pending if self.chans[c].est is None]
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

    # ------------------------------------------------------------------ #
    def _orbit_triangulate(self, c: int, s: ChannelState, R: float) -> bool:
        if self.active_sensing:
            return super()._orbit_triangulate(c, s, R)
        # --- w/o Active Sensing：用“当前机器狗位置方向”布点（反面教材，可能落盲区）---
        w = self.world
        est = s.est
        alpha = geo.bearing_to(est, w.position)     # 非信息增益方向
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

    # ------------------------------------------------------------------ #
    def _home_and_clear(self, c: int) -> bool:
        if self.replanning:
            return super()._home_and_clear(c)
        # --- w/o Replanning：committed——只用首个可用估计，一次直清，不再依新观测更新 ---
        w = self.world
        s = self.chans[c]
        s.refresh_fix()
        if s.est is None:
            # 无交会估计：用父类机制补一条基线拿到首个估计，但拿到后不再迭代收缩
            if s.obs:
                self._establish_baseline(c, s)
                s.refresh_fix()
        if s.est is not None:
            return self._try_clear(c, s.est)
        return False


# --------------------------------------------------------------------------- #
# 便捷构造：Full 与四个消融
# --------------------------------------------------------------------------- #
def make_full(problem: int, **kw) -> AblatableHunter:
    return AblatableHunter(problem=problem, **kw)


def make_ablation(name: str, problem: int, **kw) -> AblatableHunter:
    flags = {
        "full": {},
        "no_global_scan": {"global_scan": False},
        "no_global_route": {"global_route": False},
        "no_active_sensing": {"active_sensing": False},
        "no_replanning": {"replanning": False},
    }
    if name not in flags:
        raise ValueError(f"unknown ablation {name!r}; choices={list(flags)}")
    return AblatableHunter(problem=problem, **flags[name], **kw)


ABLATIONS = ["full", "no_global_scan", "no_global_route", "no_active_sensing", "no_replanning"]
