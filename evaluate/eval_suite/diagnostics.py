"""
逐动作诊断：把每个方法的耗时按【动作类型】拆到 move / detect / switch / clear，
与各自的物理地板逐动作对比，回答“时间到底浪费在哪个动作、还剩多少可优化空间”。

这是对 metrics 里 (ΔT_move, ΔT_info) 归因的【细化】。原归因只分“移动 vs 感知”，且
C_move/C_sense 显式忽略了清除未命中。这里进一步拆成四个可操作动作，并把被忽略的
“清除漏命中”代价单列出来：

    动作      物理必要地板(每局)                实际
    move    L_LB/5  (绝对移动下界)            Σ 每次动作的移动/5
    detect  100s    (20 频道各扫一次的感知地板) 5s × measure 次数
    switch  19s     (从频道1遍历到20)          1s × 换频次数
    clear   5·m     (每源 3s光学+2s激光，必付)  5s×命中 + 3s×漏命中

超支 Δ_act = 实际 - 必要 ≥ 0（成功局；move/clear 恒非负，detect/switch 当扫满≥20频道时非负）。
逐动作地板之和 T_floor = L_LB/5 + 119 + 5m（各分量独立成立的硬下界，但因“边走边测”不可
同时取到，故 T_floor 只是下界、不可达；可达的联合理想是 Oracle：move 用 L_UB/5）。

关键恒等式（与既有指标对齐，见 tests）：
    Δ_move                 == metrics 的 ΔT_move   （= move_s - L_LB/5）
    Δ_detect + Δ_switch    == metrics 的 ΔT_info   （= detect+switch - 119）
    Δ_clear                == 清除漏命中代价 3·n_miss（+重复清除，原归因未计）
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Optional

from .lowerbound import PER_SOURCE_S, SPEED_MPS
from .metrics import ORACLE_SCAN_DETECT_S, ORACLE_SCAN_S, ORACLE_SCAN_SWITCH_S
from .runner import MethodSummary

IDEAL_N_MEASURE = 20           # 确认全部 20 频道有无源的最少测量次数
IDEAL_N_SWITCH = 19            # 从初始频道 1 遍历到 20 的最少换频次数

ACTION_ORDER = ("move", "detect", "switch", "clear")
ACTION_LABEL = {"move": "移动", "detect": "检测", "switch": "切频", "clear": "清除"}


def _mean(xs) -> float:
    return statistics.fmean(xs) if xs else 0.0


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class ActionStat:
    """单个动作在某方法上的（成功局均值）耗时画像。"""
    action: str
    necessary: float               # 物理必要地板（秒）
    actual: float                  # 实际耗时（秒）
    count: float = 0.0             # 动作次数（均值）
    count_ideal: float = 0.0       # 理想次数

    @property
    def excess(self) -> float:     # 超支（可优化余量）
        return self.actual - self.necessary

    @property
    def ratio(self) -> float:      # 实际/必要
        return self.actual / self.necessary if self.necessary > 1e-9 else float("inf")


@dataclass
class MethodActionProfile:
    """一个方法的逐动作诊断画像（仅统计成功局）。"""
    method: str
    n_success: int
    T: float = 0.0                 # 成功局均值总时间
    T_floor: float = 0.0           # 逐动作硬下界之和（不可达）
    T_oracle: float = 0.0          # 可达联合理想
    oracle_move: float = 0.0       # Oracle 的移动时间 L_UB/5（可约/不可约移动的分界）
    n_clear_miss: float = 0.0
    actions: list[ActionStat] = field(default_factory=list)

    def get(self, action: str) -> Optional[ActionStat]:
        for a in self.actions:
            if a.action == action:
                return a
        return None

    @property
    def total_excess_floor(self) -> float:
        """相对逐动作硬地板的总超支 = Σ Δ_act = T - T_floor。"""
        return self.T - self.T_floor

    @property
    def total_excess_oracle(self) -> float:
        """相对可达理想 Oracle 的总超支（= metrics 的 gap_to_oracle 均值）。"""
        return self.T - self.T_oracle


def build_profile(summary: MethodSummary) -> MethodActionProfile:
    """从一个方法的成功局构造逐动作画像。无成功局 → n_success=0 的空画像。"""
    succ = [r for r in summary.results if r.success]
    prof = MethodActionProfile(method=summary.method, n_success=len(succ))
    if not succ:
        return prof

    prof.T = _mean([r.T for r in succ])
    prof.T_oracle = _mean([r.T_oracle for r in succ])
    prof.oracle_move = _mean([r.L_UB for r in succ]) / SPEED_MPS
    prof.n_clear_miss = _mean([r.bd.n_clear_miss for r in succ])

    move = ActionStat(
        "move",
        necessary=_mean([r.T_move_lb for r in succ]),
        actual=_mean([r.bd.move_s for r in succ]),
        count=_mean([r.bd.move_m for r in succ]),               # 用距离(m)代替次数
        count_ideal=_mean([r.L_LB for r in succ]),
    )
    detect = ActionStat(
        "detect", necessary=ORACLE_SCAN_DETECT_S,
        actual=_mean([r.bd.detect_s for r in succ]),
        count=_mean([r.bd.n_measure for r in succ]), count_ideal=IDEAL_N_MEASURE,
    )
    switch = ActionStat(
        "switch", necessary=ORACLE_SCAN_SWITCH_S,
        actual=_mean([r.bd.switch_s for r in succ]),
        count=_mean([r.bd.n_switch for r in succ]), count_ideal=IDEAL_N_SWITCH,
    )
    clear = ActionStat(
        "clear", necessary=_mean([PER_SOURCE_S * r.total for r in succ]),
        actual=_mean([r.bd.clear_s for r in succ]),
        count=_mean([r.bd.n_clear for r in succ]), count_ideal=_mean([r.total for r in succ]),
    )
    prof.actions = [move, detect, switch, clear]
    prof.T_floor = sum(a.necessary for a in prof.actions)
    return prof


def build_profiles(summaries: dict[str, MethodSummary]) -> dict[str, MethodActionProfile]:
    return {m: build_profile(s) for m, s in summaries.items()}


def _sorted(summaries: dict[str, MethodSummary]) -> list[MethodSummary]:
    return sorted(summaries.values(), key=lambda s: s.sort_key())


# --------------------------------------------------------------------------- #
# 表 1：逐动作【实际耗时】跨方法对比（含总时间）
# --------------------------------------------------------------------------- #
def action_time_table(summaries: dict[str, MethodSummary]) -> str:
    head = (f"{'method':<18}{'移动s':>10}{'检测s':>9}{'切频s':>8}{'清除s':>9}"
            f"{'总时间s':>11}")
    lines = [head, "-" * len(head)]
    for s in _sorted(summaries):
        p = build_profile(s)
        if p.n_success == 0:
            lines.append(f"{s.method:<18}{'—（无成功局）':>45}")
            continue
        mv, dt, sw, cl = (p.get(a) for a in ACTION_ORDER)
        lines.append(f"{s.method:<18}{mv.actual:>10.0f}{dt.actual:>9.0f}"
                     f"{sw.actual:>8.0f}{cl.actual:>9.0f}{p.T:>11.0f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 表 2：逐动作【超支/可优化余量】+ 占比（相对物理地板）
# --------------------------------------------------------------------------- #
def headroom_table(summaries: dict[str, MethodSummary]) -> str:
    head = (f"{'method':<18}{'Δ移动':>9}{'Δ检测':>9}{'Δ切频':>8}{'Δ清除':>8}"
            f"{'Σ超支':>10}{'移动占比':>9}")
    lines = [head, "-" * len(head),
             "（Δ = 实际 - 物理地板；地板 move=L_LB/5, detect=100, switch=19, clear=5m）"]
    for s in _sorted(summaries):
        p = build_profile(s)
        if p.n_success == 0:
            lines.append(f"{s.method:<18}{'—（无成功局）':>45}")
            continue
        mv, dt, sw, cl = (p.get(a).excess for a in ACTION_ORDER)
        tot = p.total_excess_floor
        share_mv = mv / tot if abs(tot) > 1e-9 else 0.0
        lines.append(f"{s.method:<18}{mv:>9.0f}{dt:>9.0f}{sw:>8.0f}{cl:>8.0f}"
                     f"{tot:>10.0f}{share_mv*100:>8.0f}%")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 表 3：动作【次数】对比（诊断“为什么超支”）
# --------------------------------------------------------------------------- #
def action_count_table(summaries: dict[str, MethodSummary]) -> str:
    head = (f"{'method':<18}{'测量次':>8}{'/理想':>7}{'换频次':>8}{'清除次':>8}"
            f"{'漏命中':>8}{'移动m':>10}{'/下界m':>10}")
    lines = [head, "-" * len(head)]
    for s in _sorted(summaries):
        p = build_profile(s)
        if p.n_success == 0:
            lines.append(f"{s.method:<18}{'—（无成功局）':>50}")
            continue
        mv, dt, sw, cl = (p.get(a) for a in ACTION_ORDER)
        lines.append(f"{s.method:<18}{dt.count:>8.1f}{IDEAL_N_MEASURE:>7d}"
                     f"{sw.count:>8.1f}{cl.count:>8.1f}{cl.count - cl.count_ideal:>8.1f}"
                     f"{mv.count:>10.0f}{mv.count_ideal:>10.0f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 自动诊断叙述：把数据读成“哪里不够好、还有多少空间”
# --------------------------------------------------------------------------- #
def _pick_focus(summaries: dict[str, MethodSummary]) -> Optional[str]:
    """诊断主角：优先 ours，否则成功率最高的现实方法（排除 lb/oracle）。"""
    if "ours" in summaries and build_profile(summaries["ours"]).n_success:
        return "ours"
    real = [s for m, s in summaries.items()
            if m not in ("lb", "oracle") and build_profile(s).n_success]
    if not real:
        return None
    return sorted(real, key=lambda s: s.sort_key())[0].method


def format_diagnosis(summaries: dict[str, MethodSummary], problem: int = 3) -> str:
    focus = _pick_focus(summaries)
    if focus is None:
        return "【逐动作诊断】无成功局可诊断。"
    p = build_profile(summaries[focus])
    mv, dt, sw, cl = (p.get(a) for a in ACTION_ORDER)
    tot = p.total_excess_floor
    pct = lambda x: (x / tot * 100 if abs(tot) > 1e-9 else 0.0)

    out = [f"【逐动作诊断】problem {problem}  主角={focus}（成功局均值，n={p.n_success}）",
           f"  相对逐动作物理地板总超支 Σ={tot:.0f}s（总时间 {p.T:.0f}s，地板 {p.T_floor:.0f}s）。",
           f"  拆分：移动 {mv.excess:.0f}s({pct(mv.excess):.0f}%) | 检测 {dt.excess:.0f}s({pct(dt.excess):.0f}%)"
           f" | 切频 {sw.excess:.0f}s({pct(sw.excess):.0f}%) | 清除 {cl.excess:.0f}s({pct(cl.excess):.0f}%)。"]

    # 最大头 → 优化重点
    biggest = max(ACTION_ORDER, key=lambda a: p.get(a).excess)
    out.append(f"  → 最大可优化头是【{ACTION_LABEL[biggest]}】，占总超支 {pct(p.get(biggest).excess):.0f}%。")

    # 移动：可约 vs 不可约（邻域松弛）
    irreducible = max(0.0, p.oracle_move - mv.necessary)      # (L_UB-L_LB)/5，圆盘邻域松弛
    reducible = max(0.0, mv.actual - p.oracle_move)           # 相对 Oracle 最优移动的绕路/定位往返
    out.append(f"  · 移动 {mv.actual:.0f}s 中，可约绕路≈{reducible:.0f}s（对照 Oracle 最优移动 {p.oracle_move:.0f}s），"
               f"邻域松弛≈{irreducible:.0f}s 属圆盘半径带来的不可约项。")

    # 检测：多扫了多少次
    extra_meas = dt.count - IDEAL_N_MEASURE
    if extra_meas > 0.05:
        out.append(f"  · 检测 {dt.count:.1f} 次（理想 20）：多测 {extra_meas:.1f} 次≈{dt.excess:.0f}s，"
                   f"来自交会定位的环绕补测——用更少测量收敛定位即可回收。")
    elif extra_meas < -0.05:
        out.append(f"  · 检测仅 {dt.count:.1f} 次（<20）：已借助“已知数量”提前停扫，省下感知 {-dt.excess:.0f}s。")

    # 清除：漏命中 = 定位精度余量
    if p.n_clear_miss > 0.05:
        out.append(f"  · 清除漏命中 {p.n_clear_miss:.2f} 次/局 ×3s = {cl.excess:.0f}s：纯定位精度余量，漏 0 次即清零。")
    else:
        out.append("  · 清除零漏命中：定位精度已把该项打到 0，无余量。")

    # 对比基线：全局规划已消掉多少移动
    base = None
    for cand in ("reactive", "no_global_scan"):
        if cand in summaries and build_profile(summaries[cand]).n_success:
            base = build_profile(summaries[cand]); break
    if base is not None:
        bmv = base.get("move")
        if mv.excess > 1e-6:
            out.append(f"  · 对比基线 {base.method}：其移动超支 {bmv.excess:.0f}s ≈ {focus} 的 "
                       f"{bmv.excess / mv.excess:.0f}×——全局侦察+规划已消掉移动大头；"
                       f"{focus} 剩余头主要在定位往返与测量效率。")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# 图：逐动作可优化余量（相对物理地板的超支，堆叠）
# --------------------------------------------------------------------------- #
def make_action_figure(summaries: dict[str, MethodSummary], out_path: str,
                       problem: int = 3, title: Optional[str] = None) -> Optional[str]:
    """
    每方法一根堆叠柱 = 相对逐动作物理地板的超支，按 move/detect/switch/clear 着色。
    直观展示“各方法的可优化余量有多大、又集中在哪个动作”。matplotlib 缺失 → None。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.font_manager import FontProperties, findfont
        from .report import os_has_font
    except Exception:
        return None

    for fam in ("Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"):
        if os_has_font(findfont, FontProperties, fam):
            plt.rcParams["font.sans-serif"] = [fam]
            plt.rcParams["axes.unicode_minus"] = False
            break

    order = _sorted(summaries)
    labels, segs = [], {a: [] for a in ACTION_ORDER}
    for s in order:
        p = build_profile(s)
        if p.n_success == 0 or s.method == "lb":     # lb 是地板本身，画超支无意义
            continue
        labels.append(s.method)
        for a in ACTION_ORDER:
            segs[a].append(max(0.0, p.get(a).excess))   # 图上钳到 ≥0（表里保留符号）
    if not labels:
        return None

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(labels)), 5.2))
    colors = {"move": "#e08a3c", "detect": "#6aa84f", "switch": "#b060b0", "clear": "#999999"}
    bottom = np.zeros(len(labels))
    for a in ACTION_ORDER:
        v = np.asarray(segs[a])
        ax.bar(x, v, bottom=bottom, label=f"Δ{ACTION_LABEL[a]}", color=colors[a],
               edgecolor="white", linewidth=0.5)
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("相对物理地板的超支时间 (s)")
    ax.set_title(title or f"问题{problem}：各方法逐动作可优化余量（成功局均值）")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    for xi, tot in zip(x, bottom):
        ax.text(xi, tot, f"{tot:.0f}s", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
