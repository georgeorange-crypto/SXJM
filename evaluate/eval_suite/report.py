"""
汇总成表 + 论文主图（堆叠柱：T/T_ABS_LB 分解为 必要移动/额外移动/必要清除/检测/切频）。

主图纵轴 = T / T_ABS_LB（下界恒为 1）；每根柱按计时来源堆叠，直观展示：
  OURS 的柱离 1 最近，且相对基线主要压缩了“额外移动”那一段。
matplotlib 为可选依赖；缺失时 make_figure 返回 None 并跳过（表格仍可用）。
"""

from __future__ import annotations

import math
import statistics
from typing import Optional

from .metrics import ORACLE_SCAN_S
from .runner import MethodSummary


def os_has_font(findfont, FontProperties, family: str) -> bool:
    """该字体族是否真的装在系统里（findfont 命中的文件名含该族，而非回退到默认）。"""
    try:
        path = findfont(FontProperties(family=family), fallback_to_default=False)
        return bool(path)
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# 文本表
# --------------------------------------------------------------------------- #
def leaderboard(summaries: dict[str, MethodSummary]) -> str:
    """按字典序（清除率↓, 成功局均值时间↑）排出榜单表。"""
    rows = sorted(summaries.values(), key=lambda s: s.sort_key())
    head = (f"{'method':<18}{'全清率':>8}{'s/源':>10}{'成功均值s':>11}"
            f"{'R_LB中位':>10}{'R_or中位':>10}{'ΔTmove':>9}{'ΔTinfo':>9}{'Cmove':>7}")
    lines = [head, "-" * len(head)]
    for s in rows:
        lines.append(
            f"{s.method:<18}{s.success_rate*100:>7.1f}%{s.time_per_cleared:>10.1f}"
            f"{s.time_mean_success:>11.1f}{s.R_LB_median:>10.3f}{s.R_oracle_median:>10.3f}"
            f"{s.dT_move_mean:>9.0f}{s.dT_info_mean:>9.0f}{s.C_move_mean:>7.2f}")
    return "\n".join(lines)


def stacked_breakdown_table(summaries: dict[str, MethodSummary]) -> str:
    """
    每方法（仅成功局均值）的计时分解，且把移动拆成 必要/额外：
      必要移动 = T_move_LB（= L_LB/5，成功局均值），额外移动 = move_s - 必要移动。
    列与论文堆叠柱一一对应。
    """
    head = (f"{'method':<18}{'总时间':>9}{'必要移动':>9}{'额外移动':>9}"
            f"{'检测':>7}{'切频':>7}{'清除':>7}")
    lines = [head, "-" * len(head)]
    order = sorted(summaries.values(), key=lambda s: s.sort_key())
    for s in order:
        succ = [r for r in s.results if r.success]
        if not succ:
            lines.append(f"{s.method:<18}{'—（无成功局）':>40}")
            continue
        move = statistics.fmean([r.bd.move_s for r in succ])
        need_move = statistics.fmean([r.T_move_lb for r in succ])
        extra_move = move - need_move
        det = statistics.fmean([r.bd.detect_s for r in succ])
        sw = statistics.fmean([r.bd.switch_s for r in succ])
        clr = statistics.fmean([r.bd.clear_s for r in succ])
        tot = statistics.fmean([r.T for r in succ])
        lines.append(f"{s.method:<18}{tot:>9.0f}{need_move:>9.0f}{extra_move:>9.0f}"
                     f"{det:>7.0f}{sw:>7.0f}{clr:>7.0f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 论文主图
# --------------------------------------------------------------------------- #
def make_figure(summaries: dict[str, MethodSummary], out_path: str,
                problem: int = 3, title: Optional[str] = None) -> Optional[str]:
    """
    堆叠柱图：纵轴 T/T_ABS_LB，堆叠 必要移动/额外移动/必要清除/检测/切频（成功局均值，归一化）。
    保存到 out_path（png）。matplotlib 缺失 → 返回 None。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import findfont, FontProperties
    except Exception:
        return None

    # 中文字体 best-effort：必须在建图【之前】设好，否则不生效（历史 bug）。
    # 只选系统真的装了的字体，避免 findfont 回退到 DejaVu 再刷一屏缺字警告。
    for fam in ("Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"):
        try:
            if os_has_font(findfont, FontProperties, fam):
                plt.rcParams["font.sans-serif"] = [fam]
                plt.rcParams["axes.unicode_minus"] = False
                break
        except Exception:
            pass

    order = sorted(summaries.values(), key=lambda s: s.sort_key())
    labels, seg_need_move, seg_extra_move, seg_clear, seg_detect, seg_switch = [], [], [], [], [], []
    for s in order:
        succ = [r for r in s.results if r.success]
        if not succ:
            continue
        denom = statistics.fmean([r.T_abs_lb for r in succ])
        if denom < 1e-9:
            continue
        need_move = statistics.fmean([r.T_move_lb for r in succ])
        move = statistics.fmean([r.bd.move_s for r in succ])
        labels.append(s.method)
        seg_need_move.append(need_move / denom)
        seg_extra_move.append(max(0.0, move - need_move) / denom)
        seg_clear.append(statistics.fmean([r.bd.clear_s for r in succ]) / denom)
        seg_detect.append(statistics.fmean([r.bd.detect_s for r in succ]) / denom)
        seg_switch.append(statistics.fmean([r.bd.switch_s for r in succ]) / denom)

    if not labels:
        return None

    import numpy as np
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(labels)), 5.2))
    segs = [
        ("必要移动 (L_LB/5)", seg_need_move, "#3b6fb0"),
        ("额外移动", seg_extra_move, "#e08a3c"),
        ("检测 (5s×measure)", seg_detect, "#6aa84f"),
        ("切频 (1s)", seg_switch, "#b060b0"),
        ("光学+清除 (5m)", seg_clear, "#999999"),
    ]
    bottom = np.zeros(len(labels))
    for name, vals, color in segs:
        v = np.asarray(vals)
        ax.bar(x, v, bottom=bottom, label=name, color=color, edgecolor="white", linewidth=0.5)
        bottom += v
    ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.2, label="物理下界 =1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("归一化时间  T / T_ABS-LB")
    ax.set_title(title or f"问题{problem}：各方法相对绝对物理下界的时间分解（成功局均值）")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    for xi, tot in zip(x, bottom):
        ax.text(xi, tot + 0.04, f"{tot:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
