"""
可视化：轨迹、覆盖布点、可行集外包络演化。matplotlib 惰性导入，无显示环境也能被 import。

图件用于论文与评审：
  - plot_trajectory：机器人轨迹 + 测点 + 清除点 + 竞技场圆。
  - plot_cover：覆盖布点与其保守半径圆。
  - plot_feasible：某频道可行集叶盒 + 外包络 MEC。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ..domain.types import Vec2


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_trajectory(log: Sequence, arena_r: float, out_path: str,
                    title: str = "Trajectory") -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_arena(ax, arena_r)
    xs, ys = [], []
    for e in log:
        loc = getattr(e, "location", None)
        if loc is None:
            continue
        xs.append(loc[0]); ys.append(loc[1])
        kind = getattr(e, "kind", "")
        note = getattr(e, "note", "")
        if kind == "clear":
            color = "green" if note == "success" else "red"
            ax.scatter([loc[0]], [loc[1]], c=color, marker="x", s=60, zorder=5)
        else:
            ax.scatter([loc[0]], [loc[1]], c="steelblue", s=12, zorder=4)
    ax.plot(xs, ys, "-", color="gray", lw=0.8, alpha=0.7, zorder=3)
    ax.set_aspect("equal"); ax.set_title(title)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_cover(points: Sequence[Vec2], arena_r: float, safe_r: float, out_path: str,
               title: str = "Coverage") -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_arena(ax, arena_r)
    for (x, y) in points:
        ax.add_patch(plt.Circle((x, y), safe_r, color="orange", alpha=0.08))
        ax.scatter([x], [y], c="darkorange", s=18, zorder=5)
    ax.set_aspect("equal"); ax.set_title(title)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_feasible(leaf_boxes: Sequence, mec_center: Optional[Vec2], mec_radius: float,
                  arena_r: float, out_path: str, title: str = "Feasible set") -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_arena(ax, arena_r)
    for b in leaf_boxes:
        ax.add_patch(plt.Rectangle((b.xl, b.yl), b.width, b.height,
                                   fill=True, color="teal", alpha=0.15, lw=0))
    if mec_center is not None and mec_radius < float("inf"):
        ax.add_patch(plt.Circle(mec_center, mec_radius, fill=False, color="crimson", lw=1.5))
        ax.scatter([mec_center[0]], [mec_center[1]], c="crimson", marker="+", s=80)
    ax.set_aspect("equal"); ax.set_title(title)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _draw_arena(ax, arena_r: float) -> None:
    import matplotlib.pyplot as plt
    ax.add_patch(plt.Circle((0, 0), arena_r, fill=False, color="black", lw=1.2))
    ax.set_xlim(-arena_r * 1.05, arena_r * 1.05)
    ax.set_ylim(-arena_r * 1.05, arena_r * 1.05)
