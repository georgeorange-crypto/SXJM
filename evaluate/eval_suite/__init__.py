"""
四层评价体系（问题 3/4）：绝对物理下界 → One-shot Oracle → 现实基线 → OURS → 消融。

统一核心指标 R = T / T_ABS-LB；并把非理想损失拆成 额外移动 ΔT_move 与 额外感知 ΔT_info。
真值（干扰源坐标/有效半径/朝向）仅在【离线、ground-truth 已知】的仿真集上取用（附件明确
不在正式测试暴露真值）——故本体系用于科学验证，正式盲测仍按题目只报清除率与平均时间。

依赖 Way3/jammerhunt（真值 environment、OURS 策略、几何工具）；import 本包即挂好 sys.path。
"""

from __future__ import annotations

from . import _paths  # noqa: F401  (side effect: put Way3 on sys.path)

__all__ = ["_paths"]
