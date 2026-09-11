"""
集中管理题面常量与几何容差（附件2.txt §1.2 及问题 2 的建模约定）。

所有数值都能在题面找到出处，避免把工程假设藏在函数默认参数里：
  - 竞技场半径 R = 1800 m；
  - 每个源的"接收半径" R_eff ∈ [1000, 1500] m，未知、仿真也不返回；
  - 示向度误差 δ ∈ [−1°, +1°]；
  - 近距阈值 5 m（更近不出示向度）、清除半径 20 m；
  - 机器狗坐标只受 |·| ≤ 2e6 约束，**不**被限制在竞技场内（§1.2 第 13 行）。

问题 2 的两个"硬信息"来自 R_eff 的区间：
  - S1 能测到该源 ⟹ r1 = ‖G−S1‖ ≤ R_eff ≤ 1500  → r1 封顶 1500（r1_max_cap_m）；
  - 要"保证"第二点也能测到，只能用 R_eff 的**下界** 1000 → 接收保证半径 ρ_acc = 1000
    （acceptance_radius_m）。300 m 之类的数字没有题面依据，只作画图/基线，不进入保证。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProblemConfig:
    """问题 1/2 共用的题面常量。数值均来自附件2.txt。"""

    # --- 竞技场与源 ---
    arena_radius_m: float = 1800.0        # 竞技场半径 R
    reception_min_m: float = 1000.0       # 接收半径下界（硬信息，用于"保证"）
    reception_max_m: float = 1500.0       # 接收半径上界（硬信息，用于 r1 封顶）

    # --- 测量 ---
    delta_deg: float = 1.0                # 示向度误差半张角 δ
    near_threshold_m: float = 5.0         # 近距阈值：更近不出示向度
    clear_radius_m: float = 20.0          # 清除半径

    # --- 问题 2 决策用（源自 R_eff 区间的两个硬界） ---
    acceptance_radius_m: float = 1000.0   # ρ_acc：保证再次接收所用的最坏接收半径 = R_eff 下界
    r1_max_cap_m: float = 1500.0          # r1 = ‖G−S1‖ 上限（≤ R_eff ≤ 1500）
    r2_min_m: float = 5.0                 # 第二点到源的最小距离（近距阈值）
    r2_max_m: float = 1000.0              # 第二点到源的最大距离（受 ρ_acc 约束）

    # --- 数值/离散化 ---
    disk_sides: int = 128                 # 圆的外切多边形边数（保证性外近似）
    default_bound_m: float = 6000.0       # 半平面交初始大正方形半边长
    eps: float = 1e-9                     # 通用绝对容差


# 一个默认实例，方便顶层函数直接引用（可被显式传参覆盖）。
DEFAULT_CONFIG = ProblemConfig()
