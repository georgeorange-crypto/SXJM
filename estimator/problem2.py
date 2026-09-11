"""
问题 2：第二检测点的选取 —— **有界不确定性下的 Minimax Next-Best-View**。

本模块分两层（互不替代）：

── 第 1 层：解析解释（为什么第二点大致在"第一测向线的侧向"）──────────────
    两点交会定位的 Fisher/雅可比给出 |det H| = |sinγ| /(r1 r2)，一阶定位区域沿测向
    方向的伸长量
        L ≈ (2δ/|sinγ|)·√(r1² + r2² + 2 r1 r2 |cosγ|)。
    由此得到直觉：交会角 γ→90°、第二距离 r2 越小越好。**但源位置未知**，无法对所有
    可能的源都保持 90°，所以这一层只用于"解释与画图"，不是最终答案。
    对应函数：crossing_angle / localization_length_L / second_point_candidates /
    recommend_second_point（后两者是逐假设-90° 构造与中位猜测基线，仅供图示/初值）。

── 第 2 层：决策（真正的第二问答案）──────────────────────────────────────
    出发点是任务本身：机器狗只要走到与真源相距 ≤20 m 处即可光学清除。所以第二次
    测向后得到定位区域 Ω₂，真正该问的不是"Ω₂ 面积多大"，而是"有没有一个点 Q 使
    Ω₂ 内所有可能源都距 Q ≤20 m"——即 Ω₂ 的最小覆盖圆（MEC）半径
        ρ(Ω₂) = min_Q max_{G∈Ω₂} ‖G−Q‖。
    ρ(Ω₂) ≤ 20 ⟺ 第二次测完只需再移动一次即可保证直接清除（无需第三次测向）。
    （MEC 半径天然同时惩罚"太大"与"太细长/太斜"：矩形 a×b 的 ρ=½√(a²+b²)，定面积下
      a=b 最小——所以"越小越方正越好"就是 ρ 本身，不需另立"方正度"指标。）

    唯一被信任的目标知识是**第一可行集** Ω₁（不引入任何概率分布）：
        Ω₁ = D_1800 ∩ W(S1, θ̂1, ±1°) ∩ {G : 5 < ‖G−S1‖ ≤ 1500}
    （S1 测到该源 ⟹ r1 ≤ R_eff ≤ 1500 是硬信息；圆弧用外切多边形做保证性外近似。）

    目标：最坏情形下的清除覆盖半径（沿用第一问的 MEC 度量）
        J(S2) = max_{G∈Ω₁, e2∈[−δ,δ]} ρ( Ω₂(S2; G, e2) ),
        Ω₂ = Ω₁ ∩ W(S2, arg(G−S2)+e2, ±δ)。
    因 Ω₂ 只通过 φ = arg(G−S2)+e2 依赖 (G,e2)，最坏情形退化为对 φ 的**一维扫描**：
        J(S2) = max_{φ ∈ [α_min−δ, α_max+δ]} ρ( Ω₁ ∩ W(S2, φ, ±δ) )，
    其中 [α_min,α_max] 是 Ω₁ 从 S2 看过去的角度轮廓。
        S2* = argmin_{S2∈𝒞_recv} J(S2)。
    若存在 S2 使 J(S2) ≤ 20 → 该点保证"二次测向后一次移动完成清除"；否则取 J 最小者。

    保证接收域（候选区域，纯由题面两个范围推出，不含任何人为阈值）：
    S1 已成功测到位于 G 的源 ⟹ R_eff ≥ ‖G−S1‖，又题面 R_eff ≥ 1000，故与首次观测相容
    的最小接收半径是 max(1000, ‖G−S1‖)。于是要保证第二次仍能接收，须对所有 G∈Ω₁：
        ‖S2−G‖ ≤ max(1000, ‖G−S1‖)，即
        𝒞_recv = ∩_{G∈Ω₁} B( G, max(1000, ‖G−S1‖) )   （⊇ 旧的 ∩B(G,1000)，更准更大）。
    机器狗**不**受竞技场约束。MEC(Ω₁) 半径 ρ* 仅作可行性诊断。

    Tokekar/Vander Hook/Isler 的 bearing-only 主动定位给出解析近似
        S_ana^± = S1 + (r₋+r₊)/2·u ± (r₊−r₋)/2·n   (u=测向单位向量, n=其左法向)
    仅作数值优化初值与对照（验证 S2*≈S_ana），不直接照抄。
    参考：https://josh.vanderhook.info/media/pdf/iros2011localization.pdf
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .config import ProblemConfig, DEFAULT_CONFIG
from .geometry import (
    deg2rad, rad2deg, norm_deg, dir_vec, wedge_halfplanes,
    halfplane_intersection, polygon_diameter,
    circle_outer_halfplanes, clip_polygon_halfplanes,
    angular_span_from_point, farthest_vertex_distance,
    min_enclosing_circle, Halfplane,
)


def crossing_angle(s1: tuple[float, float], s2: tuple[float, float],
                   p: tuple[float, float]) -> float:
    """两测向线 S1→P 与 S2→P 在 P 处的交会角（度，[0,180]）。"""
    v1x, v1y = p[0] - s1[0], p[1] - s1[1]
    v2x, v2y = p[0] - s2[0], p[1] - s2[1]
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    c = (v1x * v2x + v1y * v2y) / (n1 * n2)
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def localization_length_L(r1: float, r2: float, gamma_deg: float,
                          delta_deg: float = 1.0) -> float:
    """
    交会定位区域的一阶伸长量 L（米）。见模块头公式。
    gamma_deg：交会角（度）。gamma→0/180（近平行）时 sinγ→0，L→∞（病态）。
    """
    g = deg2rad(gamma_deg)
    s = math.sin(g)
    if abs(s) < 1e-9:
        return float("inf")
    delta = deg2rad(delta_deg)
    return (2.0 * delta / abs(s)) * math.sqrt(
        r1 * r1 + r2 * r2 + 2.0 * r1 * r2 * abs(math.cos(g)))


@dataclass
class SecondPointCandidate:
    r1: float                       # 假设的 S1→源 距离
    p: tuple[float, float]          # 该假设下的源位置（沿 svd1 方向）
    s2: tuple[float, float]         # 推荐第二检测点
    r2: float                       # |S2→源|
    gamma_deg: float                # 交会角
    L: float                        # 该点的定位伸长量


def second_point_candidates(
    s1: tuple[float, float],
    svd1_deg: float,
    r1_values: Optional[list[float]] = None,
    r2: float = 300.0,
    gamma_target_deg: float = 90.0,
    arena_radius: float = 1800.0,
    side: int = 0,
) -> list[SecondPointCandidate]:
    """
    【第 1 层 · 仅供几何解释图】逐假设-90° 构造：因 r1（S1 到源的距离）未知，
    沿 svd1 的可能源位置扫过一段区间，
    对每个假设 r1 给出一个"使交会角=gamma_target、且 |S2源|=r2"的推荐第二点。
    这些点连起来就是"月牙形候选带"。
    注意：这是对**每一个** r1 假设各自取 90°，无法同时对所有可能源成立，
    因此**不是**第二问的优化答案；真正的选点见 minimax_second_point()。

    几何：在假设源 P 处，P→S1 方向为 svd1+180°；要让 ∠(P→S1, P→S2)=γ，
    则 P→S2 方向 = (svd1+180°) ± γ，两个符号对应测向线两侧。S2 = P + r2·dir。

    - r1_values：假设的 S1→源距离序列；默认覆盖 [200,1800] 一段。
    - r2：希望的第二点到源距离（越小 L 越小，但要可达且留出移动/量程余量）。
    - gamma_target_deg：目标交会角（90° 使 L 最小）。
    - side：+1 / -1 固定取某一侧（画干净的单条带）；0=自动（两侧都在区域内时
      取 L 最小者，等价时取 +1 侧），保证同一侧连续、不出现锯齿。
    返回落在目标区域圆内的候选点。
    """
    if r1_values is None:
        r1_values = [200.0 + 100.0 * k for k in range(0, 17)]   # 200..1800

    ux, uy = dir_vec(svd1_deg)                    # S1 指向源的单位方向
    back = norm_deg(svd1_deg + 180.0)             # 源 P 指向 S1 的方向
    signs = (side,) if side in (+1, -1) else (+1, -1)

    out: list[SecondPointCandidate] = []
    for r1 in r1_values:
        px, py = s1[0] + ux * r1, s1[1] + uy * r1     # 假设源位置 P
        cand = []
        for sgn in signs:
            dx, dy = dir_vec(norm_deg(back + sgn * gamma_target_deg))
            s2 = (px + dx * r2, py + dy * r2)
            if math.hypot(s2[0], s2[1]) <= arena_radius + 1e-9:
                g = crossing_angle(s1, s2, (px, py))
                L = localization_length_L(r1, r2, g)
                cand.append((sgn, SecondPointCandidate(r1, (px, py), s2, r2, g, L)))
        if not cand:
            continue
        if side in (+1, -1):
            out.append(cand[0][1])
        else:
            # 自动：优先 +1 侧（保证连续），仅当其越界时退到 -1 侧
            plus = [c for s, c in cand if s == +1]
            out.append(plus[0] if plus else cand[0][1])
    return out


def recommend_second_point(
    s1: tuple[float, float],
    svd1_deg: float,
    r1_guess: float = 1000.0,
    r2: float = 300.0,
    arena_radius: float = 1800.0,
) -> SecondPointCandidate:
    """
    【第 1 层 · 基线/初值，非最终答案】用 r1 的中位猜测给一个可执行单点。
    注意 r1_guess=1000、r2=300 这两个数**没有题面依据**（题面只保证 R_eff∈[1000,1500]），
    仅作对照基线或数值初值。稳健选点请用 minimax_second_point()。
    """
    cands = second_point_candidates(s1, svd1_deg, r1_values=[r1_guess],
                                    r2=r2, arena_radius=arena_radius)
    if cands:
        return cands[0]
    # 退化：直接沿垂直方向找一个区域内的点
    ux, uy = dir_vec(svd1_deg)
    px, py = s1[0] + ux * r1_guess, s1[1] + uy * r1_guess
    for (dx, dy) in ((-uy, ux), (uy, -ux)):
        s2 = (px + dx * r2, py + dy * r2)
        if math.hypot(s2[0], s2[1]) <= arena_radius + 1e-9:
            g = crossing_angle(s1, s2, (px, py))
            return SecondPointCandidate(r1_guess, (px, py), s2, r2, g,
                                        localization_length_L(r1_guess, r2, g))
    return SecondPointCandidate(r1_guess, (px, py), (px, py), 0.0, 0.0, float("inf"))


def region_diameter_two_points(
    s1: tuple[float, float], svd1_deg: float,
    s2: tuple[float, float], svd2_deg: float,
    delta_deg: float = 1.0,
    arena_radius: float = 1800.0,
) -> tuple[float, list[tuple[float, float]]]:
    """
    两检测点各一次示向度 → 两楔形之交（近平行四边形小区域）的直径与顶点。
    用于校核一阶公式 L 与真实几何区域尺寸的吻合程度。
    """
    hps = wedge_halfplanes(*s1, svd1_deg, delta_deg) + \
        wedge_halfplanes(*s2, svd2_deg, delta_deg)
    poly = halfplane_intersection(hps, bound=6000.0, clip_circle_radius=arena_radius)
    if not poly:
        return 0.0, []
    diam, _, _ = polygon_diameter(poly)
    return diam, poly


# 第 1 层单点基线的别名（语义更清晰；供 Q3/对照使用）。
recommend_second_point_midpoint_baseline = recommend_second_point


# ══════════════════════════════════════════════════════════════════════════
# 第 2 层：有界不确定性下的 Minimax Next-Best-View（第二问的最终答案）
# ══════════════════════════════════════════════════════════════════════════

class P2Status(Enum):
    OK = "ok"                                       # 可行，给出 S2* 与候选区域
    GUARANTEED_DETECTION_INFEASIBLE = "infeasible"  # （保留）𝒞_recv 为空；本模型 S1∈𝒞_recv 恒成立故不触发
    DEGENERATE = "degenerate"                       # Ω₁ 为空/退化


@dataclass
class SourceUncertaintySet:
    """第一可行集 Ω₁ = D_1800 ∩ W(S1,θ̂1,±δ) ∩ {5<‖G−S1‖≤1500} 的多边形表示。

    圆弧用**外切多边形**近似，故 poly 是真 Ω₁ 的**超集**（保证性外近似）：据此算出的
    最坏清除覆盖半径 J、可行判据都是保守的（上界/充分条件）。内圈 {‖G−S1‖>5} 的挖除
    被省略（5 m 相对 1500 m 尺度可忽略，且保留只会让集合更大 → 保守）。
    """
    s1: tuple[float, float]
    svd1_deg: float
    delta_deg: float
    poly: list[tuple[float, float]]
    r_min: float                       # min_{G∈Ω₁} ‖G−S1‖
    r_max: float                       # max_{G∈Ω₁} ‖G−S1‖
    mec_center: tuple[float, float]    # 最小包围圆圆心（Chebyshev 中心）
    mec_radius: float                  # ρ* = MEC 半径
    empty: bool = False


def first_feasible_set(s1: tuple[float, float], svd1_deg: float,
                       cfg: ProblemConfig = DEFAULT_CONFIG) -> SourceUncertaintySet:
    """构造 Ω₁：第一楔形 ∩ (S1 为心、半径 1500 的外切圆) ∩ (原点为心、半径 1800 的外切圆)。"""
    hps = wedge_halfplanes(s1[0], s1[1], svd1_deg, cfg.delta_deg)
    hps += circle_outer_halfplanes(s1[0], s1[1], cfg.r1_max_cap_m, cfg.disk_sides)
    hps += circle_outer_halfplanes(0.0, 0.0, cfg.arena_radius_m, cfg.disk_sides)
    poly = halfplane_intersection(hps, bound=cfg.default_bound_m, clip_circle_radius=None)
    if not poly:
        return SourceUncertaintySet(s1, svd1_deg, cfg.delta_deg, [],
                                    0.0, 0.0, s1, 0.0, empty=True)
    dists = [math.hypot(x - s1[0], y - s1[1]) for (x, y) in poly]
    c, r = min_enclosing_circle(poly)
    return SourceUncertaintySet(s1, svd1_deg, cfg.delta_deg, poly,
                                min(dists), max(dists), c, r)


def analytic_second_point_tokekar(s1: tuple[float, float], svd1_deg: float,
                                  r_minus: float, r_plus: float,
                                  side: int = +1) -> tuple[float, float]:
    """Tokekar/Vander Hook/Isler bearing-only 主动定位的解析近似第二点：
        S_ana = S1 + (r₋+r₊)/2·u ± (r₊−r₋)/2·n
    u = 测向单位向量，n = 其左法向。仅作数值初值/对照，不作最终答案。"""
    ux, uy = dir_vec(svd1_deg)
    nx, ny = -uy, ux
    mid = 0.5 * (r_minus + r_plus)
    half = 0.5 * (r_plus - r_minus)
    return (s1[0] + mid * ux + side * half * nx,
            s1[1] + mid * uy + side * half * ny)


def reception_guaranteed(s2: tuple[float, float],
                         omega1: SourceUncertaintySet,
                         cfg: ProblemConfig = DEFAULT_CONFIG) -> bool:
    """S2 是否落在**保证接收域** 𝒞_recv = ∩_{G∈Ω₁} B(G, max(ρ_min, ‖G−S1‖))：
    即对所有 G∈Ω₁ 都有 ‖S2−G‖ ≤ max(ρ_min, ‖G−S1‖)，ρ_min = R_eff 下界 = 1000。

    这个半径来自"第一次已成功接收"这一硬信息：S1 测到位于 G 的源 ⟹ R_eff ≥ ‖G−S1‖，
    又题面 R_eff ≥ 1000，故与首次观测相容的最小接收半径是 max(1000, ‖G−S1‖)。

    精确判据（凸集，无需采样）：违反 ⟺ ∃G∈Ω₁ 使 ‖S2−G‖>ρ_min 且 ‖S2−G‖>‖G−S1‖
    （因 max(a,b)<c ⟺ a<c 且 b<c）。后一条件"G 距 S1 比距 S2 更近"是半平面
        H : (S2−S1)·G ≤ (‖S2‖²−‖S1‖²)/2。
    先把 Ω₁ 裁到 R = Ω₁∩H（仍是凸多边形），再看 R 内离 S2 最远的顶点：
      R 空 → 无违反 → True；否则 max_{v∈R}‖S2−v‖ ≤ ρ_min → True，否则 False。
    （边界等距点被计入 R 只会更保守——宁可拒真判为需保证接收，不会漏判。）
    """
    if omega1.empty or not omega1.poly:
        return False
    s1 = omega1.s1
    if math.hypot(s2[0] - s1[0], s2[1] - s1[1]) <= 1e-6:
        return True          # S2=S1：与首次观测同点，‖S2−G‖=r1(G)≤R_eff，接收必然成立
    nx, ny = s2[0] - s1[0], s2[1] - s1[1]
    c = 0.5 * ((s2[0] * s2[0] + s2[1] * s2[1]) - (s1[0] * s1[0] + s1[1] * s1[1]))
    closer_to_s1 = Halfplane(nx, ny, c)      # {G : (S2−S1)·G ≤ c} = G 距 S1 更近
    r_poly = clip_polygon_halfplanes(omega1.poly, [closer_to_s1])
    if not r_poly:
        return True
    return farthest_vertex_distance(r_poly, s2[0], s2[1]) <= cfg.reception_min_m + 1e-6


def worst_case_clearance_radius(
    s2: tuple[float, float],
    omega1: SourceUncertaintySet,
    cfg: ProblemConfig = DEFAULT_CONFIG,
    phi_step_deg: float = 0.5,
    refine: bool = True,
) -> tuple[float, Optional[float], list[tuple[float, float]]]:
    """
    J(S2) = max_{φ∈[α_min−δ, α_max+δ]} ρ( Ω₁ ∩ W(S2, φ, ±δ) )，
    其中 ρ(·) = 最小覆盖圆（MEC）半径 = min_Q max_{G∈Ω₂}‖G−Q‖。

    返回 (J, φ*_deg, 最坏 φ 处的 Ω₂ 多边形)。这是把
        max_{G∈Ω₁, e2∈[−δ,δ]} ρ(Ω₁ ∩ W(S2, arg(G−S2)+e2, ±δ))
    化成对第二楔形朝向 φ 的一维扫描（Ω₂ 只通过 φ 依赖 (G,e2)）。
    φ 的扫描范围由 Ω₁ 从 S2 看的角度轮廓 [α_min,α_max] 再各外扩 δ 给出。

    J ≤ clear_radius(20) ⟺ 无论真源落在 Ω₁ 何处、第二次测向误差如何，第二次测完后
    都存在一个落点 Q 使 Ω₂ 内所有可能源都距 Q ≤20 m，即"一次移动即可保证直接清除"。
    """
    poly = omega1.poly
    if not poly:
        return (0.0, None, [])
    delta = cfg.delta_deg
    lo, hi = angular_span_from_point(poly, s2[0], s2[1])
    phi_start, phi_end = lo - delta, hi + delta

    def rho_at(phi: float) -> tuple[float, list[tuple[float, float]]]:
        w2 = wedge_halfplanes(s2[0], s2[1], norm_deg(phi), delta)
        omega2 = clip_polygon_halfplanes(poly, w2)
        if not omega2:
            return 0.0, omega2
        _, r = min_enclosing_circle(omega2)
        return r, omega2

    best_j, best_phi, best_poly = -1.0, phi_start, []
    n_steps = max(1, int(math.ceil((phi_end - phi_start) / phi_step_deg)))
    for k in range(n_steps + 1):
        phi = phi_start + k * phi_step_deg
        r, omg = rho_at(phi)
        if r > best_j:
            best_j, best_phi, best_poly = r, phi, omg

    if refine and best_j >= 0.0:
        fine_lo, fine_hi, m = best_phi - phi_step_deg, best_phi + phi_step_deg, 20
        for k in range(m + 1):
            phi = fine_lo + (fine_hi - fine_lo) * k / m
            r, omg = rho_at(phi)
            if r > best_j:
                best_j, best_phi, best_poly = r, phi, omg

    return (max(best_j, 0.0), norm_deg(best_phi), best_poly)


@dataclass
class SecondPointRecommendation:
    """第二问的完整决策输出。"""
    status: P2Status
    feasible: bool
    s1: tuple[float, float]
    svd1_deg: float
    omega1: SourceUncertaintySet
    rho_star: float                    # ρ* = MEC(Ω₁) 半径（仅作诊断，不再当可行判据）
    clear_radius: float                # 一次移动清除阈值 = 20 m（J≤此值 ⟹ 可一次清除）
    reception_min: float               # ρ_min = R_eff 下界 = 1000
    # 决策结果
    s2_star: Optional[tuple[float, float]] = None      # argmin J
    j_star: float = float("inf")                       # 最坏清除覆盖半径的最小值（精修后）
    clearable_one_move: bool = False                   # J* ≤ clear_radius ？（能否二测后一次清除）
    eta: float = 0.05
    # 三个区域（均为粗网格上的采样点集，供画图/Q3 选点）
    reception_region: list[tuple[float, float]] = field(default_factory=list)    # 𝒞_recv（保证接收）
    clearable_region: list[tuple[float, float]] = field(default_factory=list)    # 𝒞_clear = {J ≤ 20}
    near_optimal_region: list[tuple[float, float]] = field(default_factory=list)  # 𝒞_η（回退候选带）
    s2_select: Optional[tuple[float, float]] = None    # 答案区域中距 S1 最近者
    select_dist_to_s1: float = float("inf")
    # 解析对照（Tokekar）
    s_ana_plus: Optional[tuple[float, float]] = None
    s_ana_minus: Optional[tuple[float, float]] = None
    j_ana_plus: float = float("inf")
    j_ana_minus: float = float("inf")
    # 供画 J(S2) 热力图/候选区域
    grid_points: list[tuple[float, float]] = field(default_factory=list)
    grid_J: list[float] = field(default_factory=list)
    grid_feasible: list[bool] = field(default_factory=list)
    notes: str = ""


def minimax_second_point(
    s1: tuple[float, float],
    svd1_deg: float,
    cfg: ProblemConfig = DEFAULT_CONFIG,
    grid_n: int = 25,
    eta: float = 0.05,
    phi_step_deg: float = 0.5,
    refine_iters: int = 2,
) -> SecondPointRecommendation:
    """
    第二问主入口（管线）：
        Ω₁  →  𝒞_recv（保证接收域，纯由题面 [−1°,1°] 与 [1000,1500] 推出）
            →  J(S2)=最坏清除覆盖半径  →  S2*=argmin_{𝒞_recv} J
            →  若 J*≤20：答案区域 = 𝒞_clear={J≤20}（可保证二测后一次清除）
               否则：答案区域 = 近优带 𝒞_η={J≤(1+η)J*}（需三次测向，取 J 最小）
            →  就近择点 S2^select = argmin ‖S−S1‖（字典序：先精度、再就近，无权重 λ）。

    - grid_n：𝒞_recv 包围盒上的粗网格边数（越大越精，越慢）。paper 级可调至 60+。
    - eta：回退候选带的相对松弛（仅当不存在一次清除点时启用）。
    - S1 恒属于 𝒞_recv（‖S1−G‖=r1(G)≤max(1000,r1)），故总有可行点，不会不可行。
    """
    omega1 = first_feasible_set(s1, svd1_deg, cfg)
    clear_r = cfg.clear_radius_m
    recv_min = cfg.reception_min_m

    if omega1.empty:
        return SecondPointRecommendation(
            P2Status.DEGENERATE, False, s1, svd1_deg, omega1, 0.0, clear_r, recv_min,
            notes="Ω₁ 为空（楔形与先验圆无交），无法给出第二点。")

    rho_star = omega1.mec_radius
    s_ana_plus = analytic_second_point_tokekar(s1, svd1_deg, omega1.r_min, omega1.r_max, +1)
    s_ana_minus = analytic_second_point_tokekar(s1, svd1_deg, omega1.r_min, omega1.r_max, -1)

    # ---- 𝒞_recv 的包围盒：每顶点半径 rad(v)=max(recv_min,‖v−S1‖)，𝒞_recv ⊆ ∩_v B(v,rad_v) ----
    verts = omega1.poly
    rads = [max(recv_min, math.hypot(vx - s1[0], vy - s1[1])) for (vx, vy) in verts]
    box_xlo = max(verts[i][0] - rads[i] for i in range(len(verts)))
    box_xhi = min(verts[i][0] + rads[i] for i in range(len(verts)))
    box_ylo = max(verts[i][1] - rads[i] for i in range(len(verts)))
    box_yhi = min(verts[i][1] + rads[i] for i in range(len(verts)))

    def eval_pts(pts: list[tuple[float, float]], refine: bool):
        Js, feas = [], []
        for s2 in pts:
            ok = reception_guaranteed(s2, omega1, cfg)
            feas.append(ok)
            Js.append(worst_case_clearance_radius(s2, omega1, cfg, phi_step_deg, refine)[0]
                      if ok else float("inf"))
        return Js, feas

    def make_grid(xlo, xhi, ylo, yhi, n):
        pts = []
        for i in range(n):
            x = xlo if n == 1 else xlo + (xhi - xlo) * i / (n - 1)
            for j in range(n):
                y = ylo if n == 1 else ylo + (yhi - ylo) * j / (n - 1)
                pts.append((x, y))
        return pts

    grid_points: list[tuple[float, float]] = []
    if box_xhi >= box_xlo and box_yhi >= box_ylo:
        grid_points = make_grid(box_xlo, box_xhi, box_ylo, box_yhi, grid_n)
    # 兜底纳入 S1（必在 𝒞_recv）、MEC 圆心、两个解析点（保证有可行点并直接对照解析解）
    grid_points += [s1, omega1.mec_center, s_ana_plus, s_ana_minus]
    grid_J, grid_feasible = eval_pts(grid_points, refine=False)

    best_i = min(range(len(grid_J)), key=lambda i: grid_J[i])
    s2_star = grid_points[best_i]

    # 局部精修：围绕当前最优收缩网格、开启 φ 精修
    span_x = (box_xhi - box_xlo) / max(1, grid_n - 1) if box_xhi >= box_xlo else 50.0
    span_y = (box_yhi - box_ylo) / max(1, grid_n - 1) if box_yhi >= box_ylo else 50.0
    for _ in range(max(0, refine_iters)):
        rx, ry = max(span_x * 2, 1e-6), max(span_y * 2, 1e-6)
        loc = make_grid(s2_star[0] - rx, s2_star[0] + rx,
                        s2_star[1] - ry, s2_star[1] + ry, 9)
        locJ, locF = eval_pts(loc, refine=True)
        bi = min(range(len(locJ)), key=lambda i: locJ[i])
        if locF[bi] and locJ[bi] < float("inf"):
            s2_star = loc[bi]
        span_x *= 0.34
        span_y *= 0.34

    j_star = worst_case_clearance_radius(s2_star, omega1, cfg, phi_step_deg, refine=True)[0]
    clearable = j_star <= clear_r + 1e-6

    # 三个区域（在粗网格采样上标注；供画图与 Q3 顺路择点）
    reception_region = [grid_points[i] for i in range(len(grid_points)) if grid_feasible[i]]
    clearable_region = [grid_points[i] for i in range(len(grid_points))
                        if grid_feasible[i] and grid_J[i] <= clear_r + 1e-6]
    feas_J = [grid_J[i] for i in range(len(grid_J)) if grid_feasible[i]]
    j_grid_star = min(feas_J) if feas_J else j_star
    thr = (1.0 + eta) * j_grid_star
    near_optimal_region = [grid_points[i] for i in range(len(grid_points))
                           if grid_feasible[i] and grid_J[i] <= thr]

    # 答案区域：优先"一次清除区" 𝒞_clear；否则回退近优带 𝒞_η；再否则退到 S2* 本身。
    if clearable_region:
        answer_region = clearable_region
    elif near_optimal_region:
        answer_region = near_optimal_region
    else:
        answer_region = [s2_star]

    # 字典序择点：答案区域中距 S1 最近
    s2_select = min(answer_region, key=lambda p: math.hypot(p[0] - s1[0], p[1] - s1[1]))
    select_dist = math.hypot(s2_select[0] - s1[0], s2_select[1] - s1[1])

    def j_of(p):
        return (worst_case_clearance_radius(p, omega1, cfg, phi_step_deg, refine=True)[0]
                if reception_guaranteed(p, omega1, cfg) else float("inf"))

    note = (f"可行：ρ*(Ω₁)={rho_star:.1f} m；J*={j_star:.2f} m "
            f"({'≤' if clearable else '>'} 清除半径 {clear_r:.0f} m)；"
            f"|𝒞_recv 采样|={len(reception_region)}，|𝒞_clear|={len(clearable_region)}，"
            f"|𝒞_η|={len(near_optimal_region)}（η={eta:.0%}）。"
            + ("存在能保证二测后一次移动清除的第二点。" if clearable
               else "无一次清除点，返回 J 最小者（建议三次测向）。"))

    return SecondPointRecommendation(
        P2Status.OK, True, s1, svd1_deg, omega1, rho_star, clear_r, recv_min,
        s2_star=s2_star, j_star=j_star, clearable_one_move=clearable, eta=eta,
        reception_region=reception_region, clearable_region=clearable_region,
        near_optimal_region=near_optimal_region,
        s2_select=s2_select, select_dist_to_s1=select_dist,
        s_ana_plus=s_ana_plus, s_ana_minus=s_ana_minus,
        j_ana_plus=j_of(s_ana_plus), j_ana_minus=j_of(s_ana_minus),
        grid_points=grid_points, grid_J=grid_J, grid_feasible=grid_feasible,
        notes=note)
