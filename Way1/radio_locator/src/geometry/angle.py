"""
角度工具：题目采用 东=0°、逆时针为正、范围 [0,360)。

内部计算统一用弧度；对外与模拟器交互用度。所有跨零点（如 359.5°±1°）的处理
都收敛到这里，避免散落在各模块。
"""
from __future__ import annotations

import math

TWO_PI = 2.0 * math.pi
DEG = math.pi / 180.0


def deg2rad(d: float) -> float:
    return d * DEG


def rad2deg(r: float) -> float:
    return r / DEG


def wrap_pi(theta: float) -> float:
    """归一化到 (-π, π]。"""
    t = math.fmod(theta, TWO_PI)
    if t <= -math.pi:
        t += TWO_PI
    elif t > math.pi:
        t -= TWO_PI
    return t


def wrap_2pi(theta: float) -> float:
    """归一化到 [0, 2π)。"""
    t = math.fmod(theta, TWO_PI)
    if t < 0.0:
        t += TWO_PI
    # fmod 对极小负数可能返回 2π 边界，强制收敛
    if t >= TWO_PI:
        t -= TWO_PI
    return t


def norm_deg(d: float) -> float:
    """归一化到 [0, 360)。"""
    x = math.fmod(d, 360.0)
    if x < 0.0:
        x += 360.0
    if x >= 360.0:
        x -= 360.0
    return x


def ang_diff_pi(a: float, b: float) -> float:
    """两角有向最小差 a-b，落在 (-π, π]（弧度）。"""
    return wrap_pi(a - b)


def ang_abs_diff(a: float, b: float) -> float:
    """两角无向最小夹角，落在 [0, π]（弧度）。"""
    return abs(wrap_pi(a - b))


def ang_abs_diff_deg(a: float, b: float) -> float:
    """两角无向最小夹角，落在 [0, 180]（度）。"""
    return abs(rad2deg(wrap_pi(deg2rad(a) - deg2rad(b))))


def bearing_rad(dx: float, dy: float) -> float:
    """由方向增量求方位角（弧度，[0,2π)）。东=0，逆时针为正。"""
    return wrap_2pi(math.atan2(dy, dx))


def unit(theta: float) -> tuple[float, float]:
    """单位方向向量。"""
    return (math.cos(theta), math.sin(theta))


def angle_in_arc(theta: float, lo: float, hi: float) -> bool:
    """
    判断角 theta（弧度）是否落在从 lo 逆时针到 hi 的圆弧内（含边界）。
    lo, hi 任意实数；内部按 [0,2π) 归一化，支持跨零点。
    约定：弧长为 (hi-lo) mod 2π；若归一化后 lo==hi 视为整圆（恒 True）。
    """
    lo2 = wrap_2pi(lo)
    hi2 = wrap_2pi(hi)
    t = wrap_2pi(theta)
    span = wrap_2pi(hi - lo)
    if span == 0.0:
        # lo==hi：区分“零宽点”与“整圆”需上层语义；此处按点处理。
        return math.isclose(t, lo2, abs_tol=1e-12)
    rel = wrap_2pi(t - lo2)
    return rel <= span + 1e-12
