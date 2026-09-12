"""二维向量工具（用 tuple[float,float] 表示，避免重量级依赖）。"""
from __future__ import annotations

import math
from typing import Tuple

Vec2 = Tuple[float, float]


def add(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] + b[0], a[1] + b[1])


def sub(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def mul(a: Vec2, s: float) -> Vec2:
    return (a[0] * s, a[1] * s)


def dot(a: Vec2, b: Vec2) -> float:
    return a[0] * b[0] + a[1] * b[1]


def cross(a: Vec2, b: Vec2) -> float:
    """z 分量叉积 ax*by - ay*bx。"""
    return a[0] * b[1] - a[1] * b[0]


def norm(a: Vec2) -> float:
    return math.hypot(a[0], a[1])


def dist(a: Vec2, b: Vec2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def dist2(a: Vec2, b: Vec2) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def normalize(a: Vec2) -> Vec2:
    n = norm(a)
    if n == 0.0:
        return (0.0, 0.0)
    return (a[0] / n, a[1] / n)


def perp(a: Vec2) -> Vec2:
    """逆时针旋转 90°。"""
    return (-a[1], a[0])


def rotate(a: Vec2, theta: float) -> Vec2:
    c = math.cos(theta)
    s = math.sin(theta)
    return (a[0] * c - a[1] * s, a[0] * s + a[1] * c)
