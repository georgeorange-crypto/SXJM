"""
Sutherland–Hodgman 凸多边形对半平面的裁剪。

用于 Q1：从一个足够大的初始凸多边形（目标圆外接方框）出发，
依次用每个楔形的两个半平面裁剪，得到定位区域 P = ∩ W_i。

前提：被裁剪多边形为凸（初始为凸，半平面裁剪保持凸），顶点按逆时针给出。
"""
from __future__ import annotations

from typing import List

from .halfplane import HalfPlane
from .vector import Vec2, sub, cross


def clip_polygon(poly: List[Vec2], hp: HalfPlane, eps: float = 1e-9) -> List[Vec2]:
    """用单个半平面裁剪多边形，返回内侧部分（可能为空）。"""
    if not poly:
        return []
    out: List[Vec2] = []
    n = len(poly)
    for i in range(n):
        cur = poly[i]
        nxt = poly[(i + 1) % n]
        s_cur = hp.signed(cur)
        s_nxt = hp.signed(nxt)
        cur_in = s_cur >= -eps
        nxt_in = s_nxt >= -eps
        if cur_in:
            out.append(cur)
        if cur_in != nxt_in:
            # 边与边界相交，插入交点
            denom = s_cur - s_nxt
            if abs(denom) > 1e-18:
                t = s_cur / denom
                ip = (cur[0] + t * (nxt[0] - cur[0]), cur[1] + t * (nxt[1] - cur[1]))
                out.append(ip)
    return _dedup(out)


def clip_polygon_halfplanes(poly: List[Vec2], hps: List[HalfPlane]) -> List[Vec2]:
    """依次用多个半平面裁剪。"""
    result = poly
    for hp in hps:
        result = clip_polygon(result, hp)
        if not result:
            return []
    return result


def bounding_square(radius: float, center: Vec2 = (0.0, 0.0)) -> List[Vec2]:
    """
    以 center 为中心、边长 2*radius*margin 的逆时针方框，作为裁剪初始多边形。
    取 margin 略大于 sqrt(2) 以确保包住半径 radius 的圆及无界楔形的近处部分。
    """
    r = radius
    cx, cy = center
    return [
        (cx - r, cy - r),
        (cx + r, cy - r),
        (cx + r, cy + r),
        (cx - r, cy + r),
    ]


def polygon_area(poly: List[Vec2]) -> float:
    """有向面积（逆时针为正）的绝对值。"""
    if len(poly) < 3:
        return 0.0
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def _dedup(poly: List[Vec2], eps: float = 1e-9) -> List[Vec2]:
    """去除相邻重复点。"""
    if not poly:
        return poly
    out: List[Vec2] = [poly[0]]
    for p in poly[1:]:
        if abs(p[0] - out[-1][0]) > eps or abs(p[1] - out[-1][1]) > eps:
            out.append(p)
    # 首尾重复
    if len(out) > 1 and abs(out[0][0] - out[-1][0]) <= eps and abs(out[0][1] - out[-1][1]) <= eps:
        out.pop()
    return out
