"""Legacy Way3 omni backbone as Way4's guaranteed-completion fallback (§6.5, M3B).

DESIGN.md §6 semantic flip: Way3's fixed ``omni_scan_points`` is no longer the
certificate itself — it is demoted to ``fallback_anchors``, a known-good template
that, once fully scanned NO_SIGNAL for a channel, guarantees ``D_1800`` is covered
by that channel's detection discs (Invariant C: task always completable, even when
the arbitrary-disc-cover verifier false-negatives on a zero-margin seam, note A).

Two sourcing paths, both yielding a *sound* backbone:
  * Faithful: load Way3's own ``omni_scan_points`` (center + K-ring, designed for a
    50 m margin: worst nearest-scan distance <= 950 < 1000). Loaded by file path so
    Way3's package ``__init__`` never runs.
  * Built-in: a center + 8-ring at radius 1150 m, which the SAME arbitrary-disc
    verifier certifies (with margin) — used when Way3 is not on disk.

Either way, ``assert_backbone_covers`` re-confirms the union covers the arena with
Way4's *own* conservative quadtree (never Way3's dense-grid sampler — that would be
禁止11), so soundness never rests on how the points were sourced.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import List, Optional, Tuple

from .hard_disc_cover import HardDiscCoverVerifier

Point = Tuple[float, float]

ARENA_RADIUS_M = 1800.0
DETECTION_RADIUS_M = 1000.0
DESIGN_MARGIN_M = 50.0   # Way3's robustness margin (worst nearest-scan <= 1000-50)

_CACHE: Optional[List[Point]] = None


def _ring(k: int, radius: float, phase_deg: float = 0.0) -> List[Point]:
    return [
        (
            radius * math.cos(math.radians(phase_deg + i * 360.0 / k)),
            radius * math.sin(math.radians(phase_deg + i * 360.0 / k)),
        )
        for i in range(k)
    ]


def _builtin_backbone() -> List[Point]:
    """Center + 8-ring at 1150 m. Covers ``D_1800`` with a >=90 m margin against
    the 1000 m detection radius (confirmed by the quadtree verifier in tests)."""
    return [(0.0, 0.0)] + _ring(8, 1150.0)


def _load_way3_coverage():
    """Import Way3's ``coverage.py`` by file path (no ``jammerhunt`` package
    ``__init__`` side effects). None on any failure."""
    try:
        # this file: SX/Way4/src/way4/certificate/fallback.py -> parents[4] == SX
        sx_root = Path(__file__).resolve().parents[4]
        coverage_py = sx_root / "Way3" / "jammerhunt" / "coverage.py"
        if not coverage_py.is_file():
            return None
        spec = importlib.util.spec_from_file_location("_way3_coverage", coverage_py)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # coverage.py is stdlib-only, no relative imports
        return mod
    except Exception:
        return None


def _load_way3_omni() -> Optional[List[Point]]:
    """Faithful Way3 ``omni_scan_points`` (best effort)."""
    mod = _load_way3_coverage()
    if mod is None:
        return None
    try:
        return [(float(x), float(y)) for (x, y) in mod.omni_scan_points()]
    except Exception:
        return None


def omni_fallback_anchors(prefer_way3: bool = True) -> List[Point]:
    """The guaranteed-completion omni backbone (cached). Faithful Way3 points when
    available, else the built-in sound backbone."""
    global _CACHE
    if _CACHE is None:
        pts = _load_way3_omni() if prefer_way3 else None
        _CACHE = pts if pts else _builtin_backbone()
    return list(_CACHE)


def assert_backbone_covers(
    anchors: Optional[List[Point]] = None,
    detection_radius: float = DETECTION_RADIUS_M,
    arena_radius: float = ARENA_RADIUS_M,
) -> bool:
    """Confirm the anchors' detection discs cover the arena, using Way4's OWN
    conservative quadtree (sound; never Way3's dense grid). Returns the verdict —
    tests assert it True so the fallback's soundness is checked, not assumed."""
    anchors = anchors if anchors is not None else omni_fallback_anchors()
    verifier = HardDiscCoverVerifier(radius=detection_radius, arena_radius=arena_radius)
    return verifier.is_covered(anchors)


# --------------------------------------------------------------------------- #
# P4 directional backbone (M3C) — legacy template only; P4 does NOT upgrade this
# stage (§6.9). The angular-gap (< 180°) soundness is Way3's ``verify_three_cover``
# concern and is re-examined when the directional certificate is wired in (M8).
# --------------------------------------------------------------------------- #
_DIR_CACHE: Optional[List[Point]] = None


def _load_way3_directional(src_radius: float) -> Optional[List[Point]]:
    mod = _load_way3_coverage()
    if mod is None:
        return None
    try:
        return [(float(x), float(y)) for (x, y) in mod.directional_scan_points(src_radius=src_radius)]
    except Exception:
        return None


def _builtin_directional(src_radius: float, h: float = 900.0) -> List[Point]:
    """Triangular lattice (edge ``h`` < 1000) clipped just beyond the source range.
    Mirrors Way3's construction so any in-range point sees >=3 lattice vertices
    within 1000 m, closing the 180° angular gap (verified by Way3, deferred here)."""
    pts: List[Point] = []
    # Keep a full one-cell exterior shell.  The smaller 0.55*h envelope can
    # leave rim sources with all nearby detectors on the arena-facing side,
    # which is a valid directional blind-arc counterexample.
    rmax = src_radius + h
    dy = h * math.sqrt(3.0) / 2.0
    jmax = int(math.ceil(rmax / dy)) + 1
    for j in range(-jmax, jmax + 1):
        y = j * dy
        x_off = (h / 2.0) if (j % 2) else 0.0
        imax = int(math.ceil((rmax + abs(x_off)) / h)) + 1
        for i in range(-imax, imax + 1):
            x = i * h + x_off
            if x * x + y * y <= rmax * rmax + 1e-6:
                pts.append((x, y))
    # Offline verified fixed-template reduction for h=900: these six far outer
    # vertices are redundant for the 1800 m domain's directional 3-cover.  Keep
    # the reduction deterministic and geometry-only; never remove anchors based
    # on runtime belief or observed source locations.  Dense 50 m and 25 m
    # regressions cover the resulting 31-point template (4053 and 16241 domain
    # samples respectively).
    if abs(h - 900.0) <= 1e-9:
        dy900 = h * math.sqrt(3.0) / 2.0
        pts = [
            p for p in pts
            if not (
                (abs(abs(p[0]) - 1.5 * h) <= 1e-9 and abs(p[1]) > 2.0 * dy900)
                or (abs(abs(p[0]) - 3.0 * h) <= 1e-9 and abs(p[1]) <= 1e-9)
            )
        ]
    return pts


def directional_fallback_anchors(
    src_radius: float = ARENA_RADIUS_M, prefer_way3: bool = True
) -> List[Point]:
    """The P4 directional 3-cover backbone (cached). Faithful Way3
    ``directional_scan_points`` when available, else a built-in triangular lattice."""
    global _DIR_CACHE
    if _DIR_CACHE is None:
        pts = _load_way3_directional(src_radius) if prefer_way3 else None
        _DIR_CACHE = pts if pts else _builtin_directional(src_radius)
    return list(_DIR_CACHE)


def ray_sweep_points(
    origin: Point,
    bearing_deg: float,
    length_m: float,
    *,
    lateral_half_width_m: float = 13.0,
    step_m: float = 15.0,
) -> List[Point]:
    """Generate a deterministic zig-zag rescue path around a bearing ray.

    This is a *candidate path*, not a localization or absence certificate.  It
    is intended for the P4 case where a valid bearing has already been
    observed but subsequent views may fall in the directional blind half-plane.
    The lateral spacing is deliberately explicit: callers must choose it so
    the resulting path's clearance discs cover the desired wedge.  Points are
    emitted in alternating order, with the ray endpoint included once.
    """
    if length_m < 0 or lateral_half_width_m < 0 or step_m <= 0:
        raise ValueError("length_m and lateral_half_width_m must be non-negative; step_m > 0")
    ux = math.cos(math.radians(bearing_deg))
    uy = math.sin(math.radians(bearing_deg))
    nx, ny = -uy, ux
    count = max(1, int(math.ceil(length_m / step_m)))
    points: List[Point] = []
    for i in range(count + 1):
        along = min(length_m, i * step_m)
        side = lateral_half_width_m if i % 2 == 0 else -lateral_half_width_m
        points.append((
            origin[0] + along * ux + side * nx,
            origin[1] + along * uy + side * ny,
        ))
    endpoint = (origin[0] + length_m * ux, origin[1] + length_m * uy)
    if points[-1] != endpoint:
        points.append(endpoint)
    return points


def triangular_clear_sweep_points(
    center: Point, *, radius_m: float = 50.0, spacing_m: float = 33.0
) -> List[Point]:
    """Local triangular-lattice clear candidates; never a certificate."""
    if radius_m < 0 or spacing_m <= 0:
        raise ValueError("radius_m must be non-negative and spacing_m > 0")
    dy = spacing_m * math.sqrt(3.0) / 2.0
    n = int(math.ceil(radius_m / dy))
    points: List[Point] = []
    for j in range(-n, n + 1):
        y = j * dy
        x_offset = 0.5 * spacing_m if j % 2 else 0.0
        i_max = int(math.ceil((radius_m + abs(x_offset)) / spacing_m))
        for i in range(-i_max, i_max + 1):
            x = i * spacing_m + x_offset
            if x * x + y * y <= radius_m * radius_m + 1e-9:
                points.append((center[0] + x, center[1] + y))
    points.sort(key=lambda p: (math.dist(center, p), p[1], p[0]))
    return points
