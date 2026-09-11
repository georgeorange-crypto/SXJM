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


def _builtin_directional(src_radius: float, h: float = 600.0) -> List[Point]:
    """Triangular lattice (edge ``h`` < 1000) clipped just beyond the source range.
    Mirrors Way3's construction so any in-range point sees >=3 lattice vertices
    within 1000 m, closing the 180° angular gap (verified by Way3, deferred here)."""
    pts: List[Point] = []
    rmax = src_radius + 0.55 * h
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
