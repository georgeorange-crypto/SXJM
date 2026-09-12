"""P3 arbitrary-scan-point hard coverage certificate — conservative adaptive quadtree.

Frozen design: Way4/DESIGN.md §6.1, §6.3, §6.10 (Invariants A/B, Conservatism
notes A/B, Resolution note).

Physics (§6.1): for channel ``c`` let ``S_c`` be the positions of every *real*
``scan(c) -> NO_SIGNAL``. Omnidirectional detection means "detected iff within
``R_eff``" with a guaranteed lower bound ``R_eff >= 1000``. Hence
``NO_SIGNAL @ s`` implies any source ``p_c`` satisfies ``|p_c - s| > R_eff >= 1000``,
i.e. ``p_c not in B(s, 1000)``. Therefore if the whole arena disc
``D = D(0, arena_radius)`` is contained in ``union_i B(s_i, 1000)`` then no legal
source can exist and the channel is ``ABSENT_CERTIFIED``.

Soundness contract (the ONLY thing this module guarantees):

    ``is_covered(scan_points) == True``  =>  ``D subset of union B(s, radius)``.

It is *allowed* to return ``False`` when coverage actually holds (conservative
false negatives — Conservatism notes A/B); it MUST NEVER return ``True`` when a
legal in-arena point lies outside every disc (no false positive). Both ``eps``
offsets push strictly toward conservatism:

  * prune a cell as OUTSIDE_ARENA only when ``min_dist(origin, cell) > arena_radius + eps``;
  * CERTIFY a cell only when some *single* disc's farthest-corner distance is
    ``<= radius - eps`` (single-disc test => whole cell inside that one disc).

``min_cell_size`` (Resolution note, §6.10) controls proof completeness and cost,
NOT soundness: a coarser floor only yields more false negatives, never a false
positive.

This first implementation is stdlib-only (matching Way3's stdlib ethos). The
numpy vectorisation and cross-call cell caching described in §6.8 are pure
performance optimisations that can be slotted in later without changing this
interface or its guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, sqrt
from typing import List, Sequence, Tuple

Point = Tuple[float, float]

_HALF_DIAG = sqrt(2.0) / 2.0  # half-diagonal / side ratio of a square


def distance(a: Point, b: Point) -> float:
    """Euclidean distance. (Placeholder for the future sxjm_core.geometry core.)"""
    return hypot(a[0] - b[0], a[1] - b[1])


@dataclass(frozen=True)
class CoverageCell:
    """An axis-aligned square region of the quadtree."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    depth: int

    @property
    def size(self) -> float:
        return self.xmax - self.xmin

    @property
    def center(self) -> Point:
        return (0.5 * (self.xmin + self.xmax), 0.5 * (self.ymin + self.ymax))

    def corners(self) -> Tuple[Point, Point, Point, Point]:
        return (
            (self.xmin, self.ymin),
            (self.xmin, self.ymax),
            (self.xmax, self.ymin),
            (self.xmax, self.ymax),
        )

    def subdivide(self) -> Tuple["CoverageCell", "CoverageCell", "CoverageCell", "CoverageCell"]:
        mx = 0.5 * (self.xmin + self.xmax)
        my = 0.5 * (self.ymin + self.ymax)
        d = self.depth + 1
        return (
            CoverageCell(self.xmin, mx, self.ymin, my, d),
            CoverageCell(mx, self.xmax, self.ymin, my, d),
            CoverageCell(self.xmin, mx, my, self.ymax, d),
            CoverageCell(mx, self.xmax, my, self.ymax, d),
        )

    def min_dist_to(self, p: Point) -> float:
        """Distance from ``p`` to the nearest point of the (closed) box; 0 if inside."""
        dx = max(self.xmin - p[0], 0.0, p[0] - self.xmax)
        dy = max(self.ymin - p[1], 0.0, p[1] - self.ymax)
        return hypot(dx, dy)

    def max_dist_to(self, p: Point) -> float:
        """Distance from ``p`` to the farthest point of the box (always a corner)."""
        dx = max(p[0] - self.xmin, self.xmax - p[0])
        dy = max(p[1] - self.ymin, self.ymax - p[1])
        return hypot(dx, dy)


def cell_fully_covered_by_disc(
    cell: CoverageCell, scan_point: Point, radius: float = 1000.0, eps: float = 1e-7
) -> bool:
    """True iff the whole ``cell`` lies inside ``B(scan_point, radius)``.

    The farthest point of an axis-aligned box from any fixed point is a corner,
    so ``cell.max_dist_to == max over corners``. The ``- eps`` makes acceptance
    strictly conservative (§6.3).
    """
    return cell.max_dist_to(scan_point) <= radius - eps


class HardDiscCoverVerifier:
    """Conservative adaptive-quadtree hard cover verifier (§6.3).

    The only source of ``is_complete = True`` for the P3 arbitrary-scan-point
    certificate. Never produces a false positive; may produce false negatives.
    """

    def __init__(
        self,
        radius: float = 1000.0,
        arena_radius: float = 1800.0,
        min_cell_size: float = 10.0,
        max_depth: int = 11,
        eps: float = 1e-7,
    ) -> None:
        self.radius = float(radius)
        self.arena_radius = float(arena_radius)
        self.min_cell_size = float(min_cell_size)
        self.max_depth = int(max_depth)
        self.eps = float(eps)

    # -- public API ---------------------------------------------------------

    def is_covered(self, scan_points: Sequence[Point]) -> bool:
        """Return True only if the arena disc is provably covered by the union
        of ``B(s, radius)`` over the given NO_SIGNAL scan points."""
        pts: List[Point] = [(float(x), float(y)) for (x, y) in scan_points]
        if not pts:
            return False

        origin = (0.0, 0.0)
        arena_cut = self.arena_radius + self.eps
        floor = self.min_cell_size

        root = CoverageCell(
            -self.arena_radius, self.arena_radius,
            -self.arena_radius, self.arena_radius, 0,
        )
        stack: List[CoverageCell] = [root]
        while stack:
            cell = stack.pop()
            # OUTSIDE_ARENA: prune only cells entirely beyond the arena disc.
            if cell.min_dist_to(origin) > arena_cut:
                continue
            if self._single_disc_covers(cell, pts):
                continue
            # Not certified. Refine unless we have hit the resolution floor.
            if cell.size <= floor or cell.depth >= self.max_depth:
                # An in-arena cell that no single disc covers at the finest
                # resolution -> cannot prove coverage (conservative False).
                return False
            stack.extend(cell.subdivide())
        return True

    # -- internals ----------------------------------------------------------

    def _single_disc_covers(self, cell: CoverageCell, pts: Sequence[Point]) -> bool:
        """Does some single disc B(s, radius) contain the whole cell?"""
        rho = _HALF_DIAG * cell.size  # half-diagonal: every corner within rho of center
        cx, cy = cell.center
        thr = self.radius - self.eps
        for sx, sy in pts:
            dc = hypot(cx - sx, cy - sy)
            # Cheap sound accept: all corners within dc + rho <= thr.
            if dc + rho <= thr:
                return True
            # Cheap sound reject: nearest corner >= dc - rho > thr => cannot cover.
            if dc - rho > thr:
                continue
            # Exact single-disc test on the farthest corner.
            if cell.max_dist_to((sx, sy)) <= thr:
                return True
        return False
