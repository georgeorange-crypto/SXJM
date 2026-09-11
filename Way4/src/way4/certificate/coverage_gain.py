"""CoverageGainMap — fast heuristic coverage map for the planner (DESIGN.md §6.6).

This is the NON-certifying layer of the two-layer separation (§6.2). It answers
"how much *new* arena would a NO_SIGNAL scan at q add for channel c?" on a coarse
grid, cheaply, for candidate scoring. It is deliberately incapable of certifying
absence: it exposes coverage *ratios* and *gains* only — never an ``is_complete``
(Invariant B). Only ``HardDiscCoverVerifier`` / legacy backbone / cardinality may
certify (§6.5).

numpy is used here purely for speed (§6.8 "scan points stored numpy"); the hard
verifier stays stdlib for soundness clarity.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

Point = Tuple[float, float]


class CoverageGainMap:
    """Coarse (default 40 m) in-arena grid of coverage masks, one per channel."""

    def __init__(
        self,
        arena_radius: float = 1800.0,
        spacing: float = 40.0,
        detection_radius: float = 1000.0,
    ) -> None:
        self.arena_radius = float(arena_radius)
        self.spacing = float(spacing)
        self.detection_radius = float(detection_radius)
        self._r2 = self.detection_radius * self.detection_radius

        # Grid-cell centres whose centre lies within the arena disc.
        n = int(np.floor(arena_radius / spacing))
        coords = (np.arange(-n, n + 1) + 0.0) * spacing
        gx, gy = np.meshgrid(coords, coords)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        inside = (pts[:, 0] ** 2 + pts[:, 1] ** 2) <= arena_radius * arena_radius
        self.grid: np.ndarray = pts[inside]              # (N, 2)
        self.n: int = int(self.grid.shape[0])
        self._masks: Dict[int, np.ndarray] = {}          # channel -> bool[N]

    # -- mask maintenance ---------------------------------------------------

    def _mask(self, channel: int) -> np.ndarray:
        m = self._masks.get(channel)
        if m is None:
            m = np.zeros(self.n, dtype=bool)
            self._masks[channel] = m
        return m

    def add_no_signal(self, channel: int, scan_point: Point) -> None:
        """Fold a NO_SIGNAL scan at ``scan_point`` into channel ``c``'s mask:
        every grid centre within the (lower-bound) detection radius is covered."""
        m = self._mask(channel)
        d2 = (self.grid[:, 0] - scan_point[0]) ** 2 + (self.grid[:, 1] - scan_point[1]) ** 2
        np.logical_or(m, d2 <= self._r2, out=m)

    # -- heuristic queries (NEVER certify) ----------------------------------

    def coverage_ratio(self, channel: int) -> float:
        """Fraction of in-arena grid centres already covered for ``channel``."""
        if self.n == 0:
            return 0.0
        return float(self._mask(channel).sum()) / self.n

    def coverage_gain(self, channel: int, q: Point) -> float:
        """Fraction of grid centres a NO_SIGNAL scan at ``q`` would *newly* cover
        for ``channel`` (0..1). This is the planner's exploration/certificate
        heuristic — not a proof."""
        if self.n == 0:
            return 0.0
        m = self._mask(channel)
        d2 = (self.grid[:, 0] - q[0]) ** 2 + (self.grid[:, 1] - q[1]) ** 2
        newly = (d2 <= self._r2) & (~m)
        return float(newly.sum()) / self.n

    def batch_coverage_gain(self, channels, q: Point, weights=None) -> float:
        """Weighted sum of per-channel coverage gains at ``q`` over a channel set
        (DESIGN.md §6.6 batch ``G``). ``weights`` maps channel -> weight (default 1)."""
        total = 0.0
        for c in channels:
            w = 1.0 if weights is None else float(weights.get(c, 1.0))
            total += w * self.coverage_gain(c, q)
        return total

    def remaining_holes(self, channel: int) -> List[Point]:
        """In-arena grid centres NOT yet covered for ``channel`` (planning target
        set for verification; DESIGN.md §6.7 / §10). Approximate, in-arena only —
        never the hard verifier's unresolved cells (Conservatism note B)."""
        m = self._mask(channel)
        holes = self.grid[~m]
        return [(float(x), float(y)) for x, y in holes]

    def is_fully_covered_heuristic(self, channel: int) -> bool:
        """All grid centres covered. NOTE: heuristic only — this is NOT a
        certificate and must never drive ABSENT (Invariant B). Use it only to
        decide *when to bother* running the hard verifier (§6.8)."""
        return bool(self._mask(channel).all()) and self.n > 0
