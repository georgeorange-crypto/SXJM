"""Per-channel P3 belief (DESIGN.md §4, §5) and the multi-channel container.

Representation choice (the crux of soundness):

``F_c`` is a convex polygon that is a SUPERSET of the true feasible set — the
circumscribed arena polygon clipped by every ``±(1°+ε)`` bearing wedge and every
range-disc superset ``B(S, 1500)``. Two consequences, both intended:

  * The true source is NEVER clipped out (DESIGN.md §15's headline property):
    the measured ``svd`` is within 1° of the true bearing, so the true source's
    bearing from ``S`` lies inside the ``1°+ε`` wedge; and the true source lies
    within ``R_eff ≤ 1500`` of ``S`` when detected, hence inside the range disc.
  * ``MEC(F_c).r`` is an OUTER bound of the true feasible set's MEC. So
    ``MEC.r ≤ clear_threshold`` still guarantees the true source is within the
    clear radius of the MEC centre — a sound blind-clear (DESIGN.md §3.3).

NO_SIGNAL observations only append negative exclusion discs; they never make a
channel PRESENT and never certify absence here (Invariants A/B/D live in the
certificate layer and the cardinality guard, not in belief).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import ceil, inf
from typing import Dict, List, Optional, Sequence

from sxjm_core.geometry import (
    Point,
    arena_polygon,
    angle_sep_deg,
    disc_superset_halfplanes,
    dist,
    halfplane_intersection,
    min_enclosing_circle,
    norm_deg,
    polygon_area,
    polygon_diameter,
    wedge_halfplanes,
)


@dataclass(frozen=True)
class EffectiveRegion:
    """Explicit non-convex conservative raster representation of effective F_c."""
    spacing: float
    cells: frozenset

    @property
    def area(self) -> float:
        return len(self.cells) * self.spacing * self.spacing

    @property
    def components(self) -> int:
        remaining = set(self.cells)
        count = 0
        while remaining:
            count += 1
            stack = [remaining.pop()]
            while stack:
                x, y = stack.pop()
                for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if q in remaining:
                        remaining.remove(q)
                        stack.append(q)
        return count


class ChannelStatus(str, Enum):
    """Way2's four states plus ABSENT_CERTIFIED (DESIGN.md §4; no probabilistic
    absence). PRESENT is not a state — it is the union DETECTED/LOCALIZED/CLEARED
    (§6.5 cardinality)."""

    UNKNOWN = "UNKNOWN"
    PRESENT_UNOBSERVED = "PRESENT_UNOBSERVED"  # cardinality-forced, no bearing yet
    INITIALIZED = "INITIALIZED"                # positive geometry sufficient for refine
    DETECTED = "DETECTED"        # a source is confirmed present, MEC still large
    LOCALIZED = "LOCALIZED"      # MEC.r <= clear_threshold -> clearable
    CLEARED = "CLEARED"
    ABSENT_CERTIFIED = "ABSENT_CERTIFIED"


#: Channels that a real positive observation has confirmed to hold a source
#: (Invariant D: only these count toward the cardinality of 16).
PRESENT_STATES = frozenset(
    {ChannelStatus.PRESENT_UNOBSERVED, ChannelStatus.DETECTED,
     ChannelStatus.INITIALIZED, ChannelStatus.LOCALIZED, ChannelStatus.CLEARED}
)
#: Channels needing no further action before EXIT (DESIGN.md §11 EXIT guard).
RESOLVED_STATES = frozenset({ChannelStatus.CLEARED, ChannelStatus.ABSENT_CERTIFIED})


@dataclass(frozen=True)
class ExclusionDisc:
    """A NO_SIGNAL disc: the source is provably NOT within ``radius`` of
    ``center`` (uses the guaranteed lower bound 1000, DESIGN.md §6.1)."""

    center: Point
    radius: float = 1000.0


@dataclass(frozen=True)
class BearingObs:
    """A positive bearing measurement: ``svd_deg`` within 1° of the true bearing
    of the source as seen from ``point`` (DESIGN.md §1)."""

    point: Point
    svd_deg: float
    time: float = 0.0


@dataclass
class ChannelBelief:
    """P3 belief for a single channel."""

    channel: int
    clear_threshold: float = 18.0        # Way2 value: 2 m margin under the 20 m radius
    range_radius: float = 1500.0         # R_eff upper bound, detected => within this
    detect_lower_bound: float = 1000.0   # R_eff guaranteed lower bound (NO_SIGNAL disc)
    near_radius: float = 5.0             # 'near' => source within 5 m (DESIGN.md §1)
    wedge_half_deg: float = 1.0
    eps_num_deg: float = 1e-6
    arena_radius: float = 1800.0

    status: ChannelStatus = ChannelStatus.UNKNOWN
    F_c: Optional[List[Point]] = None
    negative_discs: List[ExclusionDisc] = field(default_factory=list)
    bearings: List[BearingObs] = field(default_factory=list)
    near_points: List[Point] = field(default_factory=list)
    mec_center: Optional[Point] = None
    mec_radius: float = inf
    diameter: float = inf
    area: float = inf
    kappa: float = inf
    principal_axis: Optional[Point] = None
    connected_components: int = 1
    scan_count: int = 0
    last_scan_time: float = 0.0
    first_detect_time: Optional[float] = None
    localized_time: Optional[float] = None
    cleared_time: Optional[float] = None

    # -- observation intake (DESIGN.md §5) ---------------------------------

    def record_no_signal(self, point: Point, time: float = 0.0) -> None:
        """P3 NO_SIGNAL @ point. Appends a negative exclusion disc; never makes
        the channel PRESENT and never certifies absence (Invariants A/B)."""
        self.negative_discs.append(ExclusionDisc(point, self.detect_lower_bound))
        self.scan_count += 1
        self.last_scan_time = max(self.last_scan_time, time)
        self._update_inertia_summary()

    def contains_possible_source(self, p: Point) -> bool:
        """Return whether ``p`` remains in the effective feasible set.

        ``F_c`` remains the conservative convex outer set used by safety/MEC.
        This query additionally applies every sound NO_SIGNAL exclusion disc,
        so planning and hypothesis queries cannot sample a ruled-out location.
        """
        if dist(p, (0.0, 0.0)) > self.arena_radius + 1e-9:
            return False
        if self.F_c is not None and not _point_in_convex_polygon(p, self.F_c):
            return False
        return not self.excludes(p)

    def _effective_grid(self, spacing: float = 30.0) -> List[Point]:
        """Deterministic interior samples for the non-convex effective set."""
        r = self.arena_radius
        n = int(ceil(r / spacing))
        return [
            (ix * spacing, iy * spacing)
            for ix in range(-n, n + 1)
            for iy in range(-n, n + 1)
            if self.contains_possible_source((ix * spacing, iy * spacing))
        ]

    def effective_region(self, spacing: float = 30.0) -> EffectiveRegion:
        """Return an explicit non-convex region for planner queries."""
        return EffectiveRegion(
            float(spacing),
            frozenset((round(x / spacing), round(y / spacing))
                      for x, y in self._effective_grid(spacing)),
        )

    def effective_area(self, spacing: float = 30.0) -> float:
        r"""Conservative raster estimate of ``area(F_c \ exclusions)``.

        The safety outer set is intentionally unchanged; this estimate is for
        planning only and is deterministic at a fixed spacing.
        """
        cell = float(spacing) ** 2
        return cell * len(self._effective_grid(spacing))

    def effective_diameter(self, spacing: float = 30.0) -> float:
        """Sampled diameter of the effective set, for planning/NBV only."""
        pts = self._effective_grid(spacing)
        if len(pts) < 2:
            return 0.0
        return max(dist(a, b) for i, a in enumerate(pts) for b in pts[i + 1:])

    def effective_components(self, spacing: float = 30.0) -> int:
        """Count connected components of the conservative effective raster.

        The result is planner geometry only.  Four-neighbour connectivity avoids
        joining diagonal branches and is intentionally conservative about holes;
        hard absence and clear certificates continue to use their dedicated
        verifiers.
        """
        return self.effective_region(spacing).components

    def sample_effective_hypotheses(self, limit: int = 64, spacing: float = 30.0) -> List[Point]:
        """Return deterministic, evenly spread effective hypotheses.

        This is deliberately not a certificate and never replaces ``F_c`` for
        clear safety.  Every returned point satisfies ``contains_possible_source``.
        """
        pts = self._effective_grid(spacing)
        # A narrow bearing wedge may contain no coarse lattice centre.  Add
        # boundary/centroid probes from the outer polygon before giving up.
        if self.F_c:
            probes = list(self.F_c)
            probes.append((
                sum(p[0] for p in self.F_c) / len(self.F_c),
                sum(p[1] for p in self.F_c) / len(self.F_c),
            ))
            pts.extend(p for p in probes if self.contains_possible_source(p))
            pts = list(dict.fromkeys(pts))
        if len(pts) <= limit:
            return pts
        stride = max(1, len(pts) // limit)
        return pts[::stride][:limit]

    def record_bearing(self, point: Point, svd_deg: float, time: float = 0.0) -> None:
        """P3 positive detection with bearing ``svd_deg`` @ point."""
        self.bearings.append(BearingObs(point, norm_deg(svd_deg), time))
        self.scan_count += 1
        self.last_scan_time = max(self.last_scan_time, time)
        if self.first_detect_time is None:
            self.first_detect_time = float(time)
        self._rebuild()

    def record_near(self, point: Point, time: float = 0.0) -> None:
        """'near' @ point (<=5 m, in-arc): strongest positive info (DESIGN.md §5,
        triggers opportunistic clear in the safety layer)."""
        self.near_points.append(point)
        self.scan_count += 1
        self.last_scan_time = max(self.last_scan_time, time)
        if self.first_detect_time is None:
            self.first_detect_time = float(time)
        self._rebuild()

    # -- feasible-set reconstruction ---------------------------------------

    def _rebuild(self) -> None:
        halfplanes = []
        for b in self.bearings:
            halfplanes.extend(
                wedge_halfplanes(b.point, b.svd_deg, self.wedge_half_deg + self.eps_num_deg)
            )
            halfplanes.extend(disc_superset_halfplanes(b.point, self.range_radius))
        for np in self.near_points:
            halfplanes.extend(disc_superset_halfplanes(np, self.near_radius))

        poly = halfplane_intersection(halfplanes, arena_polygon(self.arena_radius))
        self.F_c = poly
        if poly:
            self.mec_center, self.mec_radius = min_enclosing_circle(poly)
            self.diameter = polygon_diameter(poly)
            self.area = polygon_area(poly)
        else:
            # Empty intersection would mean contradictory positives — impossible
            # with the sound +eps margins on real data. Keep it observable.
            self.mec_center, self.mec_radius, self.diameter, self.area = None, inf, inf, inf
        self._update_inertia_summary()
        self._update_status()

    def _update_inertia_summary(self) -> None:
        """Update planning-only anisotropy from deterministic effective samples."""
        # Positive updates are frequent (including Monte-Carlo soundness tests),
        # so use the already-built outer polygon vertices when available.  Fall
        # back to a coarse effective raster only before the first positive.
        pts = list(self.F_c) if self.F_c else self._effective_grid(spacing=120.0)
        # Coarse spacing keeps this planner-only summary cheap during long
        # negative-scan traces; callers can request finer verification explicitly.
        self.connected_components = self.effective_components(spacing=120.0)
        if len(pts) < 3:
            self.kappa, self.principal_axis = inf, None
            return
        import numpy as np
        arr = np.asarray(pts, dtype=float)
        cov = np.cov(arr, rowvar=False, bias=True)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vec = vecs[:, order[0]]
        self.kappa = float(vals[0] / max(vals[1], 1e-9))
        self.principal_axis = (float(vec[0]), float(vec[1]))

    def _update_status(self) -> None:
        previous = self.status
        if self.status in (ChannelStatus.CLEARED, ChannelStatus.ABSENT_CERTIFIED):
            return
        if not self.bearings and not self.near_points:
            if self.status == ChannelStatus.PRESENT_UNOBSERVED:
                return
            self.status = ChannelStatus.UNKNOWN
        elif self.F_c and self.mec_radius <= self.clear_threshold:
            self.status = ChannelStatus.LOCALIZED
        elif self.initialization_ready:
            self.status = ChannelStatus.INITIALIZED
        else:
            self.status = ChannelStatus.DETECTED
        if self.status == ChannelStatus.LOCALIZED and previous != ChannelStatus.LOCALIZED:
            self.localized_time = float(self.last_scan_time)

    @property
    def initialization_ready(self) -> bool:
        """Whether positive observations provide a non-degenerate baseline.

        Two bearings from nearly the same direction are not a reliable coarse
        initialization.  Readiness is deliberately geometric and deterministic;
        the learner cannot force this lifecycle transition.
        """
        if len(self.bearings) < 2:
            return False
        for i, left in enumerate(self.bearings):
            for right in self.bearings[i + 1:]:
                if angle_sep_deg(left.svd_deg, right.svd_deg) >= 10.0:
                    return True
        return False

    def readiness_score(self, robot_pos: Optional[Point] = None,
                        coverage_debt: float = 0.0) -> float:
        """Continuous initialization quality in ``[0, 1]``.

        This is a planning feature, not a presence or clear certificate.  It
        rewards small/compact geometry and a diverse bearing baseline while
        penalising unresolved components, coverage debt and travel distance.
        """
        import math
        if not self.F_c or not self.bearings:
            return 0.0
        area_term = 1.0 / (1.0 + max(0.0, self.area) / 1.0e6)
        diameter_term = 1.0 / (1.0 + max(0.0, self.diameter) / 1800.0)
        shape_term = 1.0 / (1.0 + max(0.0, self.kappa - 1.0) / 25.0)
        branch_term = 1.0 / max(1.0, float(self.connected_components))
        diversity = 0.0
        if len(self.bearings) >= 2:
            diversity = max(
                angle_sep_deg(a.svd_deg, b.svd_deg)
                for i, a in enumerate(self.bearings)
                for b in self.bearings[i + 1:]
            ) / 90.0
            diversity = min(1.0, diversity)
        distance_term = 1.0
        if robot_pos is not None and self.mec_center is not None:
            distance_term = 1.0 / (1.0 + dist(robot_pos, self.mec_center) / 1800.0)
        debt_term = 1.0 / (1.0 + max(0.0, float(coverage_debt)))
        score = (area_term * diameter_term * shape_term * branch_term
                 * max(diversity, 0.05) * distance_term * debt_term)
        return max(0.0, min(1.0, float(score)))

    def readiness_snapshot(self, robot_pos: Optional[Point] = None,
                           coverage_debt: float = 0.0) -> dict:
        """Structured readiness summary for planners and audit artifacts.

        Values are descriptive features only; this snapshot cannot certify
        presence/absence or localization.
        """
        return {
            "status": self.status.value,
            "area": float(self.area),
            "diameter": float(self.diameter),
            "mec_radius": float(self.mec_radius),
            "kappa": float(self.kappa),
            "connected_components": int(self.connected_components),
            "coverage_debt": max(0.0, float(coverage_debt)),
            "initialization_ready": bool(self.initialization_ready),
            "robot_distance": (float(dist(robot_pos, self.mec_center))
                               if robot_pos is not None and self.mec_center is not None
                               else None),
            "readiness_score": self.readiness_score(robot_pos, coverage_debt),
        }

    # -- queries ------------------------------------------------------------

    @property
    def is_present(self) -> bool:
        return self.status in PRESENT_STATES

    @property
    def is_clearable(self) -> bool:
        return (
            self.status == ChannelStatus.LOCALIZED
            and self.F_c is not None
            and self.mec_radius <= self.clear_threshold
        )

    @property
    def is_resolved(self) -> bool:
        return self.status in RESOLVED_STATES

    @property
    def clear_target(self) -> Optional[Point]:
        """Blind-clear point: the MEC centre (DESIGN.md §3.3). Guaranteed within
        the clear radius of the true source whenever ``is_clearable``."""
        return self.mec_center

    def excludes(self, p: Point, eps: float = 0.0) -> bool:
        """True if ``p`` is inside any NO_SIGNAL disc (source provably not there)."""
        return any(dist(p, d.center) <= d.radius - eps for d in self.negative_discs)

    # -- state transitions (called by executor / certificate manager) ------

    def mark_cleared(self, time: float = 0.0) -> None:
        self.status = ChannelStatus.CLEARED
        self.cleared_time = float(time)

    def mark_present_unobserved(self) -> bool:
        """Apply the cardinality-forced presence transition.

        This transition deliberately adds no bearing/range observation: the
        lower-bound cardinality proof establishes existence, not location.
        """
        if self.status == ChannelStatus.UNKNOWN:
            self.status = ChannelStatus.PRESENT_UNOBSERVED
            return True
        return False

    def mark_absent_certified(self) -> bool:
        """Set ABSENT_CERTIFIED — ONLY valid for an UNKNOWN channel. A PRESENT
        channel can never be absent. Returns whether the mark was applied.
        (Callers must be the certificate manager / cardinality guard, §6.5.)"""
        if self.status == ChannelStatus.UNKNOWN:
            self.status = ChannelStatus.ABSENT_CERTIFIED
            return True
        return False

class BeliefState:
    """The 20-channel belief container (DESIGN.md §4)."""

    def __init__(self, n_channels: int = 20, **channel_kwargs) -> None:
        self.n_channels = n_channels
        self.channels: Dict[int, ChannelBelief] = {
            c: ChannelBelief(channel=c, **channel_kwargs)
            for c in range(1, n_channels + 1)
        }

    def __getitem__(self, channel: int) -> ChannelBelief:
        return self.channels[channel]

    def present_channels(self) -> List[int]:
        return [c for c, b in self.channels.items() if b.is_present]

    def present_count(self) -> int:
        return len(self.present_channels())

    def clearable_channels(self) -> List[int]:
        return [c for c, b in self.channels.items() if b.is_clearable]

    def unknown_channels(self) -> List[int]:
        return [c for c, b in self.channels.items() if b.status == ChannelStatus.UNKNOWN]

    def unobserved_present_channels(self) -> List[int]:
        return [c for c, b in self.channels.items()
                if b.status == ChannelStatus.PRESENT_UNOBSERVED]

    def all_resolved(self) -> bool:
        return all(b.is_resolved for b in self.channels.values())


def _point_in_convex_polygon(p: Point, poly: Sequence[Point], eps: float = 1e-9) -> bool:
    """Boundary-inclusive point test for the CCW/CW convex polygons we build."""
    if not poly:
        return False

    signs = []
    for a, b in zip(poly, list(poly[1:]) + [poly[0]]):
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(cross) > eps:
            signs.append(cross > 0)
    return not signs or all(s == signs[0] for s in signs)
