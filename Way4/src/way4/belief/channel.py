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
from math import inf
from typing import Dict, List, Optional

from sxjm_core.geometry import (
    Point,
    arena_polygon,
    disc_superset_halfplanes,
    dist,
    halfplane_intersection,
    min_enclosing_circle,
    norm_deg,
    polygon_area,
    polygon_diameter,
    wedge_halfplanes,
)


class ChannelStatus(str, Enum):
    """Way2's four states plus ABSENT_CERTIFIED (DESIGN.md §4; no probabilistic
    absence). PRESENT is not a state — it is the union DETECTED/LOCALIZED/CLEARED
    (§6.5 cardinality)."""

    UNKNOWN = "UNKNOWN"
    DETECTED = "DETECTED"        # a source is confirmed present, MEC still large
    LOCALIZED = "LOCALIZED"      # MEC.r <= clear_threshold -> clearable
    CLEARED = "CLEARED"
    ABSENT_CERTIFIED = "ABSENT_CERTIFIED"


#: Channels that a real positive observation has confirmed to hold a source
#: (Invariant D: only these count toward the cardinality of 16).
PRESENT_STATES = frozenset(
    {ChannelStatus.DETECTED, ChannelStatus.LOCALIZED, ChannelStatus.CLEARED}
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
    scan_count: int = 0
    last_scan_time: float = 0.0

    # -- observation intake (DESIGN.md §5) ---------------------------------

    def record_no_signal(self, point: Point, time: float = 0.0) -> None:
        """P3 NO_SIGNAL @ point. Appends a negative exclusion disc; never makes
        the channel PRESENT and never certifies absence (Invariants A/B)."""
        self.negative_discs.append(ExclusionDisc(point, self.detect_lower_bound))
        self.scan_count += 1
        self.last_scan_time = max(self.last_scan_time, time)

    def record_bearing(self, point: Point, svd_deg: float, time: float = 0.0) -> None:
        """P3 positive detection with bearing ``svd_deg`` @ point."""
        self.bearings.append(BearingObs(point, norm_deg(svd_deg), time))
        self.scan_count += 1
        self.last_scan_time = max(self.last_scan_time, time)
        self._rebuild()

    def record_near(self, point: Point, time: float = 0.0) -> None:
        """'near' @ point (<=5 m, in-arc): strongest positive info (DESIGN.md §5,
        triggers opportunistic clear in the safety layer)."""
        self.near_points.append(point)
        self.scan_count += 1
        self.last_scan_time = max(self.last_scan_time, time)
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
        self._update_status()

    def _update_status(self) -> None:
        if self.status in (ChannelStatus.CLEARED, ChannelStatus.ABSENT_CERTIFIED):
            return
        if not self.bearings and not self.near_points:
            self.status = ChannelStatus.UNKNOWN
        elif self.F_c and self.mec_radius <= self.clear_threshold:
            self.status = ChannelStatus.LOCALIZED
        else:
            self.status = ChannelStatus.DETECTED

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

    def mark_cleared(self) -> None:
        self.status = ChannelStatus.CLEARED

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

    def all_resolved(self) -> bool:
        return all(b.is_resolved for b in self.channels.values())
