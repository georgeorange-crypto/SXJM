"""Shared observation vocabulary (DESIGN.md §5, §6.4, §13 core/).

A sensor scan at a point yields, per channel, one *outcome*: NO_SIGNAL, a BEARING
(positive, with ``svd_deg`` within 1° of the true bearing), or NEAR (positive,
source within 5 m and in-arc, no ``svd_deg``). ``Observation`` is that outcome —
the *what*; the *where* (channel, point) is passed alongside it (matching the
frozen ``record_observation(c, point, obs)`` interface, §6.4).

This is the single vocabulary the BeliefUpdater and CertificateManager both
consume, so the two layers can never disagree about what a scan meant.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ObservationKind(str, Enum):
    NO_SIGNAL = "NO_SIGNAL"   # no detection (channel clear / out of R_eff / off-arc)
    BEARING = "BEARING"       # positive detection with a bearing measurement
    NEAR = "NEAR"             # positive, <=5 m in-arc, signal too strong for a bearing


@dataclass(frozen=True)
class Observation:
    """One channel's outcome from a scan (the *what*, not the *where*)."""

    kind: ObservationKind
    svd_deg: Optional[float] = None   # only for BEARING
    time: float = 0.0

    # -- convenience constructors ------------------------------------------

    @classmethod
    def no_signal(cls, time: float = 0.0) -> "Observation":
        return cls(ObservationKind.NO_SIGNAL, None, time)

    @classmethod
    def bearing(cls, svd_deg: float, time: float = 0.0) -> "Observation":
        return cls(ObservationKind.BEARING, float(svd_deg), time)

    @classmethod
    def near(cls, time: float = 0.0) -> "Observation":
        return cls(ObservationKind.NEAR, None, time)

    # -- queries ------------------------------------------------------------

    @property
    def is_no_signal(self) -> bool:
        return self.kind is ObservationKind.NO_SIGNAL

    @property
    def is_bearing(self) -> bool:
        return self.kind is ObservationKind.BEARING

    @property
    def is_near(self) -> bool:
        """A 'too strong' return: source within 5 m and in-arc, no bearing. The
        strongest positive info — triggers opportunistic clear (§7, §11)."""
        return self.kind is ObservationKind.NEAR

    @property
    def is_positive(self) -> bool:
        """A positive detection (BEARING or NEAR) — confirms a source is present
        (the only thing that may count toward cardinality, Invariant D)."""
        return self.kind in (ObservationKind.BEARING, ObservationKind.NEAR)
