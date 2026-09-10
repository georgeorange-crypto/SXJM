"""The Mathematical State Estimator — per-channel feasible regions.

This is the frozen core of the pipeline. It ingests :class:`DetectionObservation`
records and maintains, for every channel, a convex feasible region for that
channel's source. The neural network never touches these objects; it only reads
features derived from them (the iron rule).

Soundness is the contract: the true source location is *never* excluded from a
channel's region. To honour that under floating-point noise and the field's
rounding (the bearing error is bounded by 1 deg but rounding can nudge it a hair
past), wedges are widened by a small ``bearing_margin_deg`` and range discs are
approximated from *outside* (circumscribed polygons). Negative information
(no-signal, clear-miss) is only sound for omni sources and for exclusion, so it
is kept as a separate list of forbidden discs used by search — never folded into
the convex regions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..core.constants import CONSTANTS
from ..core.datatypes import (
    ChannelStatus,
    DetectionObservation,
    ObservationType,
)
from .region import Region


@dataclass
class EstimatorConfig:
    """Tunables for the estimator (all metres / degrees)."""

    arena_radius: float = CONSTANTS.region_radius
    range_max: float = CONSTANTS.receive_radius_max        # 1500: R_eff upper bound
    range_min: float = CONSTANTS.receive_radius_min        # 1000: R_eff lower bound
    near_radius: float = CONSTANTS.too_strong_radius       # 5
    clear_radius: float = CONSTANTS.clear_radius           # 20
    localize_mec_radius: float = 18.0                      # localized when MEC r <= this
    bearing_delta_deg: float = CONSTANTS.bearing_error_deg  # 1.0
    bearing_margin_deg: float = 0.05                       # widen wedges for soundness
    disc_facets: int = 128
    no_signal_exclusion: bool = True                       # sound only for omni (P3)

    @property
    def wedge_half_deg(self) -> float:
        return self.bearing_delta_deg + self.bearing_margin_deg


@dataclass
class ExclusionDisc:
    """A disc known to NOT contain the source (no-signal / clear-miss)."""

    x: float
    y: float
    radius: float
    channel: int
    reason: str  # "no_signal" | "clear_miss"

    def contains(self, x: float, y: float, eps: float = 1e-6) -> bool:
        return math.hypot(x - self.x, y - self.y) <= self.radius + eps


class ChannelBelief:
    """Feasible region and status for a single channel's source."""

    def __init__(self, channel: int, cfg: EstimatorConfig) -> None:
        self.channel = int(channel)
        self.cfg = cfg
        self.status = ChannelStatus.UNKNOWN
        self.region = Region.arena(cfg.arena_radius, cfg.disc_facets)
        self.n_bearings = 0
        self.bearings: list[tuple[float, float, float]] = []  # (x, y, svd_deg)
        self.near_point: Optional[tuple[float, float]] = None  # forced estimate
        self.est_x: Optional[float] = None
        self.est_y: Optional[float] = None
        self.mec_radius: float = float("inf")
        self.region_diameter: float = float("inf")
        self.degenerate = False  # a clip emptied the region (numeric guard tripped)

    # -- positive detections ----------------------------------------------
    def add_bearing(self, x: float, y: float, svd_deg: float) -> None:
        prev = self.region.copy()
        self.region.clip_wedge(x, y, svd_deg, self.cfg.wedge_half_deg)
        # A single received signal also bounds range to R_eff <= range_max.
        self.region.clip_inside_disc(x, y, self.cfg.range_max, self.cfg.disc_facets)
        if self.region.is_empty():
            # Inconsistent under numerics: keep the tighter prior region.
            self.region = prev
            self.degenerate = True
        self.bearings.append((x, y, svd_deg))
        self.n_bearings += 1
        if self.status == ChannelStatus.UNKNOWN:
            self.status = ChannelStatus.DETECTED
        self._recompute()

    def add_near(self, x: float, y: float) -> None:
        """A 'too strong' reading: source within near_radius of (x, y)."""
        prev = self.region.copy()
        self.region.clip_inside_disc(x, y, self.cfg.near_radius, self.cfg.disc_facets)
        if self.region.is_empty():
            self.region = prev
            self.degenerate = True
        self.near_point = (x, y)
        if self.status in (ChannelStatus.UNKNOWN, ChannelStatus.DETECTED):
            self.status = ChannelStatus.LOCALIZED
        self._recompute()

    def mark_cleared(self) -> None:
        self.status = ChannelStatus.CLEARED

    # -- derived estimate --------------------------------------------------
    def _recompute(self) -> None:
        if self.near_point is not None:
            # A near reading pins the source to within near_radius; clear there.
            self.est_x, self.est_y = self.near_point
            self.mec_radius = min(self.mec_radius, self.cfg.near_radius)
            self.region_diameter = self.region.diameter()
            if self.status != ChannelStatus.CLEARED:
                self.status = ChannelStatus.LOCALIZED
            return
        if self.region.is_empty():
            return
        (cx, cy), r = self.region.min_enclosing_circle()
        self.est_x, self.est_y, self.mec_radius = cx, cy, r
        self.region_diameter = self.region.diameter()
        if self.status != ChannelStatus.CLEARED and r <= self.cfg.localize_mec_radius:
            self.status = ChannelStatus.LOCALIZED

    # -- queries -----------------------------------------------------------
    @property
    def estimate(self) -> Optional[tuple[float, float]]:
        if self.est_x is None:
            return None
        return (self.est_x, self.est_y)

    @property
    def is_localized(self) -> bool:
        return self.status == ChannelStatus.LOCALIZED

    @property
    def is_cleared(self) -> bool:
        return self.status == ChannelStatus.CLEARED

    @property
    def is_detected(self) -> bool:
        return self.status in (ChannelStatus.DETECTED, ChannelStatus.LOCALIZED)


class BeliefState:
    """The whole mathematical state: one :class:`ChannelBelief` per channel plus
    the robot pose, clock, and negative-information exclusions used for search."""

    def __init__(self, cfg: Optional[EstimatorConfig] = None, problem: int = 3) -> None:
        self.cfg = cfg or EstimatorConfig()
        self.problem = int(problem)
        self.channels: dict[int, ChannelBelief] = {
            ch: ChannelBelief(ch, self.cfg)
            for ch in range(CONSTANTS.channel_min, CONSTANTS.channel_max + 1)
        }
        self.pose_x: float = CONSTANTS.initial_x
        self.pose_y: float = CONSTANTS.initial_y
        self.current_channel: int = CONSTANTS.initial_channel
        self.virtual_time_s: float = 0.0
        self.exclusions: list[ExclusionDisc] = []
        self.scan_counts: dict[int, int] = {}
        self.n_measurements = 0
        self.n_clear_attempts = 0

    # -- ingestion ---------------------------------------------------------
    def reset(self) -> None:
        self.__init__(self.cfg, self.problem)

    def update(self, obs: DetectionObservation) -> None:
        """Fold one observation into the belief. This is the ONLY mutation path."""
        self.pose_x, self.pose_y = obs.position_x, obs.position_y
        self.virtual_time_s += float(obs.virtual_time_delta)
        ch = int(obs.channel)
        cb = self.channels.get(ch)

        t = obs.result_type
        if t == ObservationType.RESET:
            return
        if t == ObservationType.SIGNAL:
            self.current_channel = ch
            self.n_measurements += 1
            self.scan_counts[ch] = self.scan_counts.get(ch, 0) + 1
            if cb is not None and obs.bearing_deg is not None:
                cb.add_bearing(obs.position_x, obs.position_y, float(obs.bearing_deg))
        elif t == ObservationType.TOO_STRONG:
            self.current_channel = ch
            self.n_measurements += 1
            self.scan_counts[ch] = self.scan_counts.get(ch, 0) + 1
            if cb is not None:
                cb.add_near(obs.position_x, obs.position_y)
        elif t == ObservationType.NO_SIGNAL:
            self.current_channel = ch
            self.n_measurements += 1
            self.scan_counts[ch] = self.scan_counts.get(ch, 0) + 1
            self._add_no_signal(ch, obs.position_x, obs.position_y)
        elif t == ObservationType.CLEAR_SUCCESS:
            self.n_clear_attempts += 1
            if cb is not None:
                cb.mark_cleared()
        elif t == ObservationType.CLEAR_FAILURE:
            self.n_clear_attempts += 1
            self.exclusions.append(
                ExclusionDisc(obs.position_x, obs.position_y,
                              self.cfg.clear_radius, ch, "clear_miss")
            )

    def _add_no_signal(self, channel: int, x: float, y: float) -> None:
        # Sound only for omni sources: if within range_min we would have heard it.
        if not self.cfg.no_signal_exclusion or self.problem >= 4:
            return
        self.exclusions.append(
            ExclusionDisc(x, y, self.cfg.range_min, channel, "no_signal")
        )

    # -- queries used by candidate generation / heuristics -----------------
    def belief(self, channel: int) -> ChannelBelief:
        return self.channels[channel]

    def scans_on(self, channel: int) -> int:
        return self.scan_counts.get(int(channel), 0)

    def clearable(self, channel: int, margin: float = 1.0) -> bool:
        """True if a clear at the channel's estimate is *guaranteed* to hit: the
        whole feasible region lies within (clear_radius - margin) of the centre."""
        cb = self.channels.get(int(channel))
        if cb is None or cb.is_cleared or cb.estimate is None:
            return False
        return cb.mec_radius <= (self.cfg.clear_radius - margin)

    def detected_channels(self) -> list[int]:
        return [c for c, b in self.channels.items() if b.is_detected]

    def localized_channels(self) -> list[int]:
        return [c for c, b in self.channels.items() if b.is_localized]

    def uncleared_localized(self) -> list[int]:
        return [c for c, b in self.channels.items() if b.is_localized and not b.is_cleared]

    def cleared_channels(self) -> list[int]:
        return [c for c, b in self.channels.items() if b.is_cleared]

    def undetected_channels(self) -> list[int]:
        return [c for c, b in self.channels.items()
                if b.status == ChannelStatus.UNKNOWN]

    def n_cleared(self) -> int:
        return len(self.cleared_channels())

    def excluded(self, x: float, y: float, channel: Optional[int] = None) -> bool:
        for e in self.exclusions:
            if channel is not None and e.channel != channel:
                continue
            if e.contains(x, y):
                return True
        return False


def build_estimator_config(cfg) -> EstimatorConfig:
    """Build an :class:`EstimatorConfig` from a (possibly OmegaConf) geometry node."""
    ec = EstimatorConfig()
    if cfg is None:
        return ec
    def g(key, default):
        try:
            v = cfg.get(key) if hasattr(cfg, "get") else cfg[key]
            return default if v is None else v
        except Exception:
            return default
    ec.localize_mec_radius = float(g("localize_mec_radius_m", ec.localize_mec_radius))
    ec.bearing_margin_deg = float(g("bearing_margin_deg", ec.bearing_margin_deg))
    ec.disc_facets = int(g("disc_facets", ec.disc_facets))
    ec.no_signal_exclusion = bool(g("no_signal_exclusion", ec.no_signal_exclusion))
    return ec
