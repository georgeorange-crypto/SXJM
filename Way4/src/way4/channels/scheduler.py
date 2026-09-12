"""ChannelScheduler — which channels to batch-scan at a waypoint (DESIGN.md §7).

At a chosen waypoint ``q`` the robot can measure several channels before moving on;
only the *first* measurement pays the move, so the move cost is already sunk and each
extra in-range channel is scanned for just its marginal switch+detect time (batch
scan, §7). The batch is therefore the highest-value eligible channels up to the mode
cap — EARLY allows the full 20-channel band, MID focuses on the top few, VERIFICATION
scans only still-UNKNOWN & un-certified channels (§6.7 completion waypoints, never the
whole band). CLEARED / ABSENT_CERTIFIED channels are never re-scanned; a LOCALIZED
channel is cleared, not scanned; ``must_include`` pins the channel a REFINE/PURSUE
waypoint exists for.

Per-channel value ``V_c(q) = V_discover + V_refine + V_certificate`` is normalised to
``[0,1]`` (coverage fraction for UNKNOWN, fractional-diameter-shrink proxy for
DETECTED); zero-value channels (already covered at ``q``, or out of range) are dropped.

The value-per-dwell ratio ``Σ V_c / (5·|G| + N_switch)`` (with ``N_switch = |G| − 1``
if the current measuring channel is in ``G``, scanned first, else ``|G|``) is reported
on the plan for diagnostics but does **not** gate batch size. Because every UNKNOWN
channel shares the same purely-spatial coverage gain at a fixed ``q``, a ratio-
maximiser would always collapse the batch to a single channel and defeat batch scan
(禁止10 — never trade the full clear for speed): the move is sunk, so amortising it over
the whole in-range band strictly dominates one-channel-per-waypoint. This is a planner
heuristic — it certifies nothing (Invariant B) and marks nothing (禁止6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import radians, sin
from typing import Dict, List, Optional, Sequence, Tuple

from sxjm_core.geometry import angle_sep_deg, bearing_deg, dist, polygon_centroid

from ..belief import ChannelStatus
from ..core.actions import MacroActionType, get_allowed_actions

Point = Tuple[float, float]


class SchedulerMode(str, Enum):
    EARLY = "EARLY"                # broad discovery: allow the full band
    MID = "MID"                    # focus: cap the batch
    VERIFICATION = "VERIFICATION"  # completion: only UNKNOWN & un-certified (§6.7)


@dataclass
class ScanPlan:
    """The ordered channel batch for a waypoint and its dwell economics."""

    channels: List[int]
    value: float = 0.0
    dwell_time_s: float = 0.0
    n_switch: int = 0
    ratio: float = 0.0
    stop_reason: str = ""
    stop_value: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.channels


class AdaptiveScanSession:
    """Select one scan at a time and rerank after each observation."""

    def __init__(self, scheduler, q, belief, certificate, current_channel,
                 mode=SchedulerMode.EARLY, min_value=1e-9):
        self.scheduler = scheduler
        self.q = q
        self.belief = belief
        self.certificate = certificate
        self.current_channel = current_channel
        self.mode = mode
        self.min_value = float(min_value)
        self.scanned = set()

    def choose(self) -> ScanPlan:
        ranked = []
        for c in range(1, self.belief.n_channels + 1):
            if c in self.scanned:
                continue
            value, allowed = self.scheduler.channel_value(
                c, self.q, self.belief, self.certificate
            )
            if allowed and value > self.min_value:
                ranked.append((float(value), c))
        if not ranked:
            return ScanPlan([], stop_reason="no_positive_voi", stop_value=0.0)
        value, channel = max(ranked, key=lambda x: (x[0], x[1] == self.current_channel))
        switched = int(channel != self.current_channel)
        return ScanPlan(
            [channel], value=value,
            dwell_time_s=self.scheduler.measure_s + switched * self.scheduler.switch_s,
            n_switch=switched,
            stop_reason="continue_scan",
            stop_value=float(value),
        )

    def observe(self, channel: int) -> None:
        self.scanned.add(int(channel))
        self.current_channel = int(channel)


class ChannelScheduler:
    def __init__(
        self,
        range_radius: float = 1500.0,
        measure_s: float = 5.0,
        switch_s: float = 1.0,
        w_certificate: float = 1.0,   # weight on UNKNOWN coverage/discover value
        w_refine: float = 1.0,        # weight on DETECTED bearing-shrink value
        w_scan_debt: float = 0.25,
        early_cap: int = 20,
        mid_cap: int = 6,
        starvation_limit: int = 8,
        eps: float = 1e-9,
    ) -> None:
        self.range_radius = float(range_radius)
        self.measure_s = float(measure_s)
        self.switch_s = float(switch_s)
        self.w_certificate = float(w_certificate)
        self.w_refine = float(w_refine)
        self.w_scan_debt = float(w_scan_debt)
        self.early_cap = int(early_cap)
        self.mid_cap = int(mid_cap)
        self.starvation_limit = int(starvation_limit)
        self._skip_count: Dict[int, int] = {}
        self.eps = float(eps)

    # -- per-channel value (each term normalised to [0,1]) -----------------

    def value_components(self, channel: int, q: Point, belief, certificate) -> Dict[str, float]:
        """Unified per-channel value decomposition V_E,V_I,V_R,V_C,V_D."""
        b = belief[channel]
        discovery = initialization = refinement = certificate_value = 0.0
        debt = self.scan_debt(b, certificate=certificate)
        if b.status == ChannelStatus.UNKNOWN:
            discovery = certificate.coverage_gain(channel, q) if certificate is not None else 0.0
            certificate_value = discovery
        elif b.status == ChannelStatus.PRESENT_UNOBSERVED:
            initialization = certificate.coverage_gain(channel, q) if certificate is not None else 0.0
            certificate_value = initialization
        elif b.status in (ChannelStatus.DETECTED, ChannelStatus.INITIALIZED):
            refinement = self._refine_proxy(b, q)
        return {"V_E": float(discovery), "V_I": float(initialization),
                "V_R": float(refinement), "V_C": float(certificate_value),
                "V_D": float(debt)}

    def scan_debt(self, channel_belief, *, certificate=None, now_s=None) -> float:
        """Continuous starvation debt: residual coverage, elapsed time and
        incomplete scan history. Resolved channels carry zero debt."""
        if channel_belief.status in (ChannelStatus.CLEARED,
                                     ChannelStatus.ABSENT_CERTIFIED,
                                     ChannelStatus.LOCALIZED):
            return 0.0
        coverage = certificate.coverage_debt(channel_belief.channel) if certificate is not None else 1.0
        elapsed = 0.0
        if now_s is not None:
            elapsed = max(0.0, float(now_s) - float(channel_belief.last_scan_time)) / 300.0
        completeness = 1.0 / (1.0 + float(channel_belief.scan_count))
        return 0.5 * max(0.0, float(coverage)) + 0.3 * min(1.0, elapsed) + 0.2 * completeness

    def eta(self, channel: int, q: Point, belief, certificate, current_channel: int) -> float:
        """Value per marginal sensing time (measure + optional switch)."""
        comp = self.value_components(channel, q, belief, certificate)
        value = comp["V_E"] + comp["V_I"] + comp["V_R"] + comp["V_C"]
        value += self.w_scan_debt * comp["V_D"] * max(comp["V_E"], comp["V_I"], 0.0)
        cost = self.measure_s + (0.0 if int(channel) == int(current_channel) else self.switch_s)
        return value / cost if cost > 0 else 0.0

    def _refine_proxy(self, channel_belief, q: Point) -> float:
        """Cheap DETECTED-channel shrink value at ``q``: the best crossing quality
        ``sin(γ)`` between an existing line-of-sight and the new one toward the
        region centroid (0 if ``q`` is out of detection range). A perpendicular
        crossing (γ≈90°) ≈ full fractional shrink; near-collinear ≈ 0."""
        poly = channel_belief.F_c
        if not poly or channel_belief.mec_center is None or not channel_belief.bearings:
            return 0.0
        # Planner geometry follows the effective set (NO_SIGNAL exclusions),
        # while the safety outer polygon remains authoritative for certificates.
        sampler = getattr(channel_belief, "sample_effective_hypotheses", None)
        samples = sampler(limit=32, spacing=60.0) if sampler else []
        m = (
            (sum(p[0] for p in samples) / len(samples),
             sum(p[1] for p in samples) / len(samples))
            if samples else polygon_centroid(poly)
        )
        if dist(q, m) > self.range_radius:
            return 0.0
        new_los = bearing_deg(q, m)
        best = 0.0
        for b in channel_belief.bearings:
            sep = angle_sep_deg(b.svd_deg, new_los)
            best = max(best, abs(sin(radians(sep))))
        return best

    def channel_value(self, channel: int, q: Point, belief, certificate) -> Tuple[float, bool]:
        """Return ``(value, scannable)``. ``scannable`` is False for channels that
        must never be scanned again (CLEARED / ABSENT_CERTIFIED / LOCALIZED→clear)."""
        b = belief[channel]
        st = b.status
        allowed = get_allowed_actions(st)
        if st in (ChannelStatus.CLEARED, ChannelStatus.ABSENT_CERTIFIED, ChannelStatus.LOCALIZED) \
                or MacroActionType.EXPLORE not in allowed and MacroActionType.REFINE not in allowed:
            return 0.0, False
        if st == ChannelStatus.UNKNOWN:
            comp = self.value_components(channel, q, belief, certificate)
            return comp["V_E"] * (self.w_certificate + self.w_scan_debt * comp["V_D"]), True
        if st == ChannelStatus.PRESENT_UNOBSERVED:
            comp = self.value_components(channel, q, belief, certificate)
            return max(comp["V_I"], self.eps) * (self.w_certificate + self.w_scan_debt * comp["V_D"]), True
        if st in (ChannelStatus.DETECTED, ChannelStatus.INITIALIZED):
            return self.w_refine * self.value_components(channel, q, belief, certificate)["V_R"], True
        return 0.0, False

    # -- dwell economics ----------------------------------------------------

    def n_switch(self, channels: Sequence[int], current_channel: int) -> int:
        k = len(channels)
        if k == 0:
            return 0
        return k - 1 if current_channel in channels else k

    def _dwell(self, channels: Sequence[int], current_channel: int) -> Tuple[float, int]:
        ns = self.n_switch(channels, current_channel)
        return self.measure_s * len(channels) + self.switch_s * ns, ns

    def _ratio(self, channels: Sequence[int], values: Dict[int, float], current_channel: int) -> float:
        if not channels:
            return 0.0
        num = sum(values[c] for c in channels)
        dwell, _ = self._dwell(channels, current_channel)
        return num / dwell if dwell > 0 else 0.0

    def starvation_debt(self, channel: int) -> int:
        return int(self._skip_count.get(int(channel), 0))

    def debt_snapshot(self, channel: int, belief, certificate, now_s=None) -> Dict[str, float]:
        """Unified trace view of continuous and discrete starvation debt."""
        b = belief[channel]
        return {
            "continuous_scan_debt": float(self.scan_debt(b, certificate=certificate, now_s=now_s)),
            "skipped_plan_count": float(self.starvation_debt(channel)),
            "scan_count": float(getattr(b, "scan_count", 0)),
            "last_scan_time": float(getattr(b, "last_scan_time", 0.0)),
        }

    def record_plan(self, channels: Sequence[int], n_channels: Optional[int] = None) -> None:
        """Advance skip debt after a plan is selected.

        Resolved channels are naturally reset by callers supplying the current
        eligible count; this method only tracks channels that were not selected.
        """
        chosen = {int(c) for c in channels}
        limit = n_channels if n_channels is not None else max(self._skip_count.keys(), default=0)
        for c in range(1, int(limit) + 1):
            if c in chosen:
                self._skip_count[c] = 0
            else:
                self._skip_count[c] = self._skip_count.get(c, 0) + 1

    # -- selection ----------------------------------------------------------

    def select(
        self,
        q: Point,
        belief,
        certificate,
        current_channel: int,
        mode: SchedulerMode = SchedulerMode.EARLY,
        must_include: Sequence[int] = (),
    ) -> ScanPlan:
        """Batch the highest-value eligible channels at ``q`` up to the mode cap (§7).

        The move to ``q`` is already sunk, so an extra in-range channel is measured for
        only its marginal switch+detect time — batching the whole eligible band strictly
        beats one-channel-per-waypoint. We therefore take the top-value scannable
        channels up to the mode cap (EARLY 20, MID ``mid_cap``, VERIFICATION 20), always
        keeping the ``must_include`` channels. Zero-value channels (already covered here,
        or out of range) are dropped; resolved channels are never scanned. The value-
        per-second ratio is computed for the plan but does not limit its size (it would
        collapse a homogeneous UNKNOWN band to k=1 — see module docstring)."""
        n = belief.n_channels
        # eligible pool
        values: Dict[int, float] = {}
        scannable: List[int] = []
        for c in range(1, n + 1):
            v, ok = self.channel_value(c, q, belief, certificate)
            if not ok:
                continue
            if mode == SchedulerMode.VERIFICATION:
                if belief[c].status != ChannelStatus.UNKNOWN:
                    continue
                if certificate is not None and certificate.is_absent_certified(c, force=False):
                    continue
            values[c] = v
            scannable.append(c)

        pinned = [c for c in must_include if c in values]
        cap = {
            SchedulerMode.EARLY: self.early_cap,
            SchedulerMode.MID: self.mid_cap,
            SchedulerMode.VERIFICATION: self.early_cap,
        }[mode]

        optional = sorted(
            (c for c in scannable if c not in pinned and values[c] > self.eps),
            key=lambda c: -values[c],
        )

        # Starvation guard: a repeatedly skipped eligible channel is promoted
        # even when its local gain is small/zero. This affects planning only;
        # resolved channels were removed from ``values`` above.
        starved = [c for c in scannable if self.starvation_debt(c) >= self.starvation_limit]
        pinned = list(dict.fromkeys(pinned + starved))

        # Sunk-move economics (§7): fill the batch with the highest-value channels up to
        # the mode cap; pinned channels always ride along. NOT ratio-gated — a value-per-
        # dwell maximiser collapses a homogeneous UNKNOWN band to a single channel and
        # defeats batch scan (禁止10). Over-scanning only gathers more info; it never
        # marks absent (禁止6) or trades away the full clear.
        room = max(0, cap - len(pinned))
        best = pinned + optional[:room]

        # order: current channel first (saves the initial switch), then value desc
        ordered = sorted(best, key=lambda c: (c != current_channel, -values.get(c, 0.0)))
        dwell, ns = self._dwell(ordered, current_channel)
        return ScanPlan(
            channels=ordered,
            value=sum(values[c] for c in ordered),
            dwell_time_s=dwell,
            n_switch=ns,
            ratio=self._ratio(ordered, values, current_channel),
        )
