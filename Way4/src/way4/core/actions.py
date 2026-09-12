"""Macro-action vocabulary (DESIGN.md §7).

A ``MacroCandidate`` is Way2's ``CandidateAction`` enriched with a *batch* of
channels to scan at one waypoint and a decomposed gain vector — the object the
planner ranks and (later, M10) the RL residual reorders. It expands into a short
list of ``Primitive`` env commands: the batch-scan idea (§7) is simply "move once
to ``q``, then ``/measure`` several channels there", so every scan primitive in a
macro shares the same target and only the first one pays the move.

Kept deliberately free of any physics: timing lives in ``core.cost`` and execution
in ``executor`` so this stays a pure data description of *what* to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from sxjm_core.geometry import Point


class MacroActionType(str, Enum):
    """The macro intents (DESIGN.md §7). ``STOP`` is additive — the P0 spatial
    planner's location-first bundle (see ``SpatialStop``); it is never produced by the
    frozen action-centric generator, so the legacy path is unchanged."""

    EXPLORE = "EXPLORE"        # broad discovery / coverage batch scan
    INITIALIZE = "INITIALIZE"  # first bearing on a freshly-detected channel
    REFINE = "REFINE"          # NBV second/third bearing to shrink F_c (§8)
    PURSUE = "PURSUE"          # measure from a waypoint closer to a detected source
    CLEAR = "CLEAR"            # blind-clear at the MEC centre (§3.3)
    VERIFY = "VERIFY"          # completion waypoint, batch only uncertified UNKNOWNs (§6.7)
    EXIT = "EXIT"              # leave (guarded by the EXIT certificate, §11)
    STOP = "STOP"              # P0 spatial planner: a location-first multi-service bundle


def get_allowed_actions(channel_state, global_state=None):
    """Single read-only action mask for a channel lifecycle state.

    ``channel_state`` may be a ``ChannelStatus`` or its string value.  The mask
    is advisory for candidate generation; safety/certificate ownership remains
    with the executor and certificate manager.
    """
    value = getattr(channel_state, "value", channel_state)
    if value in ("CLEARED", "ABSENT_CERTIFIED"):
        return frozenset()
    if value == "LOCALIZED":
        return frozenset({MacroActionType.CLEAR, MacroActionType.STOP})
    if value in ("DETECTED", "INITIALIZED"):
        return frozenset({MacroActionType.REFINE, MacroActionType.PURSUE, MacroActionType.CLEAR, MacroActionType.STOP})
    if value == "PRESENT_UNOBSERVED":
        return frozenset({MacroActionType.INITIALIZE, MacroActionType.EXPLORE, MacroActionType.STOP})
    return frozenset({MacroActionType.EXPLORE, MacroActionType.VERIFY, MacroActionType.STOP})


class PrimitiveKind(str, Enum):
    """A low-level environment command."""

    MEASURE = "MEASURE"
    CLEAR = "CLEAR"
    EXIT = "EXIT"


@dataclass(frozen=True)
class Primitive:
    """One env call. ``channel`` is the measured channel (MEASURE) or the source
    channel to clear (CLEAR); ignored for EXIT."""

    kind: PrimitiveKind
    target: Point
    channel: int = 0


@dataclass
class MacroCandidate:
    """A ranked macro action (DESIGN.md §7). ``scan_channels`` is the ordered batch
    for scan-type macros; ``clear_channel`` names the source for a CLEAR."""

    action_type: MacroActionType
    target: Point
    scan_channels: Tuple[int, ...] = ()
    clear_channel: Optional[int] = None

    # gain decomposition (§7); the planner combines these, RL only reranks them.
    exploration_gain: float = 0.0
    refinement_gain: float = 0.0
    certificate_gain: float = 0.0
    route_gain: float = 0.0

    # filled in by the cost model / planner.
    expected_time: float = 0.0
    score: float = 0.0
    meta: dict = field(default_factory=dict)

    def primitives(self) -> List[Primitive]:
        """Expand into env commands. Scan macros become one ``MEASURE`` per channel
        at the shared ``target`` (batch scan, §7): the first pays the move, the rest
        cost only switch + detect."""
        if self.action_type == MacroActionType.EXIT:
            return [Primitive(PrimitiveKind.EXIT, self.target)]
        if self.action_type == MacroActionType.CLEAR:
            if self.clear_channel is None:
                raise ValueError("CLEAR macro requires clear_channel")
            return [Primitive(PrimitiveKind.CLEAR, self.target, int(self.clear_channel))]
        return [Primitive(PrimitiveKind.MEASURE, self.target, int(c)) for c in self.scan_channels]

    @property
    def is_scan(self) -> bool:
        return self.action_type not in (MacroActionType.CLEAR, MacroActionType.EXIT)


@dataclass
class SpatialStop(MacroCandidate):
    """A location-first *service bundle* (P0 #2/#3 — the spatial planner's basic object).

    Where a ``MacroCandidate`` is action-centric (one action at one target), a
    ``SpatialStop`` is space-centric: a neighbourhood at which the robot completes a
    whole *set* of services in one visit — a batch of MEASUREs (``scan_channels``, at
    ``target``) plus any CLEARs whose blind-clear point (§3.3) falls in that
    neighbourhood. It subclasses ``MacroCandidate`` so it flows through the planner,
    executor and pipeline unchanged (they duck-type on ``primitives()`` /
    ``action_type`` / ``expected_time``); ``action_type`` is always ``STOP``, so the
    frozen ``MacroCandidate.primitives()`` (all-MEASURE-or-single-CLEAR) is never used
    and the legacy path stays byte-identical.

    ``clear_channels[i]`` is cleared at ``clear_targets[i]`` (parallel tuples). A stop
    may be pure-scan (no clears), pure-clear (no measures), or a genuine bundle — the
    mixed MEASURE+CLEAR expansion the action-centric candidate could not express. This
    object only *describes* work; when to route to it stays the planner's job and the
    clear guard / EXIT guard are untouched (禁止6/7)."""

    clear_channels: Tuple[int, ...] = ()
    clear_targets: Tuple[Point, ...] = ()

    def primitives(self) -> List[Primitive]:
        """Expand to env commands: MEASURE the whole batch at ``target`` first (§7 batch
        scan — the first pays the move, the rest only switch + detect), then CLEAR each
        bundled source at its own blind-clear point. Measures precede clears so a stop
        senses its neighbourhood before acting on it."""
        prims: List[Primitive] = [
            Primitive(PrimitiveKind.MEASURE, self.target, int(c)) for c in self.scan_channels
        ]
        for ch, tgt in zip(self.clear_channels, self.clear_targets):
            prims.append(Primitive(PrimitiveKind.CLEAR, tgt, int(ch)))
        return prims

    @property
    def n_services(self) -> int:
        return len(self.scan_channels) + len(self.clear_channels)
