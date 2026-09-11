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
    """The seven macro intents (DESIGN.md §7)."""

    EXPLORE = "EXPLORE"        # broad discovery / coverage batch scan
    INITIALIZE = "INITIALIZE"  # first bearing on a freshly-detected channel
    REFINE = "REFINE"          # NBV second/third bearing to shrink F_c (§8)
    PURSUE = "PURSUE"          # measure from a waypoint closer to a detected source
    CLEAR = "CLEAR"            # blind-clear at the MEC centre (§3.3)
    VERIFY = "VERIFY"          # completion waypoint, batch only uncertified UNKNOWNs (§6.7)
    EXIT = "EXIT"              # leave (guarded by the EXIT certificate, §11)


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
