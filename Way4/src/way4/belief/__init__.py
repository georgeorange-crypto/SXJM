"""Way4 unified belief layer (DESIGN.md §4).

M2 lands the P3 (omnidirectional) per-channel belief; the Q4 coarse-hypothesis
layer (§3.1) arrives in M8. The certificate/absent machinery lives in
``way4.certificate`` (M3) — this layer never certifies absence (Invariant B).
"""

from .channel import (
    ChannelStatus,
    ExclusionDisc,
    BearingObs,
    ChannelBelief,
    BeliefState,
    PRESENT_STATES,
    RESOLVED_STATES,
)

__all__ = [
    "ChannelStatus",
    "ExclusionDisc",
    "BearingObs",
    "ChannelBelief",
    "BeliefState",
    "PRESENT_STATES",
    "RESOLVED_STATES",
]
