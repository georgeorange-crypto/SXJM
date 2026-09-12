"""Way4 shared core datatypes (DESIGN.md §13).

Holds cross-layer vocabulary that belief / certificate / sensing / executor all
share: the observation outcome type (M3), and the macro-action + cost/state kernel
(M4). Task constants and richer datatypes join here as later milestones need them.
"""

from .observation import Observation, ObservationKind
from .actions import (
    MacroActionType,
    MacroCandidate,
    Primitive,
    PrimitiveKind,
    SpatialStop,
    get_allowed_actions,
)
from .cost import AnalyticalCostModel, RobotState
from .events import EventBus, EventType, OptionTransition, Way4Event, transition_dataset

__all__ = [
    "Observation",
    "ObservationKind",
    "MacroActionType",
    "MacroCandidate",
    "Primitive",
    "PrimitiveKind",
    "SpatialStop",
    "get_allowed_actions",
    "AnalyticalCostModel",
    "RobotState",
    "EventType",
    "EventBus",
    "transition_dataset",
    "Way4Event",
    "OptionTransition",
]
