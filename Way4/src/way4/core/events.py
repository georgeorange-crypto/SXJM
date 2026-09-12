"""Typed event and option-transition records for the Way4 SMDP layer."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class EventType(str, Enum):
    BATCH_COMPLETED = "BATCH_COMPLETED"
    POSITIVE_DISCOVERY = "POSITIVE_DISCOVERY"
    INITIALIZED = "INITIALIZED"
    LOCALIZED = "LOCALIZED"
    SOURCE_CLEARED = "SOURCE_CLEARED"
    CHANNEL_CERTIFIED_EMPTY = "CHANNEL_CERTIFIED_EMPTY"
    CARDINALITY_CLOSURE = "CARDINALITY_CLOSURE"
    COVERAGE_THRESHOLD = "COVERAGE_THRESHOLD"
    TIME_BUDGET_WARNING = "TIME_BUDGET_WARNING"


@dataclass(frozen=True)
class Way4Event:
    event_type: EventType
    time_s: float
    channel: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptionTransition:
    """One macro/option transition; timestep is not a primitive second."""

    start_belief: Tuple[Tuple[int, str], ...]
    option_type: str
    target: Tuple[float, float]
    channel_batch: Tuple[int, ...]
    elapsed_time_s: float
    primitive_count: int
    resulting_belief: Tuple[Tuple[int, str], ...]
    terminal: bool
    full_clear: bool

    def to_record(self) -> Dict[str, Any]:
        """JSON-ready SMDP transition record for offline learners."""
        return asdict(self)


def transition_dataset(transitions) -> List[Dict[str, Any]]:
    """Serialize a sequence of option transitions without primitive expansion."""
    return [t.to_record() if hasattr(t, "to_record") else asdict(t)
            for t in transitions]


class EventBus:
    """Small synchronous event bus for diagnostics and replanning observers.

    Subscribers are observers only: publishing an event cannot mutate belief,
    certificate, or simulator state unless an explicit caller chooses to do so.
    Exceptions are not swallowed, making faulty instrumentation visible in tests.
    """

    def __init__(self) -> None:
        self._listeners: Dict[Optional[EventType], List[Callable[[Way4Event], None]]] = {}

    def subscribe(self, listener: Callable[[Way4Event], None], event_type=None) -> None:
        key = event_type
        self._listeners.setdefault(key, []).append(listener)

    def publish(self, event: Way4Event) -> None:
        for listener in self._listeners.get(None, ()):
            listener(event)
        for listener in self._listeners.get(event.event_type, ()):
            listener(event)
