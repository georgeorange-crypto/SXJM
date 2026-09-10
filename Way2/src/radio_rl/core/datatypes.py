"""Plain-data types that flow between layers.

These are deliberately torch-free so the whole mathematical core (env,
geometry, candidates, heuristic, safety, evaluation) runs without importing
torch. Tensor-valued contracts (FeatureBundle, EncoderOutput) live next to the
code that produces them, in ``features`` and ``models``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class ObservationType(IntEnum):
    """Outcome of executing one action in the environment.

    The measurement outcomes map one-to-one onto what the official engine
    returns: SIGNAL ("direction", carries a noisy bearing), NO_SIGNAL, and
    TOO_STRONG ("near", within 5 m — reachable but no usable bearing). A CLEAR
    resolves to CLEAR_SUCCESS / CLEAR_FAILURE. RESET is an internal control
    value carrying the initial pose. There is deliberately no OPTICAL outcome:
    the robot is never told the true source position; getting within the 20 m
    clear radius and issuing CLEAR is the only way a source is neutralised.
    """

    RESET = 0
    SIGNAL = 1          # "direction": within receive radius & coverage, bearing given
    NO_SIGNAL = 2       # out of range / out of coverage
    TOO_STRONG = 3      # "near": within 5 m, signal too strong for a bearing
    CLEAR_SUCCESS = 4
    CLEAR_FAILURE = 5


class ActionType(IntEnum):
    SCAN = 0     # move to target, (switch channel), perform detection there
    CLEAR = 1    # move to target, attempt to clear the source on `channel`
    EXIT = 2     # terminate the episode


class ChannelStatus(IntEnum):
    UNKNOWN = 0     # never detected a signal on this channel
    DETECTED = 1    # at least one bearing, region still large
    LOCALIZED = 2   # feasible region small enough to clear
    CLEARED = 3     # source neutralized


class CandidateSource(IntEnum):
    SEARCH = 0      # explore for unknown sources
    LOCALIZE = 1    # shrink an existing feasible region
    FOLLOWUP = 2    # exploit a fresh bearing with a good second measurement
    CLEAR = 3       # execute a clear on a localized source
    SAFETY = 4      # mathematical fallback move


SIGNAL_OUTCOMES = frozenset(
    {ObservationType.SIGNAL, ObservationType.TOO_STRONG}
)


def is_positive_detection(t: ObservationType) -> bool:
    """True when the outcome means 'the source is reachable from here'."""
    return t in SIGNAL_OUTCOMES


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
@dataclass
class Action:
    """Concrete instruction handed to the environment.

    Produced by the safety shield from the agent's chosen CandidateAction; it is
    the only thing ``env.execute`` ever sees.
    """

    action_type: ActionType
    channel: int
    target_x: float
    target_y: float
    candidate_id: int = -1          # provenance for logging/debug


@dataclass
class CandidateAction:
    """A single scored option offered to the learnable agent.

    The agent only ever outputs an *index* into a padded list of these; it never
    emits raw coordinates. All predicted_* / heuristic_* fields come from the
    analytical world model and are pure inputs to the policy.
    """

    candidate_id: int
    action_type: int                # ActionType value (int for tensorization)
    channel: int
    target_x: float
    target_y: float

    expected_time: float = 0.0                  # analytical time cost of acting
    heuristic_information_gain: float = 0.0     # analytical expected info gain
    predicted_region_reduction: float = 0.0     # expected feasible-area drop
    predicted_clear_probability: float = 0.0    # P(clear succeeds) if CLEAR

    source: int = int(CandidateSource.SEARCH)   # CandidateSource value

    def to_action(self) -> Action:
        return Action(
            action_type=ActionType(self.action_type),
            channel=self.channel,
            target_x=self.target_x,
            target_y=self.target_y,
            candidate_id=self.candidate_id,
        )


# ---------------------------------------------------------------------------
# Observations & measurement history
# ---------------------------------------------------------------------------
@dataclass
class DetectionObservation:
    """Result of one executed action, from local *or* official environment.

    Both environments must return exactly this structure so the agent cannot
    tell which one it is running in.
    """

    result_type: ObservationType
    channel: int
    position_x: float               # robot pose *after* the action
    position_y: float
    virtual_time_delta: float       # task-time consumed by the action (s)

    bearing_deg: float | None = None        # set for SIGNAL (noisy, +/-1 deg)
    clear_success: bool | None = None        # set for CLEAR_SUCCESS / CLEAR_FAILURE


@dataclass
class MeasurementRecord:
    """One entry of the running measurement log (feeds the GNN graph later)."""

    channel: int
    x: float
    y: float
    result_type: ObservationType
    virtual_time: float
    bearing_deg: float | None = None


# ---------------------------------------------------------------------------
# Episode bookkeeping (for evaluation & logging)
# ---------------------------------------------------------------------------
@dataclass
class EpisodeStats:
    sources_total: int = 0
    sources_cleared: int = 0

    virtual_time: float = 0.0       # total task time (the thing we minimize)
    wall_time: float = 0.0          # program wall-clock consumed
    route_distance: float = 0.0     # total metres travelled

    num_scans: int = 0
    num_clears: int = 0             # clear attempts
    failed_clears: int = 0
    num_switches: int = 0
    steps: int = 0

    # Virtual time at which each source was cleared (for avg-clear-time metric).
    clear_times: list[float] = field(default_factory=list)
    # Number of scans spent per cleared source (localization effort metric).
    scans_per_cleared: list[int] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.sources_total > 0 and self.sources_cleared == self.sources_total

    @property
    def clear_ratio(self) -> float:
        if self.sources_total == 0:
            return 0.0
        return self.sources_cleared / self.sources_total

    @property
    def avg_clear_time(self) -> float:
        if not self.clear_times:
            return float("inf")
        return sum(self.clear_times) / len(self.clear_times)
