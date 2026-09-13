"""Feature extraction for the M10 RL residual (DESIGN.md §12).

The residual ``ΔQ_θ(B,a)`` scores a *macro candidate* in the context of the
current belief. This module turns ``(CandidateEvaluation, BeliefState, RobotState)``
into a fixed-width, **torch-free** feature vector so it can be unit-tested and
used for offline dataset export without importing torch, and so the scorer is a
plain MLP over a flat vector (禁止9: no recurrence/graph/state-space — the belief
already compresses history, §12).

Design rules honoured here:
  * **Pure function, no side effects** — reads belief/state, never mutates them,
    never marks anything (Invariant B).
  * **The math q-value is a feature.** ``Q_math`` (the planner's per-candidate
    ``q_value``) is fed in as an input channel, so the network learns a *residual*
    relative to the analytical ranking, not a from-scratch value (§12: 数学负责
    正确/安全/几何, RL 只学 long-term trade-off).
  * **Deterministic + bounded scaling.** Times are divided by ``T_SCALE`` seconds
    and positions by ``ARENA_R`` metres — fixed, documented constants, no learned
    normaliser state — so the same inputs always give the same vector.

The layout is frozen by the exported constants (``CANDIDATE_FEATURE_DIM``,
``GLOBAL_FEATURE_DIM``, ``FEATURE_DIM``); tests assert them so a silent layout
drift can't desync a trained checkpoint.
"""

from __future__ import annotations

from typing import List, Optional, Sequence
import hashlib

from sxjm_core.geometry import dist

from ..belief import ChannelStatus
from ..core import MacroActionType
from ..planner.lower_bounds import time_debt

# --- fixed scaling constants (documented, not learned) -----------------------

#: Seconds that map to 1.0 in a time feature. Episode/cost-to-go values run to
#: a few thousand seconds (see compare_way3_way4), so 1000 keeps features O(1).
T_SCALE = 1000.0
#: Arena radius (m); positions and travel are normalised by it (§1 D_1800).
ARENA_R = 1800.0
#: Nominal candidate-set size used only to scale the "how many options" feature.
NOMINAL_CANDS = 32.0

#: The eight macro intents, in a FROZEN order for the one-hot block (§7).
_ACTION_ORDER: List[MacroActionType] = [
    MacroActionType.EXPLORE,
    MacroActionType.INITIALIZE,
    MacroActionType.REFINE,
    MacroActionType.PURSUE,
    MacroActionType.CLEAR,
    MacroActionType.VERIFY,
    MacroActionType.STOP,
    MacroActionType.EXIT,
]
_ACTION_INDEX = {a: i for i, a in enumerate(_ACTION_ORDER)}

#: Per-candidate block width = one-hot(7) + gains(4) + (immediate, future, q)(3)
#: + scan_frac(1) + travel(1) + is_scan(1).
# gains(4), route marginal/saving/synergy(3), costs(3), scan/travel/is_scan(3),
# multi-service count/density(3)
CANDIDATE_FEATURE_DIM = len(_ACTION_ORDER) + 4 + 3 + 3 + 1 + 1 + 1 + 3
CHANNEL_FEATURE_DIM = 5  # area, diameter, kappa, coverage debt, readiness
#: Global/context block: base global features (7) plus selected-channel geometry
#: (effective area, diameter, kappa, coverage debt, readiness).
GLOBAL_FEATURE_DIM = 4 + 2 + 1 + CHANNEL_FEATURE_DIM
# channel block is included in the default evaluation vector.
#: Full input vector = per-candidate ++ global ++ selected-channel geometry.
FEATURE_DIM = CANDIDATE_FEATURE_DIM + GLOBAL_FEATURE_DIM
FEATURE_SCHEMA_VERSION = 2
FEATURE_SCHEMA_SPEC = "action=" + ",".join(a.value for a in _ACTION_ORDER) + ";candidate_dim=" + str(CANDIDATE_FEATURE_DIM)
FEATURE_SCHEMA_HASH = hashlib.sha256(FEATURE_SCHEMA_SPEC.encode("utf-8")).hexdigest()

BACKBONE_DECISIONS = ("FOLLOW", "SKIP", "INSERT", "DETOUR", "RETURN")


def backbone_feature_block(candidate) -> List[float]:
    """Explicit backbone context for a residual policy (V01/V02)."""
    meta = getattr(candidate, "meta", {})
    kind = str(meta.get("backbone_kind", ""))
    decision = {"BackboneNext": "FOLLOW", "BackboneSkip": "SKIP",
                "BackboneBridge": "RETURN", "BackboneClosure": "FOLLOW",
                "PiggybackRefine": "INSERT", "PiggybackClear": "INSERT"}.get(kind, "DETOUR")
    one_hot = [1.0 if decision == name else 0.0 for name in BACKBONE_DECISIONS]
    values = [float(meta.get("backbone_index", -1.0)),
              float(meta.get("backbone_route_delta", meta.get("bridge_distance", 0.0))),
              float(meta.get("backbone_debt_gain", meta.get("certificate_debt", 0.0))),
              float(meta.get("backbone_nodes_retired", 0.0)),
              float(meta.get("rejoin_cost", meta.get("bridge_distance", 0.0))),
              float(meta.get("skip_gain", 0.0)), float(meta.get("piggyback_gain", 0.0))]
    if values[0] < -1.0 or any(value < 0.0 for value in values[1:]):
        raise ValueError("backbone features must be non-negative, except missing index=-1")
    return [1.0 if kind else 0.0] + one_hot + [values[0] / 32.0, values[1] / T_SCALE,
                                               values[2], values[3] / 32.0,
                                               values[4] / T_SCALE, values[5], values[6]]


def critic_time_state_block(*, elapsed_time_s: float, unresolved_count: int,
                            cleared_count: int, certificate_coverage: float,
                            remaining_route_lb_s: float, time_debt_s: float,
                            phase: float) -> List[float]:
    """AA02: normalized global temporal state for the value critic."""
    if any(float(v) < 0.0 for v in (elapsed_time_s, unresolved_count, cleared_count,
                                    remaining_route_lb_s, time_debt_s)):
        raise ValueError("critic time state values must be non-negative")
    if not 0.0 <= float(certificate_coverage) <= 1.0:
        raise ValueError("certificate coverage must be in [0, 1]")
    return [float(elapsed_time_s) / T_SCALE, float(unresolved_count),
            float(cleared_count), float(certificate_coverage),
            float(remaining_route_lb_s) / T_SCALE, float(time_debt_s) / T_SCALE,
            float(phase)]


def time_debt_block(*, before_est_s: float, after_est_s: float,
                    remaining_lb_s: float) -> List[float]:
    """Optional v3 block: debt before, after estimate, and signed delta."""
    before = time_debt(before_est_s, remaining_lb_s)
    after = time_debt(after_est_s, remaining_lb_s)
    return [before / T_SCALE, after / T_SCALE, (after - before) / T_SCALE]


def candidate_time_debt_block(candidate) -> List[float]:
    """N02: candidate-local before/after time debt and signed change."""
    meta = getattr(candidate, "meta", {})
    before = float(meta.get("time_debt_before_s", 0.0))
    after = float(meta.get("time_debt_after_s", 0.0))
    if before < 0.0 or after < 0.0:
        raise ValueError("candidate time debt must be non-negative")
    return [before / T_SCALE, after / T_SCALE, (after - before) / T_SCALE]


def candidate_time_block(candidate, *, total_time_s: Optional[float] = None) -> List[float]:
    """Optional v3 immediate-time block in seconds/T_SCALE.

    Candidate generators may provide ``predicted_*_time_s`` metadata. Missing
    components remain zero; the total falls back to ``expected_time`` so no
    component is invented from an aggregate.
    """
    meta = getattr(candidate, "meta", {})
    total = float(candidate.expected_time if total_time_s is None else total_time_s)
    move = float(meta.get("predicted_move_time_s", 0.0))
    measure = float(meta.get("predicted_measure_time_s", 0.0))
    switch = float(meta.get("predicted_switch_time_s", 0.0))
    service = float(meta.get("predicted_service_time_s", 0.0))
    vals = (move, measure, switch, service, total)
    if any(v < 0.0 for v in vals):
        raise ValueError("predicted times must be non-negative")
    return [v / T_SCALE for v in vals]


def route_efficiency_block(candidate) -> List[float]:
    """Optional v3 route-efficiency features, with explicit unit scaling."""
    meta = getattr(candidate, "meta", {})
    def val(name, default=0.0):
        return float(getattr(candidate, name, meta.get(name, default)))
    route_delta = val("route_delta", val("route_marginal"))
    route_rank = val("route_rank")
    detour = val("detour_ratio")
    backtrack = val("backtrack_m")
    repeated = val("repeated_edge_m")
    repeated_ratio = val("repeated_edge_ratio")
    returned = val("unnecessary_return_m")
    crossing_gain = val("avoidable_crossing_gain", val("avoidable_crossing_m"))
    corridor = val("corridor_distance")
    next_task = val("next_task_distance")
    vals = (route_delta, route_rank, detour, backtrack, repeated,
            repeated_ratio, returned, crossing_gain, corridor, next_task)
    if any(v < 0.0 for v in vals):
        raise ValueError("route efficiency features must be non-negative")
    return [route_delta / T_SCALE, route_rank, detour, backtrack / ARENA_R,
            repeated / ARENA_R, repeated_ratio, returned / ARENA_R,
            crossing_gain / ARENA_R, corridor / ARENA_R, next_task / ARENA_R]


def progress_per_second_block(candidate, *, time_s: Optional[float] = None) -> List[float]:
    """Optional v3 progress-rate features, all gain units per second."""
    meta = getattr(candidate, "meta", {})
    total = float(candidate.expected_time if time_s is None else time_s)
    names = ("coverage_gain", "certificate_gain", "information_gain",
             "clear_probability", "localization_gain")
    gains = [float(getattr(candidate, name, meta.get(name, 0.0))) for name in names]
    if total <= 0.0 or any(g < 0.0 for g in gains):
        raise ValueError("progress gains must be non-negative and time must be positive")
    return [g / total for g in gains]


def future_route_block(candidate) -> List[float]:
    """Optional v3 future-route features with explicit second/metre scaling."""
    meta = getattr(candidate, "meta", {})
    def val(name, default=0.0):
        return float(getattr(candidate, name, meta.get(name, default)))
    before = val("remaining_route_lb_before")
    after = val("remaining_route_lb_after")
    future_delta = val("future_cost_delta")
    debt_delta = val("delta_time_debt")
    nearest = val("nearest_next_task")
    cluster = val("task_cluster_size")
    if cluster < 0.0 or nearest < 0.0 or before < 0.0 or after < 0.0:
        raise ValueError("future route distances and cluster size must be non-negative")
    return [before / T_SCALE, after / T_SCALE, future_delta / T_SCALE,
            debt_delta / T_SCALE, nearest / ARENA_R, cluster]


def history_behavior_block(candidate) -> List[float]:
    """Optional v3 historical-behaviour features."""
    meta = getattr(candidate, "meta", {})
    def val(name):
        return float(getattr(candidate, name, meta.get(name, 0.0)))
    region = val("times_region_visited")
    nearby = val("times_channel_measured_nearby")
    since_time = val("time_since_last_progress")
    since_dist = val("distance_since_last_progress")
    overlap = val("recent_route_overlap_ratio")
    if any(v < 0.0 for v in (region, nearby, since_time, since_dist)):
        raise ValueError("history counts and distances must be non-negative")
    if not 0.0 <= overlap <= 1.0:
        raise ValueError("recent route overlap ratio must be in [0, 1]")
    return [region, nearby, since_time / T_SCALE, since_dist / ARENA_R, overlap]


def _action_one_hot(action_type: MacroActionType) -> List[float]:
    vec = [0.0] * len(_ACTION_ORDER)
    idx = _ACTION_INDEX.get(action_type)
    if idx is not None:
        vec[idx] = 1.0
    return vec


def candidate_block(
    candidate,
    *,
    immediate_cost: float,
    future_cost: float,
    q_value: float,
    robot_pos,
    n_channels: int,
) -> List[float]:
    """The per-candidate features (length ``CANDIDATE_FEATURE_DIM``).

    ``immediate_cost``/``future_cost``/``q_value`` come from the planner's
    ``CandidateEvaluation`` for this candidate; ``q_value`` is ``Q_math`` — the
    anchor the residual corrects."""
    feats: List[float] = []
    feats += _action_one_hot(candidate.action_type)
    feats.append(float(candidate.exploration_gain))
    feats.append(float(candidate.refinement_gain))
    feats.append(float(candidate.certificate_gain))
    feats.append(float(candidate.route_gain))
    feats.append(float(getattr(candidate, "route_marginal", candidate.meta.get("route_marginal", 0.0))) / T_SCALE)
    feats.append(float(getattr(candidate, "route_saving", candidate.meta.get("route_saving", 0.0))) / T_SCALE)
    feats.append(float(getattr(candidate, "route_synergy", candidate.meta.get("route_synergy", 0.0))) / T_SCALE)
    feats.append(float(immediate_cost) / T_SCALE)
    feats.append(float(future_cost) / T_SCALE)
    feats.append(float(q_value) / T_SCALE)
    n_scan = len(getattr(candidate, "scan_channels", ()) or ())
    feats.append(n_scan / float(max(1, n_channels)))
    feats.append(dist(robot_pos, candidate.target) / ARENA_R)
    feats.append(1.0 if getattr(candidate, "is_scan", False) else 0.0)
    feats.append(float(getattr(candidate, "n_measure_services", len(getattr(candidate, "scan_channels", ())))) / max(1, n_channels))
    feats.append(float(getattr(candidate, "n_clear_services", len(getattr(candidate, "clear_channels", ())))) / max(1, n_channels))
    feats.append(float(getattr(candidate, "service_density", 0.0)))
    return feats


def global_block(belief, state) -> List[float]:
    """Belief-context features (length ``GLOBAL_FEATURE_DIM``), identical for every
    candidate in one ``plan`` call — the network sees "where the episode stands"."""
    n = float(max(1, getattr(belief, "n_channels", 20)))
    present = len(belief.present_channels()) / n
    clearable = len(belief.clearable_channels()) / n
    unknown = len(belief.unknown_channels()) / n
    resolved = (
        sum(1 for b in belief.channels.values() if b.status in _RESOLVED) / n
    )
    return [
        present,
        clearable,
        unknown,
        resolved,
        float(state.x) / ARENA_R,
        float(state.y) / ARENA_R,
    ]


_RESOLVED = frozenset({ChannelStatus.CLEARED, ChannelStatus.ABSENT_CERTIFIED})


def evaluation_features(evaluation, belief, state, *, n_candidates: int) -> List[float]:
    """Full feature vector (length ``FEATURE_DIM``) for one ``CandidateEvaluation``.

    Concatenates the per-candidate block, the shared global block, and a single
    normalised candidate-count feature (how many options the planner weighed)."""
    n_channels = int(getattr(belief, "n_channels", 20))
    cand = candidate_block(
        evaluation.candidate,
        immediate_cost=evaluation.immediate_cost,
        future_cost=evaluation.future_cost,
        q_value=evaluation.q_value,
        robot_pos=(state.x, state.y),
        n_channels=n_channels,
    )
    glob = global_block(belief, state)
    glob.append(float(n_candidates) / NOMINAL_CANDS)
    return cand + glob + channel_context_block(evaluation, belief, state)


def evaluation_features_v3(evaluation, belief, state, *, n_candidates: int) -> List[float]:
    """Explicit augmented vector for v3 experiments (N02/O04).

    The frozen ``evaluation_features`` schema is unchanged for old checkpoints;
    v3 callers opt in and receive time-debt plus future-route blocks.
    """
    base = evaluation_features(evaluation, belief, state, n_candidates=n_candidates)
    candidate = evaluation.candidate
    return base + candidate_time_debt_block(candidate) + future_route_block(candidate)


def channel_context_block(evaluation, belief, state, certificate=None) -> List[float]:
    """Optional per-channel geometry block for learning-augmented planners."""
    candidate = evaluation.candidate
    channel = candidate.meta.get("refine_channel")
    if channel is None and candidate.scan_channels:
        channel = candidate.scan_channels[0]
    if channel is None or channel not in belief.channels:
        return [0.0] * CHANNEL_FEATURE_DIM
    b = belief[channel]
    debt = certificate.coverage_debt(channel) if certificate is not None else 0.0
    snapshot = getattr(b, "readiness_snapshot", None)
    if snapshot is not None:
        summary = snapshot((state.x, state.y), debt)
        area_raw = float(summary["area"])
        diameter_raw = float(summary["diameter"])
        kappa_raw = float(summary["kappa"])
        readiness = float(summary["initialization_ready"])
    else:
        area_raw = float(b.effective_area(spacing=60.0))
        diameter_raw = float(b.effective_diameter(spacing=60.0))
        kappa_raw = float(getattr(b, "kappa", 1.0))
        readiness = float(getattr(b, "initialization_ready", False))
    area = min(1.0, max(0.0, area_raw / (3.141592653589793 * ARENA_R ** 2)))
    diameter = min(2.0, max(0.0, diameter_raw / ARENA_R))
    kappa = min(20.0, max(1.0, kappa_raw)) / 20.0
    return [area, diameter, kappa, float(debt), readiness]


def feature_matrix(evaluations: Sequence, belief, state) -> List[List[float]]:
    """Stack ``evaluation_features`` for every evaluation → an ``n × FEATURE_DIM``
    row list (torch-free; the scorer converts to a tensor)."""
    n = len(evaluations)
    return [evaluation_features(e, belief, state, n_candidates=n) for e in evaluations]
