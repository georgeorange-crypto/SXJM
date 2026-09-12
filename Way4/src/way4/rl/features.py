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

from sxjm_core.geometry import dist

from ..belief import ChannelStatus
from ..core import MacroActionType

# --- fixed scaling constants (documented, not learned) -----------------------

#: Seconds that map to 1.0 in a time feature. Episode/cost-to-go values run to
#: a few thousand seconds (see compare_way3_way4), so 1000 keeps features O(1).
T_SCALE = 1000.0
#: Arena radius (m); positions and travel are normalised by it (§1 D_1800).
ARENA_R = 1800.0
#: Nominal candidate-set size used only to scale the "how many options" feature.
NOMINAL_CANDS = 32.0

#: The seven macro intents, in a FROZEN order for the one-hot block (§7).
_ACTION_ORDER: List[MacroActionType] = [
    MacroActionType.EXPLORE,
    MacroActionType.INITIALIZE,
    MacroActionType.REFINE,
    MacroActionType.PURSUE,
    MacroActionType.CLEAR,
    MacroActionType.VERIFY,
    MacroActionType.EXIT,
]
_ACTION_INDEX = {a: i for i, a in enumerate(_ACTION_ORDER)}

#: Per-candidate block width = one-hot(7) + gains(4) + (immediate, future, q)(3)
#: + scan_frac(1) + travel(1) + is_scan(1).
CANDIDATE_FEATURE_DIM = len(_ACTION_ORDER) + 4 + 3 + 1 + 1 + 1
#: Global belief-context block: present/clearable/unknown/resolved fractions(4)
#: + robot xy(2) + n_candidates(1).
GLOBAL_FEATURE_DIM = 4 + 2 + 1
#: Full input vector = per-candidate block ++ global block.
FEATURE_DIM = CANDIDATE_FEATURE_DIM + GLOBAL_FEATURE_DIM


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
    feats.append(float(immediate_cost) / T_SCALE)
    feats.append(float(future_cost) / T_SCALE)
    feats.append(float(q_value) / T_SCALE)
    n_scan = len(getattr(candidate, "scan_channels", ()) or ())
    feats.append(n_scan / float(max(1, n_channels)))
    feats.append(dist(robot_pos, candidate.target) / ARENA_R)
    feats.append(1.0 if getattr(candidate, "is_scan", False) else 0.0)
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
    return cand + glob


def feature_matrix(evaluations: Sequence, belief, state) -> List[List[float]]:
    """Stack ``evaluation_features`` for every evaluation → an ``n × FEATURE_DIM``
    row list (torch-free; the scorer converts to a tensor)."""
    n = len(evaluations)
    return [evaluation_features(e, belief, state, n_candidates=n) for e in evaluations]
