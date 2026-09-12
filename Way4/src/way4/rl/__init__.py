"""Way4 RL residual layer (DESIGN.md §12, M10 — optional, last).

The math stack (Belief → Certificate → Candidate → Route → FutureCost →
RecedingHorizon → Safety) owns every correctness/safety/coverage/clear guarantee.
This package adds **only** a learned residual ``ΔQ_θ`` that reranks the macro
candidates the math planner already scored:

    Q(B, a) = Q_math(B, a) + ΔQ_θ(B, a)

It is deliberately dormant by default: ``Pipeline()`` still constructs a plain
``RecedingHorizonPlanner``, and even ``RLResidualPlanner`` with a fresh scorer is a
bit-for-bit no-op (zero-initialised head / torch-absent → residual 0). Activation
is an explicit, separate step taken only after the deterministic layer is verified
at 100% full-clear and the math expert is frozen (§12: 只在 Way4-Math 稳定后).

Constraints enforced structurally: 禁止8 (reranker only, never emits coordinates),
禁止9 (MLP over the compressed belief — no Dreamer/Mamba/GNN/LSTM), 禁止10 (only
reorders already-safe candidates; the SafetyShield runs *after* and can still veto).
"""

from .features import (
    CANDIDATE_FEATURE_DIM,
    FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    evaluation_features,
    feature_matrix,
)
from .residual_planner import RLResidualPlanner
from .sampling_planner import Decision, SamplingResidualPlanner
from .scorer import ResidualScorer
from .train import (
    EpisodeRecord,
    ResidualTrainer,
    TrainConfig,
    episode_return,
)

__all__ = [
    "CANDIDATE_FEATURE_DIM",
    "GLOBAL_FEATURE_DIM",
    "FEATURE_DIM",
    "evaluation_features",
    "feature_matrix",
    "ResidualScorer",
    "RLResidualPlanner",
    "SamplingResidualPlanner",
    "Decision",
    "ResidualTrainer",
    "TrainConfig",
    "EpisodeRecord",
    "episode_return",
]
