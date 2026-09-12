"""Way4 decision layer (DESIGN.md §7–§10, §13).

Turns belief + certificate + state into ranked macro actions (``CandidateGenerator``,
§7–8), then chooses one under a receding-horizon future-cost estimate (M7). The
generator proposes only — it never certifies (Invariant B) or bypasses the clear
guard (禁止6/7); those decisions stay in the certificate and safety layers.
"""

from .candidates import CandidateGenerator
from .future_cost import (
    CostView,
    DetectedRegion,
    FutureCost,
    FutureCostEstimator,
    build_cost_view,
)
from .receding_horizon import (
    CandidateEvaluation,
    Outcome,
    OutcomePredictor,
    PlanResult,
    RecedingHorizonPlanner,
)

__all__ = [
    "CandidateGenerator",
    "CostView",
    "DetectedRegion",
    "FutureCost",
    "FutureCostEstimator",
    "build_cost_view",
    "CandidateEvaluation",
    "Outcome",
    "OutcomePredictor",
    "PlanResult",
    "RecedingHorizonPlanner",
]
