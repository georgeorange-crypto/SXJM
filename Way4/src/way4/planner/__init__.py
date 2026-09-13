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
from .spatial import InformationRidge, SpatialStopGenerator
from .opportunities import RemainingTask, insertion_cost, pareto_prune
from .task_pool import RemainingTaskPool, WaitingTask
from .baselines import CoverageGreedyResult, coverage_greedy
from .commitment import FocusController, FocusState
from .lower_bounds import time_debt
from .backbone import (BackboneEdge, BackboneManager, BackboneNode, BackboneStatus,
                       BoundaryCap, CoverageDebt, CoverageResponsibility, DirectionalTriangle,
                       BackboneGeometryAudit, audit_backbone_geometry, build_boundary_caps,
                       build_p4_triangle_mesh)
from .p3_backbone import P3BackboneResult, center_ring_baseline, optimize_p3_backbone
from .piggyback import (WAIT_FOR_BACKBONE, RefineVariantEvaluation,
                        choose_refine_variant, evaluate_refine_variant,
                        mark_wait_for_backbone)
from .endgame import EndgameController, EndgamePlan, solve_endgame

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
    "SpatialStopGenerator",
    "RemainingTask",
    "insertion_cost",
    "pareto_prune",
    "RemainingTaskPool",
    "WaitingTask",
    "InformationRidge",
    "CoverageGreedyResult",
    "coverage_greedy",
    "FocusController",
    "FocusState",
    "time_debt",
    "BackboneEdge", "BackboneManager", "BackboneNode", "BackboneStatus",
    "BoundaryCap", "CoverageDebt", "CoverageResponsibility", "DirectionalTriangle",
    "P3BackboneResult", "center_ring_baseline", "optimize_p3_backbone",
    "build_p4_triangle_mesh",
    "build_boundary_caps",
    "BackboneGeometryAudit", "audit_backbone_geometry",
    "RefineVariantEvaluation", "evaluate_refine_variant", "choose_refine_variant",
    "WAIT_FOR_BACKBONE", "mark_wait_for_backbone",
    "EndgamePlan", "solve_endgame", "EndgameController",
]
