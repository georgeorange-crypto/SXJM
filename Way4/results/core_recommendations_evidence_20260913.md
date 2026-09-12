# Core recommendations evidence audit

This audit maps the 16 requirements in the supplied log-analysis checklist to
current Way4 code and executable evidence. A code symbol alone is not marked as
end-to-end proof.

| ID | Requirement | Current evidence | Verdict |
|---|---|---|---|
| P0-1 | Optimize total route length and move time | `evaluation_metrics.py`, `metrics.py`, `test_route_metrics.py`, `test_routing.py` | implemented + unit verified |
| P0-2 | Replace one-step planning with short horizon | `RecedingHorizonPlanner(horizon=...)`, `test_receding_horizon.py`, planner-mode tests | implemented; performance evidence pending |
| P0-3 | PPO is not the primary planner | `planner_mode=route_math` and explicit `final_ppo` mode; PPO is not required by default | verified by configuration |
| P0-4 | FOUND-source backlog | `RemainingTaskPool.sync()` and `WAITING_FOR_ROUTE`; `test_task_pool.py`, pipeline assignment tests | implemented + targeted verified |
| P0-5 | Spatial batching/regional commitment | `SpatialStop`, `SpatialStopGenerator`, service bundling tests | implemented + targeted verified |
| P0-6 | Route repair after route updates | `route_repair_open()` with 2-opt/relocate/swap; `test_routing.py` | implemented + targeted verified |
| P0-7 | Joint clearance and certificate route | `UnifiedRoutePlanner`, `FutureCost.j_joint`, certificate coverage debt | implemented; broad runtime comparison pending |
| P0-8 | Geometric guaranteed discovery for P4 | directional backbone/fallback and P4 geometry tests | implemented; larger randomized gate pending |
| P0-9 | Boundary outward/tangent handling | directional fallback and P4 hypothesis tests | partially verified; no complete worst-case proof artifact |
| P1-1 | Fixed detection/coverage backbone | `directional_fallback_anchors`, certificate manager | implemented; full P3/P4 route audit pending |
| P1-2 | Selective channel scanning | scheduler `value_components()`/`eta()` and adaptive STOP tests | implemented + targeted verified |
| P1-3 | Set-membership belief | `ChannelBelief`, effective region, NO_SIGNAL exclusion tests | implemented + targeted verified |
| P1-4 | Discovery-age penalty | scheduler scan debt, task-pool wait age/starvation promotion | implemented + targeted verified |
| P1-5 | Required route metrics | `episode_row()`, `compute_route_metrics()` and route metric tests | implemented + targeted verified |
| P1-6 | Watchdog/fallback accounting | `CandidatePPOPlanner.fallback_reason`, pipeline metrics | implemented; full fallback episode audit pending |
| P1-7 | Same-seed paired benchmark | `evaluation_protocol.py`, `paired_p4_eval.py` | protocol implemented; complete PPO paired run unavailable |
| P1-8 | Community thresholds as gates | existing P3/P4 gate scripts and artifacts | reporting gate exists; threshold pass not established |

## Runtime evidence currently available

The route-math P4 gate run on seeds 2000–2002 completed with 3/3 full-clear,
zero errors, and 13/13, 14/14, and 15/15 cleared sources respectively. It is
not evidence that the 400 s/target community threshold is met.

A separate same-seed paired artifact is now complete for seeds 2050–2052:
both Math and PPO are 3/3 full-clear with zero errors. Math mean time is
14486.24 s and PPO mean time is 14636.34 s; Math mean movement is 55447.86 m
and PPO mean movement is 55901.70 m. Paired win-rate is 2/3 for PPO, but one
regression is 1116.66 s, so this does not support promoting PPO to the primary
planner. Artifact: `paired_p4_verified_2050_2052.json`.

Using the cleared-source count as the target denominator, the observed
seconds/target values are Math/PPO = 1034.04/1009.55 (2050),
1318.65/1420.16 (2051), and 965.14/943.57 (2052). None of the three Math
cases meets the 400 s/target community reference, so the performance gate is
correctly reported as failed rather than silently treated as a success.

Targeted regression evidence: `29 passed, 1 skipped` for checkpoint/routing/
SpatialStop/WAIT/Pareto/STOP/metrics/lower-bound/service-synergy coverage, plus
`31 passed, 3 skipped` for routing, planner mode, artifact, readiness, state,
identity, and freeze checks.

Geometry/certificate/cardinality/scheduler evidence: `27 passed` covering
guaranteed clear sets, P4 hypotheses, effective soundness, negative soundness
artifact, cardinality soundness, coverage debt, scheduler starvation/value,
task VOI, and evaluation protocol.

The remaining verdicts above require broader randomized soundness, full fallback
episode accounting, or a complete paired/ablation run; they must not be promoted
to complete solely from the targeted tests.
