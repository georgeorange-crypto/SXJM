# Algorithm Toolbox

This is the auditable mapping from the design checklist to executable code.
The toolbox contains deterministic kernels; `way4.certificate` remains the sole
owner of hard absence/full-clear decisions.

| Checklist family | Executable entry point | Way4 use |
|---|---|---|
| Set membership / hard belief | `way4.belief.ChannelBelief` | P3/P4 safety layer |
| Bayesian, particles, mixture, KDE | `way4.toolbox.probability` | soft ranking only |
| Information gain / VOI | `way4.toolbox.information`, `RecedingHorizonPlanner.value_of_information` | selective sensing |
| Circle/set/max coverage | `way4.toolbox.coverage`, `way4.certificate` | exploration and certificates |
| Triangle/boundary geometry | `way4.certificate.directional_certificate` | P4 certificate |
| Open TSP / Held--Karp | `way4.routing.tsp` | exact short routes |
| Insertion / 2-opt / relocate | `way4.toolbox.routing` | `UnifiedRoutePlanner` heuristic routes |
| TSPN / GTSP | `way4.routing.estimator`, `way4.toolbox.advanced.generalized_tsp` | uncertain neighborhoods |
| Rolling DP / beam / MPC | `way4.toolbox.routing.rolling_subset_dp`, `way4.toolbox.planning`, `way4.planner.receding_horizon` | receding horizon |
| Dijkstra / A* / D* Lite / LPA* | `way4.toolbox.graph` | discrete-map adapter |
| Expected / robust / CVaR / B&B | `way4.toolbox.optimization` | route ranking and offline search |
| K-means / DBSCAN | `way4.toolbox.clustering` | spatial batching |
| Residual / shield / hyper-heuristic | `way4.toolbox.planning`, `way4.rl` | bounded learned ranking |
| MST / 1-tree / oracle / metrics | `way4.toolbox.advanced`, `way4.planner.lower_bounds`, `way4.metrics` | evaluation and lower bounds |

The hard belief/certificate invariant is unchanged: probability, heuristics and
RL may change order or speed, but cannot mark a channel absent or full-clear.

Current gate status: DBSCAN is available through the explicit
`spatial_clustering="dbscan"` ablation, but is not the default pipeline mode.
The official Way3 3-seed, 20-step study reduced travel time/distance while also
reducing resolved channels and achieved 0/3 full-clear in both variants.
