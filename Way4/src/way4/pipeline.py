"""Way4 rolling pipeline (DESIGN.md §13).

The end-to-end control loop that ties the whole stack together:

    Env → BeliefUpdater → CertificateManager → CandidateGenerator →
    ChannelScheduler → AnalyticalCostModel → RouteEstimator →
    FutureCostEstimator → RecedingHorizonPlanner (→ optional RL residual) →
    SafetyShield → MacroExecutor → Env

Each tick is *Observe → Update → Plan → Execute one* (§10): generate the macro
candidates for the current belief, rank them under the receding-horizon future
cost, execute the single best macro against the env (folding every primitive
observation back into belief + certificate through the ``MacroExecutor``), then
re-plan. The loop ends on EXIT, on the env deadline, or on a step cap.

This M7 version wires the math stack. The SafetyShield / Way3-homing fallback and
the EXIT guard are M9 — until then the loop leans on two conservative rules kept
here so it can never *wrongly* stop early:

  * **EXIT gate** — an EXIT macro is honoured only when the certificate manager
    agrees every channel is CLEARED ∨ ABSENT_CERTIFIED (a force-run of the three
    §6.5 sources). If the planner proposes EXIT prematurely it is dropped and the
    next-best macro runs instead. This is the §11 EXIT guard's invariant; M9 moves
    it into the SafetyShield proper. (禁止7/10: never trade full-clear for speed.)
  * **Absent certification** — before each replan the manager pushes ABSENT_CERTIFIED
    into belief for every UNKNOWN channel the three sources certify (§6.5, the only
    place a channel is marked absent — 禁止6). This is what lets the generator stop
    scanning certified-clear channels and eventually reach a legitimate EXIT.

It marks nothing and certifies nothing itself — every absent verdict comes from the
CertificateManager, every clear from a real env hit (Invariants B/D).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
import traceback
from typing import List, Optional, Sequence, Tuple

from .belief import BeliefState, ChannelStatus
from .certificate import CertificateManager
from .channels import AdaptiveScanSession, ChannelScheduler, SchedulerMode
from .core import (AnalyticalCostModel, EventBus, EventType, MacroActionType,
                   MacroCandidate, OptionTransition, RobotState, Way4Event)
from .executor import HomingController, MacroExecutor
from .rl.final_planner import CandidatePPOPlanner
from .belief.hypothesis import HypothesisLayer
from .metrics import (
    ROLE_CLEAR,
    ROLE_COVERAGE,
    ROLE_LOCALIZE,
    RouteEvent,
    compute_route_metrics,
    compute_efficiency_metrics,
)
from .planner import CandidateGenerator, RecedingHorizonPlanner, SpatialStopGenerator, RemainingTaskPool
from .routing import RouteEstimator, ServiceOpportunity, UnifiedRoutePlanner
from .sensing import MinimaxNBV
from .analysis import summarize_decision_trace


@dataclass
class EpisodeResult:
    """Outcome of one Way4 episode (mirrors the Way3/offline_sim metric set so the
    two stacks compare field-for-field; see scripts/compare_way3_way4.py)."""

    success: bool                    # every channel resolved AND exited cleanly
    virtual_time_s: float            # authoritative env clock at stop
    move_distance_m: float           # total travelled distance
    n_measure: int                   # number of /measure primitives issued
    n_switch: int                    # number of measuring-channel changes
    n_clear: int                     # number of /clear primitives issued
    n_clear_hit: int                 # /clear calls that neutralised a source
    cleared: int                     # channels cleared
    present: int                     # channels proven PRESENT (positive obs / clear)
    resolved: int                    # channels CLEARED ∨ ABSENT_CERTIFIED
    n_channels: int
    steps: int                       # macros executed
    exited: bool                     # EXIT executed (vs deadline / step cap)
    hit_deadline: bool               # env signalled finish before EXIT
    error: Optional[str] = None      # exception text if the loop aborted

    # -- route decomposition diagnostics (P0 #9) -----------------------------
    # Pure-additive: computed from the realised primitive sequence, drives nothing
    # (no gate/belief/certificate reads these), so they cannot affect full_clear.
    # They exist to quantify the action-centric planner's routing waste and give the
    # #10 ablation an apples-to-apples baseline against the future spatial planner.
    # See way4.metrics.compute_route_metrics for definitions.
    services_per_stop: float = 0.0
    pure_refine_travel: float = 0.0
    certificate_only_travel: float = 0.0
    revisit_distance: float = 0.0
    shared_stop_ratio: float = 0.0
    clear_insertion_delta: float = 0.0
    route_n_stops: int = 0
    route_n_services: int = 0
    total_distance_m: float = 0.0
    clear_distance_m: float = 0.0
    n_longjump: int = 0
    n_crossing: int = 0
    n_empty_scan: int = 0
    source_diagnostics: dict = field(default_factory=dict)
    longest_waiting_sources: list = field(default_factory=list)
    time_move_s: float = 0.0
    time_measure_s: float = 0.0
    time_switch_s: float = 0.0
    time_clear_s: float = 0.0
    # Versioned accounting fields.  ``time_other_s`` is the residual not
    # explained by the four explicit cost buckets; it is intentionally kept
    # rather than silently folded into movement or sensing.
    time_other_s: float = 0.0
    time_accounting_error_s: float = 0.0
    equivalent_route_cost_m: float = 0.0
    planner_version: str = "way4-route-math-v1"
    metric_schema_version: str = "way4-metrics-v2"
    decision_trace: list = field(default_factory=list)
    decision_summary: dict = field(default_factory=dict)
    illegal_clear: int = 0
    safety_violation: int = 0
    clear_audit: list = field(default_factory=list)
    efficiency_metrics: dict = field(default_factory=dict)
    no_progress_time_s: float = 0.0
    backtrack_m: float = 0.0
    repeated_edge_m: float = 0.0
    unnecessary_return_m: float = 0.0

    @property
    def full_clear(self) -> bool:
        """The correctness criterion (§15): every present source cleared and every
        other channel certified absent — i.e. all channels resolved."""
        return self.resolved == self.n_channels


class Way4Pipeline:
    """Drives the Way4 math stack over an ``Env`` for one episode."""

    def __init__(
        self,
        env,
        n_channels: int = 20,
        problem: int = 3,
        time_budget_s: float = float("inf"),
        max_steps: int = 2000,
        generator: Optional[CandidateGenerator] = None,
        planner: Optional[RecedingHorizonPlanner] = None,
        planner_mode: str = "legacy",
        certificate: Optional[CertificateManager] = None,
        cost_model: Optional[AnalyticalCostModel] = None,
        opportunistic_clear: bool = True,
        batch_stop: bool = False,
        initial_state: Optional[RobotState] = None,
        stall_limit: int = 16,
        adaptive_scan: bool = False,
        routing_strategy: str = "tspn",
        nbv_objective: str = "minimax",
        coverage_strategy: str = "active_fallback",
        # DBSCAN remains an explicit ablation until multi-seed full-clear shows
        # no correctness regression; the safe default is the legacy candidate
        # set with no spatial bundle injection.
        spatial_clustering: str = "none",
        enable_no_signal: bool = True,
        enable_cardinality: bool = True,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self.env = env
        self.n_channels = int(n_channels)
        self.problem = int(problem)
        self.time_budget_s = float(time_budget_s)
        self.max_steps = int(max_steps)
        self.stall_limit = int(stall_limit)
        self.adaptive_scan = bool(adaptive_scan)
        self.routing_strategy = routing_strategy
        self.nbv_objective = nbv_objective
        self.enable_no_signal = bool(enable_no_signal)
        self.enable_cardinality = bool(enable_cardinality)
        self.event_bus = event_bus

        self.belief = BeliefState(n_channels=n_channels)
        self.hypotheses = HypothesisLayer() if problem == 4 else None
        self.certificate = certificate or CertificateManager(
            n_channels=n_channels, problem=problem,
            enable_no_signal=enable_no_signal,
            enable_cardinality=enable_cardinality,
        )
        self.cost = cost_model or AnalyticalCostModel()
        # P0 #2/#3 planner_mode: choose the candidate generator. "legacy" is the frozen
        # action-centric generator (byte-identical to the shipped path); "spatial" wraps
        # that same generator in SpatialStopGenerator (co-located services bundled into
        # SpatialStops) — a non-destructive superset, so every legacy candidate still
        # flows through and full-clear cannot regress (禁止10). An explicit ``generator=``
        # overrides the mode; any other value is rejected.
        if planner_mode not in ("legacy", "spatial", "route_math", "final_ppo"):
            raise ValueError(
                f"planner_mode must be one of legacy/spatial/route_math/final_ppo, got {planner_mode!r}"
            )
        self.planner_mode = planner_mode
        if generator is not None:
            self.generator = generator
        elif planner_mode in ("spatial", "route_math", "final_ppo"):
            self.generator = SpatialStopGenerator(CandidateGenerator(
                nbv=MinimaxNBV(objective=nbv_objective),
                cost_model=self.cost, hypothesis_layer=self.hypotheses,
                coverage_strategy=coverage_strategy), clustering=spatial_clustering)
        else:  # "legacy"
            self.generator = CandidateGenerator(cost_model=self.cost, hypothesis_layer=self.hypotheses)
        # Spatial candidates are costed under the joint route model by default;
        # legacy remains additive for reproducible ablations.
        self.planner = planner or RecedingHorizonPlanner(
            future_cost=__import__("way4.planner", fromlist=["FutureCostEstimator"]).FutureCostEstimator(
                route=RouteEstimator(speed=self.cost.speed, strategy=routing_strategy),
                speed=self.cost.speed,
                measure_s=self.cost.measure_s,
                clear_hit_s=self.cost.clear_hit_s,
                joint_route=(planner_mode in ("spatial", "route_math", "final_ppo")),
            )
        )
        if planner is None and planner_mode in ("spatial", "route_math", "final_ppo"):
            self.planner.pareto_candidates = True
            self.planner.route_synergy_weight = 1.0
        self.channel_scheduler = getattr(self.generator, "scheduler", ChannelScheduler())
        # Planner-owned WAIT_FOR_ROUTE state. This registry never writes belief
        # statuses; it is safe to discard/rebuild between replans.
        self.task_pool = RemainingTaskPool()
        self.unified_route = UnifiedRoutePlanner()
        self.last_unified_route = None
        if planner_mode == "final_ppo" and planner is None:
            self.planner = CandidatePPOPlanner(self.planner)
        self.executor = MacroExecutor(
            env,
            belief=self.belief,
            certificate=self.certificate,
            cost_model=self.cost,
            opportunistic_clear=opportunistic_clear,
            batch_stop=batch_stop,
            strict_clear_safety=True,
        )
        # §11 Way3-homing fallback: the sound directional localizer, engaged only
        # when the omni planner livelocks on a directional source (see run()). Way3
        # params by default (clear_margin=14, orbit_radius=85, orbit_delta=42°, ...).
        self._homing = HomingController(self.executor, self.belief)
        # §1.1: initial pose (0,0), initial measuring channel 1.
        self.state = initial_state or RobotState(0.0, 0.0, 1, 0)

        # metric accumulators
        self._move_m = 0.0
        self._n_measure = 0
        self._n_switch = 0
        self._n_clear = 0
        self._n_clear_hit = 0
        self._n_empty_scan = 0
        self._time_move_s = 0.0
        self._time_measure_s = 0.0
        self._time_switch_s = 0.0
        self._time_clear_s = 0.0
        self._steps = 0
        self.decision_audit = []
        self._last_progress_time_s = 0.0
        self._last_progress_distance_m = 0.0
        self._decisions_since_progress = 0
        self._no_progress_time_s = 0.0
        self._route_events: List[RouteEvent] = []          # P0 #9 diagnostics
        self._start_pos = (float(self.state.x), float(self.state.y))
        # no-progress guard state
        self._last_progress_key: Optional[Tuple] = None
        self._stall = 0
        self._adaptive_recovery_used = False
        self.events: List[Way4Event] = []
        self.transitions: List[OptionTransition] = []
        # Primitive-level audit trail; populated for every executed move/measure/clear.
        self.primitive_trace: List[dict] = []
        self.macro_trace: List[dict] = []
        # Full candidate attribution trace.  This is diagnostic only: it never
        # feeds safety, belief, or certificate decisions.
        self.decision_trace: List[dict] = []

    def _emit_event(self, event: Way4Event) -> None:
        """Record locally and notify optional read-only external observers."""
        self.events.append(event)
        if self.event_bus is not None:
            self.event_bus.publish(event)

    # -- main loop ----------------------------------------------------------

    def run(self) -> EpisodeResult:
        exited = False
        hit_deadline = False
        error: Optional[str] = None
        try:
            while self._steps < self.max_steps:
                # Update: push every newly-certifiable absent channel into belief
                # (§6.5; the only place absent is marked — 禁止6). force=True so the
                # three sources are fully evaluated each tick.
                self.certificate.apply_certifications(self.belief, force=True)
                self.certificate.apply_cardinality_presence(self.belief)
                self.task_pool.sync(self.belief)

                if self.belief.all_resolved():
                    # legitimate completion: everything CLEARED ∨ ABSENT_CERTIFIED.
                    exited = self._execute_exit()
                    break

                # Plan: rank this belief's macros under the receding-horizon Ĵ.
                macro = self._choose_macro()
                if macro is None:
                    # A learned planner may legally choose a dead-end.  The
                    # completion contract requires handing control permanently
                    # back to the deterministic base before reporting failure.
                    if hasattr(self.planner, "watchdog") and hasattr(self.planner, "base"):
                        if hasattr(self.planner, "mark_fallback"):
                            self.planner.mark_fallback("no_candidate")
                        else:
                            self.planner.watchdog.fallback = True
                        macro = self.planner.base.plan(
                            self.belief, self.certificate, self.state,
                            self.generator.generate(self.belief, self.certificate,
                                                    self.state, scan_mode=self._scan_mode()),
                        ).best
                    if macro is None:
                        error = "no_candidate"
                        break

                # Execute one macro; fold observations back into belief+certificate.
                # Snapshot belief status BEFORE execution so route roles reflect the
                # macro's intent, not the post-macro state (diagnostics only, §P0-9).
                status_before = {
                    c: self.belief[c].status for c in range(1, self.n_channels + 1)
                }
                start_belief = tuple(sorted((c, b.value) for c, b in status_before.items()))
                start_time = self.state.virtual_time_s
                if self.time_budget_s != float("inf"):
                    remaining = self.time_budget_s - start_time
                    if remaining <= max(1.0, float(macro.expected_time)) * 1.10:
                        self._emit_event(Way4Event(
                            EventType.TIME_BUDGET_WARNING, start_time,
                            payload={"remaining_s": remaining,
                                     "expected_macro_s": float(macro.expected_time)},
                        ))
                if self.adaptive_scan and macro.is_scan and macro.scan_channels:
                    session = AdaptiveScanSession(
                        ChannelScheduler(), macro.target, self.belief, self.certificate,
                        self.state.channel, mode=self._scan_mode(), min_value=0.0,
                    )
                    res = self.executor.execute_adaptive(
                        session, self.state, self.time_budget_s,
                        max_scans=max(1, self.n_channels),
                    )
                else:
                    res = self.executor.execute(macro, self.state, self.time_budget_s)
                if macro.scan_channels:
                    self.channel_scheduler.record_plan(
                        macro.scan_channels, n_channels=self.n_channels
                    )
                if self.hypotheses is not None:
                    for item in res.primitives:
                        if item.primitive.kind.name == "MEASURE" and item.observation is not None:
                            self.hypotheses.record_observation(int(item.primitive.channel),
                                                               tuple(item.primitive.target), item.observation)
                move_before = float(self._move_m)
                time_buckets_before = (self._time_move_s, self._time_measure_s,
                                       self._time_switch_s, self._time_clear_s)
                self._account(macro, res, status_before)
                # Complete the selected decision record with realised outcome.
                # This is intentionally post-execution and diagnostic only.
                selected_rows = [r for r in self.decision_trace
                                 if r["step"] == int(self._steps + 1) and r["selected"]]
                if selected_rows:
                    realized_distance = float(self._move_m - move_before)
                    after_area = {}
                    for c in range(1, self.n_channels + 1):
                        value = float(getattr(self.belief[c], "area", 0.0))
                        after_area[str(c)] = value if isfinite(value) else None
                    finite_reductions = [
                        selected_rows[0]["belief_area_before"][str(c)] - after_area[str(c)]
                        for c in range(1, self.n_channels + 1)
                        if selected_rows[0]["belief_area_before"][str(c)] is not None and
                        after_area[str(c)] is not None
                    ]
                    for row in selected_rows:
                        row["realized"] = {
                            "time_until_next_replan": float(max(0.0, res.state.virtual_time_s - start_time)),
                            "distance": float(realized_distance),
                            "certificate_gain": float(sum(
                                1 for c in range(1, self.n_channels + 1)
                                if status_before[c] != self.belief[c].status
                            )),
                            "belief_area_after": after_area,
                            "belief_area_reduction": (float(sum(finite_reductions))
                                                       if finite_reductions else None),
                        }
                self.macro_trace.append({
                    "step": int(self._steps + 1),
                    "start_time_s": float(start_time),
                    "end_time_s": float(res.state.virtual_time_s),
                    "elapsed_time_s": float(max(0.0, res.state.virtual_time_s - start_time)),
                    "option_type": macro.action_type.value,
                    "target": [float(macro.target[0]), float(macro.target[1])],
                    "scan_channels": [int(c) for c in (macro.scan_channels or ())],
                    "clear_channel": None if macro.clear_channel is None else int(macro.clear_channel),
                    "start_state": {"x": float(self.state.x), "y": float(self.state.y), "channel": int(self.state.channel)},
                    "end_state": {"x": float(res.state.x), "y": float(res.state.y), "channel": int(res.state.channel)},
                    "start_belief": {str(c): s.value for c, s in status_before.items()},
                    "primitive_count": len(res.primitives),
                    "stopped_early": bool(res.stopped_early),
                    "finished": bool(res.finished),
                })
                for pr in res.primitives:
                    prim = pr.primitive
                    self.primitive_trace.append({
                        "time_s": float(pr.virtual_time_s),
                        "action": prim.kind.value,
                        "channel": int(prim.channel),
                        "target": [float(prim.target[0]), float(prim.target[1])],
                        "cleared": bool(pr.cleared),
                        "observation": None if pr.observation is None else {
                            "kind": pr.observation.kind.value,
                            "svd_deg": pr.observation.svd_deg,
                            "time": float(pr.observation.time),
                        },
                    })
                self.state = res.state
                self._steps += 1
                end_belief = tuple(sorted((c, self.belief[c].status.value)
                                          for c in range(1, self.n_channels + 1)))
                self.transitions.append(OptionTransition(
                    start_belief=start_belief,
                    option_type=macro.action_type.value,
                    target=(float(macro.target[0]), float(macro.target[1])),
                    channel_batch=tuple(int(c) for c in macro.scan_channels),
                    elapsed_time_s=max(0.0, self.state.virtual_time_s - start_time),
                    primitive_count=len(res.primitives),
                    resulting_belief=end_belief,
                    terminal=bool(res.finished or res.stopped_early),
                    full_clear=bool(all(self.belief[c].is_resolved
                                        for c in range(1, self.n_channels + 1))),
                ))
                self._emit_event(Way4Event(
                    EventType.BATCH_COMPLETED, self.state.virtual_time_s,
                    payload={"option": macro.action_type.value,
                             "primitive_count": len(res.primitives)},
                ))
                for c, before_status in status_before.items():
                    after_status = self.belief[c].status
                    if before_status not in (ChannelStatus.DETECTED, ChannelStatus.INITIALIZED,
                                             ChannelStatus.LOCALIZED, ChannelStatus.CLEARED,
                                             ChannelStatus.PRESENT_UNOBSERVED) and after_status in (
                                             ChannelStatus.DETECTED, ChannelStatus.INITIALIZED,
                                             ChannelStatus.LOCALIZED, ChannelStatus.CLEARED):
                        self._emit_event(Way4Event(
                            EventType.POSITIVE_DISCOVERY, self.state.virtual_time_s, c
                        ))
                    if after_status == ChannelStatus.INITIALIZED and before_status != after_status:
                        self._emit_event(Way4Event(EventType.INITIALIZED, self.state.virtual_time_s, c))
                    if after_status == ChannelStatus.LOCALIZED and before_status != after_status:
                        self._emit_event(Way4Event(EventType.LOCALIZED, self.state.virtual_time_s, c))
                    if after_status == ChannelStatus.CLEARED and before_status != after_status:
                        self._emit_event(Way4Event(EventType.SOURCE_CLEARED, self.state.virtual_time_s, c))
                    if after_status == ChannelStatus.ABSENT_CERTIFIED and before_status != after_status:
                        self._emit_event(Way4Event(EventType.CHANNEL_CERTIFIED_EMPTY, self.state.virtual_time_s, c))
                if self.certificate.cardinality_state().q_min == self.certificate.cardinality_state().q_max:
                    self._emit_event(Way4Event(
                        EventType.CARDINALITY_CLOSURE, self.state.virtual_time_s,
                        payload={"q_min": self.certificate.cardinality_state().q_min,
                                 "q_max": self.certificate.cardinality_state().q_max},
                    ))
                if any(self.certificate.heuristic_coverage_ratio(c) >= 0.995
                       for c in range(1, self.n_channels + 1)):
                    self._emit_event(Way4Event(EventType.COVERAGE_THRESHOLD, self.state.virtual_time_s))

                if res.finished:
                    # env signalled the episode is over: a deadline unless we chose
                    # EXIT ourselves (the only legitimate self-stop, §11).
                    hit_deadline = macro.action_type != MacroActionType.EXIT
                    exited = macro.action_type == MacroActionType.EXIT
                    break
                if res.stopped_early:
                    hit_deadline = True
                    break

                # No-progress guard (§11 fallback; the full SafetyShield ladder is
                # M9). If nothing measurable advances for stall_limit consecutive
                # macros — no channel resolved, no new source found, no new coverage,
                # no MEC shrink — the loop is livelocked; stop and report honestly
                # rather than spin to the step cap. A healthy tick always moves one of
                # these, so this never fires on real progress, and it cannot fake a
                # success: full_clear still reads the true resolved set. The root cause
                # of the known stall (a zero-gain no-op winning on cost) is removed in
                # the generator; this only bounds any residual degenerate loop.
                key = self._progress_key()
                progressed = key != self._last_progress_key
                if progressed:
                    self._last_progress_time_s = float(self.state.virtual_time_s)
                    self._last_progress_distance_m = float(self._move_m)
                    self._decisions_since_progress = 0
                else:
                    self._decisions_since_progress += 1
                    self._no_progress_time_s += max(0.0, float(self.state.virtual_time_s - start_time))
                self.decision_audit.append({
                    "step": int(self._steps), "virtual_time_before": float(start_time),
                    "virtual_time_after": float(self.state.virtual_time_s),
                    "delta_virtual_time_s": float(self.state.virtual_time_s - start_time),
                    "delta_move_time_s": float(self._time_move_s - time_buckets_before[0]),
                    "delta_measure_time_s": float(self._time_measure_s - time_buckets_before[1]),
                    "delta_switch_time_s": float(self._time_switch_s - time_buckets_before[2]),
                    "delta_clear_time_s": float(self._time_clear_s - time_buckets_before[3]),
                    "delta_move_distance_m": float(self._move_m - move_before),
                    "no_progress": not progressed,
                    "time_since_last_progress_s": float(self.state.virtual_time_s - self._last_progress_time_s),
                    "distance_since_last_progress_m": float(self._move_m - self._last_progress_distance_m),
                    "decisions_since_last_progress": int(self._decisions_since_progress),
                })
                if hasattr(self.planner, "record_progress"):
                    self.planner.record_progress(key)
                if key == self._last_progress_key:
                    self._stall += 1
                    if self._stall >= self.stall_limit:
                        # Adaptive batching is an optimisation layer. Give the
                        # deterministic batch policy one recovery window before
                        # declaring a genuine stall. This never mutates belief
                        # or certificates; only the batching policy changes.
                        if self.adaptive_scan and not self._adaptive_recovery_used:
                            self.adaptive_scan = False
                            self._adaptive_recovery_used = True
                            self._stall = 0
                            self._last_progress_key = key
                            continue
                        # §11 Way3-homing fallback before conceding. The omni NBV has
                        # livelocked on a directional source: its viewpoints fall in
                        # the source's blind 180° arc → NO_SIGNAL, and 禁止5 bars
                        # shrinking F_c from a directional NO_SIGNAL, so the belief
                        # freezes (the P4 no_progress_stall). Way3's homing uses only
                        # positive info — it is sound for both problems (Way3 runs it
                        # on P3 at 100%) — so engaging it here can only rescue an
                        # episode that was already going to abort; a currently-passing
                        # episode never stalls, so this is 禁止10-safe.
                        if self._home_stuck_channels():
                            self._stall = 0
                            self._last_progress_key = self._progress_key()
                            continue
                        error = "no_progress_stall"
                        break
                else:
                    self._stall = 0
                    self._last_progress_key = key
        except Exception as e:  # a crash must still yield a (failed) result row
            tail = traceback.format_exc().strip().splitlines()[-3:]
            error = f"{type(e).__name__}: {e} [{'; '.join(tail)}]"

        return self._result(exited=exited, hit_deadline=hit_deadline, error=error)

    # -- planning -----------------------------------------------------------

    def _progress_key(self) -> Tuple:
        """A cheap snapshot of forward progress. Changes whenever the episode
        genuinely advances — a channel resolved, a new source proven PRESENT, new
        arena covered for an UNKNOWN channel, a DETECTED region's MEC shrunk, or a new
        backbone anchor scanned for an UNKNOWN channel (the §11 completion fallback) —
        and is otherwise identical tick to tick. Used only by the no-progress guard; it
        certifies nothing and drives no marking (Invariant B / 禁止6)."""
        resolved = 0
        cov = 0.0
        mec = 0.0
        anchors = 0
        for c in range(1, self.n_channels + 1):
            b = self.belief[c]
            if b.is_resolved:
                resolved += 1
            if b.status == ChannelStatus.UNKNOWN:
                cov += self.certificate.heuristic_coverage_ratio(c)
                anchors += self.certificate.backbone_anchor_progress(c)
            elif b.status in (ChannelStatus.DETECTED, ChannelStatus.INITIALIZED) and b.mec_radius != float("inf"):
                mec += b.mec_radius
        return (resolved, self.certificate.present_count(), round(cov, 6), round(mec, 2), anchors)

    def _scan_mode(self) -> SchedulerMode:
        """EARLY while UNKNOWN channels remain broadly unexplored; VERIFICATION once
        we're mopping up the last few uncovered UNKNOWN channels (§6.7). Heuristic —
        it only shapes scan batches, never correctness."""
        unknown = self.belief.unknown_channels()
        if not unknown:
            return SchedulerMode.VERIFICATION
        # if most channels are already resolved, switch to completion mode
        resolved = sum(
            1 for c in range(1, self.n_channels + 1) if self.belief[c].is_resolved
        )
        if resolved >= self.n_channels - max(2, self.n_channels // 5):
            return SchedulerMode.VERIFICATION
        return SchedulerMode.EARLY

    def _choose_macro(self) -> Optional[MacroCandidate]:
        cands = self.generator.generate(
            self.belief, self.certificate, self.state, scan_mode=self._scan_mode()
        )
        # Build a conservative open-route opportunity view from the current
        # validated candidate pool. Waiting only annotates future service points;
        # the legacy candidates remain available to the math planner.
        opportunities = []
        for cand in cands:
            if cand.action_type == MacroActionType.EXIT:
                continue
            services = tuple((int(ch), cand.action_type.value)
                             for ch in (cand.scan_channels or
                                         ((cand.clear_channel,) if cand.clear_channel is not None else ())))
            opportunities.append(ServiceOpportunity(
                (float(cand.target[0]), float(cand.target[1])), services,
                float(cand.expected_time),
                float(cand.meta.get("tspn_radius", 0.0)),
            ))
        route_plan = self.unified_route.plan(self.state.pos, opportunities)
        self.last_unified_route = route_plan
        route = [self.state.pos] + list(route_plan.order)
        dedicated = {}
        for c in cands:
            ch = c.meta.get("refine_channel")
            if ch is not None:
                dedicated[int(ch)] = min(dedicated.get(int(ch), float("inf")),
                                          float(c.expected_time))
        assignments = self.task_pool.assign_route_opportunities(
            route, dedicated,
            viable=lambda ch, p: any(
                c.target == p and int(ch) in tuple(c.scan_channels) for c in cands
            ),
        )
        # Expose waiting metadata to diagnostics and future route-aware ranking
        # without removing the legacy candidate (safety superset).
        for cand in cands:
            ch = cand.meta.get("refine_channel")
            if ch is not None and int(ch) in self.task_pool.waiting:
                cand.meta["wait_state"] = self.task_pool.snapshot()[str(int(ch))]
            if cand.target in assignments.values():
                cand.meta["assigned_wait_channels"] = tuple(
                    ch for ch, point in assignments.items() if point == cand.target
                )
        if not cands:
            return None
        # EXIT guard (§11): drop any EXIT the planner proposes unless the certificate
        # manager force-confirms full resolution. Belief was already reconciled above,
        # so a surviving EXIT candidate is legitimate; this is defence in depth.
        if not self._exit_allowed():
            cands = [c for c in cands if c.action_type != MacroActionType.EXIT]
            if not cands:
                return None
        for cand in cands:
            ch = cand.meta.get("refine_channel") or cand.clear_channel
            if ch is not None:
                first = self.belief[int(ch)].first_detect_time
                if first is not None:
                    cand.meta["task_age_s"] = max(0.0, self.state.virtual_time_s - first)
        result = self.planner.plan(self.belief, self.certificate, self.state, cands)
        selected = result.best
        evaluations = {id(e.candidate): e for e in getattr(result, "evaluations", ())}
        route_positions = {tuple(p): i for i, p in enumerate(route_plan.order)}
        belief_area = {}
        for c in range(1, self.n_channels + 1):
            value = float(getattr(self.belief[c], "area", 0.0))
            belief_area[str(c)] = value if isfinite(value) else None
        decision_id = len(self.decision_trace)
        for idx, cand in enumerate(cands):
            ev = evaluations.get(id(cand))
            self.decision_trace.append({
                "decision_id": decision_id,
                "step": int(self._steps + 1),
                "candidate_id": idx,
                "candidate_type": cand.action_type.value,
                "target": [float(cand.target[0]), float(cand.target[1])],
                "channel": None if cand.clear_channel is None else int(cand.clear_channel),
                "channels": [int(c) for c in cand.scan_channels],
                "hard_legal": True,
                "certificate_gain": float(cand.certificate_gain),
                "exploration_gain": float(cand.exploration_gain),
                "refinement_gain": float(cand.refinement_gain),
                "belief_area_before": belief_area,
                "information_gain_est": float(cand.meta.get("soft_information_gain", 0.0)),
                "immediate_time_cost": float(cand.expected_time),
                "route_position": route_positions.get(tuple(cand.target)),
                "route_synergy": float(getattr(cand, "route_synergy", 0.0)),
                "horizon_value": None if ev is None else float(ev.future_cost),
                "q_value": None if ev is None else float(ev.q_value),
                "math_score": None if ev is None else float(ev.q_value),
                "rl_residual": float(cand.meta.get("rl_residual", 0.0)),
                "selected": cand is selected,
                "selected_without_rl": cand is selected,
                "selected_with_rl": cand is selected,
                "rl_changed_action": False,
                "route_length": float(route_plan.length),
                "realized": None,
            })
        if hasattr(self.planner, "record_progress"):
            base_result = self.planner.base.plan(self.belief, self.certificate, self.state, cands)
            base_best = base_result.best
            for row in self.decision_trace:
                if row["step"] == int(self._steps + 1):
                    row["selected_without_rl"] = cands[row["candidate_id"]] is base_best
                    row["selected_with_rl"] = cands[row["candidate_id"]] is result.best
                    row["rl_changed_action"] = (base_best is not result.best)
            self.planner.record_progress(self._progress_key(), float(result.q_value),
                                         float(base_result.q_value))
            if self.planner.watchdog.fallback:
                return base_result.best
        return result.best

    def _exit_allowed(self) -> bool:
        """Force-run the three §6.5 sources for every not-yet-resolved channel; EXIT
        is legal only if that resolves them all (§11 EXIT guard)."""
        for c in range(1, self.n_channels + 1):
            if self.belief[c].is_resolved:
                continue
            if not self.certificate.is_absent_certified(c, force=True):
                return False
        return True

    def _execute_exit(self) -> bool:
        exit_macro = MacroCandidate(MacroActionType.EXIT, self.state.pos)
        res = self.executor.execute(exit_macro, self.state, self.time_budget_s)
        self.state = res.state
        self._steps += 1
        return True

    # -- Way3-homing fallback (§11) ----------------------------------------

    def _home_stuck_channels(self) -> bool:
        """Run Way3's sound homing on every DETECTED (localized-but-not-cleared)
        channel and report whether the belief advanced.

        Invoked only from the no-progress guard (run()), i.e. when the omni planner
        has livelocked — typically a directional source whose blind arc keeps
        returning NO_SIGNAL. The ``HomingController`` drives the env through the
        executor's folding path, so every measurement and clear updates belief +
        certificate exactly as a planned macro would (it marks nothing itself; a
        clear is a real env hit — Invariants B/D). ``home_and_clear`` respects the
        time budget and stops on the env deadline.

        Returns True iff ``_progress_key`` changed — a channel cleared, or a DETECTED
        region shrank — so the loop resumes; False means truly stuck, and the caller
        aborts honestly. Homing primitives are replayed through ``_account`` from the
        pre-homing pose, so move/measure/switch/clear counters and route events include
        fallback work as well as the authoritative advanced environment clock."""
        detected = [
            c
            for c in range(1, self.n_channels + 1)
            if self.belief[c].status in (ChannelStatus.DETECTED, ChannelStatus.INITIALIZED)
        ]
        if not detected:
            return False
        before = self._progress_key()
        for c in detected:
            before_trace = len(self.executor.interactive_trace)
            homing_start = self.state
            self.state, _cleared, finished = self._homing.home_and_clear(
                c, self.state, self.time_budget_s
            )
            extra = self.executor.interactive_trace[before_trace:]
            if extra:
                # _account replays from ``self.state``; use the pose before
                # homing so its first leg is not silently dropped.
                homing_end = self.state
                self.state = homing_start
                self._account(MacroCandidate(MacroActionType.PURSUE, self.state.pos),
                              type("R", (), {"primitives": extra})(),
                              {i: self.belief[i].status for i in range(1, self.n_channels + 1)})
                self.state = homing_end
            if finished:
                break
        return self._progress_key() != before

    # -- metrics ------------------------------------------------------------

    def _account(self, macro: MacroCandidate, res, status_before=None) -> None:
        # Reconstruct travelled distance + primitive counts from the realised
        # primitive sequence (the env clock is authoritative for time; this is only
        # for the metric row). Walk the primitives in order from the pre-macro pose.
        # Also record a RouteEvent per primitive for the P0 #9 route decomposition
        # (way4.metrics) — pure bookkeeping, drives nothing.
        walk = (self.state.x, self.state.y)
        chan = self.state.channel
        prev_time = float(self.state.virtual_time_s)
        for pr in res.primitives:
            prim = pr.primitive
            tgt = (float(prim.target[0]), float(prim.target[1]))
            self._move_m += _dist(walk, tgt)
            move_time = _dist(walk, tgt) / max(float(self.cost.speed), 1e-9)
            elapsed = max(0.0, float(pr.virtual_time_s) - prev_time)
            self._time_move_s += move_time
            walk = tgt
            if prim.kind.name == "MEASURE":
                self._n_measure += 1
                if pr.observation is not None and getattr(pr.observation, "is_no_signal", False):
                    self._n_empty_scan += 1
                switched = int(prim.channel) != int(chan)
                if switched:
                    self._n_switch += 1
                    self._time_switch_s += float(self.cost.switch_s)
                    chan = int(prim.channel)
                self._time_measure_s += max(0.0, elapsed - move_time - (float(self.cost.switch_s) if switched else 0.0))
                self._route_events.append(RouteEvent(
                    loc=tgt, kind="MEASURE", channel=int(prim.channel),
                    role=self._measure_role(int(prim.channel), status_before),
                ))
            elif prim.kind.name == "CLEAR":
                self._n_clear += 1
                self._time_clear_s += max(0.0, elapsed - move_time)
                if pr.cleared:
                    self._n_clear_hit += 1
                # /clear does NOT change measuring channel (§1.1)
                self._route_events.append(RouteEvent(
                    loc=tgt, kind="CLEAR", channel=int(prim.channel),
                    role=ROLE_CLEAR, cleared=bool(pr.cleared),
                ))
            prev_time = float(pr.virtual_time_s)

    def _measure_role(self, channel: int, status_before) -> str:
        """Planner-agnostic role of a MEASURE for the route decomposition: a known
        source at macro start (DETECTED/LOCALIZED) -> localization, else coverage.
        Reads the pre-macro belief snapshot so the role reflects intent; drives
        nothing (Invariant B untouched — this never marks or certifies anything)."""
        st = status_before.get(channel) if status_before is not None else None
        if st is None:
            st = self.belief[channel].status
        if st in (ChannelStatus.DETECTED, ChannelStatus.INITIALIZED, ChannelStatus.LOCALIZED):
            return ROLE_LOCALIZE
        return ROLE_COVERAGE

    def _result(self, exited: bool, hit_deadline: bool, error: Optional[str]) -> EpisodeResult:
        # final reconciliation so resolved-count reflects all certifiable channels
        try:
            self.certificate.apply_certifications(self.belief, force=True)
        except Exception:
            pass
        cleared = sum(
            1 for c in range(1, self.n_channels + 1)
            if self.belief[c].status == ChannelStatus.CLEARED
        )
        present = self.certificate.present_count()
        resolved = sum(
            1 for c in range(1, self.n_channels + 1) if self.belief[c].is_resolved
        )
        success = exited and resolved == self.n_channels and error is None
        rm = compute_route_metrics(self._route_events, self._start_pos)
        efficiency = compute_efficiency_metrics(
            total_time_s=float(self.state.virtual_time_s),
            no_progress_time_s=float(self._no_progress_time_s),
            move_distance_m=rm.total_distance,
            backtrack_m=rm.backtrack_m,
            repeated_edge_m=rm.repeated_edge_m,
            unnecessary_return_m=rm.unnecessary_return_m,
            scan_time_s=self._time_measure_s + self._time_switch_s,
            useful_observations=max(0, self._n_measure - self._n_empty_scan),
        )
        diagnostics = {}
        for c in range(1, self.n_channels + 1):
            b = self.belief[c]
            first = b.first_detect_time
            diagnostics[str(c)] = {
                "first_detect_s": first, "localized_s": b.localized_time,
                "cleared_s": b.cleared_time,
                "detect_to_clear_s": (b.cleared_time - first) if first is not None and b.cleared_time is not None else None,
                "status": b.status.value,
            }
        top5 = sorted(
            (dict(v, channel=int(c)) for c, v in diagnostics.items()
             if v["detect_to_clear_s"] is not None),
            key=lambda x: (-float(x["detect_to_clear_s"]), x["channel"]),
        )[:5]
        total_time = float(self.state.virtual_time_s)
        accounted = (self._time_move_s + self._time_measure_s
                     + self._time_switch_s + self._time_clear_s)
        time_other = max(0.0, total_time - accounted)
        # If explicit buckets exceed the authoritative clock, expose the
        # defect instead of hiding it in a negative ``other`` bucket.
        accounting_error = total_time - (accounted + time_other)
        return EpisodeResult(
            success=success,
            virtual_time_s=self.state.virtual_time_s,
            move_distance_m=self._move_m,
            n_measure=self._n_measure,
            n_switch=self._n_switch,
            n_clear=self._n_clear,
            n_clear_hit=self._n_clear_hit,
            cleared=cleared,
            present=present,
            resolved=resolved,
            n_channels=self.n_channels,
            steps=self._steps,
            exited=exited,
            hit_deadline=hit_deadline,
            error=error,
            services_per_stop=rm.services_per_stop,
            pure_refine_travel=rm.pure_refine_travel,
            certificate_only_travel=rm.certificate_only_travel,
            revisit_distance=rm.revisit_distance,
            shared_stop_ratio=rm.shared_stop_ratio,
            clear_insertion_delta=rm.clear_insertion_delta,
            route_n_stops=rm.n_stops,
            route_n_services=rm.n_services,
            total_distance_m=rm.total_distance,
            clear_distance_m=rm.clear_distance,
            n_longjump=rm.n_longjump,
            n_crossing=rm.n_crossing,
            n_empty_scan=self._n_empty_scan,
            source_diagnostics=diagnostics,
            longest_waiting_sources=top5,
            time_move_s=self._time_move_s,
            time_measure_s=self._time_measure_s,
            time_switch_s=self._time_switch_s,
            time_clear_s=self._time_clear_s,
            time_other_s=time_other,
            time_accounting_error_s=accounting_error,
            equivalent_route_cost_m=self._move_m + 30.0 * self._n_measure,
            planner_version=("way4-final-ppo-v1" if self.planner_mode == "final_ppo"
                             else "way4-route-math-v1"),
            metric_schema_version="way4-metrics-v2",
            decision_trace=list(self.decision_trace),
            decision_summary=summarize_decision_trace(self.decision_trace),
            decision_audit=list(self.decision_audit),
            no_progress_time_s=float(self._no_progress_time_s),
            illegal_clear=sum((not item['legal']) and not item.get('rejected', False)
                              for item in self.executor.clear_audit),
            safety_violation=sum((not item['legal']) and not item.get('rejected', False)
                                 for item in self.executor.clear_audit),
            clear_audit=list(self.executor.clear_audit),
            efficiency_metrics=efficiency.__dict__,
            backtrack_m=rm.backtrack_m,
            repeated_edge_m=rm.repeated_edge_m,
            unnecessary_return_m=rm.unnecessary_return_m,
        )


def _dist(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
