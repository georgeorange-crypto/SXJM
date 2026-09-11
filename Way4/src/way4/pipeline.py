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
from typing import List, Optional, Sequence, Tuple

from .belief import BeliefState, ChannelStatus
from .certificate import CertificateManager
from .channels import ChannelScheduler, SchedulerMode
from .core import AnalyticalCostModel, MacroActionType, MacroCandidate, RobotState
from .executor import MacroExecutor
from .planner import CandidateGenerator, RecedingHorizonPlanner


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
        time_budget_s: float = float("inf"),
        max_steps: int = 2000,
        generator: Optional[CandidateGenerator] = None,
        planner: Optional[RecedingHorizonPlanner] = None,
        certificate: Optional[CertificateManager] = None,
        cost_model: Optional[AnalyticalCostModel] = None,
        opportunistic_clear: bool = True,
        initial_state: Optional[RobotState] = None,
        stall_limit: int = 16,
    ) -> None:
        self.env = env
        self.n_channels = int(n_channels)
        self.time_budget_s = float(time_budget_s)
        self.max_steps = int(max_steps)
        self.stall_limit = int(stall_limit)

        self.belief = BeliefState(n_channels=n_channels)
        self.certificate = certificate or CertificateManager(n_channels=n_channels)
        self.cost = cost_model or AnalyticalCostModel()
        self.generator = generator or CandidateGenerator(cost_model=self.cost)
        self.planner = planner or RecedingHorizonPlanner()
        self.executor = MacroExecutor(
            env,
            belief=self.belief,
            certificate=self.certificate,
            cost_model=self.cost,
            opportunistic_clear=opportunistic_clear,
        )
        # §1.1: initial pose (0,0), initial measuring channel 1.
        self.state = initial_state or RobotState(0.0, 0.0, 1, 0)

        # metric accumulators
        self._move_m = 0.0
        self._n_measure = 0
        self._n_switch = 0
        self._n_clear = 0
        self._n_clear_hit = 0
        self._steps = 0
        # no-progress guard state
        self._last_progress_key: Optional[Tuple] = None
        self._stall = 0

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

                if self.belief.all_resolved():
                    # legitimate completion: everything CLEARED ∨ ABSENT_CERTIFIED.
                    exited = self._execute_exit()
                    break

                # Plan: rank this belief's macros under the receding-horizon Ĵ.
                macro = self._choose_macro()
                if macro is None:
                    error = "no_candidate"      # M9 fallback replaces this abort
                    break

                # Execute one macro; fold observations back into belief+certificate.
                res = self.executor.execute(macro, self.state, self.time_budget_s)
                self._account(macro, res)
                self.state = res.state
                self._steps += 1

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
                if key == self._last_progress_key:
                    self._stall += 1
                    if self._stall >= self.stall_limit:
                        error = "no_progress_stall"
                        break
                else:
                    self._stall = 0
                    self._last_progress_key = key
        except Exception as e:  # a crash must still yield a (failed) result row
            error = f"{type(e).__name__}: {e}"

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
            elif b.status == ChannelStatus.DETECTED and b.mec_radius != float("inf"):
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
        if not cands:
            return None
        # EXIT guard (§11): drop any EXIT the planner proposes unless the certificate
        # manager force-confirms full resolution. Belief was already reconciled above,
        # so a surviving EXIT candidate is legitimate; this is defence in depth.
        if not self._exit_allowed():
            cands = [c for c in cands if c.action_type != MacroActionType.EXIT]
            if not cands:
                return None
        result = self.planner.plan(self.belief, self.certificate, self.state, cands)
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

    # -- metrics ------------------------------------------------------------

    def _account(self, macro: MacroCandidate, res) -> None:
        # Reconstruct travelled distance + primitive counts from the realised
        # primitive sequence (the env clock is authoritative for time; this is only
        # for the metric row). Walk the primitives in order from the pre-macro pose.
        walk = (self.state.x, self.state.y)
        chan = self.state.channel
        for pr in res.primitives:
            prim = pr.primitive
            tgt = (float(prim.target[0]), float(prim.target[1]))
            self._move_m += _dist(walk, tgt)
            walk = tgt
            if prim.kind.name == "MEASURE":
                self._n_measure += 1
                if int(prim.channel) != int(chan):
                    self._n_switch += 1
                    chan = int(prim.channel)
            elif prim.kind.name == "CLEAR":
                self._n_clear += 1
                if pr.cleared:
                    self._n_clear_hit += 1
                # /clear does NOT change measuring channel (§1.1)

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
        )


def _dist(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
