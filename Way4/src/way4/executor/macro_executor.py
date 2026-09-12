"""Macro executor (DESIGN.md §7, §13 pipeline tail).

Expands a ``MacroCandidate`` into env primitives, applies them, folds every
primitive observation straight into the belief and the certificate manager, and
tracks the robot state. The environment (offline_sim on the way4 branch; the Way3
engine for cross-checks) stays authoritative for the realised virtual clock — the
cost model is used only to pre-check the time budget ("超时即停", §7).

Batch scan (§7): a scan macro measures several channels at one waypoint, so one
move is amortised over many measurements; opportunistic clear on a ``near`` return
is available behind a flag (the full clear-guard logic is M9's safety layer).

The ``Env`` protocol is expressed in Way4 terms (returning ``Observation`` and a
virtual clock); ``Way3EngineAdapter`` bridges Way3's raw engine to it without Way4
depending on Way3's response format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple

from ..core.actions import MacroCandidate, Primitive, PrimitiveKind
from ..core.cost import AnalyticalCostModel, RobotState, _us
from ..core.observation import Observation, ObservationKind

#: Channels that must NEVER be scanned again — the exact criterion the channel
#: scheduler masks at selection time (``ChannelScheduler.channel_value`` returns
#: ``scannable=False`` for these). P0-D re-checks it per primitive so a channel
#: resolved MID-batch (an opportunistic clear -> CLEARED, or a certificate that
#: certifies ABSENT / a bearing that drops MEC to LOCALIZED) is not wastefully
#: re-measured by the rest of a batch that was planned before it resolved.
#: DETECTED and UNKNOWN are deliberately absent: they still carry refine / coverage
#: value, and pruning them would drop VOI-positive work (§8, §16 禁令10).
from ..belief import ChannelStatus

_PRUNE_STATES = frozenset(
    {ChannelStatus.CLEARED, ChannelStatus.ABSENT_CERTIFIED, ChannelStatus.LOCALIZED}
)


class Env(Protocol):
    """Minimal environment interface in Way4 terms."""

    def measure(self, x: float, y: float, channel: int) -> Tuple[Optional[Observation], float]:
        """``(observation, virtual_time_s)``; observation is ``None`` iff the run
        finished (deadline) before the command executed."""
        ...

    def clear(self, x: float, y: float, channel: int) -> Tuple[bool, float]:
        """``(hit, virtual_time_s)`` — ``hit`` True iff a source was neutralised."""
        ...


@dataclass
class PrimitiveResult:
    primitive: Primitive
    observation: Optional[Observation] = None
    cleared: bool = False
    virtual_time_s: float = 0.0


@dataclass
class ExecutionResult:
    """Outcome of executing one macro."""

    state: RobotState
    primitives: List[PrimitiveResult] = field(default_factory=list)
    stopped_early: bool = False       # time budget hit before all primitives ran
    finished: bool = False            # env signalled finish (EXIT / deadline)

    @property
    def observations(self) -> List[Observation]:
        return [r.observation for r in self.primitives if r.observation is not None]

    @property
    def n_cleared(self) -> int:
        return sum(1 for r in self.primitives if r.cleared)


class MacroExecutor:
    """Applies macro actions against an ``Env``, updating belief + certificate."""

    def __init__(
        self,
        env: Env,
        belief=None,
        certificate=None,
        cost_model: Optional[AnalyticalCostModel] = None,
        opportunistic_clear: bool = False,
        adaptive_stop: bool = False,
    ) -> None:
        self.env = env
        self.belief = belief
        self.certificate = certificate
        self.cost = cost_model or AnalyticalCostModel()
        self.opportunistic_clear = opportunistic_clear
        # P0-D: when True, once every remaining channel in the batch is resolved the
        # macro stops early (nothing left worth measuring). Default False keeps the
        # conservative "run the batch, only skip resolved primitives" posture. Never
        # stops while a still-valuable (UNKNOWN/DETECTED) channel remains (§16 禁令10).
        self.adaptive_stop = adaptive_stop

    # -- public API --------------------------------------------------------

    def execute(
        self, macro: MacroCandidate, state: RobotState, time_budget_s: float = float("inf")
    ) -> ExecutionResult:
        """Run ``macro`` from ``state``. Stops before any primitive whose predicted
        cost would push the virtual clock past ``time_budget_s`` (超时即停)."""
        results: List[PrimitiveResult] = []
        finished = False
        stopped = False

        prims = macro.primitives()
        for i, prim in enumerate(prims):
            if prim.kind == PrimitiveKind.EXIT:
                finished = True
                break
            # P0-D in-batch mask prune: skip a MEASURE whose channel is already
            # resolved on the LIVE belief/certificate (e.g. cleared by this batch's
            # own opportunistic clear, or localised). Judged BEFORE the budget check
            # — a skipped primitive costs nothing, so it neither consumes budget nor
            # trips 超时即停. Never prunes UNKNOWN/DETECTED (§16 禁令10).
            if prim.kind == PrimitiveKind.MEASURE and self._is_resolved(prim.channel):
                # P0-D adaptive stop (opt-in): if NOTHING valuable is left in the
                # batch (every remaining primitive is a resolved-channel MEASURE),
                # end the macro so the pipeline can replan on the changed belief
                # rather than spin through the tail. Guarded to keep any CLEAR or any
                # unresolved (valuable) channel — so it never abandons full-clear work
                # (§16 禁令10); it defers nothing that a replan won't re-propose.
                if self.adaptive_stop and self._batch_tail_exhausted(prims, i):
                    break
                continue
            if not self._within_budget(state, prim, time_budget_s):
                stopped = True
                break
            if prim.kind == PrimitiveKind.MEASURE:
                state, pr, finished = self._do_measure(state, prim)
                results.append(pr)
                if finished:
                    break
                # opportunistic clear (§7): a 'near' return means the source is
                # within 5 m < 20 m, a guaranteed hit — clear it now if enabled.
                if self.opportunistic_clear and pr.observation is not None and pr.observation.is_near:
                    if self._within_budget(state, Primitive(PrimitiveKind.CLEAR, prim.target, prim.channel), time_budget_s, hit=True):
                        state, cpr, finished = self._do_clear(state, Primitive(PrimitiveKind.CLEAR, prim.target, prim.channel))
                        results.append(cpr)
                        if finished:
                            break
            elif prim.kind == PrimitiveKind.CLEAR:
                state, pr, finished = self._do_clear(state, prim)
                results.append(pr)
                if finished:
                    break

        return ExecutionResult(state=state, primitives=results, stopped_early=stopped, finished=finished)

    # -- primitives --------------------------------------------------------

    def _do_measure(self, state: RobotState, prim: Primitive):
        obs, vts = self.env.measure(prim.target[0], prim.target[1], prim.channel)
        if obs is None:
            # deadline: the command did not execute; clock unchanged bar env's report
            return state.at_time_s(vts), PrimitiveResult(prim, None, False, vts), True
        # env sets pose + measuring channel; env clock is authoritative.
        nxt = RobotState(float(prim.target[0]), float(prim.target[1]), int(prim.channel), _us(vts))
        self._fold_observation(prim.channel, prim.target, obs)
        return nxt, PrimitiveResult(prim, obs, False, vts), False

    def _do_clear(self, state: RobotState, prim: Primitive):
        hit, vts = self.env.clear(prim.target[0], prim.target[1], prim.channel)
        # pose moves; measuring channel is UNCHANGED (§1.1 crux).
        nxt = RobotState(float(prim.target[0]), float(prim.target[1]), int(state.channel), _us(vts))
        if hit:
            self._fold_clear(prim.channel)
        return nxt, PrimitiveResult(prim, None, hit, vts), False

    # -- belief / certificate side effects ---------------------------------

    def _fold_observation(self, channel: int, point, obs: Observation) -> None:
        if self.belief is not None:
            cb = self.belief[channel]
            if obs.kind is ObservationKind.NO_SIGNAL:
                cb.record_no_signal(point, obs.time)
            elif obs.kind is ObservationKind.BEARING:
                cb.record_bearing(point, float(obs.svd_deg), obs.time)
            elif obs.kind is ObservationKind.NEAR:
                cb.record_near(point, obs.time)
        if self.certificate is not None:
            self.certificate.record_observation(channel, point, obs)

    def _fold_clear(self, channel: int) -> None:
        if self.belief is not None:
            self.belief[channel].mark_cleared()
        if self.certificate is not None:
            self.certificate.mark_cleared(channel)

    # -- budget ------------------------------------------------------------

    def _within_budget(self, state: RobotState, prim: Primitive, budget_s: float, hit: bool = False) -> bool:
        if budget_s == float("inf"):
            return True
        if prim.kind == PrimitiveKind.MEASURE:
            cost_us = self.cost.measure_us(state, prim.target, prim.channel)
        elif prim.kind == PrimitiveKind.CLEAR:
            cost_us = self.cost.clear_us(state, prim.target, hit)
        else:
            cost_us = 0
        return (state.vt_us + cost_us) / 1_000_000.0 <= budget_s + 1e-9

    # -- in-batch mask prune (P0-D, §7 "每 primitive obs 立即更新 belief") -------

    def _is_resolved(self, channel: int) -> bool:
        """True iff ``channel`` must not be scanned again, judged on the LIVE belief
        and certificate (not the pre-batch snapshot). A channel resolved earlier IN
        this same batch is then skipped instead of wastefully re-measured.

        Two sound, read-only sources:
          * belief status in ``_PRUNE_STATES`` (CLEARED via an opportunistic clear,
            or LOCALIZED once a bearing drops MEC below the clear threshold) — the
            exact non-scannable set the scheduler masks at selection time;
          * ``absent_by_cardinality`` — the pigeonhole cascade: a positive detection
            mid-batch that makes the 16th channel PRESENT leaves every remaining
            UNKNOWN provably source-free (a pure ``len(_present) >= max_sources``
            read; NOT the side-effecting ``is_absent_certified``, whose quadtree run
            can't complete a *different* channel's coverage within one batch anyway).

        Sound-direction only: UNKNOWN and DETECTED are never pruned (they still hold
        coverage / refine value, §8), and an absent-by-cardinality channel has no
        source to clear — so pruning can never drop a full-clear (§16 禁令10). This
        declines to *measure*; it never marks belief ABSENT (禁止6 — that stays with
        the manager's ``apply_certifications``). No belief -> never prune."""
        if self.belief is None:
            return False
        if self.belief[channel].status in _PRUNE_STATES:
            return True
        if self.certificate is not None and self.certificate.absent_by_cardinality(channel):
            return True
        return False
