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
    ) -> None:
        self.env = env
        self.belief = belief
        self.certificate = certificate
        self.cost = cost_model or AnalyticalCostModel()
        self.opportunistic_clear = opportunistic_clear

    # -- public API --------------------------------------------------------

    def execute(
        self, macro: MacroCandidate, state: RobotState, time_budget_s: float = float("inf")
    ) -> ExecutionResult:
        """Run ``macro`` from ``state``. Stops before any primitive whose predicted
        cost would push the virtual clock past ``time_budget_s`` (超时即停)."""
        results: List[PrimitiveResult] = []
        finished = False
        stopped = False

        for prim in macro.primitives():
            if prim.kind == PrimitiveKind.EXIT:
                finished = True
                break
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

    # -- interactive stepping (Way3 homing fallback, §11) ------------------
    # Single primitives with belief+certificate folding, for the HomingController's
    # adaptive measure->decide->measure loop (which cannot be pre-expanded into one
    # macro's primitive list). They reuse the same _do_measure/_do_clear folding as
    # execute(), so a homed observation updates belief/certificate identically to a
    # planned macro (Invariants B/D unchanged), and are budget-aware like execute().

    def step_measure(
        self, state: RobotState, point, channel: int, time_budget_s: float = float("inf")
    ):
        """Measure ``channel`` at ``point``; fold the outcome; advance state. Returns
        ``(state, observation_or_None, done)``. ``observation`` is None and ``done`` is
        True when the env hit its deadline or the budget blocks the measure — the
        caller should stop."""
        prim = Primitive(PrimitiveKind.MEASURE, (float(point[0]), float(point[1])), int(channel))
        if not self._within_budget(state, prim, time_budget_s):
            return state, None, True
        state, pr, finished = self._do_measure(state, prim)
        return state, pr.observation, finished

    def step_clear(
        self, state: RobotState, point, channel: int, time_budget_s: float = float("inf")
    ):
        """Clear ``channel`` at ``point``; fold on hit; advance state. Returns
        ``(state, hit, done)``."""
        prim = Primitive(PrimitiveKind.CLEAR, (float(point[0]), float(point[1])), int(channel))
        if not self._within_budget(state, prim, time_budget_s, hit=True):
            return state, False, True
        state, pr, finished = self._do_clear(state, prim)
        return state, pr.cleared, finished

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
