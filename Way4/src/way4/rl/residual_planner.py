"""``RLResidualPlanner`` — the M10 residual wrapper (DESIGN.md §12, §13).

Sits exactly where §13's pipeline diagram places it::

    RecedingHorizonPlanner (→ optional RL residual) → SafetyShield → MacroExecutor

It **wraps** a ``RecedingHorizonPlanner`` and exposes the *same* ``plan()``
signature, so it is a drop-in for ``Pipeline(planner=...)`` with **zero changes to
the pipeline, executor, or safety layer**. What it adds is the residual re-rank:

    Q(B, a) = Q_math(B, a) + ΔQ_θ(B, a)          (§12)

and then ``a* = argmin_a Q``. Because it only reorders the candidate list the math
planner already scored, it can never invent a coordinate (禁止8) and never reaches
past the SafetyShield that runs *after* it — the CLEAR/EXIT guards still veto
whatever it picks (禁止7; §11 "这些硬安全规则 RL 不得覆盖").

Fail-safe posture (every one of these degrades to the pure-math choice):
  * scorer unavailable (no torch) or ``enabled=False`` → residuals are 0;
  * a zero-initialised (untrained) scorer → residuals are 0;
  * **any** exception while scoring → caught, residuals treated as 0.
So the *worst* an untrained or broken residual can do is reproduce Way4-Math. It
can never lower full-clear (禁止10) because it changes only the *order* of already
safe, already generated candidates, and the guarantee lives downstream (§11).

The returned ``PlanResult`` keeps the math ``evaluations`` untouched (their
``q_value`` stays ``Q_math`` for logging/telemetry); only ``best``/``q_value`` are
recomputed from the residual-adjusted totals, and the chosen candidate's applied
residual is recorded in ``best.meta['rl_residual']`` for observability.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..planner import PlanResult, RecedingHorizonPlanner
from .features import feature_matrix
from .scorer import ResidualScorer


class RLResidualPlanner:
    """Residual re-ranker around a math planner. Duck-types ``.plan()``."""

    def __init__(
        self,
        base_planner: Optional[RecedingHorizonPlanner] = None,
        scorer: Optional[ResidualScorer] = None,
        *,
        enabled: bool = True,
    ) -> None:
        self.base = base_planner or RecedingHorizonPlanner()
        self.scorer = scorer or ResidualScorer()   # zero-init => no-op by default
        self.enabled = bool(enabled)

    # -- capability --------------------------------------------------------

    @property
    def active(self) -> bool:
        """True iff the residual can actually change the ranking (enabled AND a
        live scorer). When False this is behaviourally identical to ``base``."""
        return self.enabled and self.scorer is not None and self.scorer.available

    # -- planning ----------------------------------------------------------

    def plan(self, belief, certificate, state, candidates: Sequence) -> PlanResult:
        """Math-plan, then residual-rerank. Any failure returns the math result."""
        result = self.base.plan(belief, certificate, state, candidates)
        if not self.active or not result.evaluations:
            return result

        try:
            residuals = self._residuals(result.evaluations, belief, state)
        except Exception:
            # §11: RL unavailable/degenerate -> fall back to the math ranking.
            return result

        if not residuals or all(r == 0.0 for r in residuals):
            return result

        best_i = min(
            range(len(result.evaluations)),
            key=lambda i: result.evaluations[i].q_value + residuals[i],
        )
        best_eval = result.evaluations[best_i]
        best_cand = best_eval.candidate
        # record the applied residual for observability without mutating q_value.
        try:
            best_cand.meta["rl_residual"] = float(residuals[best_i])
        except Exception:
            pass
        return PlanResult(
            best=best_cand,
            q_value=best_eval.q_value + residuals[best_i],
            evaluations=result.evaluations,     # math q_values preserved for logging
            horizon=result.horizon,
            outcome_mode=result.outcome_mode,
        )

    def _residuals(self, evaluations, belief, state) -> List[float]:
        feats = feature_matrix(evaluations, belief, state)
        return self.scorer.residuals(feats)
