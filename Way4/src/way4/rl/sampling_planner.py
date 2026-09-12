"""Training-time sampling planner for the M10 residual (DESIGN.md §12).

At *deployment* the residual is greedy: ``a* = argmin_a [Q_math + ΔQ_θ]`` (that is
``RLResidualPlanner``). To learn ``ΔQ_θ`` with a policy gradient we need to (a) act
stochastically so the objective is differentiable in the policy, and (b) record
each decision's features + choice so REINFORCE can reweight it by the episode
return.

This planner does exactly that and **nothing else that could touch correctness**:

  * It ranks over the SAME candidate list the math generator produced — it samples
    an index into that list, never a coordinate (禁止8).
  * The sampling distribution is ``π(a) = softmax(−(Q_math(a)+ΔQ_θ(a)) / T)``. Note
    ``Q`` is a *cost* (lower = better), hence the minus sign: low-cost candidates
    get high probability. As ``T → 0`` this becomes the deployment argmin, so the
    thing we train is the thing we ship.
  * It sits where ``RecedingHorizonPlanner`` sits, before the SafetyShield, so the
    CLEAR/EXIT guards still veto whatever it samples (禁止7; §11). Sampling can only
    reorder *safe* candidates; the full-clear guarantee is unaffected structurally.

It records, per ``plan`` call, a :class:`Decision` (feature matrix + chosen index),
which the trainer replays to build ``log π(a_chosen)`` with gradient. The scorer's
grad-enabled ``_forward`` is used at replay time, not here — inference here stays
no-grad and cheap so an episode rollout runs at full speed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from ..planner import PlanResult, RecedingHorizonPlanner
from .features import feature_matrix
from .scorer import ResidualScorer


@dataclass
class Decision:
    """One recorded decision for the policy gradient replay."""

    features: List[List[float]]     # n_candidates × FEATURE_DIM
    q_math: List[float]             # analytical cost per candidate (the anchor)
    chosen: int                     # sampled index into the candidate list
    temperature: float

    def to_record(self) -> dict:
        return {
            "features": self.features,
            "q_math": self.q_math,
            "chosen": int(self.chosen),
            "temperature": float(self.temperature),
            "n_candidates": len(self.q_math),
        }


def expert_dataset(decisions: Sequence[Decision]) -> List[dict]:
    """Export all candidate scores/features, not only selected actions."""
    return [d.to_record() for d in decisions]


class SamplingResidualPlanner:
    """REINFORCE-time planner: samples a candidate under the residual policy and
    records the decision. Duck-types ``.plan()`` so it drops into ``Way4Pipeline``."""

    def __init__(
        self,
        scorer: ResidualScorer,
        base_planner: Optional[RecedingHorizonPlanner] = None,
        *,
        temperature: float = 120.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.base = base_planner or RecedingHorizonPlanner()
        self.scorer = scorer
        self.temperature = float(temperature)
        self.rng = rng or random.Random()
        self.decisions: List[Decision] = []

    def reset(self) -> None:
        self.decisions = []

    def plan(self, belief, certificate, state, candidates: Sequence) -> PlanResult:
        result = self.base.plan(belief, certificate, state, candidates)
        evals = result.evaluations
        if not evals:
            return result
        if len(evals) == 1:
            # no choice to make; still step, but nothing to learn from.
            return result

        feats = feature_matrix(evals, belief, state)
        q_math = [float(e.q_value) for e in evals]
        residuals = self.scorer.residuals(feats) if self.scorer.available else [0.0] * len(evals)
        # policy over COSTS: lower total cost -> higher probability (hence -Q/T).
        totals = [q_math[i] + residuals[i] for i in range(len(evals))]
        probs = _softmax_neg(totals, self.temperature)
        chosen = _sample(probs, self.rng)

        self.decisions.append(
            Decision(features=feats, q_math=q_math, chosen=chosen, temperature=self.temperature)
        )

        best_eval = evals[chosen]
        return PlanResult(
            best=best_eval.candidate,
            q_value=totals[chosen],
            evaluations=evals,
            horizon=result.horizon,
            outcome_mode=result.outcome_mode,
        )


def _softmax_neg(costs: Sequence[float], temperature: float) -> List[float]:
    """``softmax(-cost / T)`` — numerically stable. Low cost => high probability."""
    t = max(1e-6, float(temperature))
    logits = [-c / t for c in costs]
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    s = sum(exps)
    if s <= 0:
        n = len(costs)
        return [1.0 / n] * n
    return [e / s for e in exps]


def _sample(probs: Sequence[float], rng: random.Random) -> int:
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return i
    return len(probs) - 1
