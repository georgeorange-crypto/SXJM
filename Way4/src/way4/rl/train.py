"""REINFORCE trainer for the M10 residual ``ΔQ_θ`` (DESIGN.md §12).

This is the *last* milestone and runs **only after Way4-Math is verified stable**
(§12: 只在 Way4-Math 稳定后) — which it is (P3/P4 100% full-clear, 152 tests green).
The user explicitly requested training; this module is what that request runs.

Algorithm — episodic **REINFORCE** (Monte-Carlo policy gradient), deliberately the
minimal policy-gradient method, *not* PPO (no clipping, no critic, no GAE): the
recorded preference is "先别急着上 PPO", and REINFORCE is the honest reading of §12's
"RL 只学 long-term trade-off" — one scalar return per episode, one gradient step.

Policy: for each decision the pipeline made, ``π(a) = softmax(−(Q_math+ΔQ_θ)/T)``
over the math generator's candidates (``SamplingResidualPlanner``). The learnable
part is ``ΔQ_θ`` (the MLP). Loss::

    L = − Σ_episodes Σ_decisions (R_episode − b) · log π(a_chosen)

with ``b`` a batch-mean baseline (variance reduction; not a learned critic).

Return — bakes in **禁止10 (never trade full-clear for speed)** so it is impossible
to learn a faster-but-failing policy::

    R = − virtual_time / 1000                      if full_clear   (faster => higher)
    R = − FAIL_PENALTY − n_unresolved              otherwise       (strictly worse)

``FAIL_PENALTY`` (default 100) makes the *best possible* failure (return ≈ −100)
strictly worse than the *worst plausible* success (episode time ~16000 s ⇒ ≈ −16),
so the gradient always pushes toward full-clear first, speed second.

Safety of the whole procedure: sampling only reorders already-safe candidates and
the SafetyShield/clear-guard run downstream (§11), so even a bad gradient step can
only cost virtual time in a rollout, never full-clear — and the trained checkpoint
is accepted (:func:`validate`) only if its GREEDY full-clear ≥ the math baseline.
Nothing here writes the shipped default; activation stays a separate explicit step.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from .sampling_planner import Decision, SamplingResidualPlanner
from .scorer import ResidualScorer

try:
    import torch
    _TORCH_OK = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_OK = False


FAIL_PENALTY = 100.0
TIME_SCALE = 1000.0


@dataclass
class EpisodeRecord:
    decisions: List[Decision]
    full_clear: bool
    virtual_time_s: float
    n_unresolved: int
    ret: float = 0.0
    transitions: List[object] = field(default_factory=list)


@dataclass
class TrainConfig:
    iterations: int = 20
    episodes_per_iter: int = 4
    lr: float = 1e-3
    temperature: float = 120.0
    fail_penalty: float = FAIL_PENALTY
    seed: int = 0
    #: seeds the rollouts draw from (cycled). Kept small + fixed for reproducibility.
    train_seeds: Tuple[int, ...] = tuple(range(2000, 2016))
    problem: int = 4
    field_kind: str = "smooth"
    max_steps: int = 3000


def episode_return(full_clear: bool, virtual_time_s: float, n_unresolved: int,
                   fail_penalty: float = FAIL_PENALTY) -> float:
    """The 禁止10-safe scalar return (see module docstring)."""
    if full_clear:
        return -float(virtual_time_s) / TIME_SCALE
    return -float(fail_penalty) - float(n_unresolved)


class ResidualTrainer:
    """Runs REINFORCE over Way4 episodes to fit the residual scorer.

    ``rollout_fn(scorer, temperature, seed) -> EpisodeRecord`` runs ONE episode with
    the sampling planner and returns its recorded decisions + outcome. It is injected
    so the trainer has no hard dependency on the Way3 engine (tests pass a stub); the
    real one is :func:`make_way4_rollout`."""

    def __init__(
        self,
        scorer: ResidualScorer,
        rollout_fn: Callable[[ResidualScorer, float, int], EpisodeRecord],
        config: Optional[TrainConfig] = None,
    ) -> None:
        if not _TORCH_OK or not scorer.available:
            raise RuntimeError("torch unavailable; cannot train the residual scorer")
        self.scorer = scorer
        self.rollout_fn = rollout_fn
        self.cfg = config or TrainConfig()
        self.opt = torch.optim.Adam(list(self.scorer.net.parameters()) + [self.scorer.action_bias], lr=self.cfg.lr)
        self.history: List[dict] = []

    def train(self, log: Optional[Callable[[dict], None]] = None) -> List[dict]:
        rng = random.Random(self.cfg.seed)
        seeds = list(self.cfg.train_seeds)
        for it in range(self.cfg.iterations):
            batch: List[EpisodeRecord] = []
            for _ in range(self.cfg.episodes_per_iter):
                seed = seeds[rng.randrange(len(seeds))]
                rec = self.rollout_fn(self.scorer, self.cfg.temperature, seed)
                rec.ret = episode_return(
                    rec.full_clear, rec.virtual_time_s, rec.n_unresolved, self.cfg.fail_penalty
                )
                batch.append(rec)
            stats = self._update(batch)
            stats["iter"] = it
            self.history.append(stats)
            if log:
                log(stats)
        return self.history

    # -- one REINFORCE gradient step over a batch of episodes --------------

    def _update(self, batch: Sequence[EpisodeRecord]) -> dict:
        rets = [r.ret for r in batch]
        baseline = sum(rets) / len(rets) if rets else 0.0
        # advantage-weighted log-prob loss, summed over every decision in the batch.
        self.opt.zero_grad()
        loss = torch.zeros((), dtype=torch.float32)
        n_dec = 0
        n_clear = 0
        for rec in batch:
            adv = rec.ret - baseline
            if rec.full_clear:
                n_clear += 1
            if adv == 0.0:
                continue
            for dec in rec.decisions:
                logp = self._log_prob(dec)
                loss = loss - adv * logp
                n_dec += 1
        if n_dec > 0 and loss.requires_grad:
            loss.backward()
            self.opt.step()
        return {
            "loss": float(loss.detach()) if n_dec else 0.0,
            "mean_return": baseline,
            "full_clear_rate": n_clear / len(batch) if batch else 0.0,
            "n_decisions": n_dec,
        }

    def _log_prob(self, dec: Decision):
        """``log π(a_chosen)`` WITH gradient through ``ΔQ_θ``. Reconstructs the exact
        sampling distribution the planner used (same features, same ``Q_math`` anchor,
        same temperature), differentiable in the scorer parameters."""
        x = torch.tensor(dec.features, dtype=torch.float32)
        delta = self.scorer._forward(x)                      # (n_cand,) with grad
        q_math = torch.tensor(dec.q_math, dtype=torch.float32)
        totals = q_math + delta
        logits = -totals / max(1e-6, dec.temperature)        # cost -> preference
        logprobs = torch.log_softmax(logits, dim=0)
        return logprobs[dec.chosen]
