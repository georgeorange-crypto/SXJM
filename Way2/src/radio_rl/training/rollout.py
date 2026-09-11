"""Rollout mechanics: rewards, GAE, and turning a pipeline episode into PPO data.

The trainer runs ordinary :meth:`Pipeline.run_episode` episodes with the PPO
agent in recording mode; this module converts the resulting
(records, trace, stats) into a list of :class:`Transition` with advantages and
returns. The pipeline itself is untouched — reward shaping lives here, in the
training layer, where the objective (minimise task time while clearing every
source) is expressed as a dense per-step reward plus a terminal clear bonus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from ..algorithms.ppo import PPOAgent, StepRecord
from ..core.datatypes import ActionType, ObservationType
from ..features.spec import FeatureBundle
from ..pipeline import Pipeline, StepTrace


@dataclass
class RewardConfig:
    """Reward shaping for the task objective.

    ``exit_penalty`` and ``shaping_coef`` default to 0.0 so a bare
    ``RewardConfig()`` reproduces the Milestone-2 reward exactly; the training
    config (``configs/algorithm/ppo.yaml``) turns them on. They exist to break
    the degenerate "exit on step 1" local optimum: giving up with sources still
    uncleared is charged ``exit_penalty * (1 - clear_ratio)`` (a full clear exits
    free), and belief-progress potential shaping pays a dense, theory-preserving
    reward for shrinking the feasible regions so that localisation work — which
    otherwise earns nothing until a far-off clear — has a gradient.

    ``full_clear_bonus`` is a *non-linear* terminal payment made once iff every
    source is cleared. Unlike the linear ``success_bonus * clear_ratio``, it makes
    13/13 qualitatively better than 12/13 — matching the real objective (clear
    *all* sources, then minimise time), which the linear terms cannot express.
    All of these read only training-layer episode stats, never the observation.
    """

    time_coef: float = 1.0             # penalty weight on normalised task time
    time_scale: float = 1000.0         # seconds that map to one unit of penalty
    clear_success_bonus: float = 1.0   # per source neutralised
    clear_fail_penalty: float = 0.5    # per wasted clear
    success_bonus: float = 5.0         # terminal, scaled by clear ratio (linear)
    exit_penalty: float = 0.0          # terminal, scaled by (1 - clear ratio)
    full_clear_bonus: float = 0.0      # terminal, paid once iff EVERY source cleared (cliff)
    shaping_coef: float = 0.0          # weight (beta) on potential-based shaping

    @classmethod
    def from_cfg(cls, cfg: Optional[Any]) -> "RewardConfig":
        def g(key: str, default: float) -> float:
            if cfg is None:
                return default
            try:
                v = cfg.get(key) if hasattr(cfg, "get") else cfg[key]
            except Exception:
                return default
            return default if v is None else float(v)

        return cls(
            time_coef=g("time_coef", cls.time_coef),
            time_scale=g("time_scale", cls.time_scale),
            clear_success_bonus=g("clear_success_bonus", cls.clear_success_bonus),
            clear_fail_penalty=g("clear_fail_penalty", cls.clear_fail_penalty),
            success_bonus=g("success_bonus", cls.success_bonus),
            exit_penalty=g("exit_penalty", cls.exit_penalty),
            full_clear_bonus=g("full_clear_bonus", cls.full_clear_bonus),
            shaping_coef=g("shaping_coef", cls.shaping_coef),
        )


@dataclass
class Transition:
    bundle: FeatureBundle
    action: int
    log_prob: float
    value: float
    reward: float = 0.0
    advantage: float = 0.0
    ret: float = 0.0


def episode_rewards(trace: list[StepTrace], stats: Any, rc: RewardConfig) -> list[float]:
    """Dense per-step reward aligned index-by-index with ``trace``.

    Time cost comes from the *increment* in cumulative task time (so the terminal
    EXIT, which consumes none, is charged nothing and never re-charges the prior
    step). Clear bonuses read the executed step's observation, except on an EXIT
    step whose stored observation is stale. A terminal bonus proportional to the
    clear ratio rewards finishing the job; an EXIT step is additionally charged
    ``exit_penalty * (1 - clear_ratio)`` so that abandoning the field with sources
    still live is strictly worse than the do-nothing baseline (a full clear pays
    nothing), which is what kills the "exit immediately for a free zero" optimum.
    Finally a ``full_clear_bonus`` is added to the terminal step iff *every* source
    was cleared — a non-linear cliff separating a completed task from a near-miss.
    """
    total = getattr(stats, "sources_total", 0) or 0
    cleared = getattr(stats, "sources_cleared", 0) or 0
    ratio = (cleared / total) if total > 0 else 0.0

    rewards: list[float] = []
    prev_vt = 0.0
    for st in trace:
        dt = st.virtual_time_s - prev_vt
        if dt < 0.0:
            dt = 0.0
        prev_vt = st.virtual_time_s
        r = -rc.time_coef * (dt / rc.time_scale)
        if st.action.action_type == ActionType.EXIT:
            r -= rc.exit_penalty * (1.0 - ratio)
        else:
            if st.observation.result_type == ObservationType.CLEAR_SUCCESS:
                r += rc.clear_success_bonus
            elif st.observation.result_type == ObservationType.CLEAR_FAILURE:
                r -= rc.clear_fail_penalty
        rewards.append(r)

    if rewards:
        rewards[-1] += rc.success_bonus * ratio
        if total > 0 and cleared >= total:
            rewards[-1] += rc.full_clear_bonus   # non-linear cliff: all-clear != near-miss
    return rewards


def _potential(bundle: Optional[FeatureBundle]) -> float:
    """Belief-progress potential Phi(s) in ``[0, num_channels]``.

    Sum over channels of a per-channel localisation-progress score in ``[0, 1]``:
    ``0`` when the feasible region is still the whole arena (undetected), rising
    towards ``1`` as the region shrinks (``1 - mec_radius/arena``), and pinned to
    ``1`` once the channel is cleared. It reads only the recorded feature tensor
    (``channel_feats`` col 3 = cleared one-hot, col 5 = normalised MEC radius),
    so no belief object is needed and the frozen pipeline stays untouched.
    """
    if bundle is None:
        return 0.0
    ch = bundle.channel_feats
    cleared = ch[:, 3]                       # cleared one-hot
    mec_norm = ch[:, 5]                       # mec_radius / arena, in [0, 1]
    progress = torch.where(cleared > 0.5, torch.ones_like(mec_norm), 1.0 - mec_norm)
    return float(progress.sum().item())


def apply_potential_shaping(
    transitions: list[Transition], gamma: float, beta: float
) -> None:
    """Add potential-based shaping ``beta * (gamma*Phi' - Phi)`` in place.

    This is Ng-Harada-Russell (1999) shaping: because the bonus is the difference
    of a potential, it provably leaves the optimal policy unchanged and only
    reshapes the learning gradient — here paying the agent, step by step, for
    turning uncertain regions into localised (then cleared) ones instead of only
    at the distant terminal clear. The terminal transition gets no shaping (its
    ``Phi'`` would be the absorbing state), so there is no end-of-episode spike;
    the accrued bonus telescopes to ``~beta * (Phi_final - Phi_initial)`` and
    stays bounded by ``beta * num_channels``.
    """
    if beta == 0.0 or len(transitions) < 2:
        return
    phis = [_potential(t.bundle) for t in transitions]
    for i in range(len(transitions) - 1):
        transitions[i].reward += beta * (gamma * phis[i + 1] - phis[i])


def compute_gae(transitions: list[Transition], gamma: float, lam: float) -> None:
    """Fill ``advantage`` and ``ret`` in place (episode treated as terminal)."""
    adv = 0.0
    next_value = 0.0  # bootstrap 0 at episode end
    for t in reversed(transitions):
        delta = t.reward + gamma * next_value - t.value
        adv = delta + gamma * lam * adv
        t.advantage = adv
        t.ret = adv + t.value
        next_value = t.value


def make_transitions(
    records: list[StepRecord],
    trace: list[StepTrace],
    stats: Any,
    rc: RewardConfig,
    gamma: float,
    lam: float,
) -> list[Transition]:
    """Pair recorded decisions with rewards and compute GAE for one episode."""
    n = min(len(records), len(trace))
    rewards = episode_rewards(trace, stats, rc)
    transitions = [
        Transition(
            bundle=records[i].bundle,
            action=records[i].action,
            log_prob=records[i].log_prob,
            value=records[i].value,
            reward=rewards[i],
        )
        for i in range(n)
    ]
    apply_potential_shaping(transitions, gamma, rc.shaping_coef)
    compute_gae(transitions, gamma, lam)
    return transitions


def collect_episode(
    pipe: Pipeline,
    agent: PPOAgent,
    rc: RewardConfig,
    gamma: float,
    lam: float,
    seed: Optional[int] = None,
    max_steps: int = 2000,
) -> tuple[list[Transition], Any]:
    """Run one recording episode and return (transitions, stats)."""
    agent.begin_recording()
    stats, trace = pipe.run_episode(seed=seed, max_steps=max_steps, collect_trace=True)
    records = agent.take_records()
    transitions = make_transitions(records, trace, stats, rc, gamma, lam)
    return transitions, stats
