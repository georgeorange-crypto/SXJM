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

from ..algorithms.ppo import PPOAgent, StepRecord
from ..core.datatypes import ActionType, ObservationType
from ..features.spec import FeatureBundle
from ..pipeline import Pipeline, StepTrace


@dataclass
class RewardConfig:
    """Reward shaping for the task objective."""

    time_coef: float = 1.0             # penalty weight on normalised task time
    time_scale: float = 1000.0         # seconds that map to one unit of penalty
    clear_success_bonus: float = 1.0   # per source neutralised
    clear_fail_penalty: float = 0.5    # per wasted clear
    success_bonus: float = 5.0         # terminal, scaled by clear ratio

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
    clear ratio rewards finishing the job.
    """
    rewards: list[float] = []
    prev_vt = 0.0
    for st in trace:
        dt = st.virtual_time_s - prev_vt
        if dt < 0.0:
            dt = 0.0
        prev_vt = st.virtual_time_s
        r = -rc.time_coef * (dt / rc.time_scale)
        if st.action.action_type != ActionType.EXIT:
            if st.observation.result_type == ObservationType.CLEAR_SUCCESS:
                r += rc.clear_success_bonus
            elif st.observation.result_type == ObservationType.CLEAR_FAILURE:
                r -= rc.clear_fail_penalty
        rewards.append(r)

    if rewards:
        total = getattr(stats, "sources_total", 0) or 0
        cleared = getattr(stats, "sources_cleared", 0) or 0
        ratio = (cleared / total) if total > 0 else 0.0
        rewards[-1] += rc.success_bonus * ratio
    return rewards


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
