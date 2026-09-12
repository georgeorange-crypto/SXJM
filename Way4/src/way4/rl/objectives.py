"""Contract-level objectives for the optional learning layer.

The math planner remains authoritative.  These pure functions make the SMDP
training target explicit and testable: elapsed time is the per-transition cost,
full-clear is a terminal cliff, and a critic predicts negative remaining time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RewardConfig:
    full_clear_bonus: float = 100_000.0
    failure_terminal_bonus: float = 0.0


def transition_reward(delta_t_s: float) -> float:
    """Primary SMDP reward: elapsed mission time, with no progress shaping."""
    return -max(0.0, float(delta_t_s))


def terminal_reward(full_clear: bool, cfg: RewardConfig = RewardConfig()) -> float:
    """Terminal cliff: incomplete missions never receive the full-clear bonus."""
    return cfg.full_clear_bonus if bool(full_clear) else cfg.failure_terminal_bonus


def episode_return(
    elapsed_s: float, full_clear: bool, cfg: RewardConfig = RewardConfig()
) -> float:
    return transition_reward(elapsed_s) + terminal_reward(full_clear, cfg)


def critic_target(analytic_remaining_s: float, learned_residual_s: float = 0.0) -> float:
    """Analytic + learned residual value, where value means negative remaining time."""
    return -max(0.0, float(analytic_remaining_s)) + float(learned_residual_s)


def critic_ablation_values(analytic_remaining_s: float, learned_residual_s: float) -> dict:
    """Comparable targets for analytic-only, learned-only and hybrid ablations."""
    analytic = -max(0.0, float(analytic_remaining_s))
    residual = float(learned_residual_s)
    return {
        "analytic_future_cost_only": analytic,
        "learned_critic_only": residual,
        "analytic_plus_learned_residual": analytic + residual,
    }
