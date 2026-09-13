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
    full_clear_bonus: float = 0.0
    failure_terminal_bonus: float = -100.0
    longjump_penalty: float = 1.0
    crossing_penalty: float = 1.0
    tail_cost_weight: float = 1.0
    no_progress_multiplier: float = 2.0
    repeat_measure_multiplier: float = 1.0
    max_time_multiplier: float = 3.0
    shaping_eta: float = 0.05


def state_potential(*, certificate_progress: float = 0.0, localized_count: int = 0,
                    cleared_count: int = 0, unresolved_count: int = 0,
                    directional_count: int = 0) -> float:
    """D02 potential; inputs are planner progress counts, not reward guesses."""
    return (float(certificate_progress) + float(localized_count) + float(cleared_count)
            - float(unresolved_count) - float(directional_count))


def potential_shaping(phi_before: float, phi_after: float, *, gamma: float = 1.0,
                      eta: float = 0.05) -> float:
    """Compute F=eta*(gamma*Phi(s')-Phi(s)) with a small explicit scale."""
    if not 0.0 < float(gamma) <= 1.0 or float(eta) < 0.0:
        raise ValueError("gamma must be in (0,1] and eta non-negative")
    return float(eta) * (float(gamma) * float(phi_after) - float(phi_before))


def dense_time_reward(delta_t_s: float, *, no_progress: bool = False,
                      repeat_measure: bool = False,
                      detour_time_s: float = 0.0,
                      cfg: RewardConfig = RewardConfig(), phi_before=None,
                      phi_after=None, gamma: float = 1.0) -> float:
    """Dense elapsed-time reward with bounded waste multiplier (C01--C06)."""
    multiplier = 1.0
    if no_progress:
        multiplier += float(cfg.no_progress_multiplier)
    if repeat_measure:
        multiplier += float(cfg.repeat_measure_multiplier)
    multiplier = min(float(cfg.max_time_multiplier), multiplier)
    detour = max(0.0, float(detour_time_s))
    reward = (-max(0.0, float(delta_t_s)) * multiplier - detour) / 1000.0
    if phi_before is not None or phi_after is not None:
        if phi_before is None or phi_after is None:
            raise ValueError("both phi_before and phi_after are required")
        reward += potential_shaping(phi_before, phi_after, gamma=gamma,
                                    eta=cfg.shaping_eta)
    return reward


def transition_reward(delta_t_s: float, *, delta_route_m: float = 0.0,
                      delta_tail_s: float = 0.0, longjumps: int = 0,
                      crossings: int = 0, cfg: RewardConfig = RewardConfig()) -> float:
    """Route-aligned SMDP reward; proxy service counts are intentionally absent."""
    return (-max(0.0, float(delta_t_s))
            - cfg.tail_cost_weight * max(0.0, float(delta_tail_s))
            - max(0.0, float(delta_route_m))
            - cfg.longjump_penalty * max(0, int(longjumps))
            - cfg.crossing_penalty * max(0, int(crossings)))


def terminal_reward(full_clear: bool, cfg: RewardConfig = RewardConfig(),
                    n_unresolved: int = 0) -> float:
    """Terminal cliff: incomplete missions never receive the full-clear bonus."""
    if bool(full_clear):
        return cfg.full_clear_bonus
    if n_unresolved < 0:
        raise ValueError('unresolved count must be nonnegative')
    return cfg.failure_terminal_bonus - int(n_unresolved)


def episode_return(
    elapsed_s: float, full_clear: bool, cfg: RewardConfig = RewardConfig(),
    n_unresolved: int = 0
) -> float:
    return transition_reward(elapsed_s) + terminal_reward(full_clear, cfg, n_unresolved)


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
