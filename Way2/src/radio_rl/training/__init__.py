"""The training layer (torch). Rollout collection, PPO update, and the loop.

Imported only on the learnable path. Nothing in the mathematical core depends on
this package, so the default pipeline stays torch-free.
"""

from __future__ import annotations

from .ppo import PPOConfig, PPOTrainer
from .rollout import (
    RewardConfig,
    Transition,
    collect_episode,
    compute_gae,
    episode_rewards,
    make_transitions,
)
from .trainer import TrainConfig, TrainResult, evaluate, set_seed, train

__all__ = [
    "PPOConfig",
    "PPOTrainer",
    "RewardConfig",
    "Transition",
    "collect_episode",
    "compute_gae",
    "episode_rewards",
    "make_transitions",
    "TrainConfig",
    "TrainResult",
    "train",
    "evaluate",
    "set_seed",
]
