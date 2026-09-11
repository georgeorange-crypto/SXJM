"""High-level training loop: pipeline rollouts -> PPO updates -> checkpoints.

Ties the pieces together without touching the frozen pipeline. Each iteration
collects a few episodes through :class:`Pipeline` with the PPO agent recording,
converts them to advantages/returns, runs one PPO update, and periodically logs
and evaluates. The agent updated here is the very instance the pipeline steps, so
training and acting share one set of weights.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from ..algorithms.ppo import PPOAgent
from ..pipeline import Pipeline
from .ppo import PPOConfig, PPOTrainer
from .rollout import RewardConfig, collect_episode


@dataclass
class TrainConfig:
    iterations: int = 100
    episodes_per_iter: int = 8
    max_steps: int = 2000
    seed: int = 0
    log_every: int = 1
    eval_every: int = 0          # 0 disables periodic eval
    eval_episodes: int = 5
    checkpoint: Optional[str] = None

    @classmethod
    def from_cfg(cls, cfg: Optional[Any]) -> "TrainConfig":
        def g(key: str, default):
            if cfg is None:
                return default
            try:
                v = cfg.get(key) if hasattr(cfg, "get") else cfg[key]
            except Exception:
                return default
            return default if v is None else v

        return cls(
            iterations=int(g("iterations", cls.iterations)),
            episodes_per_iter=int(g("episodes_per_iter", cls.episodes_per_iter)),
            max_steps=int(g("max_steps", cls.max_steps)),
            seed=int(g("seed", cls.seed)),
            log_every=int(g("log_every", cls.log_every)),
            eval_every=int(g("eval_every", cls.eval_every)),
            eval_episodes=int(g("eval_episodes", cls.eval_episodes)),
            checkpoint=(None if g("checkpoint", None) is None else str(g("checkpoint", None))),
        )


@dataclass
class TrainResult:
    agent: PPOAgent
    history: list[dict] = field(default_factory=list)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass


def _node(cfg: Any, key: str) -> Any:
    if cfg is None:
        return None
    try:
        return cfg.get(key) if hasattr(cfg, "get") else cfg[key]
    except Exception:
        return None


def evaluate(pipe: Pipeline, agent: PPOAgent, episodes: int, max_steps: int,
             base_seed: int = 100000) -> dict:
    """Deterministic (argmax) evaluation; returns mean clear ratio & task time."""
    was_recording = agent._recording
    agent._recording = False
    ratios, times, cleared = [], [], []
    for e in range(episodes):
        stats, _ = pipe.run_episode(seed=base_seed + e, max_steps=max_steps)
        total = stats.sources_total or 0
        ratios.append((stats.sources_cleared / total) if total > 0 else 0.0)
        times.append(stats.virtual_time)
        cleared.append(stats.sources_cleared)
    agent._recording = was_recording
    n = max(1, episodes)
    return {
        "eval_clear_ratio": sum(ratios) / n,
        "eval_virtual_time": sum(times) / n,
        "eval_sources_cleared": sum(cleared) / n,
    }


def train(cfg: Any, progress: bool = False) -> TrainResult:
    """Train a PPO agent from a composed config (``cfg``). Returns the agent."""
    algo = _node(cfg, "algorithm")
    tc = TrainConfig.from_cfg(_node(algo, "train"))
    pc = PPOConfig.from_cfg(_node(algo, "ppo"))
    rc = RewardConfig.from_cfg(_node(algo, "reward"))

    set_seed(tc.seed)
    pipe = Pipeline(cfg)
    agent = pipe.agent
    if not isinstance(agent, PPOAgent):
        raise TypeError(
            f"train() needs a PPO agent; got {type(agent).__name__}. "
            "Compose with algorithm=ppo."
        )
    trainer = PPOTrainer(agent, pc)

    history: list[dict] = []
    ep_counter = 0
    for it in range(tc.iterations):
        buffer = []
        ep_return, ep_ratio, ep_time, ep_len = [], [], [], []
        for _ in range(tc.episodes_per_iter):
            seed = tc.seed + 1 + ep_counter
            ep_counter += 1
            transitions, stats = collect_episode(
                pipe, agent, rc, pc.gamma, pc.gae_lambda,
                seed=seed, max_steps=tc.max_steps,
            )
            buffer.extend(transitions)
            ep_return.append(sum(t.reward for t in transitions))
            total = stats.sources_total or 0
            ep_ratio.append((stats.sources_cleared / total) if total > 0 else 0.0)
            ep_time.append(stats.virtual_time)
            ep_len.append(len(transitions))

        metrics = trainer.update(buffer)
        m = max(1, tc.episodes_per_iter)
        rec = {
            "iter": it,
            "mean_return": sum(ep_return) / m,
            "mean_clear_ratio": sum(ep_ratio) / m,
            "mean_virtual_time": sum(ep_time) / m,
            "mean_ep_len": sum(ep_len) / m,
            **metrics,
        }
        if tc.eval_every and (it + 1) % tc.eval_every == 0:
            rec.update(evaluate(pipe, agent, tc.eval_episodes, tc.max_steps))
        history.append(rec)

        if progress and tc.log_every and (it % tc.log_every == 0):
            print(
                f"[iter {it:4d}] return={rec['mean_return']:+.3f} "
                f"clear={rec['mean_clear_ratio']:.2f} "
                f"vt={rec['mean_virtual_time']:.0f} "
                f"len={rec['mean_ep_len']:.0f} "
                f"pi={rec['policy_loss']:+.3f} vf={rec['value_loss']:.3f} "
                f"H={rec['entropy']:.3f}",
                flush=True,
            )

    if tc.checkpoint:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(tc.checkpoint)), exist_ok=True)
        torch.save(agent.state_dict(), tc.checkpoint)
        if progress:
            print(f"saved checkpoint -> {tc.checkpoint}", flush=True)

    return TrainResult(agent=agent, history=history)
