"""Evaluation harness — run a policy over many seeds and summarize.

Environment-agnostic: it drives whatever :class:`Pipeline` is configured, so the
same harness scores the greedy baseline today and a trained PPO agent later by
changing only ``algorithm=...`` in the config.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from ..core.datatypes import EpisodeStats
from ..pipeline import Pipeline
from .metrics import MetricsSummary, summarize


def _node(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    try:
        v = cfg.get(key) if hasattr(cfg, "get") else cfg[key]
    except Exception:
        return default
    return default if v is None else v


@dataclass
class EvalConfig:
    n_episodes: int = 20
    seeds: Optional[Sequence[int]] = None
    max_steps: int = 100_000
    stress: bool = False
    verbose: bool = True
    base_seed: int = 0
    max_wall_s: Optional[float] = None   # abort the batch if it overruns

    @staticmethod
    def from_cfg(cfg: Any) -> "EvalConfig":
        ev = _node(cfg, "evaluation")
        seeds = _node(ev, "seeds")
        return EvalConfig(
            n_episodes=int(_node(ev, "n_episodes", 20)),
            seeds=list(seeds) if seeds is not None else None,
            max_steps=int(_node(ev, "max_steps", 100_000)),
            stress=bool(_node(ev, "stress", False)),
            verbose=bool(_node(ev, "verbose", True)),
            base_seed=int(_node(cfg, "seed", 0)),
        )


class Evaluator:
    def __init__(self, cfg: Any, pipeline: Optional[Pipeline] = None) -> None:
        self.cfg = cfg
        self.ecfg = EvalConfig.from_cfg(cfg)
        self.pipeline = pipeline or Pipeline(cfg)

    def _seeds(self) -> list[int]:
        if self.ecfg.seeds:
            return list(self.ecfg.seeds)
        return [self.ecfg.base_seed + i for i in range(self.ecfg.n_episodes)]

    def run(self) -> MetricsSummary:
        seeds = self._seeds()
        episodes: list[EpisodeStats] = []
        t0 = time.monotonic()
        for i, seed in enumerate(seeds):
            stats, _ = self.pipeline.run_episode(seed=seed, max_steps=self.ecfg.max_steps)
            episodes.append(stats)
            if self.ecfg.verbose:
                print(f"[{i + 1:>3}/{len(seeds)}] seed={seed:<5} "
                      f"cleared {stats.sources_cleared}/{stats.sources_total}  "
                      f"vtime={stats.virtual_time:.1f}s  steps={stats.steps}")
            if self.ecfg.max_wall_s is not None and time.monotonic() - t0 > self.ecfg.max_wall_s:
                if self.ecfg.verbose:
                    print("[eval] wall-time budget hit; stopping early")
                break
        summary = summarize(episodes)
        if self.ecfg.verbose:
            print("\n" + summary.format())
        return summary


def evaluate(cfg: Any) -> MetricsSummary:
    return Evaluator(cfg).run()
