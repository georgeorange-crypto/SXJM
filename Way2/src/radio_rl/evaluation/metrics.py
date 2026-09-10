"""Aggregate metrics over a batch of episodes.

The competition ranks by task (virtual) time to clear all sources, with the
clear ratio as the primary gate. These aggregates report exactly that, plus the
diagnostics we tune against (failed clears, scans per source, wall-clock).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean, median, pstdev
from typing import Sequence

from ..core.datatypes import EpisodeStats


def _safe_mean(xs: Sequence[float]) -> float:
    return mean(xs) if xs else 0.0


@dataclass
class MetricsSummary:
    n_episodes: int = 0
    success_rate: float = 0.0            # fraction of episodes clearing ALL sources
    clear_ratio_mean: float = 0.0        # mean fraction of sources cleared
    sources_cleared: int = 0
    sources_total: int = 0

    virtual_time_mean: float = 0.0       # over fully-solved episodes
    virtual_time_median: float = 0.0
    virtual_time_std: float = 0.0
    virtual_time_all_mean: float = 0.0   # over every episode

    avg_clear_time_mean: float = 0.0
    scans_per_source_mean: float = 0.0
    failed_clear_rate: float = 0.0       # failed clears / clear attempts
    wall_time_mean: float = 0.0
    steps_mean: float = 0.0

    per_episode: list[EpisodeStats] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d.pop("per_episode", None)
        return d

    def format(self) -> str:
        return (
            f"episodes            : {self.n_episodes}\n"
            f"success rate        : {self.success_rate:.1%}  "
            f"(all sources cleared)\n"
            f"clear ratio (mean)  : {self.clear_ratio_mean:.1%}  "
            f"({self.sources_cleared}/{self.sources_total} sources)\n"
            f"virtual time solved : mean {self.virtual_time_mean:.1f} s | "
            f"median {self.virtual_time_median:.1f} s | std {self.virtual_time_std:.1f} s\n"
            f"virtual time all    : mean {self.virtual_time_all_mean:.1f} s\n"
            f"avg clear time      : {self.avg_clear_time_mean:.1f} s\n"
            f"scans / source      : {self.scans_per_source_mean:.2f}\n"
            f"failed clear rate   : {self.failed_clear_rate:.1%}\n"
            f"wall time / ep      : {self.wall_time_mean:.3f} s\n"
            f"steps / ep          : {self.steps_mean:.1f}"
        )


def summarize(episodes: Sequence[EpisodeStats]) -> MetricsSummary:
    s = MetricsSummary(n_episodes=len(episodes), per_episode=list(episodes))
    if not episodes:
        return s

    solved = [e for e in episodes if e.success]
    s.success_rate = len(solved) / len(episodes)
    s.clear_ratio_mean = _safe_mean([e.clear_ratio for e in episodes])
    s.sources_cleared = sum(e.sources_cleared for e in episodes)
    s.sources_total = sum(e.sources_total for e in episodes)

    solved_vt = [e.virtual_time for e in solved]
    s.virtual_time_mean = _safe_mean(solved_vt)
    s.virtual_time_median = median(solved_vt) if solved_vt else 0.0
    s.virtual_time_std = pstdev(solved_vt) if len(solved_vt) > 1 else 0.0
    s.virtual_time_all_mean = _safe_mean([e.virtual_time for e in episodes])

    finite_clear = [e.avg_clear_time for e in episodes if math.isfinite(e.avg_clear_time)]
    s.avg_clear_time_mean = _safe_mean(finite_clear)

    per_source = []
    for e in episodes:
        if e.sources_cleared:
            per_source.append(e.num_scans / e.sources_cleared)
    s.scans_per_source_mean = _safe_mean(per_source)

    attempts = sum(e.num_clears for e in episodes)
    fails = sum(e.failed_clears for e in episodes)
    s.failed_clear_rate = (fails / attempts) if attempts else 0.0

    s.wall_time_mean = _safe_mean([e.wall_time for e in episodes])
    s.steps_mean = _safe_mean([e.steps for e in episodes])
    return s
