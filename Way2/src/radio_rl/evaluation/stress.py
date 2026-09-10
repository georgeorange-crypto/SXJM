"""Stress evaluation — worst-case robustness over adversarial case families.

Uses the same pipeline but injects hand-crafted hard cases (boundary clusters,
minimum receive radius, collinear sources, adversarial error fields, evasive
directional headings, ...) so we can see where a policy breaks before the
official run. Directional families are skipped for Problem 3 (all-omni).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..env.generators import STRESS_TYPES, generate_stress_case
from ..env.local_env import LocalEnv
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


def stress_types_for(problem: int) -> tuple[str, ...]:
    if problem >= 4:
        return STRESS_TYPES
    return tuple(t for t in STRESS_TYPES if not t.startswith("dir_"))


def run_stress(
    cfg: Any,
    n_seeds: int = 5,
    types: Optional[Sequence[str]] = None,
    field_kind: str = "adversarial",
    verbose: bool = True,
    max_steps: int = 100_000,
) -> dict[str, MetricsSummary]:
    """Run each stress family over ``n_seeds`` seeds; return per-family summaries."""
    problem = int(_node(cfg, "problem", 3))
    base_seed = int(_node(cfg, "seed", 0))
    families = list(types) if types is not None else list(stress_types_for(problem))

    results: dict[str, MetricsSummary] = {}
    for fam in families:
        episodes = []
        for i in range(n_seeds):
            case = generate_stress_case(
                fam, seed=base_seed + i, problem=problem, field_kind=field_kind
            )
            env = LocalEnv(case=case)
            pipe = Pipeline(cfg, env=env)
            stats, _ = pipe.run_episode(seed=base_seed + i, max_steps=max_steps)
            episodes.append(stats)
        summary = summarize(episodes)
        results[fam] = summary
        if verbose:
            print(f"[stress:{fam:>12}] success {summary.success_rate:5.1%}  "
                  f"clear {summary.clear_ratio_mean:5.1%}  "
                  f"vtime~{summary.virtual_time_all_mean:8.1f}s  "
                  f"failclr {summary.failed_clear_rate:4.1%}")
    return results
