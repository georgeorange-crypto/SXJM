"""Algorithm layer (pluggable).

Default: the greedy mathematical baseline (no torch). PPO and other learnable
algorithms register here and are built by name; importing this package must
never require torch, so learnable algorithms are imported lazily inside
:func:`build_agent`.
"""

from __future__ import annotations

from typing import Any

from ..core.registry import ALGORITHMS
from .base import Agent
from .heuristic import GreedyMathAgent  # registers "greedy_math" (torch-free)


def _get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    try:
        v = cfg.get(key) if hasattr(cfg, "get") else cfg[key]
    except Exception:
        return default
    return default if v is None else v


def build_agent(cfg: Any = None, **kwargs: Any) -> Agent:
    """Construct the agent selected by ``cfg`` (the ``algorithm`` config node)."""
    algo_type = str(_get(cfg, "type", "heuristic")).lower()
    if algo_type == "heuristic":
        name = str(_get(cfg, "name", "greedy_math"))
        return ALGORITHMS.create(name, **kwargs)
    if algo_type in ("ppo", "learned", "rl"):
        # lazy: only import torch-backed agents when actually requested
        from .ppo import build_ppo_agent  # noqa: F401  (registers + builds)

        return build_ppo_agent(cfg, **kwargs)
    # fall back to a registered name if given directly
    name = str(_get(cfg, "name", algo_type))
    return ALGORITHMS.create(name, **kwargs)


__all__ = ["Agent", "GreedyMathAgent", "build_agent"]
