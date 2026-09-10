"""World-model layer (pluggable).

Default: the deterministic :class:`AnalyticalWorldModel`. Learned world models
(residual dynamics, Dreamer latent) register here later; a missing heavy
dependency for one of those must never break this import or the analytical path.
"""

from __future__ import annotations

from ..core.registry import WORLD_MODELS
from .analytical import AnalyticalWorldModel
from .base import WorldModel


def build_world_model(cfg=None) -> WorldModel:
    """Construct the world model named by ``cfg.type`` (default: analytical)."""
    name = "analytical"
    if cfg is not None:
        try:
            name = str(cfg.get("type", "analytical") if hasattr(cfg, "get") else cfg["type"])
        except Exception:
            name = "analytical"
    return WORLD_MODELS.create(name)


__all__ = ["WorldModel", "AnalyticalWorldModel", "build_world_model"]
