"""World-model layer (pluggable).

Default: the deterministic :class:`AnalyticalWorldModel`, imported eagerly and
torch-free. Learned world models (:mod:`residual`, :mod:`dreamer`) import torch,
so they are imported **lazily by name** only when selected — importing this
package, and building the default model on every pipeline construction, never
pulls in torch. A missing/broken heavy dependency for a learned model surfaces
only when that model is explicitly requested; it can never break the analytical
path (advanced plugins never break the core).
"""

from __future__ import annotations

import importlib

from ..core.registry import WORLD_MODELS
from .analytical import AnalyticalWorldModel
from .base import WorldModel

# name -> module that registers it (imported on demand; each imports torch).
_LAZY_MODULES = {
    "residual": "radio_rl.world_model.residual",
    "dreamer": "radio_rl.world_model.dreamer",
}


def _cfg_get(cfg, key, default):
    if cfg is None:
        return default
    try:
        v = cfg.get(key, default) if hasattr(cfg, "get") else cfg[key]
    except Exception:
        return default
    return default if v is None else v


def build_world_model(cfg=None) -> WorldModel:
    """Construct the world model named by ``cfg.type`` (default: analytical).

    Analytical is built with no arguments (torch-free). Learned models receive
    the config node so they can size their networks; if a learned model does not
    accept ``cfg=`` we fall back to a no-argument construction.
    """
    name = str(_cfg_get(cfg, "type", "analytical")).lower()

    if name in _LAZY_MODULES and name not in WORLD_MODELS:
        importlib.import_module(_LAZY_MODULES[name])  # registers the class

    if name == "analytical":
        return WORLD_MODELS.create(name)

    try:
        return WORLD_MODELS.create(name, cfg=cfg)
    except TypeError:
        return WORLD_MODELS.create(name)


__all__ = ["WorldModel", "AnalyticalWorldModel", "build_world_model"]
