"""A tiny name -> factory registry, one per pluggable category.

Plugins register themselves with a decorator::

    @MEMORIES.register("lstm")
    class LSTMMemory(TemporalMemory):
        ...

and are built by name from config::

    memory = MEMORIES.create(cfg.memory.type, cfg=cfg.memory)

Golden rule for optional heavy dependencies (torch-geometric, mamba-ssm, ...):
the plugin *module* must import cleanly without them, and the heavy import must
happen inside ``__init__`` / build — never at module top level. That way an
unavailable advanced plugin can never break the core PPO+MLP path.
"""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        key = name.lower()

        def deco(factory: Callable[..., T]) -> Callable[..., T]:
            if key in self._factories:
                raise KeyError(f"{self.kind} '{name}' already registered")
            self._factories[key] = factory
            return factory

        return deco

    def register_factory(self, name: str, factory: Callable[..., T]) -> None:
        key = name.lower()
        if key in self._factories:
            raise KeyError(f"{self.kind} '{name}' already registered")
        self._factories[key] = factory

    def get(self, name: str) -> Callable[..., T]:
        key = str(name).lower()
        if key not in self._factories:
            raise KeyError(
                f"unknown {self.kind} '{name}'. "
                f"available: {sorted(self._factories)}"
            )
        return self._factories[key]

    def create(self, name: str, *args, **kwargs) -> T:
        return self.get(name)(*args, **kwargs)

    def available(self) -> list[str]:
        return sorted(self._factories)

    def __contains__(self, name: str) -> bool:
        return str(name).lower() in self._factories


# --- One registry per pluggable category ----------------------------------
ALGORITHMS: Registry = Registry("algorithm")
FEATURE_BUILDERS: Registry = Registry("feature_builder")
CHANNEL_ENCODERS: Registry = Registry("channel_encoder")
SPATIAL_ENCODERS: Registry = Registry("spatial_encoder")
GRAPH_ENCODERS: Registry = Registry("graph_encoder")
MEMORIES: Registry = Registry("memory")
FUSIONS: Registry = Registry("fusion")
WORLD_MODELS: Registry = Registry("world_model")

ERROR_FIELDS: Registry = Registry("error_field")
SOURCE_GENERATORS: Registry = Registry("source_generator")
INFO_METRICS: Registry = Registry("info_metric")


def require(module_name: str, extra: str) -> object:
    """Import an optional dependency or raise a friendly, actionable error."""
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise ImportError(
            f"optional dependency '{module_name}' is not installed; "
            f"install it with `pip install radio_rl[{extra}]` "
            f"(the core PPO+MLP path does not need it)."
        ) from exc
