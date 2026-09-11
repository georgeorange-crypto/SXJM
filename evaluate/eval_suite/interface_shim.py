"""
从 Way3/jammerhunt.interface 复用统一 World 契约（World / MeasureObs / ClearObs）。

单一来源，避免重复定义导致 isinstance/契约漂移。import 时确保 Way3 已在 sys.path。
"""

from __future__ import annotations

from . import _paths  # noqa: F401  (side effect: Way3 on sys.path)

from jammerhunt.interface import (  # noqa: E402
    ClearObs,
    HttpClient,
    HttpWorld,
    LocalWorld,
    MeasureObs,
    RunnerWorld,
    World,
)

__all__ = [
    "World", "MeasureObs", "ClearObs",
    "LocalWorld", "RunnerWorld", "HttpWorld", "HttpClient",
]
