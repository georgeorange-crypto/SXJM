"""Way4 macro executor (DESIGN.md §7, §13)."""

from .macro_executor import (
    Env,
    ExecutionResult,
    MacroExecutor,
    PrimitiveResult,
)
from .way3_engine import (
    Way3EngineAdapter,
    load_way3_environment,
)

__all__ = [
    "Env",
    "ExecutionResult",
    "MacroExecutor",
    "PrimitiveResult",
    "Way3EngineAdapter",
    "load_way3_environment",
]
