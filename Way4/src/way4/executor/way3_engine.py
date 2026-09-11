"""Bridge from Way3's authoritative offline engine to the Way4 ``Env`` protocol.

Way3's ``jammerhunt/environment.py`` is a stdlib-only, self-certified faithful
copy of the offline simulator (its own test reproduces the §1.1 golden clock
``[105, 111, 194, 199]``). We load it *by file path* — exactly as the certificate
fallback loads Way3's ``coverage.py`` — so no ``jammerhunt`` package ``__init__``
runs, and wrap its raw ``(response, outcome)`` returns as Way4 ``Observation`` +
virtual clock. This honours 禁止1 (use the real simulator, never a reimplementation)
and gives M4 a true end-to-end cross-check plus the §13 Way3-vs-Way4 harness a
ready engine.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional, Tuple

from ..core.observation import Observation


def load_way3_environment():
    """Import Way3's ``environment.py`` by path. Returns the module or ``None``."""
    try:
        # this file: SX/Way4/src/way4/executor/way3_engine.py -> parents[4] == SX
        sx_root = Path(__file__).resolve().parents[4]
        env_py = sx_root / "Way3" / "jammerhunt" / "environment.py"
        if not env_py.is_file():
            return None
        name = "_way3_environment"
        spec = importlib.util.spec_from_file_location(name, env_py)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        # Register before exec: ``environment.py`` uses ``from __future__ import
        # annotations`` + ``@dataclass``, and the dataclass machinery resolves the
        # stringified annotations via ``sys.modules[cls.__module__]``.
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)   # stdlib-only, no relative imports
        except Exception:
            sys.modules.pop(name, None)
            raise
        return mod
    except Exception:
        return None


def _outcome_to_observation(out, time_s: float) -> Observation:
    """Map a Way3 ``MeasureOutcome`` to a Way4 ``Observation`` (note Way3 labels a
    bearing return ``"direction"``)."""
    result = out.result
    if result == "no_signal":
        return Observation.no_signal(time=time_s)
    if result == "near":
        return Observation.near(time=time_s)
    if result == "direction":
        return Observation.bearing(float(out.svd_deg), time=time_s)
    raise ValueError(f"unknown Way3 measure result {result!r}")


class Way3EngineAdapter:
    """Wrap a Way3 ``Engine`` as a Way4 ``Env``. The engine must be ``enter``-ed."""

    def __init__(self, engine) -> None:
        self.engine = engine

    def measure(self, x: float, y: float, channel: int) -> Tuple[Optional[Observation], float]:
        resp, out = self.engine.measure(x, y, channel)
        if out is None:   # deadline: command did not execute
            return None, float(resp.get("virtual_time_s", self.engine.virtual_time_s))
        vts = float(resp["virtual_time_s"])
        return _outcome_to_observation(out, vts), vts

    def clear(self, x: float, y: float, channel: int) -> Tuple[bool, float]:
        resp, out = self.engine.clear(x, y, channel)
        if out is None:   # deadline
            return False, float(resp.get("virtual_time_s", self.engine.virtual_time_s))
        return (out.result == "success"), float(resp["virtual_time_s"])
