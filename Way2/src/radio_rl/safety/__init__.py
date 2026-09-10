"""Safety layer — the last legality gate before the environment."""

from __future__ import annotations

from typing import Any

from .shield import SafetyShield


def build_safety_shield(cfg: Any = None) -> SafetyShield:
    return SafetyShield(cfg)


__all__ = ["SafetyShield", "build_safety_shield"]
