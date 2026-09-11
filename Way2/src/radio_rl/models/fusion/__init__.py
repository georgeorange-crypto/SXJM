"""Fusions (pluggable). Importing this package registers the defaults."""

from __future__ import annotations

from .concat import ConcatFusion

__all__ = ["ConcatFusion"]
