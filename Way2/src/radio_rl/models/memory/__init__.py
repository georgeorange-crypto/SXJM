"""Temporal memories (pluggable). Importing this package registers the defaults."""

from __future__ import annotations

from .none import NoMemory

__all__ = ["NoMemory"]
