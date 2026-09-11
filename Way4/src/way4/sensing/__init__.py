"""Active sensing — minimax next-best-view for bearing refinement (DESIGN.md §8)."""

from .nbv import MinimaxNBV, NBVResult

__all__ = ["MinimaxNBV", "NBVResult"]
