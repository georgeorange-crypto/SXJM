"""Fusions (pluggable). Importing this package registers all variants.

The default (:mod:`concat`) concatenates and projects the context parts; the
gated and attention variants weight them with learned, data-dependent gates. All
share the ``list[[B, di]] -> [B, out_dim]`` contract, so each is a one-line config
swap (``algorithm.model.fusion.type=gated``).
"""

from __future__ import annotations

from .attention import AttentionFusion
from .concat import ConcatFusion
from .gated import GatedFusion

__all__ = ["ConcatFusion", "GatedFusion", "AttentionFusion"]
