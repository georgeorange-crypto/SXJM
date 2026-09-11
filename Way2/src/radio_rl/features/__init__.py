"""The features layer: belief/candidates -> fixed-shape tensors.

The bridge between the torch-free mathematical core and the learnable models.
:class:`FeatureSpec` / :class:`FeatureBundle` are the tensor contract (torch,
no geometry); :class:`DefaultFeatureBuilder` is the extractor (geometry -> tensors).

Importing this package imports torch. It is only imported on the learnable path,
lazily, so the default math pipeline stays torch-free.
"""

from __future__ import annotations

from .base import FeatureBuilder
from .builder import DefaultFeatureBuilder, build_feature_builder
from .spec import FeatureBundle, FeatureSpec

__all__ = [
    "FeatureBuilder",
    "FeatureSpec",
    "FeatureBundle",
    "DefaultFeatureBuilder",
    "build_feature_builder",
]
