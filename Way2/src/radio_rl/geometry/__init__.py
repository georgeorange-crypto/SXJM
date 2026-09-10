"""Geometry layer — the mathematical state estimator and its primitives.

Pure geometry and belief-tracking. Torch-free; imports nothing from ``env`` or
``models`` (architecture principle B: models see tensors, never geometry).
"""

from __future__ import annotations

from .belief import (
    BeliefState,
    ChannelBelief,
    EstimatorConfig,
    ExclusionDisc,
    build_estimator_config,
)
from .information_gain import (
    bearing_spread_deg,
    expected_region_shrink,
    triangulation_quality,
)
from .region import (
    Halfplane,
    Region,
    convex_hull,
    diameter_circle_covers,
    disc_polygon,
    min_enclosing_circle,
    norm_deg,
    polygon_area,
    polygon_centroid,
    polygon_diameter,
    wedge_halfplanes,
)

__all__ = [
    # region primitives
    "Halfplane",
    "Region",
    "wedge_halfplanes",
    "convex_hull",
    "polygon_area",
    "polygon_centroid",
    "polygon_diameter",
    "min_enclosing_circle",
    "diameter_circle_covers",
    "disc_polygon",
    "norm_deg",
    # belief
    "BeliefState",
    "ChannelBelief",
    "EstimatorConfig",
    "ExclusionDisc",
    "build_estimator_config",
    # information gain
    "bearing_spread_deg",
    "triangulation_quality",
    "expected_region_shrink",
]
