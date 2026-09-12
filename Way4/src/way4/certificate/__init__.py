"""Way4 coverage-certificate layer (DESIGN.md §6, M3).

The ONLY component that may certify a channel ABSENT (禁止6). Three sound sources,
one fast planner heuristic that never certifies:

  * ``HardDiscCoverVerifier`` — conservative adaptive quadtree; the sole geometric
    certifier (M3A). False negatives allowed, false positives never.
  * ``omni_fallback_anchors`` / ``assert_backbone_covers`` — Way3's guaranteed
    completion template (M3B, Invariant C).
  * ``CertificateManager`` — per-channel state + the three-source disjunction
    (arbitrary disc cover ∨ legacy backbone ∨ cardinality), §6.4/§6.5.
  * ``CoverageGainMap`` — fast 40 m grid for planner scoring only (§6.6); never
    certifies (Invariant B).
"""

from .coverage_gain import CoverageGainMap
from .hard_disc_cover import (
    HardDiscCoverVerifier,
    CoverageCell,
    cell_fully_covered_by_disc,
    distance,
)
from .fallback import (
    omni_fallback_anchors,
    directional_fallback_anchors,
    ray_sweep_points,
    triangular_clear_sweep_points,
    assert_backbone_covers,
    ARENA_RADIUS_M,
    DETECTION_RADIUS_M,
)
from .manager import (
    CertificateManager,
    OmniChannelCertificate,
    CertificateSource,
    CardinalityState,
)
from .directional_certificate import (
    DirectionalCertificateResult,
    DirectionalCounterexample,
    check_point,
    disk_grid_samples,
    nearby_convex_hull,
    point_in_triangle,
    safe_triangle,
    verify_samples,
)

__all__ = [
    "CoverageGainMap",
    "HardDiscCoverVerifier",
    "CoverageCell",
    "cell_fully_covered_by_disc",
    "distance",
    "omni_fallback_anchors",
    "directional_fallback_anchors",
    "ray_sweep_points",
    "triangular_clear_sweep_points",
    "assert_backbone_covers",
    "ARENA_RADIUS_M",
    "DETECTION_RADIUS_M",
    "CertificateManager",
    "OmniChannelCertificate",
    "CertificateSource",
    "CardinalityState",
    "DirectionalCertificateResult",
    "DirectionalCounterexample",
    "check_point",
    "disk_grid_samples",
    "nearby_convex_hull",
    "point_in_triangle",
    "safe_triangle",
    "verify_samples",
]
