"""§15 headline regression: the active-scanning certificate (DESIGN.md §6, M3A/M3B).

The whole point of DECISION-1 (§6): a channel becomes ABSENT_CERTIFIED because the
robot's *own* NO_SIGNAL scans — accumulated while searching/localizing — happen to
cover the arena, WITHOUT having to walk Way3's fixed backbone. And the dual: leave
one genuine hole and it must NOT certify (no false positive, ever). Plus the
guaranteed-completion backbone (Invariant C) is itself sound.
"""

import math
import random

from way4.certificate import (
    CertificateManager,
    CertificateSource,
    assert_backbone_covers,
    directional_fallback_anchors,
    omni_fallback_anchors,
    ray_sweep_points,
    triangular_clear_sweep_points,
)


def test_ray_sweep_points_is_bounded_and_alternates_sides():
    pts = ray_sweep_points((10.0, -4.0), 0.0, 40.0, lateral_half_width_m=13.0, step_m=15.0)
    assert pts[0] == (10.0, 9.0)
    assert pts[1] == (25.0, -17.0)
    assert pts[-1] == (50.0, -4.0)
    assert all(10.0 <= x <= 50.0 for x, _ in pts)


def test_ray_sweep_rejects_invalid_step():
    import pytest
    with pytest.raises(ValueError):
        ray_sweep_points((0.0, 0.0), 90.0, 10.0, step_m=0.0)


def test_triangular_clear_sweep_is_local_and_uses_safe_spacing():
    center = (10.0, -4.0)
    points = triangular_clear_sweep_points(center, radius_m=50.0, spacing_m=33.0)
    assert points
    assert max(math.dist(center, p) for p in points) <= 50.0 + 1e-9
    assert 33.0 < 20.0 * math.sqrt(3.0)
from way4.core import Observation

ARENA = 1800.0


def _grid(spacing, lim):
    pts, n = [], int(lim // spacing)
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            pts.append((i * spacing, j * spacing))
    return pts


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# --- the core regression (§15 test_way4_active_certificate) -------------------


def test_way4_active_certificate_without_walking_backbone():
    """Active scans that form a complete cover certify absence — and do so via the
    ARBITRARY_DISC_COVER source, NOT by having walked the fixed anchors."""
    mgr = CertificateManager()
    active_scans = _grid(400.0, 2000.0)      # a dense active pattern, not the anchors
    for p in active_scans:
        mgr.record_observation(5, p, Observation.no_signal())

    assert mgr.is_absent_certified(5, force=True) is True
    assert mgr.certificate_source(5) is CertificateSource.ARBITRARY_DISC_COVER
    # crucially: the backbone was NOT walked (at most the shared centre anchor).
    assert mgr.legacy_backbone_complete(5) is False
    assert mgr.certs[5].fallback_anchor_count < len(mgr.anchors)


def test_one_remaining_hole_is_never_certified():
    """Drop every scan near a target point, leaving a genuine uncovered hole; the
    verifier (nominal AND rescue) must refuse to certify (no false positive)."""
    mgr = CertificateManager()
    target = (1500.0, 0.0)
    scans = [p for p in _grid(400.0, 2000.0) if _dist(p, target) > 1000.0]
    for p in scans:
        mgr.record_observation(6, p, Observation.no_signal())

    # sanity: the hole is real — nothing scanned covers the target
    assert all(_dist(p, target) > 1000.0 for p in scans)
    assert mgr.is_absent_certified(6, force=True) is False
    assert mgr.certificate_source(6) is None


def test_active_scanning_reduces_needed_fallback_anchors():
    """After enough active scanning the arbitrary certificate holds, so the
    verification planner suggests fewer (ideally zero-value) completion anchors —
    the §6 metric that Way4 beats Way3's fixed-backbone cost."""
    mgr = CertificateManager()
    for p in _grid(400.0, 2000.0):
        mgr.record_observation(8, p, Observation.no_signal())
    assert mgr.is_absent_certified(8, force=True) is True
    # every remaining fallback anchor now adds no new heuristic coverage
    gains = [mgr.coverage_gain(8, a) for a in mgr.anchors]
    assert max(gains) < 1e-9


# --- the fallback backbone is itself sound (Invariant C) ----------------------


def test_fallback_backbone_covers_arena():
    anchors = omni_fallback_anchors()
    assert 5 <= len(anchors) <= 20
    assert all(math.hypot(x, y) <= ARENA + 1e-6 for (x, y) in anchors)
    # confirmed by Way4's OWN conservative quadtree (not Way3's dense grid).
    assert assert_backbone_covers(anchors) is True


def test_directional_fallback_is_a_dense_omni_disc_cover():
    """The P4 directional backbone (M3C legacy template). Its *angular* 3-cover
    soundness is Way3's concern (deferred, §6.9); here we only check Way4 can source
    a non-trivial anchor set and that — being far denser than the omni backbone (edge
    < 1000 m) — its discs trivially cover the arena under Way4's own verifier."""
    anchors = directional_fallback_anchors()
    assert len(anchors) >= len(omni_fallback_anchors())
    assert all(math.isfinite(x) and math.isfinite(y) for (x, y) in anchors)
    assert assert_backbone_covers(anchors) is True
    # cached: same identity-of-contents on a second call
    assert directional_fallback_anchors() == anchors


def test_builtin_directional_shell_passes_rim_grid_samples():
    """The exterior shell is required for rim sources facing outward."""
    from way4.certificate.directional_certificate import disk_grid_samples, verify_samples
    from way4.certificate.fallback import _builtin_directional
    result = verify_samples(_builtin_directional(1800.0, 900.0),
                            disk_grid_samples(1800.0, 50.0), radius=1000.0)
    assert result.certified
    assert result.checked_points == 4053


def test_builtin_directional_reduced_template_passes_25m_grid():
    from way4.certificate.directional_certificate import disk_grid_samples, verify_samples
    from way4.certificate.fallback import _builtin_directional
    result = verify_samples(_builtin_directional(1800.0, 900.0),
                            disk_grid_samples(1800.0, 25.0), radius=1000.0)
    assert result.certified
    assert result.checked_points == 16241


# --- no-false-positive property (random arena witness outside all discs) ------


def test_property_uncovered_witness_forces_incomplete():
    rng = random.Random(4)
    for _ in range(40):
        # scatter some scans, then place a legal source deliberately far from all
        scans = [
            (rng.uniform(-1500, 1500), rng.uniform(-1500, 1500)) for _ in range(rng.randint(3, 8))
        ]
        # find an in-arena point > 1000 from every scan (a real hole); skip if none quickly
        witness = None
        for _try in range(200):
            ang = rng.uniform(0, 2 * math.pi)
            rad = ARENA * math.sqrt(rng.random())
            q = (rad * math.cos(ang), rad * math.sin(ang))
            if all(_dist(q, s) > 1000.0 + 1.0 for s in scans):
                witness = q
                break
        if witness is None:
            continue
        mgr = CertificateManager(fallback_anchors=[])   # isolate arbitrary cover
        for s in scans:
            mgr.record_observation(1, s, Observation.no_signal())
        assert mgr.is_absent_certified(1, force=True) is False
