import pytest

from way4.planner import BackboneStatus, build_boundary_caps


def test_boundary_caps_are_first_class_and_may_be_outside_arena():
    caps = build_boundary_caps(arena_radius=1800, detector_radius=1000,
                               n_caps=12, outward_offset=250)
    assert len(caps) == 12
    assert all(c.detector_points and c.covered_arc for c in caps)
    assert all(c.certificate_status == BackboneStatus.UNSATISFIED for c in caps)
    assert caps[0].detector_points[0][0] > 1800


def test_boundary_cap_parameters_are_validated():
    with pytest.raises(ValueError):
        build_boundary_caps(n_caps=2)
