from way4.certificate import (
    check_point, point_in_triangle, safe_triangle, verify_samples,
)


def test_triangle_certificate_and_edge_inclusion():
    tri = ((0.0, 0.0), (1000.0, 0.0), (500.0, 866.0254038))
    assert safe_triangle(tri, eps=1e-6)
    assert point_in_triangle((500.0, 288.675), tri)
    assert point_in_triangle((500.0, 0.0), tri)


def test_directional_counterexample_returns_escape_direction():
    witness = check_point((0.0, 0.0), ((100.0, 0.0), (0.0, 100.0)))
    assert witness is not None
    assert 0.0 <= witness.escape_direction_deg < 360.0


def test_surrounded_local_geometry_passes():
    detectors = ((1000.0, 0.0), (-500.0, 866.0254038), (-500.0, -866.0254038))
    result = verify_samples(detectors, [(0.0, 0.0)])
    assert result.certified
    assert result.counterexample is None


def test_boundary_outward_directional_source_is_not_certified_by_inward_scans():
    """A rim source can be visible only through a narrow outward-facing case.

    Three detectors clustered on the arena-facing side are all within range,
    but leave a >180 degree angular gap as seen from the rim point.  The
    certificate must expose that witness instead of treating distance cover as
    directional cover.
    """
    rim = (1800.0, 0.0)
    inward = ((1000.0, 0.0), (1200.0, 200.0), (1200.0, -200.0))
    witness = check_point(rim, inward, radius=1000.0)
    assert witness is not None
    assert witness.point == rim
    assert len(witness.nearby_detectors) == 3
