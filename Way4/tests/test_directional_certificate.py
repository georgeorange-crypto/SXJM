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
