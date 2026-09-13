from way4.planner import build_p4_triangle_mesh


def test_p4_triangle_mesh_has_deterministic_center_ring_and_unique_detectors():
    triangles, detectors = build_p4_triangle_mesh(arena_radius=1800, n_sectors=12)
    assert len(triangles) == 12
    assert len(detectors) == 13
    assert all(len(t.vertices) == 3 and t.covered_region for t in triangles)
    assert triangles == build_p4_triangle_mesh(arena_radius=1800, n_sectors=12)[0]
