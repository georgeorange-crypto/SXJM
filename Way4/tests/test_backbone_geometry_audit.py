from way4.planner import (audit_backbone_geometry, build_boundary_caps,
                          build_p4_triangle_mesh)


def test_backbone_geometry_audit_checks_all_required_dimensions():
    triangles, _ = build_p4_triangle_mesh(arena_radius=600, n_sectors=12)
    caps = build_boundary_caps(arena_radius=600, n_caps=12, outward_offset=50)
    audit = audit_backbone_geometry(triangles, caps, arena_radius=600,
                                    detector_radius=1000,
                                    samples=[(0., 0.)])
    assert audit.passed
    assert audit.side_lengths_ok and audit.boundary_support_ok
    assert audit.full_arena_inclusion_ok
