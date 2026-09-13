from way4.planner import center_ring_baseline, optimize_p3_backbone


def test_p3_backbone_optimizer_is_reproducible():
    baseline = center_ring_baseline(arena_radius=600, detection_radius=400,
                                    n_ring=8, ring_radius=400)
    out = optimize_p3_backbone(baseline, arena_radius=600, detection_radius=400,
                               grid_step=100)
    assert out.n_points > 0 and out.objective >= 0
    assert out.points == optimize_p3_backbone(
        baseline, arena_radius=600, detection_radius=400, grid_step=100).points
