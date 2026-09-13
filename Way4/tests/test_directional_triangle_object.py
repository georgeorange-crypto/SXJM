from way4.planner import BackboneStatus, DirectionalTriangle


def test_directional_triangle_records_three_detectors_region_guarantee_and_status():
    t = DirectionalTriangle("t1", ((0., 0.), (1., 0.), (0., 1.)),
                            covered_region=((0., 0.), (1., 1.)))
    assert t.detector_a == (0., 0.)
    assert t.detector_b == (1., 0.)
    assert t.detector_c == (0., 1.)
    assert t.directional_guarantee
    assert t.certificate_status == BackboneStatus.UNSATISFIED
