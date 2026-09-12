from way4.planner.spatial import SpatialStopGenerator


def test_spatial_clustering_has_explicit_ablation_switch():
    assert SpatialStopGenerator(clustering="none")._bundle([], None) == []
    try:
        SpatialStopGenerator(clustering="unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid clustering mode must fail explicitly")
