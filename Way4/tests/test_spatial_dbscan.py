from way4.planner.spatial import SpatialStopGenerator


class C:
    def __init__(self, x, y, channels=(1,)):
        self.target = (x, y)
        self.scan_channels = channels
        self.refinement_gain = 0.0


def test_spatial_generator_uses_dbscan_and_preserves_noise():
    gen = SpatialStopGenerator(cluster_radius=10.0)
    clusters = gen._cluster_scans([C(0, 0), C(1, 0), C(100, 100)])
    assert sorted(c.n_members for c in clusters) == [1, 2]
