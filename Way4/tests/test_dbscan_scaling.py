from time import perf_counter
from way4.toolbox import dbscan


def test_dbscan_handles_large_repeated_candidate_pool():
    points = [(float(i % 50), float(i // 50)) for i in range(2000)]
    t0 = perf_counter()
    labels = dbscan(points, eps=2.0, min_samples=2)
    assert len(labels) == len(points)
    assert perf_counter() - t0 < 2.0
