import inspect
from way4.pipeline import Way4Pipeline


def test_pipeline_exposes_spatial_clustering_ablation():
    # Constructor-level contract; the simulator behavior is covered by spatial
    # integration tests and the existing pipeline smoke suite.
    assert "spatial_clustering" in inspect.signature(Way4Pipeline.__init__).parameters
