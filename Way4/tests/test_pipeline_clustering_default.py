import inspect
from way4.pipeline import Way4Pipeline


def test_pipeline_defaults_to_safe_nonclustered_mode():
    parameter = inspect.signature(Way4Pipeline.__init__).parameters["spatial_clustering"]
    assert parameter.default == "none"
