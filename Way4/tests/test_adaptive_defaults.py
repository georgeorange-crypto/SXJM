import inspect
from way4.pipeline import Way4Pipeline


def test_formal_pipeline_defaults_enable_adaptive_scan_and_batch_stop():
    params = inspect.signature(Way4Pipeline.__init__).parameters
    assert params['adaptive_scan'].default is True
    assert params['batch_stop'].default is True
