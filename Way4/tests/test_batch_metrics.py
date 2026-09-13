import pytest
from way4.metrics import compute_batch_metrics


def test_batch_metrics_records_size_and_early_stop_reasons():
    out = compute_batch_metrics([1, 3, 2], ["near", "replan"])
    assert out["scan_batch_count"] == 3
    assert out["mean_batch_size"] == pytest.approx(2)
    assert out["median_batch_size"] == 2
    assert out["early_stop_count"] == 2
    assert out["batch_stop_reason"] == ["near", "replan"]
