from way4.certificate import CertificateManager
from way4.core import Observation


def test_coverage_debt_is_persisted_and_decreases():
    mgr = CertificateManager(n_channels=3)
    before = mgr.coverage_debt(3)
    mgr.record_observation(3, (0.0, 0.0), Observation.no_signal())
    after = mgr.coverage_debt(3)
    assert 0.0 <= after < before <= 1.0
    assert mgr.uncovered_area(3) >= 0.0
    snap = mgr.coverage_snapshot(3)
    assert {"coverage_debt", "uncovered_area", "active_scan_count"} <= snap.keys()
