from way4.certificate import CertificateManager
from way4.core import Observation
from way4.planner import coverage_greedy


def test_coverage_greedy_is_deterministic_and_only_reads_certificates():
    cert = CertificateManager(n_channels=2)
    before = cert.certs[1].hard_complete
    a = coverage_greedy(cert, [1, 2], max_points=3)
    b = coverage_greedy(cert, [1, 2], max_points=3)
    assert a == b
    assert a.waypoints
    assert a.covered_gain > 0.0
    assert cert.certs[1].hard_complete is before


def test_coverage_greedy_marginal_gain_falls_after_scans():
    cert = CertificateManager(n_channels=1)
    first = coverage_greedy(cert, [1], max_points=1)
    cert.record_observation(1, first.waypoints[0], Observation.no_signal())
    second = coverage_greedy(cert, [1], start=first.waypoints[0], max_points=1)
    assert second.covered_gain <= first.covered_gain
