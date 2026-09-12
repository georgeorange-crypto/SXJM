from way4.metrics import RouteEvent, ROLE_COVERAGE, compute_route_metrics


def test_route_diagnostics_distinguish_reverse_and_repeated_legs():
    events = [
        RouteEvent((10., 0.), 'MEASURE', 1, ROLE_COVERAGE),
        RouteEvent((20., 0.), 'MEASURE', 1, ROLE_COVERAGE),
        RouteEvent((10., 0.), 'MEASURE', 1, ROLE_COVERAGE),
    ]
    m = compute_route_metrics(events, (0., 0.))
    assert m.backtrack_m == 10.
    assert m.repeated_edge_m == 10.
    assert m.unnecessary_return_m == 20.
