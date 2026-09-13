from way4.metrics import (RouteEvent, ROLE_COVERAGE, compute_efficiency_metrics,
                          compute_route_metrics, compute_unnecessary_return_m)
from way4.evaluation_metrics import episode_row
from way4.routing.unified import UnifiedRoutePlanner


def test_route_diagnostics_distinguish_reverse_and_repeated_legs():
    events = [
        RouteEvent((10., 0.), 'MEASURE', 1, ROLE_COVERAGE),
        RouteEvent((20., 0.), 'MEASURE', 1, ROLE_COVERAGE),
        RouteEvent((10., 0.), 'MEASURE', 1, ROLE_COVERAGE),
    ]
    m = compute_route_metrics(events, (0., 0.))
    assert m.backtrack_m == 10.
    assert m.repeated_edge_m == 10.


def test_repeated_edge_is_partial_polyline_overlap_and_ratio_is_bounded():
    events = [
        RouteEvent((100., 0.), 'MEASURE', 1, ROLE_COVERAGE),
        RouteEvent((0., 0.), 'MEASURE', 1, ROLE_COVERAGE),
        RouteEvent((50., 0.), 'MEASURE', 1, ROLE_COVERAGE),
    ]
    m = compute_route_metrics(events, (0., 0.))
    assert m.repeated_edge_m == 150.
    assert compute_efficiency_metrics(
        total_time_s=10, move_distance_m=m.total_distance,
        repeated_edge_m=m.repeated_edge_m,
    ).repeated_edge_ratio == 0.6


def test_self_intersection_is_diagnostic_only_and_is_exported():
    events = [
        RouteEvent((10., 10.), 'MEASURE', 1, ROLE_COVERAGE),
        RouteEvent((0., 10.), 'MEASURE', 1, ROLE_COVERAGE),
        RouteEvent((10., 0.), 'MEASURE', 1, ROLE_COVERAGE),
    ]
    m = compute_route_metrics(events, (0., 0.))
    assert m.self_intersection_count == 1
    assert m.avoidable_crossing_count == 1
    assert m.avoidable_crossing_m > 0.0
    row = episode_row(type('R', (), {'n_channels': 1, 'virtual_time_s': 1,
                                      'total_distance_m': m.total_distance,
                                      'self_intersection_count': m.self_intersection_count})())
    assert row['self_intersection_count'] == 1


def test_unnecessary_return_requires_ready_task_evidence():
    points = [(0., 0.), (40., 0.), (200., 0.), (100., 0.)]
    assert compute_unnecessary_return_m(points) == 0.0
    assert compute_unnecessary_return_m(points, [((100., 0.), 1, 3)], near_tol=50.) == 100.


def test_unified_route_score_is_seconds():
    planner = UnifiedRoutePlanner()
    route = [(0., 0.), (100., 0.)]
    assert planner.route_cost_seconds(route) == 20.0  # 100 m / 5 m/s
    assert planner.route_score(route) == planner.route_cost_seconds(route)
