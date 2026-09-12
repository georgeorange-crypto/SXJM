from way4.belief.channel import ChannelBelief


def test_readiness_snapshot_has_unified_geometry_lifecycle_and_distance_fields():
    b = ChannelBelief(channel=1, arena_radius=1800.0)
    snapshot = b.readiness_snapshot(robot_pos=(0.0, 0.0), coverage_debt=0.25)
    assert {
        "area", "diameter", "mec_radius", "kappa", "connected_components",
        "coverage_debt", "initialization_ready", "robot_distance", "readiness_score",
    } <= set(snapshot)
    assert snapshot["coverage_debt"] == 0.25
    assert 0.0 <= snapshot["readiness_score"] <= 1.0
