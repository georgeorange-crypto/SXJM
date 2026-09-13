from way4.planner import BackboneManager, BackboneNode, BackboneStatus


def test_backbone_nodes_track_responsibility_not_mandatory_visits():
    m = BackboneManager([BackboneNode("a", (0., 0.), coverage_cells=(1, 2),
                                    certificate_holes=("h",))])
    assert m.remaining_route_nodes()[0].status == BackboneStatus.UNSATISFIED
    assert m.debt("a").remaining_cells == 2
    m.satisfy_by_observation(["a"], source="remote-b")
    assert not m.remaining_route_nodes()


def test_backbone_visit_status_and_duplicate_guard():
    m = BackboneManager()
    m.add_node(BackboneNode("a", (1., 2.)))
    m.mark_visited("a")
    assert m.nodes["a"].status == BackboneStatus.VISITED


def test_dynamic_refine_observation_repays_backbone_responsibility():
    m = BackboneManager([BackboneNode("a", (0., 0.), coverage_cells=(1, 2),
                                    certificate_holes=("h",))])
    assert m.apply_observation(coverage_cells=(1,), certificate_holes=("h",)) == ["a"]
    assert m.nodes["a"].status == BackboneStatus.PARTIALLY_SATISFIED
    assert m.debt("a").remaining_cells == 1
    assert m.apply_observation(coverage_cells=(2,)) == ["a"]
    assert m.nodes["a"].status == BackboneStatus.SATISFIED_BY_OTHER_OBSERVATION


def test_backbone_initializes_one_global_open_route_in_seconds():
    m = BackboneManager([
        BackboneNode("a", (100., 0.)), BackboneNode("b", (200., 0.)),
    ])
    assert m.initialize_open_route(speed_mps=10.) == ("a", "b")
    assert m.route_cost_s == 20.0


def test_backbone_prune_removes_satisfied_nodes_without_reordering_remaining_route():
    m = BackboneManager([
        BackboneNode("a", (100., 0.)), BackboneNode("b", (200., 0.)),
        BackboneNode("c", (300., 0.)),
    ])
    m.initialize_open_route(speed_mps=10.)
    m.mark_visited("b")
    assert m.prune_route(speed_mps=10.) == ("a", "c")
    assert m.route_cost_s == 30.0


def test_dynamic_information_can_only_accept_a_shortened_subsequence():
    m = BackboneManager([BackboneNode("a", (100., 0.)), BackboneNode("b", (200., 0.))])
    m.initialize_open_route(speed_mps=10.)
    assert m.accept_shortened_route(("a",), 10.0)
    assert not m.accept_shortened_route(("a", "new"), 10.0)
    assert not m.accept_shortened_route(("a",), 11.0)
