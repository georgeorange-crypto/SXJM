from way4.pipeline import Way4Pipeline


class E:
    def measure(self, x, y, channel): return None, 0.0
    def clear(self, x, y, channel): return False, 0.0


def test_pipeline_exposes_unified_route_diagnostic_after_choice():
    p = Way4Pipeline(E(), n_channels=2)
    # no candidate is required: route planner state is initialized and safe
    assert p.last_unified_route is None
    assert p.unified_route is not None
    p._choose_macro()
    assert p.last_unified_route is not None
    assert p.last_unified_route.length >= 0.0
