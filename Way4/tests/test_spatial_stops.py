"""P0 #2/#3 — SpatialStop object + spatial candidate generation.

These are the space-centric additions behind ``planner=spatial``. They are unit-level
and deliberately do NOT touch the pipeline: the ``planner_mode`` wiring lands with its
own integration test once the shared ``pipeline.py`` is free (it is currently held by a
concurrent session's uncommitted work). What is pinned here:

  * ``SpatialStop.primitives()`` expands to *measures-then-clears* — the mixed batch the
    action-centric ``MacroCandidate`` cannot express.
  * ``SpatialStopGenerator`` is **non-destructive**: every legacy candidate is returned
    unchanged (passthrough), and bundles are *added* on top. No service is ever dropped
    — the guarantee that makes full-clear provably safe under the spatial planner.
  * co-located scans merge into one bundle (union of channels); a co-located clear folds
    into that stop; a standalone clear stays a legacy CLEAR candidate.
  * the planner grows a STOP outcome branch that predicts a bundle's effect (holes near
    the stop covered, folded clears removed).
"""

from way4.core import (
    AnalyticalCostModel,
    MacroActionType,
    MacroCandidate,
    PrimitiveKind,
    RobotState,
    SpatialStop,
)
from way4.planner import CostView, OutcomePredictor, SpatialStopGenerator


# --- SpatialStop.primitives() ---------------------------------------------------


def test_spatial_stop_primitives_measures_then_clears():
    stop = SpatialStop(
        action_type=MacroActionType.STOP,
        target=(10.0, 0.0),
        scan_channels=(3, 7),
        clear_channels=(3,),
        clear_targets=((100.0, 5.0),),
    )
    prims = stop.primitives()
    kinds = [p.kind for p in prims]
    assert kinds == [PrimitiveKind.MEASURE, PrimitiveKind.MEASURE, PrimitiveKind.CLEAR]
    # both measures at the stop's shared waypoint, in batch order
    assert prims[0].target == (10.0, 0.0) and prims[0].channel == 3
    assert prims[1].target == (10.0, 0.0) and prims[1].channel == 7
    # the clear runs at its own blind-clear point, on its own channel
    assert prims[2].target == (100.0, 5.0) and prims[2].channel == 3
    assert stop.n_services == 3


def test_spatial_stop_pure_scan_has_no_clears():
    stop = SpatialStop(
        action_type=MacroActionType.STOP, target=(0.0, 0.0), scan_channels=(1, 2, 4)
    )
    prims = stop.primitives()
    assert all(p.kind == PrimitiveKind.MEASURE for p in prims)
    assert [p.channel for p in prims] == [1, 2, 4]


# --- SpatialStopGenerator: bundling + non-destructive passthrough ---------------


class _StubBase:
    """Minimal stand-in for ``CandidateGenerator``: returns a fixed candidate list and
    exposes a ``.cost`` (the generator reads only these two)."""

    def __init__(self, cands):
        self._cands = list(cands)
        self.cost = AnalyticalCostModel()

    def generate(self, belief, certificate, state, scan_mode=None):
        return list(self._cands)


def _scan(target, channels, refine=0.0):
    m = MacroCandidate(MacroActionType.EXPLORE, target, scan_channels=tuple(channels))
    m.refinement_gain = float(refine)
    return m


def _clear(target, channel):
    return MacroCandidate(MacroActionType.CLEAR, target, clear_channel=channel)


_STATE = RobotState(0.0, 0.0, 1, 0)


def test_bundles_colocated_scans_with_union_channels():
    legacy = [_scan((500.0, 0.0), (1, 2)), _scan((600.0, 0.0), (2, 3))]
    gen = SpatialStopGenerator(_StubBase(legacy), cluster_radius=1000.0)
    out = gen.generate(None, None, _STATE)
    stops = [c for c in out if isinstance(c, SpatialStop)]
    assert len(stops) == 1
    # union of both members' channels, de-duplicated, order-preserving
    assert stops[0].scan_channels == (1, 2, 3)
    assert stops[0].action_type == MacroActionType.STOP
    assert stops[0].expected_time > 0.0


def test_passthrough_is_non_destructive():
    """Every legacy candidate object survives verbatim in the spatial output (identity),
    so the spatial planner can always replicate the legacy plan — full-clear can't drop."""
    legacy = [
        _scan((500.0, 0.0), (1, 2)),
        _scan((600.0, 0.0), (2, 3)),
        _clear((9000.0, 0.0), 5),
    ]
    gen = SpatialStopGenerator(_StubBase(legacy), cluster_radius=1000.0)
    out = gen.generate(None, None, _STATE)
    for c in legacy:
        assert c in out                      # nothing dropped, nothing mutated away
    # and at least one bundle was added on top
    assert any(isinstance(c, SpatialStop) for c in out)


def test_standalone_clear_stays_legacy():
    """A clear with no scan cluster in range is not bundled — it remains a legacy CLEAR
    candidate (routing pure-clear sequences is Phase D, not here)."""
    legacy = [_scan((500.0, 0.0), (1,)), _scan((600.0, 0.0), (2,)), _clear((9000.0, 0.0), 5)]
    gen = SpatialStopGenerator(_StubBase(legacy), cluster_radius=1000.0)
    out = gen.generate(None, None, _STATE)
    stops = [c for c in out if isinstance(c, SpatialStop)]
    # the bundle carries the co-located scans but NOT the far clear
    assert stops and all(5 not in s.clear_channels for s in stops)
    # the far clear is still on the table as its own legacy candidate
    assert _clear_present(out, channel=5)


def test_colocated_clear_folds_into_stop():
    legacy = [
        _scan((500.0, 0.0), (1, 2)),
        _scan((550.0, 0.0), (2,)),
        _clear((560.0, 10.0), 2),            # within cluster_radius of the scan cluster
    ]
    gen = SpatialStopGenerator(_StubBase(legacy), cluster_radius=1000.0)
    out = gen.generate(None, None, _STATE)
    stops = [c for c in out if isinstance(c, SpatialStop)]
    assert len(stops) == 1
    s = stops[0]
    assert s.clear_channels == (2,)
    assert s.clear_targets == ((560.0, 10.0),)
    # the mixed bundle expands to measures then the folded clear
    kinds = [p.kind for p in s.primitives()]
    assert PrimitiveKind.MEASURE in kinds and kinds[-1] == PrimitiveKind.CLEAR


def test_lone_scan_no_clear_emits_no_bundle():
    """A single isolated scan with no clear to fold would only duplicate a legacy
    candidate, so no SpatialStop is emitted for it."""
    legacy = [_scan((500.0, 0.0), (1,))]
    gen = SpatialStopGenerator(_StubBase(legacy), cluster_radius=1000.0)
    out = gen.generate(None, None, _STATE)
    assert not any(isinstance(c, SpatialStop) for c in out)
    assert legacy[0] in out


# --- planner STOP outcome branch ------------------------------------------------


def test_predictor_stop_branch_covers_holes_and_removes_clear():
    base = CostView(
        pos=(0.0, 0.0),
        clearable_targets=[(100.0, 0.0), (9000.0, 0.0)],
        unknown_holes=[(50.0, 0.0), (9000.0, 50.0)],
        n_unknown=2,
        anchors=[],
    )
    pred = OutcomePredictor(detection_radius=1000.0)
    stop = SpatialStop(
        action_type=MacroActionType.STOP,
        target=(0.0, 0.0),
        scan_channels=(1, 2),
        clear_channels=(1,),
        clear_targets=((100.0, 0.0),),
    )
    outs = pred.predict(stop, belief=None, base=base)
    assert len(outs) == 1 and outs[0].label == "stop"
    v = outs[0].view
    # the batch covers holes within detection_radius of the stop; far holes remain
    assert (50.0, 0.0) not in v.unknown_holes
    assert (9000.0, 50.0) in v.unknown_holes
    # the folded clear removes its target; the unrelated clearable remains
    assert (100.0, 0.0) not in v.clearable_targets
    assert (9000.0, 0.0) in v.clearable_targets
    # the base view is not mutated (copy-on-predict)
    assert (50.0, 0.0) in base.unknown_holes


# --- helpers --------------------------------------------------------------------


def _clear_present(cands, channel):
    for c in cands:
        if c.action_type == MacroActionType.CLEAR and getattr(c, "clear_channel", None) == channel:
            return True
    return False
