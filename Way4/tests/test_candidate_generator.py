"""§7–8 CandidateGenerator — belief/certificate/state → ranked macro actions.

Pins the family rules: a CLEAR per clearable channel at its MEC centre with a
cost-model time; a REFINE at the minimax-NBV viewpoint that pins its own channel in
the scan batch; multipurpose EXPLORE waypoints that actually carry UNKNOWN coverage
gain; EXIT only once everything is resolved; and — the soundness rule — a resolved
channel (CLEARED / ABSENT_CERTIFIED / LOCALIZED) never lands in a scan batch.
"""

from math import hypot

import pytest

from way4.belief import BeliefState, ChannelStatus
from way4.certificate import CertificateManager
from way4.channels import SchedulerMode
from way4.core import AnalyticalCostModel, MacroActionType, RobotState
from way4.planner import CandidateGenerator
from way4.sensing import MinimaxNBV


def _fresh(n=6):
    return BeliefState(n_channels=n), CertificateManager(n_channels=n)


def _by_type(cands, t):
    return [c for c in cands if c.action_type == t]


def test_cardinality_forced_channel_gets_initialize_option():
    belief, cert = _fresh(3)
    belief[1].status = ChannelStatus.PRESENT_UNOBSERVED
    cands = CandidateGenerator(max_explore=4).generate(
        belief, cert, RobotState()
    )
    assert any(c.action_type is MacroActionType.INITIALIZE for c in cands)


# -- CLEAR --------------------------------------------------------------------


def test_clear_generated_for_clearable_at_mec_centre():
    belief, cert = _fresh()
    belief[3].record_near((10.0, 0.0))        # a 5 m near-disc -> small MEC -> LOCALIZED
    assert belief[3].is_clearable
    state = RobotState()

    cands = CandidateGenerator().generate(belief, cert, state)
    clears = _by_type(cands, MacroActionType.CLEAR)
    assert len(clears) == 1
    m = clears[0]
    assert m.clear_channel == 3
    assert m.target == belief[3].clear_target        # MEC centre (§3.3)
    # time is the cost model's clear-hit time, not a guess
    assert m.expected_time == pytest.approx(
        AnalyticalCostModel().clear_time_s(state, belief[3].clear_target, hit=True)
    )


def test_clear_only_for_clearable_channels():
    belief, cert = _fresh()
    belief[2].record_bearing((0.0, 0.0), 30.0)       # DETECTED, MEC huge -> not clearable
    state = RobotState()
    clears = _by_type(CandidateGenerator().generate(belief, cert, state), MacroActionType.CLEAR)
    assert clears == []


# -- REFINE -------------------------------------------------------------------


def test_refine_targets_nbv_point_and_pins_its_channel():
    belief, cert = _fresh()
    belief[2].record_bearing((0.0, 0.0), 40.0)       # single bearing -> DETECTED wedge
    assert belief[2].status == ChannelStatus.DETECTED
    state = RobotState()

    nbv = MinimaxNBV()
    gen = CandidateGenerator(nbv=nbv)
    cands = gen.generate(belief, cert, state)
    refines = _by_type(cands, MacroActionType.REFINE)
    assert len(refines) == 1
    m = refines[0]

    expected = nbv.choose(belief[2], state.pos)       # deterministic, stateless
    assert expected is not None
    assert m.target == expected.point
    assert m.refinement_gain == pytest.approx(expected.expected_shrink)
    assert 2 in m.scan_channels                        # its own channel is pinned
    assert m.meta["refine_channel"] == 2
    # batch-scan time comes from the authoritative cost model
    assert m.expected_time == pytest.approx(
        AnalyticalCostModel().batch_scan_time_s(state, m.target, m.scan_channels)
    )


# -- EXPLORE ------------------------------------------------------------------


def test_explore_waypoints_have_unknown_coverage_gain_and_costed_time():
    belief, cert = _fresh()                            # everything UNKNOWN
    state = RobotState()
    cands = CandidateGenerator().generate(belief, cert, state)
    explores = _by_type(cands, MacroActionType.EXPLORE)
    assert explores, "fresh belief should yield exploration waypoints"
    assert all(m.certificate_gain > 0.0 for m in explores)   # real UNKNOWN coverage
    for m in explores:
        assert m.scan_channels
        assert m.expected_time == pytest.approx(
            AnalyticalCostModel().batch_scan_time_s(state, m.target, m.scan_channels)
        )


def test_no_explore_when_no_unknown_channels():
    belief, cert = _fresh()
    for c in range(1, 7):
        belief[c].record_bearing((0.0, 0.0), 10.0 * c)   # all DETECTED, none UNKNOWN
    cands = CandidateGenerator().generate(belief, cert, RobotState())
    assert _by_type(cands, MacroActionType.EXPLORE) == []


def test_verification_mode_emits_verify_not_explore():
    belief, cert = _fresh()
    cands = CandidateGenerator().generate(
        belief, cert, RobotState(), scan_mode=SchedulerMode.VERIFICATION
    )
    assert _by_type(cands, MacroActionType.EXPLORE) == []
    assert _by_type(cands, MacroActionType.VERIFY), "verification mode should emit VERIFY waypoints"


# -- EXIT ---------------------------------------------------------------------


def test_exit_only_when_all_resolved():
    belief, cert = _fresh()
    cands = CandidateGenerator().generate(belief, cert, RobotState())
    assert _by_type(cands, MacroActionType.EXIT) == []

    for c in range(1, 7):
        belief[c].mark_cleared()
    assert belief.all_resolved()
    cands2 = CandidateGenerator().generate(belief, cert, RobotState())
    exits = _by_type(cands2, MacroActionType.EXIT)
    assert len(exits) == 1
    # nothing left to scan or clear once resolved
    assert all(m.action_type == MacroActionType.EXIT for m in cands2)


# -- soundness: resolved channels never scanned -------------------------------


def test_resolved_channels_never_appear_in_scan_batches():
    belief, cert = _fresh()
    belief[4].mark_cleared()                           # CLEARED
    belief[5].status = ChannelStatus.ABSENT_CERTIFIED  # certified absent
    belief[6].record_near((20.0, 0.0))                 # LOCALIZED -> clear, never scan
    belief[2].record_bearing((0.0, 0.0), 15.0)         # DETECTED (active)
    # ch1, ch3 remain UNKNOWN (active)
    state = RobotState()

    cands = CandidateGenerator().generate(belief, cert, state)
    scanned = set()
    for m in cands:
        scanned.update(m.scan_channels)
    assert 4 not in scanned and 5 not in scanned and 6 not in scanned

    # 6 is clearable -> it may only appear as a CLEAR channel, never a scan
    clear_channels = {m.clear_channel for m in _by_type(cands, MacroActionType.CLEAR)}
    assert clear_channels == {6}


# -- COMPLETION fallback (§11 Safe fallback / §6.5 Invariant C) ----------------
# Regression lock for the no_candidate abort (seeds 1013/1015/1027/1048). When
# active scanning has saturated the 40 m heuristic map so every EXPLORE waypoint
# gains zero new coverage, yet the hard verifier still can't certify a seam and the
# legacy backbone is incomplete, generate() must fall back to the guaranteed-
# completion backbone rather than return an empty set (which aborts the pipeline
# loop no_candidate). See candidates._completion_candidates and pipeline.run.


def _saturate_heuristic_map(cert, channels):
    """Force the saturated-heuristic corner through the map's own public API: fold a
    NO_SIGNAL for every backbone anchor into each channel's *heuristic* coverage map
    (the anchors provably cover D_1800 — Invariant C). This drives the coarse
    coverage ratio to 1.0, so every EXPLORE waypoint has zero gain, WITHOUT touching
    the hard certificate — no real scan points, no visited anchors — so the channel
    stays UNKNOWN and un-certifiable: exactly the abort state seeds 1013/1015/1027/
    1048 reached in the field."""
    for c in channels:
        for a in cert.anchors:
            cert.map.add_no_signal(c, a)


def test_completion_fallback_fires_when_active_families_are_empty():
    n = 6
    belief, cert = _fresh(n)
    _saturate_heuristic_map(cert, range(1, n + 1))
    # precondition: the heuristic map is saturated, but nothing is certified absent
    # and no backbone anchor is recorded as scanned (only the heuristic map moved).
    for c in range(1, n + 1):
        assert cert.heuristic_coverage_ratio(c) == pytest.approx(1.0)
        assert not cert.is_absent_certified(c, force=True)   # even force can't certify
        assert cert.backbone_anchor_progress(c) == 0
    assert not belief.all_resolved()

    gen = CandidateGenerator()
    cands = gen.generate(belief, cert, RobotState(), scan_mode=SchedulerMode.VERIFICATION)

    # the whole point: never empty while a channel is unresolved (no more no_candidate).
    assert cands, "generate must not return empty while a channel is unresolved"
    # and we genuinely reached the fallback — no active-family candidate survived.
    assert _by_type(cands, MacroActionType.CLEAR) == []
    assert _by_type(cands, MacroActionType.REFINE) == []
    assert _by_type(cands, MacroActionType.EXPLORE) == []
    # the fallback proposes backbone-completion VERIFY waypoints (tagged in meta).
    assert all(m.action_type == MacroActionType.VERIFY for m in cands)
    assert all("completion_anchor" in m.meta for m in cands)
    assert all(m.certificate_gain > 0.0 for m in cands)


def test_completion_candidates_walk_nearest_unvisited_backbone_anchors():
    belief, cert = _fresh(6)
    gen = CandidateGenerator()
    state = RobotState()  # at the origin

    out = gen._completion_candidates(belief, cert, state)
    assert out, "unresolved channels + unvisited anchors -> a non-empty fallback"
    assert all(m.action_type == MacroActionType.VERIFY for m in out)

    anchors = [(float(a[0]), float(a[1])) for a in cert.anchors]

    def _nearest(pt):
        return min(anchors, key=lambda p: hypot(p[0] - pt[0], p[1] - pt[1]))

    for m in out:
        near = _nearest(m.target)
        assert hypot(m.target[0] - near[0], m.target[1] - near[1]) < 1e-6   # a real anchor
        assert m.meta["completion_anchor"] == anchors.index(near)
        assert m.certificate_gain > 0.0                                      # never a zero-gain no-op
        assert m.expected_time > 0.0                                         # costed by the model

    # nearest-first: the leading candidate is the anchor closest to the robot.
    nearest = _nearest((state.x, state.y))
    assert hypot(out[0].target[0] - nearest[0], out[0].target[1] - nearest[1]) < 1e-6


def test_completion_fallback_is_last_resort_only():
    """On a fresh belief the active EXPLORE family carries real coverage gain, so
    generate() must NOT emit any backbone-completion candidate — the fallback is
    reached only when the active families are exhausted."""
    belief, cert = _fresh(6)
    cands = CandidateGenerator().generate(belief, cert, RobotState())
    assert _by_type(cands, MacroActionType.EXPLORE), "fresh belief yields real EXPLORE"
    assert all("completion_anchor" not in m.meta for m in cands)
