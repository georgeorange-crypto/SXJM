"""§10 receding-horizon planner — outcome prediction + Q(a)=C(a)+f_o[Ĵ] ranking.

Locks the decision behaviour: EXIT/CLEAR give one deterministic outcome; a scan
gives ≤3 representative outcomes with the right cheap perturbations (NO_SIGNAL
erases nearby holes; a detection on an UNKNOWN channel spends an unknown and adds a
region; a ``near`` makes the spot clearable); robust aggregation ≥ expected; a
clearable source under the robot wins the argmin; and horizon-2 both runs and keeps
routing memoised (an identical replan re-solves nothing).
"""

import pytest

from way4.belief import BeliefState, ChannelStatus
from way4.certificate import CertificateManager
from way4.core import MacroActionType, MacroCandidate, RobotState
from way4.planner import (
    CandidateGenerator,
    CostView,
    DetectedRegion,
    OutcomePredictor,
    RecedingHorizonPlanner,
)


def _pick(outs, label):
    return next(o for o in outs if o.label == label)


# -- OutcomePredictor: terminal actions ---------------------------------------


def test_predict_exit_is_single_outcome():
    pred = OutcomePredictor()
    belief = BeliefState(n_channels=3)
    base = CostView(pos=(0.0, 0.0))
    outs = pred.predict(MacroCandidate(MacroActionType.EXIT, (0.0, 0.0)), belief, base)
    assert len(outs) == 1 and outs[0].label == "exit" and outs[0].prob == 1.0


def test_predict_clear_removes_the_cleared_target():
    pred = OutcomePredictor()
    belief = BeliefState(n_channels=3)
    base = CostView(pos=(0.0, 0.0), clearable_targets=[(100.0, 0.0), (0.0, 400.0)])
    cand = MacroCandidate(MacroActionType.CLEAR, (100.0, 0.0), clear_channel=2)
    outs = pred.predict(cand, belief, base)
    assert len(outs) == 1 and outs[0].label == "hit"
    v = outs[0].view
    assert v.pos == (100.0, 0.0)
    assert (100.0, 0.0) not in v.clearable_targets      # the cleared one is gone
    assert (0.0, 400.0) in v.clearable_targets          # the other remains
    assert base.clearable_targets == [(100.0, 0.0), (0.0, 400.0)]   # base untouched


# -- OutcomePredictor: scans on an UNKNOWN focus channel -----------------------


def test_predict_scan_on_unknown_channel_perturbations():
    pred = OutcomePredictor(detection_radius=1000.0)
    belief = BeliefState(n_channels=3)                  # ch1 UNKNOWN
    q = (500.0, 0.0)
    base = CostView(
        pos=(0.0, 0.0),
        unknown_holes=[(500.0, 0.0), (2000.0, 0.0)],    # one within 1000 m of q, one not
        n_unknown=3,
    )
    cand = MacroCandidate(MacroActionType.EXPLORE, q, scan_channels=(1,))
    outs = pred.predict(cand, belief, base)
    assert {o.label for o in outs} == {"no_signal", "detect", "near"}
    assert len(outs) <= 3

    ns = _pick(outs, "no_signal").view
    assert ns.pos == q
    assert ns.unknown_holes == [(2000.0, 0.0)]          # nearby hole erased, far one kept

    det = _pick(outs, "detect").view
    assert det.n_unknown == 2                           # one UNKNOWN spent
    assert len(det.detected) == 1                       # a fresh nominal region appears

    near = _pick(outs, "near").view
    assert q in near.clearable_targets                  # clearable on the spot
    assert near.n_unknown == 2


# -- OutcomePredictor: scans on a DETECTED focus channel (REFINE) --------------


def test_predict_refine_on_detected_channel_shrinks_region():
    pred = OutcomePredictor()
    belief = BeliefState(n_channels=3)
    belief[2].record_bearing((0.0, 0.0), 40.0)          # -> DETECTED
    assert belief[2].status == ChannelStatus.DETECTED
    q = (500.0, 0.0)
    base = CostView(pos=(0.0, 0.0), detected=[DetectedRegion(q, 200.0, 400.0, 0.3)])
    cand = MacroCandidate(
        MacroActionType.REFINE, q, scan_channels=(2,),
        refinement_gain=100.0, meta={"refine_channel": 2},
    )
    outs = pred.predict(cand, belief, base)
    assert {o.label for o in outs} == {"no_signal", "detect", "near"}

    det = _pick(outs, "detect").view                    # refine shrinks the region
    assert len(det.detected) == 1
    assert det.detected[0].diameter == pytest.approx(300.0)   # 400 - refinement_gain

    near = _pick(outs, "near").view                     # near => region becomes clearable
    assert q in near.clearable_targets
    assert len(near.detected) == 0


# -- aggregation: robust (minimax) vs expected --------------------------------


def test_robust_is_max_expected_is_weighted_mean():
    rob = RecedingHorizonPlanner(outcome_mode="robust")
    exp = RecedingHorizonPlanner(outcome_mode="expected")
    jvals = [("a", 0.5, 10.0), ("b", 0.5, 30.0)]
    assert rob._aggregate(jvals) == 30.0
    assert exp._aggregate(jvals) == pytest.approx(20.0)
    assert rob._aggregate(jvals) >= exp._aggregate(jvals)


# -- integration: the cheap clear under the robot wins the argmin -------------


def test_clear_next_to_robot_beats_faraway_exploration():
    belief = BeliefState(n_channels=3)
    cert = CertificateManager(n_channels=3)
    belief[1].record_near((5.0, 0.0))                   # clearable right next to (0,0)
    # ch2, ch3 remain UNKNOWN -> explore candidates exist but sit on far rings
    state = RobotState()
    cands = CandidateGenerator().generate(belief, cert, state)

    result = RecedingHorizonPlanner().plan(belief, cert, state, cands)
    assert result.best is not None
    assert result.best.action_type == MacroActionType.CLEAR
    assert result.best.clear_channel == 1
    # every candidate got a Q value
    assert len(result.evaluations) == len(cands)
    assert result.best.expected_time <= result.q_value    # Q = C + Ĵ ≥ C


# -- horizon-2 runs and routing stays memoised --------------------------------


def test_horizon2_plans_and_reuses_route_cache():
    belief = BeliefState(n_channels=6)
    cert = CertificateManager(n_channels=6)
    belief[1].record_near((100.0, 0.0))                 # clearable
    belief[2].record_bearing((0.0, 0.0), 30.0)          # DETECTED
    belief[3].record_bearing((0.0, 0.0), 60.0)          # DETECTED
    # ch4, ch5, ch6 UNKNOWN
    state = RobotState()
    cands = CandidateGenerator().generate(belief, cert, state)

    planner = RecedingHorizonPlanner(horizon=2, beam_width=8)
    result = planner.plan(belief, cert, state, cands)
    assert result.best is not None
    assert result.horizon == 2

    # an identical replan must be fully served from the tour cache: zero new solves
    solves_after_first = planner.fce.route.solves
    planner.plan(belief, cert, state, cands)
    assert planner.fce.route.solves == solves_after_first
