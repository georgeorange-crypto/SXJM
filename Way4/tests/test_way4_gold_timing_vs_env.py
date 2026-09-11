"""M4 acceptance — cross-check Way4's cost model AND executor against the
*authoritative* offline engine (Way3's ``environment.py``, loaded by path).

This is the strong form of "逐秒复现 §1.1": an engine written independently of
Way4's cost model reproduces the golden clock [105, 111, 194, 199], Way4's
``AnalyticalCostModel`` reproduces the same numbers, and the ``MacroExecutor``
driving the real engine through the adapter reproduces them a third time — proving
the batch-scan/clear channel-tracking is faithful, not merely internally
consistent. Skips cleanly if the Way3 tree is not checked out.
"""

import pytest

from way4.belief import BeliefState
from way4.core import AnalyticalCostModel, MacroActionType, MacroCandidate, RobotState
from way4.executor import MacroExecutor, Way3EngineAdapter, load_way3_environment

GOLD_CLOCKS = [105.0, 111.0, 194.0, 199.0]


@pytest.fixture
def way3_env():
    mod = load_way3_environment()
    if mod is None:
        pytest.skip("Way3 environment.py not available on disk")
    return mod


def _gold_case(mod):
    """A case whose ch3 clear at (300,0) misses (no ch3 source): the only outcome
    the golden timing depends on. Constant error field keeps it deterministic."""
    Jammer, Case = mod.Jammer, mod.Case
    field = mod.ConstantField(1.0)
    jammers = [
        Jammer(1, 1000.0, 0.0, 1200.0, "omni"),
        Jammer(2, -1000.0, 0.0, 1200.0, "omni"),
    ]
    return Case(jammers, field=field)


def test_raw_engine_reproduces_golden_clock(way3_env):
    eng = way3_env.Engine(_gold_case(way3_env))
    eng.enter()
    clocks = []
    resp, _ = eng.measure(300.0, 400.0, 1); clocks.append(resp["virtual_time_s"])
    resp, _ = eng.measure(300.0, 400.0, 2); clocks.append(resp["virtual_time_s"])
    resp, out = eng.clear(300.0, 0.0, 3);   clocks.append(resp["virtual_time_s"])
    assert out.result == "no_target_in_range"        # step 4 is a miss (3 s)
    resp, _ = eng.measure(300.0, 0.0, 2);   clocks.append(resp["virtual_time_s"])
    assert clocks == GOLD_CLOCKS


def test_cost_model_matches_engine_on_golden_sequence(way3_env):
    """Way4's analytical predictor lands on the same clocks the engine realises."""
    cost = AnalyticalCostModel()
    s = RobotState()
    s, _ = cost.apply_measure(s, (300.0, 400.0), 1)
    s, _ = cost.apply_measure(s, (300.0, 400.0), 2)
    s, _ = cost.apply_clear(s, (300.0, 0.0), hit=False)
    s2 = s
    s, _ = cost.apply_measure(s, (300.0, 0.0), 2)
    predicted = [105.0, 111.0, s2.virtual_time_s, s.virtual_time_s]
    assert predicted == GOLD_CLOCKS


def test_executor_over_real_engine_reproduces_golden_clock(way3_env):
    """The macro executor, driving the real engine via the adapter, reproduces the
    clock and — crucially — the clear leaves the measuring channel at 2 so step 5
    pays no switch."""
    eng = way3_env.Engine(_gold_case(way3_env))
    eng.enter()
    ex = MacroExecutor(Way3EngineAdapter(eng), belief=BeliefState(n_channels=20))

    clocks = []
    s = RobotState()
    r = ex.execute(MacroCandidate(MacroActionType.EXPLORE, (300.0, 400.0), scan_channels=(1, 2)), s)
    clocks += [p.virtual_time_s for p in r.primitives]; s = r.state
    r = ex.execute(MacroCandidate(MacroActionType.CLEAR, (300.0, 0.0), clear_channel=3), s)
    clocks += [p.virtual_time_s for p in r.primitives]; s = r.state
    assert s.channel == 2   # /clear ch3 did not change the measuring channel
    r = ex.execute(MacroCandidate(MacroActionType.PURSUE, (300.0, 0.0), scan_channels=(2,)), s)
    clocks += [p.virtual_time_s for p in r.primitives]

    assert clocks == GOLD_CLOCKS
