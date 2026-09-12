"""M7 — end-to-end smoke test of the Way4 rolling pipeline (§13).

Drives the whole math stack (generator → planner → executor → belief/certificate)
against the authoritative Way3 engine through the shipped adapter, on a tiny known
case. Proves the Observe→Update→Plan→Execute loop terminates, folds observations,
clears the present sources, certifies the rest absent, and EXITs cleanly — the
prerequisite for the Way3-vs-Way4 comparison. Skips if the Way3 tree is absent.
"""

import pytest

from way4.core import MacroActionType
from way4.core.actions import MacroCandidate
from way4.core.cost import RobotState
from way4.executor import Way3EngineAdapter, load_way3_environment
from way4.executor.macro_executor import ExecutionResult
from way4.pipeline import Way4Pipeline


@pytest.fixture
def way3_env():
    mod = load_way3_environment()
    if mod is None:
        pytest.skip("Way3 environment.py not available on disk")
    return mod


def _two_source_case(mod):
    """Two omni sources on channels 1 & 2, generous R_eff, constant error field."""
    Jammer, Case = mod.Jammer, mod.Case
    field = mod.ConstantField(1.0)
    jammers = [
        Jammer(1, 600.0, 0.0, 1400.0, "omni"),
        Jammer(2, -600.0, 300.0, 1400.0, "omni"),
    ]
    return Case(jammers, field=field)


def test_pipeline_runs_and_terminates(way3_env):
    eng = way3_env.Engine(_two_source_case(way3_env))
    eng.enter()
    pipe = Way4Pipeline(Way3EngineAdapter(eng), n_channels=20, max_steps=4000)
    result = pipe.run()

    assert result.error is None, f"pipeline aborted: {result.error}"
    assert result.steps > 0
    # loop must halt (not hit the step cap)
    assert result.steps < 4000


def test_pipeline_clears_present_sources(way3_env):
    case = _two_source_case(way3_env)
    eng = way3_env.Engine(case)
    eng.enter()
    pipe = Way4Pipeline(Way3EngineAdapter(eng), n_channels=20, max_steps=4000)
    result = pipe.run()

    # ground truth: both real sources neutralised
    assert case.cleared_count == case.total, (
        f"only cleared {case.cleared_count}/{case.total}"
    )
    assert result.n_clear_hit >= case.total
    # the two present channels were discovered
    assert result.present >= case.total


class _NoProgressEnv:
    """Env stub whose commands never finish (used only to satisfy the executor
    constructor; the executor itself is stubbed in the stall test)."""

    def measure(self, x, y, channel):
        return None, 0.0

    def clear(self, x, y, channel):
        return False, 0.0


def test_no_progress_guard_bounds_a_livelock(monkeypatch):
    """A degenerate loop that never advances the belief must terminate at the stall
    limit with an honest failure — never a false full_clear, never the step cap.
    This is the termination guarantee the M7 comparison relies on; the seed-1003
    stall's *root* cause is removed in the generator, this only bounds the residual."""
    pipe = Way4Pipeline(_NoProgressEnv(), n_channels=20, max_steps=5000, stall_limit=8)

    dummy = MacroCandidate(MacroActionType.EXPLORE, (0.0, 0.0), scan_channels=(3,))
    monkeypatch.setattr(pipe, "_choose_macro", lambda: dummy)

    def _noop_execute(macro, state, budget):
        # advance the clock (as a real measure would) but change nothing in belief.
        return ExecutionResult(
            state=RobotState(0.0, 0.0, 1, state.vt_us + 5_000_000),
            primitives=[],
            finished=False,
        )

    monkeypatch.setattr(pipe.executor, "execute", _noop_execute)

    result = pipe.run()

    assert result.error == "no_progress_stall"
    assert result.steps <= pipe.stall_limit + 1     # bounded, not the 5000 step cap
    assert not result.full_clear                     # honest: nothing was resolved
    assert not result.success
