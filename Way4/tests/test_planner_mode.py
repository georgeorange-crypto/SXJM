"""P0 #2/#3 — pipeline ``planner_mode`` wiring (the follow-up a803f9b deferred).

``planner_mode`` selects the candidate generator the ``Way4Pipeline`` runs:

  * ``"legacy"``  -> the frozen action-centric ``CandidateGenerator`` (byte-identical
    to the shipped path — the default, so nothing changes unless asked).
  * ``"spatial"`` -> ``SpatialStopGenerator`` wrapping that *same* ``CandidateGenerator``
    (co-located services bundled into ``SpatialStop``s, P0 #2/#3). It is a
    non-destructive superset: every legacy candidate still flows through, so the
    spatial planner can always replicate the legacy plan and full-clear cannot regress.

An explicit ``generator=`` overrides the mode; any other value is rejected up front.

The construction tests need no Way3 — the pipeline builds against a stub env (as the
smoke test's no-progress case already relies on). The full-clear A/B gate (禁止10:
spatial full-clear >= legacy full-clear on the same ground truth) drives the real Way3
engine and skips where Way3 is absent (as in the linked worktree); it is the gate that
must pass in the shared tree before Phase D builds on the spatial routing lane.
"""

import pytest

from way4.executor import Way3EngineAdapter, load_way3_environment
from way4.planner import CandidateGenerator, SpatialStopGenerator
from way4.pipeline import Way4Pipeline


class _StubEnv:
    """Satisfies the ``MacroExecutor`` constructor; never invoked by the construction
    tests (``Way4Pipeline.__init__`` only stores the env, it does not call it)."""

    def measure(self, x, y, channel):
        return None, 0.0

    def clear(self, x, y, channel):
        return False, 0.0


# --- construction / wiring (no Way3 needed) ------------------------------------


def test_default_mode_is_legacy_plain_generator():
    pipe = Way4Pipeline(_StubEnv())
    assert pipe.planner_mode == "legacy"
    assert isinstance(pipe.generator, CandidateGenerator)
    # the plain generator is NOT a spatial wrapper: the default path is unchanged.
    assert not isinstance(pipe.generator, SpatialStopGenerator)


def test_spatial_mode_wraps_a_candidate_generator():
    pipe = Way4Pipeline(_StubEnv(), planner_mode="spatial")
    assert pipe.planner_mode == "spatial"
    assert isinstance(pipe.generator, SpatialStopGenerator)
    # it WRAPS (never replaces) the frozen generator — the non-destructive guarantee.
    assert isinstance(pipe.generator.base, CandidateGenerator)
    assert pipe.planner.fce.joint_route is True


def test_legacy_mode_keeps_additive_future_cost():
    pipe = Way4Pipeline(_StubEnv())
    assert pipe.planner.fce.joint_route is False


def test_explicit_generator_overrides_mode():
    g = CandidateGenerator()
    pipe = Way4Pipeline(_StubEnv(), planner_mode="spatial", generator=g)
    assert pipe.generator is g          # explicit wins; it is not re-wrapped


def test_unknown_planner_mode_raises():
    with pytest.raises(ValueError):
        Way4Pipeline(_StubEnv(), planner_mode="bogus")


# --- full-clear A/B gate (禁止10) — real Way3 engine, skips if absent -----------


@pytest.fixture
def way3_env():
    mod = load_way3_environment()
    if mod is None:
        pytest.skip("Way3 environment.py not available on disk")
    return mod


def _two_source_case(mod):
    """Two omni sources on channels 1 & 2 (same ground truth the smoke test uses)."""
    Jammer, Case = mod.Jammer, mod.Case
    field = mod.ConstantField(1.0)
    jammers = [
        Jammer(1, 600.0, 0.0, 1400.0, "omni"),
        Jammer(2, -600.0, 300.0, 1400.0, "omni"),
    ]
    return Case(jammers, field=field)


def _run(mod, mode):
    case = _two_source_case(mod)
    eng = mod.Engine(case)
    eng.enter()
    pipe = Way4Pipeline(
        Way3EngineAdapter(eng), n_channels=20, max_steps=4000, planner_mode=mode
    )
    return case, pipe.run()


@pytest.mark.parametrize("mode", ["legacy", "spatial"])
def test_both_modes_fully_clear_the_two_source_case(way3_env, mode):
    """禁止10: whichever planner runs, every real source is neutralised. The spatial
    lane may reshape the route but can never drop a clear (its candidate set is a
    superset of legacy's, and EXIT/no-progress guards still backstop every tick)."""
    case, result = _run(way3_env, mode)
    assert result.error is None, f"{mode}: pipeline aborted: {result.error}"
    assert case.cleared_count == case.total, (
        f"{mode}: only cleared {case.cleared_count}/{case.total}"
    )
    assert result.full_clear, f"{mode}: run did not report full_clear"
