"""Regression guard for the Way3-homing fallback (DESIGN.md §6.10, §11).

Two layers:

  * Pure geometry — the ported ``jammerhunt.geometry`` helpers the controller's
    decisions rest on (ray crossing, localization-region MEC, the single→double
    bearing transition). Fast, no Way3 dependency.
  * End-to-end — a previously-failing P4 seed run through the real Way3 engine.
    Seed 2000 livelocked at 12/13 with ``no_progress_stall`` before the fallback
    (the omni NBV kept probing a directional source's blind arc → NO_SIGNAL, and
    禁止5 bars shrinking F_c from that). If it now fully clears, the fallback fired
    and did its job — nothing else changed. Skips cleanly if the Way3 tree is absent.

Both assert GROUND TRUTH (``case.cleared_count == case.total``), never Way4's own
``belief.all_resolved()`` — the same contract as scripts/compare_way3_way4.py.
"""

import math

import pytest

from way4.executor import HomingController, MacroExecutor, Way3EngineAdapter, load_way3_environment
from way4.executor.homing import _best_region, _Fix, _ray_intersect
from way4.pipeline import Way4Pipeline


# --------------------------------------------------------------------------- #
# pure geometry (the ported Way3 decision helpers)
# --------------------------------------------------------------------------- #
def _bearing(p, s):
    return math.degrees(math.atan2(s[1] - p[1], s[0] - p[0])) % 360.0


def test_ray_intersect_lands_on_the_true_crossing():
    src = (500.0, 300.0)
    p1, p2 = (0.0, 0.0), (500.0, 0.0)
    x = _ray_intersect(p1, _bearing(p1, src), p2, _bearing(p2, src))
    assert x is not None
    assert math.hypot(x[0] - src[0], x[1] - src[1]) < 1e-6


def test_ray_intersect_none_when_parallel():
    assert _ray_intersect((0.0, 0.0), 30.0, (10.0, 10.0), 30.0) is None


def test_best_region_centre_is_the_source_radius_reflects_pm1deg():
    src = (500.0, 300.0)
    p1, p2 = (0.0, 0.0), (500.0, 0.0)
    obs = [(p1, _bearing(p1, src)), (p2, _bearing(p2, src))]
    region = _best_region(obs)
    assert region is not None
    ctr, r = region
    # the ±1° wedge crossing is centred on the true source ...
    assert math.hypot(ctr[0] - src[0], ctr[1] - src[1]) < 2.0
    # ... and its radius is the finite ±1° spread at this range (a few metres),
    # never a degenerate blow-up.
    assert 0.0 < r < 30.0


def test_fix_single_bearing_is_unreliable_then_two_localises():
    src = (500.0, 300.0)
    p1, p2 = (0.0, 0.0), (500.0, 0.0)
    fix = _Fix(obs=[(p1, _bearing(p1, src))])
    fix.refresh()
    # one bearing => no crossing => must drive the perpendicular-baseline path
    assert fix.est is None and fix.region_r == float("inf")
    fix.obs.append((p2, _bearing(p2, src)))
    fix.refresh()
    assert fix.est is not None and fix.region_r < 30.0
    assert math.hypot(fix.est[0] - src[0], fix.est[1] - src[1]) < 2.0


def test_best_region_skips_near_parallel_pairs():
    # two almost-collinear bearings give a wild far crossing; best_region must
    # reject them (crossing < 2°) rather than trust a phantom fix.
    src = (500.0, 300.0)
    p1 = (0.0, 0.0)
    p2 = (1.0, 0.6)                       # essentially on the p1->src line
    obs = [(p1, _bearing(p1, src)), (p2, _bearing(p2, src))]
    assert _best_region(obs) is None


# --------------------------------------------------------------------------- #
# end-to-end: the fallback restores a previously-livelocked P4 clear
# --------------------------------------------------------------------------- #
@pytest.fixture
def way3_env():
    mod = load_way3_environment()
    if mod is None:
        pytest.skip("Way3 environment.py not available on disk")
    return mod


@pytest.mark.parametrize("seed", [2000, 2003])
def test_p4_previously_stalled_seed_now_fully_clears(way3_env, seed):
    """Seeds 2000/2003 aborted with no_progress_stall before the homing fallback
    (each cleared all-but-one directional source, then spun). They must now fully
    clear — the §11 fallback engaging on the stuck DETECTED channel."""
    case = way3_env.generate_case(seed=seed, problem=4, field_kind="smooth")
    eng = way3_env.Engine(case)
    eng.enter()
    result = Way4Pipeline(
        Way3EngineAdapter(eng), n_channels=20, problem=4, max_steps=3000
    ).run()

    assert result.error is None, f"pipeline aborted: {result.error}"
    assert case.cleared_count == case.total, (
        f"seed {seed}: only cleared {case.cleared_count}/{case.total} "
        f"(homing fallback did not restore the full clear)"
    )


def test_homing_controller_is_constructed_by_the_pipeline(way3_env):
    """The pipeline owns a HomingController wired to its executor+belief with the
    Way3 parameter set (a broken import/param would surface here, not only in a
    slow episode)."""
    eng = way3_env.Engine(way3_env.generate_case(seed=2000, problem=4, field_kind="smooth"))
    eng.enter()
    pipe = Way4Pipeline(Way3EngineAdapter(eng), n_channels=20, problem=4)
    assert isinstance(pipe._homing, HomingController)
    assert pipe._homing.exec is pipe.executor
    assert pipe._homing.belief is pipe.belief
    assert (pipe._homing.clear_margin_m, pipe._homing.orbit_radius_m,
            pipe._homing.orbit_delta_deg, pipe._homing.max_homing_iters) == (14.0, 85.0, 42.0, 8)
    assert isinstance(pipe.executor, MacroExecutor)
