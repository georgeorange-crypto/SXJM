"""M10 RL residual — unit + property tests (DESIGN.md §12, §15, §16).

The residual's whole contract is: it may reorder the math planner's candidates and
nothing else. These tests lock that contract so a trained checkpoint (or a bug)
can never breach it:

  * **No-op default is EXACT** — a fresh/zero-init scorer reproduces the pure-math
    argmin bit-for-bit, over many random configs (禁止10: never lower full-clear).
  * **Reranker only** — the chosen action is always one of the input candidates;
    the residual never fabricates a coordinate (禁止8).
  * **Feature layout is frozen** — dimensions match the exported constants, so a
    trained net can't silently desync (§15 unit layer).
  * **Q_math is preserved** — the wrapper never rewrites the analytical q_values.
  * **Fail-safe** — a raising scorer degrades to the math ranking (§11 fallback).
  * **The mechanism actually works** — a non-zero residual does flip the argmin,
    and only ever to another candidate in the set.
"""

import math

import pytest

from way4.belief import BeliefState
from way4.core import MacroActionType, MacroCandidate, RobotState
from way4.planner import RecedingHorizonPlanner
from way4.certificate import CertificateManager
from way4.rl import (
    CANDIDATE_FEATURE_DIM,
    FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    RLResidualPlanner,
    ResidualScorer,
    feature_matrix,
)
from way4.rl.features import evaluation_features


# --- fixtures ----------------------------------------------------------------


def _belief(n=20):
    return BeliefState(n_channels=n)


def _certificate(belief):
    return CertificateManager(n_channels=belief.n_channels)


def _state(x=0.0, y=0.0, ch=1):
    return RobotState(x, y, ch, 0)


def _candidates():
    """A spread of macro intents at different distances/gains — enough that the
    argmin is non-trivial and residual reranking is observable."""
    return [
        MacroCandidate(
            MacroActionType.EXPLORE, (600.0, 0.0), scan_channels=(1, 2, 3),
            exploration_gain=3.0, expected_time=180.0,
        ),
        MacroCandidate(
            MacroActionType.REFINE, (120.0, 90.0), scan_channels=(2,),
            refinement_gain=40.0, expected_time=60.0, meta={"refine_channel": 2},
        ),
        MacroCandidate(
            MacroActionType.PURSUE, (300.0, -200.0), scan_channels=(3,),
            exploration_gain=1.0, expected_time=95.0,
        ),
        MacroCandidate(
            MacroActionType.VERIFY, (0.0, 800.0), scan_channels=(4, 5),
            certificate_gain=2.0, expected_time=220.0,
        ),
    ]


class _StubScorer:
    """A scorer that returns a fixed residual vector — for testing the rerank
    mechanism without training. ``available`` mimics a live torch net."""

    def __init__(self, residuals, available=True):
        self._residuals = residuals
        self.available = available

    def residuals(self, features):
        # honour the batch length; pad/truncate defensively
        out = list(self._residuals)[: len(features)]
        out += [0.0] * (len(features) - len(out))
        return out


class _RaisingScorer:
    available = True

    def residuals(self, features):
        raise RuntimeError("boom")


# --- feature layout ----------------------------------------------------------


def test_feature_dims_are_consistent():
    assert FEATURE_DIM == CANDIDATE_FEATURE_DIM + GLOBAL_FEATURE_DIM


def test_stop_has_nonzero_one_hot_feature():
    belief, state = _belief(), _state()
    cert = _certificate(belief)
    cand = MacroCandidate(MacroActionType.STOP, (0.0, 0.0), scan_channels=(1,))
    res = RecedingHorizonPlanner().plan(belief, cert, state, [cand])
    row = evaluation_features(res.evaluations[0], belief, state, n_candidates=1)
    assert sum(row[:8]) == pytest.approx(1.0)


def test_feature_matrix_row_width_matches_FEATURE_DIM():
    belief, state = _belief(), _state()
    cert = _certificate(belief)
    planner = RecedingHorizonPlanner()
    res = planner.plan(belief, cert, state, _candidates())
    mat = feature_matrix(res.evaluations, belief, state)
    assert len(mat) == len(res.evaluations)
    assert all(len(row) == FEATURE_DIM for row in mat)
    # features are finite real numbers
    assert all(math.isfinite(v) for row in mat for v in row)


def test_evaluation_features_length():
    belief, state = _belief(), _state()
    cert = _certificate(belief)
    res = RecedingHorizonPlanner().plan(belief, cert, state, _candidates())
    f = evaluation_features(res.evaluations[0], belief, state, n_candidates=len(res.evaluations))
    assert len(f) == FEATURE_DIM


# --- no-op default is EXACT (禁止10) ------------------------------------------


def test_fresh_scorer_is_exact_noop_vs_math():
    """A default RLResidualPlanner (zero-init scorer) must choose the identical
    candidate the bare math planner chooses — the core 禁止10 safety property."""
    belief, state = _belief(), _state()
    cert = _certificate(belief)
    cands = _candidates()

    math_planner = RecedingHorizonPlanner()
    math_res = math_planner.plan(belief, cert, state, cands)

    rl = RLResidualPlanner()   # fresh zero-init scorer or torch-absent -> no-op
    rl_res = rl.plan(belief, cert, state, cands)

    assert rl_res.best is math_res.best
    assert rl_res.q_value == pytest.approx(math_res.q_value)


def test_fresh_scorer_noop_over_many_random_configs():
    import random

    rng = random.Random(20260912)
    math_planner = RecedingHorizonPlanner()
    rl = RLResidualPlanner()
    for _ in range(50):
        belief = _belief()
        state = _state(rng.uniform(-1000, 1000), rng.uniform(-1000, 1000))
        cert = _certificate(belief)
        cands = []
        for _k in range(rng.randint(2, 6)):
            at = rng.choice(list(MacroActionType))
            cands.append(
                MacroCandidate(
                    at,
                    (rng.uniform(-1500, 1500), rng.uniform(-1500, 1500)),
                    scan_channels=tuple(range(1, rng.randint(1, 4))),
                    exploration_gain=rng.uniform(0, 5),
                    refinement_gain=rng.uniform(0, 50),
                    certificate_gain=rng.uniform(0, 3),
                    expected_time=rng.uniform(10, 300),
                )
            )
        m = math_planner.plan(belief, cert, state, cands)
        r = rl.plan(belief, cert, state, cands)
        assert r.best is m.best


# --- reranker only, never fabricates a coordinate (禁止8) ---------------------


def test_best_is_always_one_of_the_input_candidates():
    belief, state = _belief(), _state()
    cert = _certificate(belief)
    cands = _candidates()
    # a strong residual favouring the LAST candidate
    stub = _StubScorer([0.0, 0.0, 0.0, -10_000.0])
    rl = RLResidualPlanner(scorer=stub)
    res = rl.plan(belief, cert, state, cands)
    assert res.best in cands
    assert res.best is cands[-1]        # residual pulled the argmin to it


def test_residual_flips_argmin_only_among_candidates():
    belief, state = _belief(), _state()
    cert = _certificate(belief)
    cands = _candidates()
    math_res = RecedingHorizonPlanner().plan(belief, cert, state, cands)
    math_best_idx = cands.index(math_res.best)
    # push a large negative residual onto a DIFFERENT candidate
    other = (math_best_idx + 1) % len(cands)
    residuals = [0.0] * len(cands)
    residuals[other] = -1e6
    rl = RLResidualPlanner(scorer=_StubScorer(residuals))
    res = rl.plan(belief, cert, state, cands)
    assert res.best is cands[other]
    assert res.best in cands


# --- Q_math preserved --------------------------------------------------------


def test_evaluations_qvalues_are_untouched_by_rerank():
    belief, state = _belief(), _state()
    cert = _certificate(belief)
    cands = _candidates()
    math_res = RecedingHorizonPlanner().plan(belief, cert, state, cands)
    math_qs = [e.q_value for e in math_res.evaluations]

    rl = RLResidualPlanner(scorer=_StubScorer([0.0, -500.0, 0.0, 0.0]))
    rl_res = rl.plan(belief, cert, state, cands)
    rl_qs = [e.q_value for e in rl_res.evaluations]
    assert rl_qs == pytest.approx(math_qs)   # analytical q kept for logging


def test_applied_residual_recorded_in_meta():
    belief, state = _belief(), _state()
    cert = _certificate(belief)
    cands = _candidates()
    rl = RLResidualPlanner(scorer=_StubScorer([0.0, 0.0, 0.0, -9999.0]))
    res = rl.plan(belief, cert, state, cands)
    assert res.best.meta.get("rl_residual") == pytest.approx(-9999.0)


# --- fail-safe (§11 fallback) ------------------------------------------------


def test_raising_scorer_falls_back_to_math():
    belief, state = _belief(), _state()
    cert = _certificate(belief)
    cands = _candidates()
    math_res = RecedingHorizonPlanner().plan(belief, cert, state, cands)
    rl = RLResidualPlanner(scorer=_RaisingScorer())
    rl_res = rl.plan(belief, cert, state, cands)
    assert rl_res.best is math_res.best


def test_disabled_planner_is_noop_even_with_active_scorer():
    belief, state = _belief(), _state()
    cert = _certificate(belief)
    cands = _candidates()
    math_res = RecedingHorizonPlanner().plan(belief, cert, state, cands)
    rl = RLResidualPlanner(scorer=_StubScorer([0.0, 0.0, 0.0, -1e6]), enabled=False)
    assert rl.active is False
    rl_res = rl.plan(belief, cert, state, cands)
    assert rl_res.best is math_res.best     # enabled=False -> math ranking


def test_empty_candidates_returns_empty_plan():
    belief, state = _belief(), _state()
    cert = _certificate(belief)
    rl = RLResidualPlanner(scorer=_StubScorer([]))
    res = rl.plan(belief, cert, state, [])
    assert res.best is None


# --- scorer behaviour --------------------------------------------------------


def test_zero_init_scorer_returns_all_zero_residuals():
    scorer = ResidualScorer(zero_init=True)
    if not scorer.available:
        pytest.skip("torch unavailable; residuals trivially zero")
    feats = [[0.1] * FEATURE_DIM for _ in range(5)]
    assert scorer.residuals(feats) == [0.0] * 5


def test_scorer_residuals_are_bounded_by_clip():
    scorer = ResidualScorer(zero_init=False, residual_clip=30.0)
    if not scorer.available:
        pytest.skip("torch unavailable")
    feats = [[5.0] * FEATURE_DIM for _ in range(8)]
    out = scorer.residuals(feats)
    assert all(abs(v) <= 30.0 + 1e-4 for v in out)


def test_scorer_empty_batch():
    scorer = ResidualScorer()
    assert scorer.residuals([]) == []


def test_active_reflects_scorer_availability():
    rl = RLResidualPlanner()
    assert rl.active == rl.scorer.available   # enabled by default; tracks torch


# --- pipeline drop-in integration (§13: zero pipeline edits) -----------------


def _way3_env_or_skip():
    from way4.executor import load_way3_environment

    mod = load_way3_environment()
    if mod is None:
        pytest.skip("Way3 environment.py not available on disk")
    return mod


def _two_source_case(mod):
    Jammer, Case = mod.Jammer, mod.Case
    field = mod.ConstantField(1.0)
    jammers = [
        Jammer(1, 600.0, 0.0, 1400.0, "omni"),
        Jammer(2, -600.0, 300.0, 1400.0, "omni"),
    ]
    return Case(jammers, field=field)


def test_rl_planner_is_pipeline_dropin_and_noop_matches_math():
    """A fresh (no-op) RLResidualPlanner injected via ``Way4Pipeline(planner=...)``
    must drive the SAME end-to-end outcome as the default math pipeline — the
    pipeline-level 禁止10 guarantee: a dormant residual can't change full-clear."""
    from way4.executor import Way3EngineAdapter
    from way4.pipeline import Way4Pipeline

    mod = _way3_env_or_skip()

    # math baseline
    eng_a = mod.Engine(_two_source_case(mod))
    eng_a.enter()
    res_math = Way4Pipeline(Way3EngineAdapter(eng_a), n_channels=20, max_steps=4000).run()

    # same case, RL residual planner (fresh scorer -> exact no-op)
    eng_b = mod.Engine(_two_source_case(mod))
    eng_b.enter()
    rl = RLResidualPlanner()   # drop-in: duck-types .plan()
    res_rl = Way4Pipeline(
        Way3EngineAdapter(eng_b), n_channels=20, max_steps=4000, planner=rl
    ).run()

    assert res_rl.error is None
    assert res_rl.full_clear == res_math.full_clear
    assert res_rl.n_clear_hit == res_math.n_clear_hit
    assert res_rl.steps == res_math.steps        # identical decision trajectory
