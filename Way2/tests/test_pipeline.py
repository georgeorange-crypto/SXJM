"""Integration: the frozen pipeline and its guarantees.

Covers the end-to-end loop (env -> estimator -> candidates -> agent -> safety),
plus the invariants each stage promises: the safety shield keeps every action in
the arena, the generator is never empty and always offers EXIT, and the greedy
agent clears a localized source and exits when nothing is left to do.
"""

from __future__ import annotations

import math

from radio_rl.algorithms.heuristic import GreedyMathAgent
from radio_rl.candidates.generator import CandidateGenerator
from radio_rl.core.config import compose
from radio_rl.core.datatypes import (
    ActionType,
    CandidateAction,
    CandidateSource,
    DetectionObservation,
    ObservationType,
)
from radio_rl.geometry.belief import BeliefState
from radio_rl.pipeline import Pipeline
from radio_rl.safety.shield import SafetyShield


def test_pipeline_runs_and_clears():
    cfg = compose()
    pipe = Pipeline(cfg)
    stats, trace = pipe.run_episode(seed=0, max_steps=8000, collect_trace=True)
    assert pipe.env.finished
    assert stats.failed_clears == 0            # localization guarantees clears hit
    assert stats.sources_cleared > 0
    assert stats.sources_cleared <= stats.sources_total
    assert stats.virtual_time > 0.0
    assert len(trace) == stats.steps + (1 if trace and trace[-1].action.action_type == ActionType.EXIT else 0)


def test_safety_shield_keeps_actions_in_arena():
    shield = SafetyShield()
    cand = CandidateAction(0, int(ActionType.SCAN), 5, 5000.0, -4000.0)
    action = shield.apply(cand)
    assert math.hypot(action.target_x, action.target_y) <= 1800.0 + 1e-6
    assert 1 <= action.channel <= 20


def test_safety_shield_forces_exit_near_deadline():
    shield = SafetyShield()
    cand = CandidateAction(0, int(ActionType.SCAN), 5, 100.0, 0.0)
    action = shield.apply(cand, remaining_real_s=1.0)  # below emergency margin
    assert action.action_type == ActionType.EXIT


def test_generator_never_empty_and_offers_exit():
    gen = CandidateGenerator()
    belief = BeliefState(problem=3)          # fresh: everything undetected
    cs = gen.generate(belief)
    assert len(cs) >= 1
    assert any(c.action_type == int(ActionType.EXIT) for c in cs.candidates)
    assert all(math.hypot(c.target_x, c.target_y) <= 1800.0 + 1e-6 for c in cs.candidates)


def test_heuristic_clears_localized_channel():
    belief = BeliefState(problem=3)
    # a 'too strong' reading localizes channel 5 at the pose
    belief.update(DetectionObservation(ObservationType.TOO_STRONG, 5, 120.0, 0.0, 6.0))
    assert belief.belief(5).is_localized

    gen = CandidateGenerator()
    agent = GreedyMathAgent()
    cs = gen.generate(belief)
    clears = [c for c in cs.candidates if c.action_type == int(ActionType.CLEAR)
              and c.channel == 5]
    assert clears, "expected a CLEAR candidate for the localized channel"

    idx = agent.select(cs, belief)
    chosen = cs.candidates[idx]
    assert chosen.action_type == int(ActionType.CLEAR)
    assert chosen.channel == 5


def test_heuristic_exits_when_nothing_to_do():
    belief = BeliefState(problem=3)
    agent = GreedyMathAgent()
    # a candidate set with only an EXIT (no information anywhere)
    from radio_rl.candidates.generator import CandidateSet

    exit_only = CandidateSet(
        candidates=[CandidateAction(0, int(ActionType.EXIT), 1, 0.0, 0.0,
                                    source=int(CandidateSource.SAFETY))],
        k_max=32,
        n_real=1,
    )
    assert agent.select(exit_only, belief) == 0
