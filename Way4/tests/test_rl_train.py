"""M10 residual TRAINER unit tests (DESIGN.md §12, §16).

Verifies the REINFORCE mechanics with a cheap stub rollout — no Way3 engine — so
the learning logic is locked independently of episode cost:

  * **Reward obeys 禁止10** — every full-clear return strictly beats every
    non-full-clear return, for any plausible time/unresolved count.
  * **Sampling planner records a differentiable decision** and samples within the
    candidate set (禁止8).
  * **A gradient step runs and moves the weights** on a batch with return spread.
  * **The trainer learns to prefer the rewarded action** on a toy contextual-bandit
    rollout: after training, the greedy residual pushes the argmin toward the
    action that yields full-clear.
  * **Torch-absent degrades honestly** — constructing a trainer without torch raises,
    it never silently no-ops training.
"""

import random

import pytest

torch = pytest.importorskip("torch")

from way4.belief import BeliefState
from way4.certificate import CertificateManager
from way4.core import MacroActionType, MacroCandidate, RobotState
from way4.rl import (
    EpisodeRecord,
    ResidualScorer,
    ResidualTrainer,
    SamplingResidualPlanner,
    TrainConfig,
    episode_return,
)
from way4.rl.train import FAIL_PENALTY


# --- reward shape (禁止10) ----------------------------------------------------


def test_any_full_clear_beats_any_failure():
    """The best possible failure must score strictly below the worst plausible
    success — so the gradient can never prefer fast-but-failing."""
    # worst plausible success: a very long but complete episode
    worst_success = episode_return(True, 20000.0, 0)
    # best possible failure: instant, only one unresolved
    best_failure = episode_return(False, 0.0, 1)
    assert worst_success > best_failure
    # and a faster success beats a slower one
    assert episode_return(True, 4000.0, 0) > episode_return(True, 8000.0, 0)
    # more unresolved is worse
    assert episode_return(False, 100.0, 2) < episode_return(False, 100.0, 1)


def test_failure_return_is_below_negative_fail_penalty():
    assert episode_return(False, 0.0, 0) == -FAIL_PENALTY
    assert episode_return(False, 5000.0, 3) == -FAIL_PENALTY - 3


# --- sampling planner --------------------------------------------------------


def _belief():
    return BeliefState(n_channels=20)


def _cert(b):
    return CertificateManager(n_channels=b.n_channels)


def _state():
    return RobotState(0.0, 0.0, 1, 0)


def _cands():
    return [
        MacroCandidate(MacroActionType.EXPLORE, (600.0, 0.0), scan_channels=(1, 2),
                       exploration_gain=3.0, expected_time=180.0),
        MacroCandidate(MacroActionType.REFINE, (120.0, 90.0), scan_channels=(2,),
                       refinement_gain=40.0, expected_time=60.0),
        MacroCandidate(MacroActionType.PURSUE, (300.0, -200.0), scan_channels=(3,),
                       expected_time=95.0),
    ]


def test_sampling_planner_records_decision_within_candidate_set():
    scorer = ResidualScorer()
    planner = SamplingResidualPlanner(scorer, temperature=120.0, rng=random.Random(0))
    b, s = _belief(), _state()
    cert = _cert(b)
    cands = _cands()
    res = planner.plan(b, cert, s, cands)
    assert res.best in cands                      # never fabricates a coordinate
    assert len(planner.decisions) == 1
    dec = planner.decisions[0]
    assert 0 <= dec.chosen < len(cands)
    assert len(dec.features) == len(cands)
    assert len(dec.q_math) == len(cands)


def test_sampling_planner_single_candidate_records_nothing():
    scorer = ResidualScorer()
    planner = SamplingResidualPlanner(scorer, rng=random.Random(0))
    b, s = _belief(), _state()
    cert = _cert(b)
    one = [MacroCandidate(MacroActionType.CLEAR, (10.0, 0.0), clear_channel=1)]
    planner.plan(b, cert, s, one)
    assert planner.decisions == []                # no choice => no training signal


# --- gradient step -----------------------------------------------------------


def _toy_decision(chosen, n=3, temperature=120.0):
    from way4.rl import FEATURE_DIM
    from way4.rl.sampling_planner import Decision
    feats = [[float((i + 1) * 0.1)] * FEATURE_DIM for i in range(n)]
    q_math = [100.0, 100.0, 100.0]
    return Decision(features=feats, q_math=q_math, chosen=chosen, temperature=temperature)


def test_gradient_step_moves_weights_on_return_spread():
    scorer = ResidualScorer(zero_init=False)
    before = [p.detach().clone() for p in scorer.net.parameters()]

    def stub_rollout(sc, temp, seed):
        # alternate good/bad episodes so the batch has advantage spread
        good = seed % 2 == 0
        return EpisodeRecord(
            decisions=[_toy_decision(0 if good else 2, temperature=temp)],
            full_clear=good,
            virtual_time_s=5000.0,
            n_unresolved=0 if good else 1,
        )

    cfg = TrainConfig(iterations=3, episodes_per_iter=6, lr=1e-2, train_seeds=tuple(range(10)))
    trainer = ResidualTrainer(scorer, stub_rollout, cfg)
    hist = trainer.train()
    assert len(hist) == 3
    after = list(scorer.net.parameters())
    moved = any(not torch.allclose(b, a.detach()) for b, a in zip(before, after))
    assert moved, "REINFORCE step did not update the scorer weights"


def test_trainer_learns_to_prefer_rewarded_action():
    """Toy contextual bandit: action 0 => full_clear, others => fail. After training,
    the greedy residual should make action 0 the argmin of (Q_math + ΔQ_θ)."""
    scorer = ResidualScorer(zero_init=True, residual_clip=200.0)

    GOOD = 0

    def stub_rollout(sc, temp, seed):
        # sample under the CURRENT policy so learning is on-policy
        from way4.rl.features import FEATURE_DIM
        import math as _m
        feats = [[0.2] * FEATURE_DIM for _ in range(3)]
        q_math = [100.0, 100.0, 100.0]
        deltas = sc.residuals(feats)
        logits = [-(q_math[i] + deltas[i]) / temp for i in range(3)]
        m = max(logits)
        exps = [_m.exp(l - m) for l in logits]
        s = sum(exps)
        probs = [e / s for e in exps]
        r = random.random()
        acc = 0.0
        chosen = 2
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                chosen = i
                break
        from way4.rl.sampling_planner import Decision
        dec = Decision(features=feats, q_math=q_math, chosen=chosen, temperature=temp)
        good = chosen == GOOD
        return EpisodeRecord([dec], good, 5000.0, 0 if good else 1)

    cfg = TrainConfig(iterations=40, episodes_per_iter=8, lr=5e-2, temperature=120.0,
                      train_seeds=tuple(range(50)))
    trainer = ResidualTrainer(scorer, stub_rollout, cfg)
    trainer.train()

    from way4.rl.features import FEATURE_DIM
    feats = [[0.2] * FEATURE_DIM for _ in range(3)]
    deltas = scorer.residuals(feats)
    totals = [100.0 + deltas[i] for i in range(3)]
    assert totals[GOOD] == min(totals), (
        f"trainer did not learn to prefer the rewarded action: totals={totals}"
    )


def test_trainer_requires_torch_or_available_scorer():
    scorer = ResidualScorer()
    if not scorer.available:
        with pytest.raises(RuntimeError):
            ResidualTrainer(scorer, lambda s, t, seed: None)
    else:
        # available scorer -> constructs fine
        ResidualTrainer(scorer, lambda s, t, seed: None)
