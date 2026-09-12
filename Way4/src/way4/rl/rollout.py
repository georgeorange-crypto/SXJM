"""Way4-engine rollout + validation for the M10 residual trainer (DESIGN.md §12).

Bridges the engine-agnostic :class:`ResidualTrainer` to the authoritative Way3
engine (the same one ``scripts/compare_way3_way4.py`` uses for the M7 gate), so the
policy is trained and validated on ground truth, not on Way4's internal belief.

  * :func:`make_way4_rollout` — one stochastic episode under the
    ``SamplingResidualPlanner``; returns the recorded decisions + the GROUND-TRUTH
    outcome (``case.cleared_count == case.total``), never ``belief.all_resolved``.
  * :func:`validate` — GREEDY evaluation (the deployment ``RLResidualPlanner``, no
    sampling) of a scorer against the math baseline over a seed set. This is the
    acceptance gate: a checkpoint is only worth keeping if it does not lower
    full-clear (禁止10). Returns per-stack full-clear + mean time.

Kept out of ``train.py`` so the trainer has no hard engine dependency (its unit
tests inject a stub rollout); this module is imported only by the training script.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .residual_planner import RLResidualPlanner
from .sampling_planner import SamplingResidualPlanner
from .scorer import ResidualScorer
from .train import EpisodeRecord


def _load_engine():
    """Import the Way3 engine (adds it to sys.path if needed). Returns the module
    or None when the Way3 tree is absent."""
    try:
        from jammerhunt import environment as env
        return env
    except Exception:
        import pathlib
        import sys

        here = pathlib.Path(__file__).resolve()
        sx_root = here.parents[4]              # SX/
        way3 = sx_root / "Way3"
        for p in (sx_root, way3):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        try:
            from jammerhunt import environment as env
            return env
        except Exception:
            return None


def make_way4_rollout(problem: int = 4, field_kind: str = "smooth", max_steps: int = 3000):
    """Build a ``rollout_fn(scorer, temperature, seed) -> EpisodeRecord`` bound to a
    fresh Way3 engine per call. Ground-truth outcome; records sampling decisions."""
    from way4.executor import Way3EngineAdapter
    from way4.pipeline import Way4Pipeline

    env = _load_engine()
    if env is None:
        raise RuntimeError("Way3 environment unavailable; cannot roll out episodes")

    def rollout(scorer: ResidualScorer, temperature: float, seed: int) -> EpisodeRecord:
        case = env.generate_case(seed=seed, problem=problem, field_kind=field_kind)
        engine = env.Engine(case)
        engine.enter()
        planner = SamplingResidualPlanner(
            scorer, temperature=temperature, rng=random.Random(seed)
        )
        pipe = Way4Pipeline(
            Way3EngineAdapter(engine), n_channels=20, problem=problem,
            max_steps=max_steps, planner=planner,
        )
        result = pipe.run()
        full_clear = (case.cleared_count == case.total)     # GROUND TRUTH
        n_unresolved = case.total - case.cleared_count
        return EpisodeRecord(
            decisions=planner.decisions,
            full_clear=full_clear,
            virtual_time_s=result.virtual_time_s,
            n_unresolved=n_unresolved,
        )

    return rollout


@dataclass
class ValidationResult:
    n: int
    math_full_clear: int
    rl_full_clear: int
    math_mean_time: float
    rl_mean_time: float

    @property
    def passes(self) -> bool:
        """禁止10 gate: the RL residual must not lower full-clear."""
        return self.rl_full_clear >= self.math_full_clear

    @property
    def both_cleared_speedup(self) -> Optional[float]:
        if self.math_mean_time <= 0:
            return None
        return self.rl_mean_time / self.math_mean_time


def validate(
    scorer: ResidualScorer,
    seeds: Sequence[int],
    *,
    problem: int = 4,
    field_kind: str = "smooth",
    max_steps: int = 3000,
) -> ValidationResult:
    """GREEDY head-to-head: math planner vs. ``RLResidualPlanner(scorer)`` on the
    same seeds. Time is meaned only over episodes BOTH stacks fully cleared (the only
    fair set, mirroring the M7 script)."""
    from way4.executor import Way3EngineAdapter
    from way4.pipeline import Way4Pipeline
    from way4.planner import RecedingHorizonPlanner

    env = _load_engine()
    if env is None:
        raise RuntimeError("Way3 environment unavailable; cannot validate")

    math_fc = rl_fc = 0
    both_times: List[Tuple[float, float]] = []
    for seed in seeds:
        # math baseline (fresh case)
        case_m = env.generate_case(seed=seed, problem=problem, field_kind=field_kind)
        eng_m = env.Engine(case_m); eng_m.enter()
        rm = Way4Pipeline(
            Way3EngineAdapter(eng_m), n_channels=20, problem=problem,
            max_steps=max_steps, planner=RecedingHorizonPlanner(),
        ).run()
        m_ok = (case_m.cleared_count == case_m.total)

        # RL greedy (fresh case, identical ground truth at same seed)
        case_r = env.generate_case(seed=seed, problem=problem, field_kind=field_kind)
        eng_r = env.Engine(case_r); eng_r.enter()
        rr = Way4Pipeline(
            Way3EngineAdapter(eng_r), n_channels=20, problem=problem,
            max_steps=max_steps, planner=RLResidualPlanner(scorer=scorer),
        ).run()
        r_ok = (case_r.cleared_count == case_r.total)

        math_fc += int(m_ok)
        rl_fc += int(r_ok)
        if m_ok and r_ok:
            both_times.append((rm.virtual_time_s, rr.virtual_time_s))

    n = len(seeds)
    m_mean = sum(a for a, _ in both_times) / len(both_times) if both_times else 0.0
    r_mean = sum(b for _, b in both_times) / len(both_times) if both_times else 0.0
    return ValidationResult(n, math_fc, rl_fc, m_mean, r_mean)
