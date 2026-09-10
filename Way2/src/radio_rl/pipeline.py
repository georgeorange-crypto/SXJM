"""The one frozen pipeline.

    Simulator -> Estimator -> Candidate Generator -> [Feature Builder] ->
    Agent -> Safety Shield -> Action

This is the single control loop the whole project runs through. Components are
built from config, so swapping the environment, the candidate parameters, the
world model, the agent, or the safety policy is a one-line change and never
touches this file. The Feature Builder stage is skipped for math agents
(``needs_features == False``) and used for learnable ones.

Nothing here imports torch: the default run path is pure math. A learnable agent
brings its own tensor stack in behind the ``needs_features`` flag.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from .algorithms import build_agent
from .algorithms.base import Agent
from .candidates import build_candidate_generator
from .candidates.generator import CandidateGenerator
from .core.datatypes import (
    Action,
    ActionType,
    DetectionObservation,
    EpisodeStats,
    ObservationType,
)
from .env import build_env
from .env.base import RadioEnv
from .geometry.belief import BeliefState, build_estimator_config
from .safety import build_safety_shield
from .safety.shield import SafetyShield
from .world_model import build_world_model


def _node(cfg: Any, key: str) -> Any:
    if cfg is None:
        return None
    try:
        return cfg.get(key) if hasattr(cfg, "get") else cfg[key]
    except Exception:
        return None


@dataclass
class StepTrace:
    """One executed step, for logging / RL trajectory collection."""

    action: Action
    observation: DetectionObservation
    candidate_index: int
    n_candidates: int
    virtual_time_s: float


class Pipeline:
    """Owns the components and runs episodes through the frozen loop."""

    def __init__(
        self,
        cfg: Any,
        env: Optional[RadioEnv] = None,
        agent: Optional[Agent] = None,
        generator: Optional[CandidateGenerator] = None,
        shield: Optional[SafetyShield] = None,
        feature_builder: Optional[object] = None,
    ) -> None:
        self.cfg = cfg
        self.problem = int(_node(cfg, "problem") or 3)

        self.env = env or build_env(cfg)
        self.world_model = build_world_model(_node(cfg, "world_model"))
        self.generator = generator or build_candidate_generator(
            _node(cfg, "candidates"), world_model=self.world_model
        )
        self.agent = agent or build_agent(_node(cfg, "algorithm"))
        self.shield = shield or build_safety_shield(_node(cfg, "safety"))
        self.estimator_cfg = build_estimator_config(_node(cfg, "geometry"))
        self.feature_builder = feature_builder  # built lazily by learnable path

        if getattr(self.agent, "needs_features", False) and self.feature_builder is None:
            raise RuntimeError(
                f"agent {type(self.agent).__name__} needs a FeatureBuilder but none "
                "was provided (the features layer is a later milestone)."
            )

    def new_belief(self) -> BeliefState:
        return BeliefState(cfg=self.estimator_cfg, problem=self.problem)

    # -- one episode -------------------------------------------------------
    def run_episode(
        self,
        seed: Optional[int] = None,
        max_steps: int = 100_000,
        collect_trace: bool = False,
    ) -> tuple[EpisodeStats, list[StepTrace]]:
        wall_start = time.monotonic()
        obs = self.env.reset(seed)
        belief = self.new_belief()
        belief.update(obs)
        self.agent.reset()

        stats = EpisodeStats()
        trace: list[StepTrace] = []
        scans_on_channel: dict[int, int] = {}
        prev_channel = belief.current_channel
        prev_x, prev_y = belief.pose_x, belief.pose_y

        for _ in range(max_steps):
            if self.env.finished:
                break

            candidates = self.generator.generate(belief)
            features = None
            if getattr(self.agent, "needs_features", False):
                features = self.feature_builder.build(belief, candidates)  # type: ignore[union-attr]
            idx = self.agent.select(candidates, belief, features)
            idx = max(0, min(idx, len(candidates.candidates) - 1))
            cand = candidates.candidates[idx]

            action = self.shield.apply(
                cand, remaining_real_s=self.env.remaining_real_duration_s()
            )

            if action.action_type == ActionType.EXIT:
                self.env.execute(action)
                if collect_trace:
                    trace.append(StepTrace(action, obs, idx, candidates.n_real,
                                           belief.virtual_time_s))
                break

            # travel distance for the route metric (pre-execute pose -> target)
            dx, dy = action.target_x - prev_x, action.target_y - prev_y
            stats.route_distance += (dx * dx + dy * dy) ** 0.5

            obs = self.env.execute(action)
            belief.update(obs)

            self._tally(stats, action, obs, prev_channel, scans_on_channel, belief)
            prev_channel = belief.current_channel
            prev_x, prev_y = action.target_x, action.target_y
            stats.steps += 1

            if collect_trace:
                trace.append(StepTrace(action, obs, idx, candidates.n_real,
                                       belief.virtual_time_s))

        self._finalize(stats, belief, wall_start)
        return stats, trace

    # -- bookkeeping -------------------------------------------------------
    def _tally(self, stats: EpisodeStats, action: Action,
               obs: DetectionObservation, prev_channel: int,
               scans_on_channel: dict[int, int], belief: BeliefState) -> None:
        if action.action_type == ActionType.SCAN:
            stats.num_scans += 1
            if action.channel != prev_channel:
                stats.num_switches += 1
            scans_on_channel[action.channel] = scans_on_channel.get(action.channel, 0) + 1
        elif action.action_type == ActionType.CLEAR:
            stats.num_clears += 1
            if obs.result_type == ObservationType.CLEAR_SUCCESS:
                stats.clear_times.append(belief.virtual_time_s)
                stats.scans_per_cleared.append(scans_on_channel.get(action.channel, 0))
            else:
                stats.failed_clears += 1

    def _finalize(self, stats: EpisodeStats, belief: BeliefState,
                  wall_start: float) -> None:
        stats.virtual_time = self.env.virtual_time_s
        stats.wall_time = time.monotonic() - wall_start
        stats.sources_cleared = belief.n_cleared()
        case = getattr(self.env, "case", None)
        if case is not None and getattr(case, "jammers", None) is not None:
            stats.sources_total = len(case.jammers)
        else:
            stats.sources_total = stats.sources_cleared


def run(cfg: Any, seed: Optional[int] = None) -> EpisodeStats:
    """Convenience: build a pipeline from cfg and run one episode."""
    stats, _ = Pipeline(cfg).run_episode(seed=seed)
    return stats
