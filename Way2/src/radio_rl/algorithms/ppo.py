"""PPO agent: the learnable policy that plugs into the frozen pipeline.

At inference the agent turns the step's :class:`FeatureBundle` into logits over
the candidate set and returns the argmax index (or a sample). During training the
trainer flips it into *recording* mode: it then samples, and stores
(bundle, action, log-prob, value) per step so an episode run through the ordinary
:class:`~radio_rl.pipeline.Pipeline` yields an on-policy trajectory without the
pipeline knowing anything about training.

The agent owns the :class:`FeatureSpec` (the model is sized from it), so it also
produces the matching feature builder for the pipeline via
:meth:`make_feature_builder` — that is the pipeline's "feature builder built
lazily by the learnable path". Torch is imported here; this module is only
imported on the learnable path, lazily, from :func:`radio_rl.algorithms.build_agent`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from ..core.registry import ALGORITHMS
from ..candidates.generator import CandidateSet
from ..features.spec import FeatureBundle, FeatureSpec, batch_from_bundle
from ..geometry.belief import BeliefState
from ..models.agent_model import AgentModel, build_agent_model
from .base import Agent


@dataclass
class StepRecord:
    """One recorded on-policy decision (rewards are filled in by the trainer)."""

    bundle: FeatureBundle
    action: int
    log_prob: float
    value: float


@ALGORITHMS.register("ppo")
class PPOAgent(Agent):
    """Pointer actor-critic policy selecting one candidate per step."""

    needs_features = True

    def __init__(
        self,
        model: AgentModel,
        spec: FeatureSpec,
        feature_cfg: Any = None,
        device: str = "cpu",
        deterministic_eval: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.spec = spec
        self.feature_cfg = feature_cfg
        self.device = device
        self.deterministic_eval = bool(deterministic_eval)

        self._state: Any = None            # recurrent memory state (None if memoryless)
        self._recording = False
        self.records: list[StepRecord] = []

    # -- pipeline hooks ----------------------------------------------------
    def reset(self) -> None:
        self._state = None

    def make_feature_builder(self):
        """Return a feature builder whose spec matches this model (pipeline hook)."""
        from ..features import build_feature_builder

        return build_feature_builder(self.feature_cfg, spec=self.spec)

    def select(
        self,
        candidates: CandidateSet,
        belief: BeliefState,
        features: Optional[object] = None,
    ) -> int:
        if not isinstance(features, FeatureBundle):
            # the pipeline supplies features for needs_features agents; guard anyway
            return 0
        sample = self._recording  # sample while collecting rollouts, argmax at eval
        with torch.no_grad():
            batch = batch_from_bundle(features, memory_state=self._state).to(self.device)
            logits, value, enc = self.model.policy_value(batch)
            self._state = enc.memory_state
            dist = torch.distributions.Categorical(logits=logits)
            if sample:
                action = dist.sample()
            elif self.deterministic_eval:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
            idx = int(action.item())
            if self._recording:
                self.records.append(
                    StepRecord(
                        bundle=features,
                        action=idx,
                        log_prob=float(dist.log_prob(action).item()),
                        value=float(value.item()),
                    )
                )
        return idx

    # -- training-mode control (used by the trainer) ----------------------
    def begin_recording(self) -> None:
        self._recording = True
        self.records = []
        self._state = None

    def take_records(self) -> list[StepRecord]:
        recs = self.records
        self.records = []
        self._recording = False
        return recs

    # -- checkpoints -------------------------------------------------------
    def state_dict(self) -> dict:
        return self.model.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.model.load_state_dict(sd)


def _node(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    try:
        v = cfg.get(key) if hasattr(cfg, "get") else cfg[key]
    except Exception:
        return default
    return default if v is None else v


def build_ppo_agent(cfg: Any = None, **kwargs: Any) -> PPOAgent:
    """Build a :class:`PPOAgent` from the ``algorithm`` config node.

    Expected shape (see ``configs/algorithm/ppo.yaml``)::

        type: ppo
        device: cpu
        deterministic_eval: true
        features: {pos_scale: 1800, time_scale: 10000, ...}
        model:    {hidden_dim: 128, encoder: {...}, memory: {...}, fusion: {...}}
    """
    feature_cfg = _node(cfg, "features")
    spec = FeatureSpec.from_cfg(feature_cfg)
    model = build_agent_model(_node(cfg, "model"), spec)

    device = str(_node(cfg, "device", "cpu"))
    deterministic = bool(_node(cfg, "deterministic_eval", True))
    agent = PPOAgent(
        model=model,
        spec=spec,
        feature_cfg=feature_cfg,
        device=device,
        deterministic_eval=deterministic,
    )

    ckpt = _node(cfg, "checkpoint")
    if ckpt:
        import os

        if os.path.isfile(str(ckpt)):
            agent.load_state_dict(torch.load(str(ckpt), map_location=device))
    return agent
