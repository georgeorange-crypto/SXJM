"""Candidate-PPO components for Way4-Final.

This module deliberately keeps the safety boundary outside the learner: the
generator supplies only safe spatial opportunities and the policy selects an
index, never coordinates.  It is independent of the legacy residual scorer so
the latter remains a reproducible ablation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

try:
    import torch
    from torch import nn
    _TORCH = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    _TORCH = False


@dataclass(frozen=True)
class CandidateChannelFeatures:
    """Complete candidate x channel interaction (fixed, serialisable schema)."""
    worth_measuring: float = 0.0
    guaranteed_detect: float = 0.0
    information_gain: float = 0.0
    expected_mec_after: float = 0.0
    clear_probability: float = 0.0
    certificate_gain: float = 0.0
    crossing_quality: float = 0.0
    already_measured: float = 0.0
    visibility_gain: float = 0.0
    directional_entropy: float = 0.0

    def as_list(self):
        return [float(getattr(self, k)) for k in self.__dataclass_fields__]


@dataclass
class SpatialCandidate:
    """A location-first action: one visit may serve many channel tasks."""
    point: tuple[float, float]
    services: dict[int, tuple[str, ...]]
    interactions: dict[int, CandidateChannelFeatures]
    route_delta: float = 0.0
    route_rank: float = 0.0
    corridor_distance: float = 0.0
    next_task_distance: float = 0.0
    math_score: float = 0.0
    travel_time: float = 0.0
    service_time: float = 0.0

    @property
    def channels(self):
        return tuple(sorted(self.interactions))

    def vector(self, robot_pos=(0.0, 0.0), n_channels=20):
        x, y = self.point
        dx, dy = x - robot_pos[0], y - robot_pos[1]
        base = [dx / 1800., dy / 1800., (dx*dx+dy*dy)**.5 / 1800.,
                self.travel_time/1000., self.service_time/1000.,
                self.route_delta/1000., self.route_rank/64.,
                self.corridor_distance/1800., self.next_task_distance/1800.,
                self.math_score/1000.]
        for c in range(1, n_channels + 1):
            base.extend(self.interactions.get(c, CandidateChannelFeatures()).as_list())
        return base


if _TORCH:
    class CandidateActorCritic(nn.Module):
        """Channel transformer + candidate/channel attention actor-critic."""
        def __init__(self, candidate_dim: int, channel_dim: int = 10,
                     hidden: int = 128, heads: int = 4, layers: int = 2):
            super().__init__()
            self.candidate_dim, self.channel_dim, self.hidden = candidate_dim, channel_dim, hidden
            self.cand = nn.Linear(10, hidden)
            self.chan = nn.Linear(channel_dim, hidden)
            h = max(1, min(heads, hidden))
            while hidden % h: h -= 1
            enc = nn.TransformerEncoderLayer(hidden, h, 2*hidden, batch_first=True,
                                             activation="gelu", norm_first=True)
            self.channel_encoder = nn.TransformerEncoder(enc, layers)
            self.cross = nn.MultiheadAttention(hidden, h, batch_first=True)
            self.actor = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
            self.critic = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

        def forward(self, x):
            # x: [B, A, 10 + C*10], first 10 are spatial/route features.
            b, a, _ = x.shape
            channels = x[:, :, 10:].reshape(b*a, -1, self.channel_dim)
            ch = self.channel_encoder(self.chan(channels))
            q = self.cand(x[:, :, :10]).reshape(b*a, 1, self.hidden)
            fused, _ = self.cross(q, ch, ch)
            z = fused[:, 0].reshape(b, a, self.hidden)
            return self.actor(z).squeeze(-1), self.critic(z.mean(1)).squeeze(-1)


@dataclass
class PPOConfig:
    gamma: float = .99
    gae_lambda: float = .95
    clip_eps: float = .2
    lr: float = 3e-4
    epochs: int = 4
    entropy_coef: float = .01
    value_coef: float = .5
    max_grad_norm: float = .5


def compute_gae(rewards: Sequence[float], values: Sequence[float], cfg=PPOConfig()):
    adv = [0.0] * len(rewards); carry = 0.0
    for i in range(len(rewards)-1, -1, -1):
        nxt = values[i+1] if i+1 < len(values) else 0.0
        carry = rewards[i] + cfg.gamma*nxt - values[i] + cfg.gamma*cfg.gae_lambda*carry
        adv[i] = carry
    return adv, [a+v for a, v in zip(adv, values)]


if _TORCH:
    class CandidatePPOTrainer:
        """Small generic masked PPO update for recorded candidate decisions."""
        def __init__(self, model, cfg=PPOConfig()):
            self.model, self.cfg = model, cfg
            self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        def update(self, observations, actions, old_log_probs, returns, advantages, mask=None):
            x = torch.as_tensor(observations, dtype=torch.float32)
            a = torch.as_tensor(actions, dtype=torch.long)
            old = torch.as_tensor(old_log_probs, dtype=torch.float32)
            ret = torch.as_tensor(returns, dtype=torch.float32)
            adv = torch.as_tensor(advantages, dtype=torch.float32)
            if len(adv) > 1: adv = (adv-adv.mean())/(adv.std()+1e-8)
            losses = []
            for _ in range(self.cfg.epochs):
                logits, value = self.model(x)
                if mask is not None:
                    logits = logits.masked_fill(~torch.as_tensor(mask, dtype=torch.bool), -1e9)
                logp = torch.log_softmax(logits, -1).gather(1, a[:, None]).squeeze(1)
                ratio = torch.exp(logp-old)
                clipped = torch.clamp(ratio, 1-self.cfg.clip_eps, 1+self.cfg.clip_eps)
                policy = -torch.minimum(ratio*adv, clipped*adv).mean()
                value_loss = (value-ret).pow(2).mean()
                entropy = -(torch.softmax(logits,-1)*torch.log_softmax(logits,-1)).sum(-1).mean()
                loss = policy + self.cfg.value_coef*value_loss - self.cfg.entropy_coef*entropy
                self.optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step(); losses.append(float(loss.detach()))
            return {"loss": sum(losses)/len(losses), "n": len(actions)}
else:  # keep the math-only installation importable
    CandidateActorCritic = None
    CandidatePPOTrainer = None
