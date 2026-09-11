"""The PPO update: clipped surrogate policy loss + value loss + entropy bonus.

Consumes a flat list of :class:`Transition` (from any number of episodes),
collates their feature bundles into padded batches, and runs several epochs of
minibatch SGD. It updates the agent's model in place, so the same
:class:`PPOAgent` instance the pipeline uses reflects the new weights on the next
rollout. Standard PPO — nothing here knows about the task; it only sees tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F

from ..algorithms.ppo import PPOAgent
from ..features.spec import collate
from .rollout import Transition


@dataclass
class PPOConfig:
    lr: float = 3e-4
    clip_eps: float = 0.2
    epochs: int = 4
    minibatch_size: int = 256
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_adv: bool = True

    @classmethod
    def from_cfg(cls, cfg: Optional[Any]) -> "PPOConfig":
        def g(key: str, default):
            if cfg is None:
                return default
            try:
                v = cfg.get(key) if hasattr(cfg, "get") else cfg[key]
            except Exception:
                return default
            return default if v is None else v

        return cls(
            lr=float(g("lr", cls.lr)),
            clip_eps=float(g("clip_eps", cls.clip_eps)),
            epochs=int(g("epochs", cls.epochs)),
            minibatch_size=int(g("minibatch_size", cls.minibatch_size)),
            vf_coef=float(g("vf_coef", cls.vf_coef)),
            ent_coef=float(g("ent_coef", cls.ent_coef)),
            max_grad_norm=float(g("max_grad_norm", cls.max_grad_norm)),
            gamma=float(g("gamma", cls.gamma)),
            gae_lambda=float(g("gae_lambda", cls.gae_lambda)),
            normalize_adv=bool(g("normalize_adv", cls.normalize_adv)),
        )


class PPOTrainer:
    def __init__(self, agent: PPOAgent, cfg: PPOConfig) -> None:
        self.agent = agent
        self.model = agent.model
        self.device = agent.device
        self.cfg = cfg
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

    def update(self, transitions: list[Transition]) -> dict:
        """Run PPO epochs over ``transitions``; return mean-loss metrics."""
        if not transitions:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "n": 0}
        c = self.cfg
        n = len(transitions)
        bundles = [t.bundle for t in transitions]
        actions = torch.tensor([t.action for t in transitions], dtype=torch.long,
                               device=self.device)
        old_logp = torch.tensor([t.log_prob for t in transitions], dtype=torch.float32,
                                device=self.device)
        returns = torch.tensor([t.ret for t in transitions], dtype=torch.float32,
                               device=self.device)
        adv = torch.tensor([t.advantage for t in transitions], dtype=torch.float32,
                           device=self.device)
        if c.normalize_adv and n > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        p_losses, v_losses, ents = [], [], []
        idx_all = torch.arange(n)
        mb = max(1, min(c.minibatch_size, n))
        for _ in range(c.epochs):
            perm = idx_all[torch.randperm(n)]
            for start in range(0, n, mb):
                sel = perm[start:start + mb].tolist()
                batch = collate([bundles[i] for i in sel]).to(self.device)
                a = actions[sel]
                logp, entropy, value = self.model.evaluate_actions(batch, a)

                ratio = torch.exp(logp - old_logp[sel])
                a_sel = adv[sel]
                surr1 = ratio * a_sel
                surr2 = torch.clamp(ratio, 1.0 - c.clip_eps, 1.0 + c.clip_eps) * a_sel
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, returns[sel])
                entropy_mean = entropy.mean()
                loss = policy_loss + c.vf_coef * value_loss - c.ent_coef * entropy_mean

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.max_grad_norm)
                self.opt.step()

                p_losses.append(float(policy_loss.item()))
                v_losses.append(float(value_loss.item()))
                ents.append(float(entropy_mean.item()))

        return {
            "policy_loss": sum(p_losses) / len(p_losses),
            "value_loss": sum(v_losses) / len(v_losses),
            "entropy": sum(ents) / len(ents),
            "n": n,
        }
