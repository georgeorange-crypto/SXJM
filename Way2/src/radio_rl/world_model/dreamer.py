"""Dreamer-style latent world model (``dreamer``): learned imagination.

A model-based plug-in. It **inherits** :class:`AnalyticalWorldModel` completely,
so candidate *annotation* (the features the policy actually consumes) stays the
exact deterministic physics — the learned part never touches the numbers the
agent scores on. What it adds is a compact recurrent latent dynamics model
(RSSM-lite: a deterministic GRU core with reward and observation heads) that can
*imagine* rollouts in latent space for planning or auxiliary training, in the
spirit of Dreamer / world-model RL.

The dynamics live in ``self.dynamics`` (an :class:`nn.Module`); the world model
itself is a plain object holding a reference to it (same pattern as
:class:`ResidualWorldModel`), which sidesteps the ABC + ``nn.Module`` metaclass
clash and keeps the annotate path free of any learned state.

Torch is imported here, but this module is loaded lazily by
:func:`radio_rl.world_model.build_world_model` only when ``world_model.type`` is
``dreamer``, so the torch-free core path never imports it.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.registry import WORLD_MODELS
from .analytical import AnalyticalWorldModel


class LatentDynamics(nn.Module):
    """Deterministic recurrent latent model (RSSM-lite).

    A GRU core carries a deterministic latent ``h``; an action embedding drives
    the transition, and lightweight heads decode a predicted reward and a
    reconstructed observation. Everything is standard torch and CPU-friendly, so
    it always constructs and runs (no ``mamba-ssm`` / heavyweight deps).

    Shapes: latent ``[B, latent_dim]``, action ``[B, action_dim]``, observation
    ``[B, obs_dim]``.
    """

    def __init__(self, latent_dim: int = 32, action_dim: int = 6,
                 hidden_dim: int = 64, obs_dim: int = 16) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.obs_dim = int(obs_dim)
        h = int(hidden_dim)
        self.act_embed = nn.Linear(self.action_dim, h)
        self.cell = nn.GRUCell(h, self.latent_dim)
        self.obs_encoder = nn.Linear(self.obs_dim, self.latent_dim)  # posterior
        self.reward_head = nn.Sequential(
            nn.Linear(self.latent_dim, h), nn.SiLU(), nn.Linear(h, 1))
        self.obs_head = nn.Linear(self.latent_dim, self.obs_dim)     # decoder

    def initial_latent(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(int(batch), self.latent_dim, device=device)

    def step(self, h: torch.Tensor, action: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor]:
        """One prior transition: ``(h, action) -> (h', predicted reward)``."""
        e = F.silu(self.act_embed(action))
        h_next = self.cell(e, h)
        reward = self.reward_head(h_next).squeeze(-1)
        return h_next, reward

    def observe(self, h: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Posterior correction: fold an observed feature vector into the latent."""
        return h + self.obs_encoder(obs)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.obs_head(h)

    def imagine(self, h0: torch.Tensor,
                policy: Callable[[torch.Tensor], torch.Tensor],
                horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll the latent forward under ``policy`` for ``horizon`` steps.

        ``policy`` maps a latent ``[B, latent_dim]`` to an action ``[B,
        action_dim]``. Returns imagined ``latents [B, H, latent_dim]`` and
        ``rewards [B, H]``.
        """
        h = h0
        lat, rew = [], []
        for _ in range(int(horizon)):
            action = policy(h)
            h, r = self.step(h, action)
            lat.append(h)
            rew.append(r)
        return torch.stack(lat, dim=1), torch.stack(rew, dim=1)


@WORLD_MODELS.register("dreamer")
class DreamerWorldModel(AnalyticalWorldModel):
    """Analytical annotation + a learned latent dynamics model for imagination."""

    def __init__(self, cfg: Any = None, latent_dim: int = 32, action_dim: int = 6,
                 hidden: int = 64, obs_dim: int = 16, **_: object) -> None:
        super().__init__()
        if cfg is not None:
            def g(key, default):
                try:
                    v = cfg.get(key) if hasattr(cfg, "get") else cfg[key]
                    return default if v is None else v
                except Exception:
                    return default
            latent_dim = int(g("latent_dim", latent_dim))
            action_dim = int(g("action_dim", action_dim))
            hidden = int(g("hidden", hidden))
            obs_dim = int(g("obs_dim", obs_dim))
        self.dynamics = LatentDynamics(latent_dim, action_dim, hidden, obs_dim)

    # -- imagination helpers (delegate to the latent module) ----------------
    def initial_latent(self, batch: int, device=None) -> torch.Tensor:
        return self.dynamics.initial_latent(batch, device)

    def imagine(self, h0: torch.Tensor,
                policy: Callable[[torch.Tensor], torch.Tensor],
                horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dynamics.imagine(h0, policy, horizon)

    # -- fit the dynamics on observed transitions ---------------------------
    def learn_dynamics(self, obs_seq: torch.Tensor, act_seq: torch.Tensor,
                       rew_seq: torch.Tensor, steps: int = 50, lr: float = 1e-3
                       ) -> dict:
        """Teacher-forced dynamics learning over observed rollouts.

        ``obs_seq [B, T, obs_dim]``, ``act_seq [B, T, action_dim]``, ``rew_seq
        [B, T]``. At each ``t`` the model predicts observation ``t+1`` and reward
        ``t`` from the prior transition; loss is reconstruction + reward MSE.
        """
        B, T, _ = obs_seq.shape
        opt = torch.optim.Adam(self.dynamics.parameters(), lr=lr)
        last = 0.0
        for _ in range(int(steps)):
            h = self.dynamics.initial_latent(B, obs_seq.device)
            obs_loss = obs_seq.new_zeros(())
            rew_loss = obs_seq.new_zeros(())
            for t in range(T - 1):
                h, r = self.dynamics.step(h, act_seq[:, t])
                obs_loss = obs_loss + F.mse_loss(self.dynamics.decode(h), obs_seq[:, t + 1])
                rew_loss = rew_loss + F.mse_loss(r, rew_seq[:, t])
                h = self.dynamics.observe(h, obs_seq[:, t + 1])   # posterior update
            denom = max(1, T - 1)
            loss = (obs_loss + rew_loss) / denom
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.item())
        return {"batch": int(B), "horizon": int(T), "loss": last}
