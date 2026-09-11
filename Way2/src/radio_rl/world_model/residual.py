"""Residual world model (``residual``): learned corrections on the analytical model.

Subclasses :class:`AnalyticalWorldModel` and adds a small MLP that nudges the
closed-form time / clear-probability / region-reduction predictions toward what
the environment actually does (bearing noise, timing rounding, geometry the
closed form approximates). The correction head is **zero-initialised**, so an
untrained :class:`ResidualWorldModel` reproduces the analytical predictions
*exactly*: the learned path can only refine the baseline, never silently degrade
it (the world-model echo of "advanced plugins never break the core").

Corrections are applied in a form that preserves validity: time and region
reduction are scaled multiplicatively (stay >= 0), clear probability is shifted in
logit space (stays in (0, 1)); predictions already pinned at 0 or 1 are left
untouched.

Torch is imported here, but this module is loaded lazily by
:func:`radio_rl.world_model.build_world_model` only when ``world_model.type`` is
``residual``, so the torch-free core path never imports it.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from ..core.constants import CONSTANTS
from ..core.datatypes import ActionType
from ..core.registry import WORLD_MODELS
from ..geometry.belief import BeliefState
from .analytical import AnalyticalWorldModel

_FEAT_DIM = 9        # see _features()
_SCALE = 0.2         # correction sensitivity (small: gentle refinement of base)
_ARENA = CONSTANTS.region_radius
_MAX_SCANS = 30.0


def _fin(value: Any, scale: float, cap: float = 4.0) -> float:
    """Normalise ``value/scale`` to a finite, bounded float (inf/NaN -> 0)."""
    try:
        r = float(value) / float(scale)
    except Exception:
        return 0.0
    if not math.isfinite(r):
        return 0.0
    return max(-cap, min(cap, r))


@WORLD_MODELS.register("residual")
class ResidualWorldModel(AnalyticalWorldModel):
    """Analytical predictions plus a learned residual correction."""

    def __init__(self, cfg: Any = None, hidden: int = 32, **_: object) -> None:
        super().__init__()
        if cfg is not None:
            try:
                v = cfg.get("hidden") if hasattr(cfg, "get") else cfg["hidden"]
                hidden = int(v) if v is not None else hidden
            except Exception:
                pass
        self.net = nn.Sequential(
            nn.Linear(_FEAT_DIM, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), 3),          # (d_time, d_clear, d_region)
        )
        nn.init.zeros_(self.net[-1].weight)     # start == analytical exactly
        nn.init.zeros_(self.net[-1].bias)
        self._scale = _SCALE

    # -- feature row for one (belief, action) ------------------------------
    def _features(self, belief: BeliefState, action_type: int, channel: int,
                  tx: float, ty: float) -> torch.Tensor:
        at = int(action_type)
        onehot = [
            1.0 if at == int(ActionType.SCAN) else 0.0,
            1.0 if at == int(ActionType.CLEAR) else 0.0,
            1.0 if at == int(ActionType.EXIT) else 0.0,
        ]
        dist_target = _fin(math.hypot(tx - belief.pose_x, ty - belief.pose_y), _ARENA)
        cb = belief.channels.get(int(channel)) if hasattr(belief, "channels") else None
        mec = _fin(getattr(cb, "mec_radius", 0.0), _ARENA) if cb is not None else 0.0
        regd = _fin(getattr(cb, "region_diameter", 0.0), 2.0 * _ARENA) if cb is not None else 0.0
        est = getattr(cb, "estimate", None) if cb is not None else None
        if est is not None:
            dist_est = _fin(math.hypot(tx - est[0], ty - est[1]), _ARENA)
            has_est = 1.0
        else:
            dist_est, has_est = 0.0, 0.0
        scans = _fin(belief.scans_on(int(channel)), _MAX_SCANS)
        row = [*onehot, dist_target, mec, regd, dist_est, has_est, scans]
        return torch.tensor(row, dtype=torch.float32)

    def _delta(self, belief: BeliefState, action_type: int, channel: int,
               tx: float, ty: float, idx: int) -> float:
        with torch.no_grad():
            return float(self.net(self._features(belief, action_type, channel, tx, ty))[idx].item())

    # -- corrected predictions ---------------------------------------------
    def predict_time(self, belief: BeliefState, action_type: int, channel: int,
                     tx: float, ty: float, clear_prob: float = 0.0) -> float:
        base = AnalyticalWorldModel.predict_time(
            self, belief, action_type, channel, tx, ty, clear_prob)
        d = self._delta(belief, action_type, channel, tx, ty, 0)
        return float(base * math.exp(self._scale * d))

    def predict_clear_probability(self, belief: BeliefState, channel: int,
                                  tx: float, ty: float) -> float:
        base = AnalyticalWorldModel.predict_clear_probability(self, belief, channel, tx, ty)
        if base <= 1e-6 or base >= 1.0 - 1e-6:
            return base                              # pinned: leave the baseline alone
        d = self._delta(belief, int(ActionType.CLEAR), channel, tx, ty, 1)
        logit = math.log(base / (1.0 - base)) + 0.5 * d
        return 1.0 / (1.0 + math.exp(-logit))

    def predict_region_reduction(self, belief: BeliefState, channel: int,
                                 tx: float, ty: float) -> float:
        base = AnalyticalWorldModel.predict_region_reduction(self, belief, channel, tx, ty)
        if base <= 0.0:
            return base
        d = self._delta(belief, int(ActionType.SCAN), channel, tx, ty, 2)
        return float(base * math.exp(self._scale * d))

    # -- calibration from observed outcomes --------------------------------
    def fit(self, samples: list[dict], steps: int = 100, lr: float = 1e-2) -> dict:
        """Calibrate the time correction against observed task times.

        Each sample is ``{belief, channel, target_x, target_y, time,
        action_type?}``; ``time`` is the observed duration (s). The clear /
        region heads calibrate the same way against observed clear outcomes and
        region shrink — left to the caller who has that ground truth.
        """
        if not samples:
            return {"n": 0}

        def feat(s):
            return self._features(s["belief"], int(s.get("action_type", ActionType.SCAN)),
                                  int(s["channel"]), float(s["target_x"]), float(s["target_y"]))

        feats = torch.stack([feat(s) for s in samples])
        base = torch.tensor([
            AnalyticalWorldModel.predict_time(
                self, s["belief"], int(s.get("action_type", ActionType.SCAN)),
                int(s["channel"]), float(s["target_x"]), float(s["target_y"]))
            for s in samples], dtype=torch.float32)
        obs = torch.tensor([float(s["time"]) for s in samples], dtype=torch.float32)

        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        last = 0.0
        for _ in range(int(steps)):
            pred = base * torch.exp(self._scale * self.net(feats)[:, 0])
            loss = torch.mean((pred - obs) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.item())
        return {"n": len(samples), "loss_time": last}
