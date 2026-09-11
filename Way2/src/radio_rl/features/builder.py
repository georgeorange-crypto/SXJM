"""FeatureBuilder: the geometry -> tensor bridge.

This is the *only* module in the features/models layers that reads
:class:`~radio_rl.geometry.belief.BeliefState` and
:class:`~radio_rl.candidates.generator.CandidateSet`. It turns them into the
fixed-shape tensors of :class:`~radio_rl.features.spec.FeatureBundle`; every
model downstream sees tensors only (architecture principle B: Models never
import Geometry).

It honours the iron rule from the other side: the builder *reads* the belief and
never mutates it, and it never invents candidates — it only tensorizes the ones
the generator already produced. The layout is documented in ``spec.py`` and the
two must be kept in sync.

Torch is imported at module top level, which is fine: this module is only
imported on the learnable path (``needs_features == True``), lazily, behind the
registry.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from ..candidates.generator import CandidateSet
from ..core.constants import CONSTANTS
from ..core.datatypes import ActionType, ChannelStatus
from ..core.registry import FEATURE_BUILDERS
from ..geometry.belief import BeliefState
from .base import FeatureBuilder
from .spec import FeatureBundle, FeatureSpec

_NUM_ACTIONS = 3   # SCAN, CLEAR, EXIT
_NUM_SOURCES = 5   # SEARCH, LOCALIZE, FOLLOWUP, CLEAR, SAFETY
_NUM_STATUS = 4    # UNKNOWN, DETECTED, LOCALIZED, CLEARED


def _sat(value: float, scale: float) -> float:
    """value/scale clamped to [0, 1]; non-finite -> 1.0 (treat as 'maxed out')."""
    if not math.isfinite(value):
        return 1.0
    v = value / scale
    if v < 0.0:
        return 0.0
    return 1.0 if v > 1.0 else v


@FEATURE_BUILDERS.register("default")
class DefaultFeatureBuilder(FeatureBuilder):
    """Analytical, per-channel + per-candidate feature extraction.

    Produces one :class:`FeatureBundle` per decision step. All work is O(channels
    + candidates); no torch autograd is involved (features are constants w.r.t.
    the policy). Tensors are built on CPU; the trainer/agent moves them to device.
    """

    def __init__(self, spec: Optional[FeatureSpec] = None) -> None:
        self.spec = spec or FeatureSpec.default()

    # -- public API used by the pipeline ----------------------------------
    def build(self, belief: BeliefState, candidates: CandidateSet) -> FeatureBundle:
        g = self._global_feats(belief)
        ch = self._channel_feats(belief)
        cf, idx = self._candidate_feats(belief, candidates)
        n = cf.shape[0]
        return FeatureBundle(
            global_feats=g,
            channel_feats=ch,
            cand_feats=cf,
            cand_channel_idx=idx,
            cand_mask=torch.ones(n, dtype=torch.bool),
            n_real=n,
        )

    # -- global (belief-level) --------------------------------------------
    def _global_feats(self, b: BeliefState) -> torch.Tensor:
        s = self.spec
        n_ch = float(s.num_channels)
        pose_r = math.hypot(b.pose_x, b.pose_y)
        feats = [
            b.n_cleared() / n_ch,
            len(b.uncleared_localized()) / n_ch,
            len(b.detected_channels()) / n_ch,
            len(b.undetected_channels()) / n_ch,
            _sat(b.virtual_time_s, s.time_scale),
            max(-1.0, min(1.0, b.pose_x / s.pos_scale)),
            max(-1.0, min(1.0, b.pose_y / s.pos_scale)),
            _sat(pose_r, s.pos_scale),
            b.current_channel / n_ch,
            _sat(float(b.n_measurements), s.count_scale),
            _sat(float(len(b.exclusions)), s.count_scale),
        ]
        return torch.tensor(feats, dtype=torch.float32)

    # -- per-channel ------------------------------------------------------
    def _channel_feats(self, b: BeliefState) -> torch.Tensor:
        s = self.spec
        rows = torch.zeros(s.num_channels, s.channel_dim, dtype=torch.float32)
        for ch in range(CONSTANTS.channel_min, CONSTANTS.channel_max + 1):
            i = CONSTANTS.channel_index(ch)
            cb = b.belief(ch)
            row = rows[i]

            status = int(cb.status)
            if 0 <= status < _NUM_STATUS:
                row[status] = 1.0                            # 0..3 status one-hot

            row[4] = _sat(float(cb.n_bearings), s.max_scans)
            row[5] = _sat(cb.mec_radius, s.pos_scale)
            row[6] = _sat(cb.region_diameter, CONSTANTS.diameter)

            est = cb.estimate
            if est is not None:
                dx, dy = est[0] - b.pose_x, est[1] - b.pose_y
                dist = math.hypot(dx, dy)
                row[7] = 1.0                                 # has estimate
                row[8] = _sat(dist, s.pos_scale)
                if dist > 1e-9:
                    row[9] = dy / dist                       # bearing sin
                    row[10] = dx / dist                      # bearing cos

            row[11] = _sat(float(b.scans_on(ch)), s.max_scans)
            row[12] = 1.0 if b.clearable(ch) else 0.0
            row[13] = 1.0 if ch == b.current_channel else 0.0
        return rows

    # -- per-candidate ----------------------------------------------------
    def _candidate_feats(
        self, b: BeliefState, candidates: CandidateSet
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.spec
        cands = candidates.candidates
        n = len(cands)
        feats = torch.zeros(n, s.candidate_dim, dtype=torch.float32)
        idx = torch.zeros(n, dtype=torch.long)

        for j, c in enumerate(cands):
            row = feats[j]
            at = int(c.action_type)
            if 0 <= at < _NUM_ACTIONS:
                row[at] = 1.0                                # 0..2 action one-hot
            src = int(c.source)
            if 0 <= src < _NUM_SOURCES:
                row[3 + src] = 1.0                           # 3..7 source one-hot

            row[8] = max(-1.0, min(1.0, c.target_x / s.pos_scale))
            row[9] = max(-1.0, min(1.0, c.target_y / s.pos_scale))
            dx, dy = c.target_x - b.pose_x, c.target_y - b.pose_y
            dist = math.hypot(dx, dy)
            row[10] = _sat(dist, s.pos_scale)
            row[11] = _sat(c.heuristic_information_gain, s.pos_scale)
            row[12] = max(0.0, min(1.0, c.predicted_clear_probability))
            row[13] = _sat(c.predicted_region_reduction, s.pos_scale)
            row[14] = _sat(c.expected_time, s.time_scale)
            if dist > 1e-9:
                row[15] = dy / dist                          # target bearing sin
                row[16] = dx / dist                          # target bearing cos

            # channel row to gather in the model; EXIT's channel is arbitrary but
            # always a valid index, so the gather is always well-defined.
            ci = CONSTANTS.channel_index(int(c.channel))
            idx[j] = max(0, min(s.num_channels - 1, ci))

        return feats, idx


def build_feature_builder(
    cfg: Optional[object] = None, spec: Optional[FeatureSpec] = None
) -> FeatureBuilder:
    """Build a feature builder from a ``features`` config node.

    ``cfg.type`` selects the implementation (default ``"default"``); scales come
    from the same node. Kept tiny so the learnable path can call it lazily.
    """
    name = "default"
    if cfg is not None:
        try:
            v = cfg.get("type") if hasattr(cfg, "get") else cfg["type"]
            if v:
                name = str(v)
        except Exception:
            pass
    spec = spec or FeatureSpec.from_cfg(cfg)
    return FEATURE_BUILDERS.create(name, spec=spec)
