"""Feature contract: fixed tensor shapes shared by the builder and the models.

This module is the *only* torch-touching thing the features layer exposes to the
models layer. It deliberately imports **no geometry** so that ``models`` can
import :class:`FeatureBundle` / :class:`FeatureSpec` without ever reaching into
``geometry`` (architecture principle B: Models never import Geometry). The
:class:`~radio_rl.features.builder.DefaultFeatureBuilder` is what bridges geometry
to these tensors; it lives next door and may import geometry freely.

The feature dimensions are *fixed by design* (a documented layout, below), not
inferred from data. That lets the policy/value network be sized from
:meth:`FeatureSpec.default` at construction time, before any observation exists,
and keeps the pointer network agnostic to how many candidates a step happens to
produce (it processes a variable-length, masked set).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Feature layout (keep in sync with DefaultFeatureBuilder)
# ---------------------------------------------------------------------------
# Global (belief-level) features, Dg:
#   0 cleared fraction            5 pose_x / arena
#   1 localized-uncleared frac    6 pose_y / arena
#   2 detected fraction           7 pose_radius / arena
#   3 undetected fraction         8 current_channel / num_channels
#   4 virtual_time / time_scale   9 n_measurements (saturating)
#                                 10 n_exclusions (saturating)
_GLOBAL_DIM = 11

# Per-channel features, Dch (one row per channel, num_channels rows):
#   0..3 status one-hot (unknown/detected/localized/cleared)
#   4 n_bearings (saturating)     9  est bearing sin (from pose)
#   5 mec_radius / arena          10 est bearing cos (from pose)
#   6 region_diameter / diameter  11 scans_on / max_scans (saturating)
#   7 has_estimate                12 clearable flag
#   8 est distance / arena        13 is current channel
_CHANNEL_DIM = 14

# Per-candidate features, Dc (the candidate's channel row is gathered from
# channel_feats separately, via cand_channel_idx, and concatenated in the model
# — so these rows never duplicate per-channel state):
#   0..2  action_type one-hot (scan/clear/exit)
#   3..7  source one-hot (search/localize/followup/clear/safety)
#   8  target_x / arena              12 predicted_clear_probability
#   9  target_y / arena              13 predicted_region_reduction / arena
#   10 target distance / arena       14 expected_time / time_scale
#   11 heuristic_information_gain    15 target bearing sin (from pose)
#      (saturating)                  16 target bearing cos (from pose)
_CANDIDATE_DIM = 17


@dataclass(frozen=True)
class FeatureSpec:
    """Immutable description of the feature tensor shapes and the scales used.

    ``num_channels`` is a problem constant (20). The three ``*_dim`` fields are
    fixed by the layout above. Scales are the only tunables (set from the
    ``features`` config); they never change tensor shapes, so a model sized from
    :meth:`default` stays valid across scale choices.
    """

    global_dim: int = _GLOBAL_DIM
    channel_dim: int = _CHANNEL_DIM
    candidate_dim: int = _CANDIDATE_DIM
    num_channels: int = 20

    pos_scale: float = 1800.0        # arena radius (m)
    time_scale: float = 10000.0      # ~ one episode of task time (s)
    max_scans: float = 30.0          # saturates the scan-count feature
    count_scale: float = 50.0        # saturates measurement/exclusion counts

    @classmethod
    def default(cls) -> "FeatureSpec":
        return cls()

    @classmethod
    def from_cfg(cls, cfg: Optional[object]) -> "FeatureSpec":
        """Build from a ``features`` config node (only scales are configurable)."""
        def g(key: str, default: float) -> float:
            if cfg is None:
                return default
            try:
                v = cfg.get(key) if hasattr(cfg, "get") else cfg[key]
            except Exception:
                return default
            return default if v is None else float(v)

        return cls(
            pos_scale=g("pos_scale", cls.pos_scale),
            time_scale=g("time_scale", cls.time_scale),
            max_scans=g("max_scans", cls.max_scans),
            count_scale=g("count_scale", cls.count_scale),
        )


@dataclass
class FeatureBundle:
    """Unbatched tensors describing one decision step.

    ``select`` builds one of these per step; the trainer collates a list of them
    into a padded batch (see :func:`radio_rl.training.rollout.collate`). All
    tensors are float32 except ``cand_channel_idx`` (long) and ``cand_mask``
    (bool). ``n_real == cand_feats.shape[0]`` for a freshly built bundle.
    """

    global_feats: torch.Tensor       # [Dg]
    channel_feats: torch.Tensor      # [num_channels, Dch]
    cand_feats: torch.Tensor         # [N, Dc]
    cand_channel_idx: torch.Tensor   # [N] long, 0-based channel of each candidate
    cand_mask: torch.Tensor          # [N] bool, True for real candidates
    n_real: int

    @property
    def num_candidates(self) -> int:
        return int(self.cand_feats.shape[0])

    def to(self, device: torch.device | str) -> "FeatureBundle":
        return FeatureBundle(
            global_feats=self.global_feats.to(device),
            channel_feats=self.channel_feats.to(device),
            cand_feats=self.cand_feats.to(device),
            cand_channel_idx=self.cand_channel_idx.to(device),
            cand_mask=self.cand_mask.to(device),
            n_real=self.n_real,
        )


@dataclass
class BatchedFeatures:
    """A padded batch of :class:`FeatureBundle` steps, ready for the model.

    Candidate counts differ per step, so ``cand_*`` are padded to the batch's
    max candidate count with ``cand_mask`` marking the real entries; the actor
    masks the padding to ``-inf`` before the softmax. ``memory_state`` is the
    recurrent state threaded in for stateful memories (``None`` for memoryless
    models and for freshly collated training batches).
    """

    global_feats: torch.Tensor       # [B, Dg]
    channel_feats: torch.Tensor      # [B, C, Dch]
    cand_feats: torch.Tensor         # [B, N, Dc]
    cand_channel_idx: torch.Tensor   # [B, N] long
    cand_mask: torch.Tensor          # [B, N] bool
    memory_state: object = None

    @property
    def batch_size(self) -> int:
        return int(self.global_feats.shape[0])

    def to(self, device: torch.device | str) -> "BatchedFeatures":
        return BatchedFeatures(
            global_feats=self.global_feats.to(device),
            channel_feats=self.channel_feats.to(device),
            cand_feats=self.cand_feats.to(device),
            cand_channel_idx=self.cand_channel_idx.to(device),
            cand_mask=self.cand_mask.to(device),
            memory_state=self.memory_state,
        )


def collate(bundles: list[FeatureBundle], memory_state: object = None) -> BatchedFeatures:
    """Stack a list of bundles into one padded batch (pads candidates to max N)."""
    if not bundles:
        raise ValueError("collate() needs at least one bundle")
    b = len(bundles)
    n_max = max(bd.num_candidates for bd in bundles)
    dc = bundles[0].cand_feats.shape[1]
    ref = bundles[0].cand_feats

    global_feats = torch.stack([bd.global_feats for bd in bundles], dim=0)
    channel_feats = torch.stack([bd.channel_feats for bd in bundles], dim=0)
    cand_feats = ref.new_zeros((b, n_max, dc))
    cand_channel_idx = torch.zeros((b, n_max), dtype=torch.long)
    cand_mask = torch.zeros((b, n_max), dtype=torch.bool)
    for i, bd in enumerate(bundles):
        n = bd.num_candidates
        cand_feats[i, :n] = bd.cand_feats
        cand_channel_idx[i, :n] = bd.cand_channel_idx
        cand_mask[i, :n] = bd.cand_mask
    return BatchedFeatures(
        global_feats=global_feats,
        channel_feats=channel_feats,
        cand_feats=cand_feats,
        cand_channel_idx=cand_channel_idx,
        cand_mask=cand_mask,
        memory_state=memory_state,
    )


def batch_from_bundle(bundle: FeatureBundle, memory_state: object = None) -> BatchedFeatures:
    """Wrap a single bundle as a batch of one (for step-time inference)."""
    return BatchedFeatures(
        global_feats=bundle.global_feats.unsqueeze(0),
        channel_feats=bundle.channel_feats.unsqueeze(0),
        cand_feats=bundle.cand_feats.unsqueeze(0),
        cand_channel_idx=bundle.cand_channel_idx.unsqueeze(0),
        cand_mask=bundle.cand_mask.unsqueeze(0),
        memory_state=memory_state,
    )
