"""Feature layer: the belief/candidates -> tensor bridge.

Checks the bundle's fixed shapes and dtypes, that features are finite and
bounded, that the documented per-channel encoding is what the builder emits, and
that collate pads a mixed-length batch and masks the padding.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from radio_rl.candidates.generator import CandidateGenerator
from radio_rl.core.constants import CONSTANTS
from radio_rl.core.datatypes import (
    ChannelStatus,
    DetectionObservation,
    ObservationType,
)
from radio_rl.features import DefaultFeatureBuilder, FeatureSpec, build_feature_builder
from radio_rl.features.spec import FeatureBundle, collate


def _belief_with_observations() -> "object":
    from radio_rl.geometry.belief import BeliefState

    b = BeliefState(problem=3)
    b.update(DetectionObservation(ObservationType.SIGNAL, 3, 0.0, 0.0, 5.0, bearing_deg=30.0))
    b.update(DetectionObservation(ObservationType.TOO_STRONG, 5, 100.0, 0.0, 6.0))
    return b


def _fake_bundle(n: int, spec: FeatureSpec) -> FeatureBundle:
    return FeatureBundle(
        global_feats=torch.randn(spec.global_dim),
        channel_feats=torch.randn(spec.num_channels, spec.channel_dim),
        cand_feats=torch.randn(n, spec.candidate_dim),
        cand_channel_idx=torch.zeros(n, dtype=torch.long),
        cand_mask=torch.ones(n, dtype=torch.bool),
        n_real=n,
    )


def test_bundle_shapes_and_dtypes():
    b = _belief_with_observations()
    cs = CandidateGenerator().generate(b)
    fb = DefaultFeatureBuilder()
    spec = fb.spec
    bundle = fb.build(b, cs)

    n = len(cs.candidates)
    assert bundle.global_feats.shape == (spec.global_dim,)
    assert bundle.channel_feats.shape == (spec.num_channels, spec.channel_dim)
    assert bundle.cand_feats.shape == (n, spec.candidate_dim)
    assert bundle.cand_channel_idx.shape == (n,)
    assert bundle.cand_mask.shape == (n,)
    assert bundle.n_real == n

    assert bundle.global_feats.dtype == torch.float32
    assert bundle.channel_feats.dtype == torch.float32
    assert bundle.cand_feats.dtype == torch.float32
    assert bundle.cand_channel_idx.dtype == torch.long
    assert bundle.cand_mask.dtype == torch.bool
    assert bool(bundle.cand_mask.all())


def test_features_finite_and_indices_in_range():
    b = _belief_with_observations()
    cs = CandidateGenerator().generate(b)
    bundle = DefaultFeatureBuilder().build(b, cs)
    for t in (bundle.global_feats, bundle.channel_feats, bundle.cand_feats):
        assert bool(torch.isfinite(t).all())
    assert int(bundle.cand_channel_idx.min()) >= 0
    assert int(bundle.cand_channel_idx.max()) <= CONSTANTS.num_channels - 1


def test_channel_encoding_matches_layout():
    from radio_rl.geometry.belief import BeliefState

    b = BeliefState(problem=3)
    # a 'too strong' reading localizes channel 5 at (100, 0) and sets current=5
    b.update(DetectionObservation(ObservationType.TOO_STRONG, 5, 100.0, 0.0, 6.0))
    bundle = DefaultFeatureBuilder().build(b, CandidateGenerator().generate(b))

    row = bundle.channel_feats[CONSTANTS.channel_index(5)]
    assert row[int(ChannelStatus.LOCALIZED)].item() == 1.0   # status one-hot
    assert row[7].item() == 1.0                               # has estimate
    assert row[12].item() == 1.0                              # clearable (near reading)
    assert row[13].item() == 1.0                              # is current channel

    # an untouched channel is UNKNOWN with no estimate
    other = bundle.channel_feats[CONSTANTS.channel_index(1)]
    assert other[int(ChannelStatus.UNKNOWN)].item() == 1.0
    assert other[7].item() == 0.0


def test_collate_pads_and_masks():
    spec = FeatureSpec.default()
    short, long = _fake_bundle(2, spec), _fake_bundle(5, spec)
    batch = collate([short, long])

    assert batch.global_feats.shape == (2, spec.global_dim)
    assert batch.channel_feats.shape == (2, spec.num_channels, spec.channel_dim)
    assert batch.cand_feats.shape == (2, 5, spec.candidate_dim)
    assert batch.cand_mask.shape == (2, 5)
    assert int(batch.cand_mask[0].sum()) == 2
    assert int(batch.cand_mask[1].sum()) == 5
    assert not bool(batch.cand_mask[0, 2:].any())              # padding masked off
    assert int(torch.count_nonzero(batch.cand_feats[0, 2:])) == 0  # padding zeroed


def test_build_feature_builder_reads_scales():
    fb = build_feature_builder({"type": "default", "pos_scale": 100.0})
    assert isinstance(fb, DefaultFeatureBuilder)
    assert fb.spec.pos_scale == 100.0
