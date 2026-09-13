import pytest
from way4.rl import SeedSplits


def test_default_seed_splits_are_disjoint():
    assert SeedSplits().validate()
    splits = SeedSplits()
    assert not set(splits.train) & set(splits.val)
    assert not set(splits.val) & set(splits.test)


def test_seed_overlap_is_rejected():
    with pytest.raises(ValueError):
        SeedSplits(train=(1,), val=(1,), test=(), development=()).validate()
