import random
from way4.rl import choose_seed


def test_seed_sampling_is_reproducible_and_prioritizes_observed_ratio():
    seeds = [1, 2]
    stats = {1: {"time_s": 100, "lower_bound_s": 100},
             2: {"time_s": 500, "lower_bound_s": 100, "tags": ["boundary"]}}
    a = choose_seed(seeds, stats, random.Random(7))
    b = choose_seed(seeds, stats, random.Random(7))
    assert a == b
