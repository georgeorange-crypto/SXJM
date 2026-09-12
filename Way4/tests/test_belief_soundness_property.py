import math
import random

from way4.belief import ChannelBelief
from sxjm_core.geometry import bearing_deg


def test_random_legal_observations_never_false_exclude_true_source():
    rng = random.Random(20260912)
    for _ in range(40):
        r = 1700.0 * math.sqrt(rng.random())
        a = rng.random() * 2.0 * math.pi
        truth = (r * math.cos(a), r * math.sin(a))
        belief = ChannelBelief(1)
        previous_area = None
        for _step in range(8):
            q = (rng.uniform(-2200.0, 2200.0), rng.uniform(-2200.0, 2200.0))
            d = math.hypot(q[0] - truth[0], q[1] - truth[1])
            if d <= 1000.0:
                theta = bearing_deg(q, truth) + rng.uniform(-1.0, 1.0)
                belief.record_bearing(q, theta)
            else:
                belief.record_no_signal(q)
            assert belief.contains_possible_source(truth)
            if previous_area is not None and belief.F_c is not None:
                assert belief.area <= previous_area + 1e-6
            previous_area = belief.area
