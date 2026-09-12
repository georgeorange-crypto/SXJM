import math
import random

from way4.belief import ChannelBelief


def test_random_no_signal_never_excludes_true_source_from_effective_set():
    rng = random.Random(731)
    for _ in range(300):
        angle = rng.random() * 2.0 * math.pi
        radius = 1700.0 * math.sqrt(rng.random())
        source = (radius * math.cos(angle), radius * math.sin(angle))
        belief = ChannelBelief(1)
        for _ in range(rng.randint(1, 6)):
            # A NO_SIGNAL is physically sound only outside guaranteed detection.
            while True:
                p = (rng.uniform(-1800, 1800), rng.uniform(-1800, 1800))
                if math.dist(source, p) > 1000.0:
                    break
            belief.record_no_signal(p)
        assert belief.contains_possible_source(source)


def test_directional_blind_arc_keeps_source_when_bearing_is_physically_aligned():
    """Negative/arc geometry must remain conservative, including narrow wedges."""
    source = (420.0, 315.0)
    belief = ChannelBelief(1)
    scan_points = [(-500.0, -500.0), (1000.0, -500.0), (-500.0, 900.0)]
    for p in scan_points:
        bearing = math.degrees(math.atan2(source[1] - p[1], source[0] - p[0]))
        belief.record_bearing(p, bearing)
    assert belief.contains_possible_source(source)
    assert belief.effective_components(spacing=60.0) >= 1
