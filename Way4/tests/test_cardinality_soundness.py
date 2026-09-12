import random

from way4.belief import BeliefState, ChannelStatus
from way4.certificate import CertificateManager


def test_random_10_to_16_cardinality_bounds_are_sound():
    rng = random.Random(90210)
    for n_present in range(10, 17):
        for _ in range(20):
            mgr = CertificateManager(n_channels=20)
            belief = BeliefState(n_channels=20)
            present = set(rng.sample(range(1, 21), n_present))
            mgr._present.update(present)
            for c in present:
                mgr.certs[c].present = True
            state = mgr.cardinality_state()
            assert state.present == n_present
            assert state.unknown == 20 - n_present - state.absent
            assert 0 <= state.q_min <= state.q_max <= state.unknown
            assert state.q_min == max(0, 10 - n_present)
            assert state.q_max <= min(state.unknown, 16 - n_present)
            # With no absence certificate, cardinality alone only certifies the
            # complement once 16 real channels are present.
            for c in set(range(1, 21)) - present:
                assert mgr.is_absent_certified(c) is (n_present >= 16)
                assert belief[c].status == ChannelStatus.UNKNOWN
