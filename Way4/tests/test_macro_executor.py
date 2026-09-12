"""M4 — MacroExecutor logic: primitive expansion, batch scan, channel tracking,
opportunistic clear, and the time budget. Timing here is exercised through a
lightweight fake env (it reuses Way4's own cost kernel); the independent timing
cross-check against Way3's real engine lives in test_way4_gold_timing_vs_env.py.
"""

from way4.belief import BeliefState, ChannelStatus
from way4.certificate import CertificateManager
from way4.core import (
    AnalyticalCostModel,
    MacroActionType,
    MacroCandidate,
    PrimitiveKind,
    RobotState,
)
from way4.core.observation import Observation
from way4.executor import MacroExecutor


class FakeEnv:
    """Deterministic Env double. Tracks pose/channel/clock with the same µs
    arithmetic as the engine (measure sets channel, clear does not); returns
    scripted observations and clear hits."""

    def __init__(self, obs_for=None, hit_for=None):
        self.cost = AnalyticalCostModel()
        self.state = RobotState()
        self._obs_for = obs_for or (lambda x, y, ch: Observation.no_signal())
        self._hit_for = hit_for or (lambda x, y, ch: False)
        self.calls = []

    def measure(self, x, y, ch):
        self.state, _ = self.cost.apply_measure(self.state, (x, y), ch)
        self.calls.append(("measure", x, y, ch))
        obs = self._obs_for(x, y, ch)
        # stamp the env clock onto the observation time
        return Observation(obs.kind, obs.svd_deg, self.state.virtual_time_s), self.state.virtual_time_s

    def clear(self, x, y, ch):
        hit = bool(self._hit_for(x, y, ch))
        self.state, _ = self.cost.apply_clear(self.state, (x, y), hit=hit)
        self.calls.append(("clear", x, y, ch))
        return hit, self.state.virtual_time_s


# --- primitive expansion ------------------------------------------------------


def test_macro_expands_to_primitives():
    m = MacroCandidate(MacroActionType.EXPLORE, (10.0, 0.0), scan_channels=(3, 7, 1))
    prims = m.primitives()
    assert [p.kind for p in prims] == [PrimitiveKind.MEASURE] * 3
    assert [p.channel for p in prims] == [3, 7, 1]
    assert all(p.target == (10.0, 0.0) for p in prims)   # batch scan: one waypoint

    c = MacroCandidate(MacroActionType.CLEAR, (5.0, 5.0), clear_channel=4)
    assert [p.kind for p in c.primitives()] == [PrimitiveKind.CLEAR]
    assert c.primitives()[0].channel == 4

    assert MacroCandidate(MacroActionType.EXIT, (0.0, 0.0)).primitives()[0].kind is PrimitiveKind.EXIT


# --- batch scan updates several channels from one move ------------------------


def test_batch_scan_folds_every_channel_into_belief_and_certificate():
    belief = BeliefState(n_channels=5)
    cert = CertificateManager(n_channels=5)
    env = FakeEnv()   # everything no_signal
    ex = MacroExecutor(env, belief=belief, certificate=cert)

    macro = MacroCandidate(MacroActionType.EXPLORE, (300.0, 400.0), scan_channels=(1, 2, 3))
    res = ex.execute(macro, RobotState())

    assert len(res.primitives) == 3
    # one move (to (300,400)) then three measurements: clock 105/111/117
    assert [round(p.virtual_time_s, 6) for p in res.primitives] == [105.0, 111.0, 117.0]
    assert res.state.channel == 3           # last measured channel
    for c in (1, 2, 3):
        assert belief[c].scan_count == 1
        assert len(cert.certs[c].negative_scan_points) == 1


def test_gold_sequence_through_executor_tracks_channel():
    """The full §1.1 sequence as macros: clock 105/111/194/199 and step-5 switch=0
    (clear ch3 leaves the measuring channel at 2)."""
    belief = BeliefState(n_channels=5)
    # ch3 clear misses (no source); measures are no_signal (timing is outcome-free)
    env = FakeEnv(hit_for=lambda x, y, ch: False)
    ex = MacroExecutor(env, belief=belief)

    s = RobotState()
    s = ex.execute(MacroCandidate(MacroActionType.EXPLORE, (300.0, 400.0), scan_channels=(1, 2)), s).state
    assert (s.virtual_time_s, s.channel) == (111.0, 2)
    s = ex.execute(MacroCandidate(MacroActionType.CLEAR, (300.0, 0.0), clear_channel=3), s).state
    assert (s.virtual_time_s, s.channel) == (194.0, 2)   # clear kept channel == 2
    res = ex.execute(MacroCandidate(MacroActionType.PURSUE, (300.0, 0.0), scan_channels=(2,)), s)
    assert (res.state.virtual_time_s, res.state.channel) == (199.0, 2)


# --- opportunistic clear on 'near' (§7) ---------------------------------------


def test_opportunistic_clear_on_near():
    belief = BeliefState(n_channels=3)
    # channel 2 returns 'near' at the waypoint; a source within 5 m -> a sure clear
    env = FakeEnv(
        obs_for=lambda x, y, ch: Observation.near() if ch == 2 else Observation.no_signal(),
        hit_for=lambda x, y, ch: ch == 2,
    )
    ex = MacroExecutor(env, belief=belief, opportunistic_clear=True)

    macro = MacroCandidate(MacroActionType.EXPLORE, (10.0, 0.0), scan_channels=(1, 2, 3))
    res = ex.execute(macro, RobotState())

    kinds = [(p.primitive.kind, p.primitive.channel) for p in res.primitives]
    assert (PrimitiveKind.CLEAR, 2) in kinds       # a clear was injected after the near
    assert res.n_cleared == 1
    assert belief[2].status is ChannelStatus.CLEARED


def test_opportunistic_clear_off_by_default():
    belief = BeliefState(n_channels=3)
    env = FakeEnv(obs_for=lambda x, y, ch: Observation.near() if ch == 2 else Observation.no_signal())
    ex = MacroExecutor(env, belief=belief)   # opportunistic_clear defaults off
    res = ex.execute(MacroCandidate(MacroActionType.EXPLORE, (10.0, 0.0), scan_channels=(1, 2, 3)), RobotState())
    assert res.n_cleared == 0
    assert all(p.primitive.kind is PrimitiveKind.MEASURE for p in res.primitives)


# --- time budget: 超时即停 ----------------------------------------------------


def test_time_budget_stops_before_exceeding():
    env = FakeEnv()
    ex = MacroExecutor(env)
    # move to (300,400) = 100 s, then measures at 5/6/6 s -> 105/111/117.
    # a 111 s budget must admit the first two measures and stop before the third.
    macro = MacroCandidate(MacroActionType.EXPLORE, (300.0, 400.0), scan_channels=(1, 2, 3))
    res = ex.execute(macro, RobotState(), time_budget_s=111.0)
    assert res.stopped_early is True
    assert len(res.primitives) == 2
    assert res.state.virtual_time_s == 111.0
