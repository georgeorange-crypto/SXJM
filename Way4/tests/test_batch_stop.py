"""P0 #8 — batch-internal STOP in the MacroExecutor.

When ``batch_stop`` is on, a scan primitive is dropped if its channel is already
resolved: CLEARED / ABSENT_CERTIFIED in belief, or force-certified absent by the
certificate. The skip predicate is *exactly* the §11 EXIT guard's, so a dropped
scan is one EXIT would already accept as done — it can never lose a clear (禁止10).

Two invariants are pinned here: (1) default OFF is byte-identical legacy — every
channel is still measured even when resolved; (2) a CLEAR is never dropped, even
for a channel that looks resolved (绝不跳过必需清除).
"""

from way4.belief import BeliefState, ChannelStatus
from way4.certificate import CertificateManager
from way4.core import MacroActionType, MacroCandidate, PrimitiveKind, RobotState
from way4.core.observation import Observation
from way4.executor import MacroExecutor


class FakeEnv:
    """Deterministic Env double (mirrors test_macro_executor.FakeEnv): measure sets
    channel, clear does not; scripted observations / clear hits."""

    def __init__(self, obs_for=None, hit_for=None):
        from way4.core import AnalyticalCostModel
        self.cost = AnalyticalCostModel()
        self.state = RobotState()
        self._obs_for = obs_for or (lambda x, y, ch: Observation.no_signal())
        self._hit_for = hit_for or (lambda x, y, ch: False)
        self.calls = []

    def measure(self, x, y, ch):
        self.state, _ = self.cost.apply_measure(self.state, (x, y), ch)
        self.calls.append(("measure", x, y, ch))
        obs = self._obs_for(x, y, ch)
        return Observation(obs.kind, obs.svd_deg, self.state.virtual_time_s), self.state.virtual_time_s

    def clear(self, x, y, ch):
        hit = bool(self._hit_for(x, y, ch))
        self.state, _ = self.cost.apply_clear(self.state, (x, y), hit=hit)
        self.calls.append(("clear", x, y, ch))
        return hit, self.state.virtual_time_s


def _measured(env):
    return [ch for (kind, x, y, ch) in env.calls if kind == "measure"]


# --- default OFF is legacy: resolved channels are still measured ----------------


def test_batch_stop_off_by_default_measures_every_channel():
    belief = BeliefState(n_channels=5)
    belief[2].mark_cleared()                     # ch2 resolved, but flag is off
    env = FakeEnv()
    ex = MacroExecutor(env, belief=belief)       # batch_stop defaults off
    res = ex.execute(MacroCandidate(MacroActionType.EXPLORE, (10.0, 0.0), scan_channels=(1, 2, 3)), RobotState())
    assert _measured(env) == [1, 2, 3]           # ch2 measured despite being cleared
    assert len(res.primitives) == 3


# --- ON: skip a scan whose channel belief already resolved ----------------------


def test_batch_stop_skips_belief_resolved_channel():
    belief = BeliefState(n_channels=5)
    belief[2].mark_cleared()                     # CLEARED => is_resolved
    env = FakeEnv()
    ex = MacroExecutor(env, belief=belief, batch_stop=True)
    res = ex.execute(MacroCandidate(MacroActionType.EXPLORE, (10.0, 0.0), scan_channels=(1, 2, 3)), RobotState())
    assert _measured(env) == [1, 3]              # ch2 dropped
    assert len(res.primitives) == 2
    assert all(p.primitive.channel != 2 for p in res.primitives)


# --- ON: skip a scan the certificate force-certifies absent ---------------------


class _CertStub:
    """Certifies a fixed set absent; records the rest (the executor folds observations
    for the channels it does measure)."""

    def __init__(self, certified):
        self.certified = set(certified)
        self.recorded = []

    def is_absent_certified(self, ch, force=False):
        return ch in self.certified

    def record_observation(self, ch, point, obs):
        self.recorded.append(ch)

    def mark_cleared(self, ch):
        pass


def test_batch_stop_skips_certificate_certified_channel():
    belief = BeliefState(n_channels=5)           # all UNKNOWN => skip is cert-driven
    cert = _CertStub(certified={2})
    env = FakeEnv()
    ex = MacroExecutor(env, belief=belief, certificate=cert, batch_stop=True)
    res = ex.execute(MacroCandidate(MacroActionType.VERIFY, (10.0, 0.0), scan_channels=(1, 2, 3)), RobotState())
    assert _measured(env) == [1, 3]              # ch2 force-certified => dropped
    assert cert.recorded == [1, 3]
    assert len(res.primitives) == 2


# --- ON: a CLEAR is NEVER dropped, even for a resolved channel (禁止10) ----------


def test_batch_stop_never_drops_a_clear():
    belief = BeliefState(n_channels=5)
    belief[4].mark_cleared()                     # looks resolved...
    env = FakeEnv(hit_for=lambda x, y, ch: ch == 4)
    ex = MacroExecutor(env, belief=belief, batch_stop=True)
    res = ex.execute(MacroCandidate(MacroActionType.CLEAR, (5.0, 5.0), clear_channel=4), RobotState())
    assert ("clear", 5.0, 5.0, 4) in env.calls   # ...but the CLEAR still runs
    assert [p.primitive.kind for p in res.primitives] == [PrimitiveKind.CLEAR]
