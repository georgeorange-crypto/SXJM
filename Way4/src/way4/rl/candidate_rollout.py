"""Recording and batching utilities for Candidate-PPO.

The audit record is deliberately independent from the policy: it describes one
decision's realised cost/progress and is therefore also usable by deterministic
ablations and post-hoc trace validators.
"""
from dataclasses import dataclass, asdict
from typing import Sequence
from math import isfinite
from .candidate_ppo import compute_gae

@dataclass
class CandidateTransition:
    observation: list
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool = False
    advantage: float = 0.0
    return_: float = 0.0
    audit: "TransitionAudit | None" = None


@dataclass(frozen=True)
class TransitionAudit:
    """Decision-level accounting contract (B01--B05).

    ``delta_time_s`` must equal the four explicit components within tolerance;
    progress fields are counts/deltas supplied by the executor, never inferred
    from reward.  This prevents a policy from manufacturing progress by reward
    shaping alone.
    """
    virtual_time_before: float
    virtual_time_after: float
    delta_move_time_s: float = 0.0
    delta_measure_time_s: float = 0.0
    delta_switch_time_s: float = 0.0
    delta_clear_time_s: float = 0.0
    delta_move_distance_m: float = 0.0
    delta_clear: int = 0
    delta_certificate: float = 0.0
    delta_localization: float = 0.0
    delta_mec: float = 0.0
    delta_entropy: float = 0.0
    delta_hypothesis: float = 0.0
    no_progress: bool = False
    repeat_measure: bool = False
    time_since_last_progress_s: float = 0.0
    distance_since_last_progress_m: float = 0.0
    decisions_since_last_progress: int = 0

    @property
    def delta_virtual_time_s(self) -> float:
        return float(self.virtual_time_after) - float(self.virtual_time_before)

    @property
    def accounted_time_s(self) -> float:
        return (self.delta_move_time_s + self.delta_measure_time_s
                + self.delta_switch_time_s + self.delta_clear_time_s)

    def validate(self, tol: float = 1e-6) -> None:
        if not isfinite(tol) or tol < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        buckets = (self.delta_move_time_s, self.delta_measure_time_s,
                   self.delta_switch_time_s, self.delta_clear_time_s)
        if not all(isfinite(v) for v in
                   (self.virtual_time_before, self.virtual_time_after, *buckets)):
            raise ValueError("decision times must be finite")
        if any(v < 0 for v in buckets):
            raise ValueError("cost buckets must be nonnegative")
        if self.delta_virtual_time_s < -tol:
            raise ValueError("virtual time must be monotonic")
        if abs(self.delta_virtual_time_s - self.accounted_time_s) > tol:
            raise ValueError("decision time does not reconcile with cost buckets")

    def to_record(self) -> dict:
        out = asdict(self)
        out["delta_virtual_time_s"] = self.delta_virtual_time_s
        out["accounted_time_s"] = self.accounted_time_s
        return out

def finish_episode(transitions: list[CandidateTransition], gamma=.99, lam=.95):
    adv, ret = compute_gae([t.reward for t in transitions],
                           [t.value for t in transitions],
                           type('C', (), {'gamma':gamma,'gae_lambda':lam})())
    for t, a, r in zip(transitions, adv, ret): t.advantage, t.return_ = a, r
    return transitions


def timed_policy_transitions(records, end_time_s, *, full_clear,
                             n_unresolved, gamma=1.0, lam=.97):
    """Charge elapsed time until the next policy decision, including fallback.

    These are semi-Markov decision intervals. A final deterministic rescue is
    charged to the action preceding the handoff; time before the first policy
    decision is outside its control. No progress proxy is invented here.
    """
    from .objectives import dense_time_reward
    if not isfinite(end_time_s):
        raise ValueError('episode end time must be finite')
    transitions = []
    for i, record in enumerate(records):
        before = float(record['virtual_time_before'])
        after = (float(records[i + 1]['virtual_time_before'])
                 if i + 1 < len(records) else float(end_time_s))
        if not isfinite(before) or not isfinite(after) or after < before:
            raise ValueError('policy decision clocks must be finite and monotonic')
        transitions.append(CandidateTransition(
            record['observation'], record['action'], record['log_prob'],
            record['value'], dense_time_reward(after - before)))
    if transitions:
        transitions[-1].done = True
        if not full_clear:
            transitions[-1].reward -= 100.0 + max(0, int(n_unresolved))
        finish_episode(transitions, gamma=gamma, lam=lam)
    return transitions

def collate(transitions: Sequence[CandidateTransition]):
    if not transitions: return ([], [], [], [], [], [])
    width=max(len(t.observation) for t in transitions); dim=len(transitions[0].observation[0])
    obs=[]; mask=[]
    for t in transitions:
        n=len(t.observation); obs.append(t.observation+[[0.0]*dim]*(width-n)); mask.append([True]*n+[False]*(width-n))
    return (obs, [t.action for t in transitions],
            [t.log_prob for t in transitions], [t.return_ for t in transitions],
            [t.advantage for t in transitions], mask)
