"""Recording and batching utilities for Candidate-PPO."""
from dataclasses import dataclass
from typing import Sequence
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

def finish_episode(transitions: list[CandidateTransition], gamma=.99, lam=.95):
    adv, ret = compute_gae([t.reward for t in transitions],
                           [t.value for t in transitions],
                           type('C', (), {'gamma':gamma,'gae_lambda':lam})())
    for t, a, r in zip(transitions, adv, ret): t.advantage, t.return_ = a, r
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
