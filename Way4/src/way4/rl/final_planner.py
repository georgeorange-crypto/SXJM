"""Way4-Final planner adapter and deterministic progress watchdog."""
from dataclasses import dataclass
from typing import Optional

@dataclass
class WatchdogConfig:
    patience: int = 8
    max_detour_s: float = 300.0
    hysteresis_s: float = 5.0

class ProgressWatchdog:
    def __init__(self, cfg=None):
        self.cfg = cfg or WatchdogConfig(); self.stale = 0; self._last = None
        self.fallback = False
    def observe(self, progress_key, candidate_cost=None, conservative_cost=None):
        if self.fallback: return True
        if self._last == progress_key: self.stale += 1
        else: self.stale = 0
        self._last = progress_key
        if candidate_cost is not None and conservative_cost is not None:
            if candidate_cost - conservative_cost > self.cfg.max_detour_s:
                self.fallback = True
        if self.stale >= self.cfg.patience: self.fallback = True
        return self.fallback
    def reset(self):
        self.stale = 0; self._last = None; self.fallback = False

class CandidatePPOPlanner:
    """Safe deployment adapter. PPO may reorder only the supplied safe candidates.

    If torch/model/checkpoint is unavailable, it permanently delegates to the
    deterministic planner for the episode; it never invents coordinates or EXIT.
    """
    def __init__(self, base_planner, policy=None, watchdog=None):
        self.base = base_planner; self.policy = policy
        self.watchdog = watchdog or ProgressWatchdog()
        self.decisions = []
        self.last_error = None
    def reset_episode(self): self.watchdog.reset()
    def record_progress(self, key, candidate_cost=None, conservative_cost=None):
        return self.watchdog.observe(key, candidate_cost, conservative_cost)
    def plan(self, belief, certificate, state, candidates):
        result = self.base.plan(belief, certificate, state, candidates)
        if self.watchdog.fallback or self.policy is None or not result.evaluations:
            return result
        try:
            idx = int(self.policy(result.evaluations, belief, state))
            if idx < 0 or idx >= len(result.evaluations): return result
            self.decisions.append({"n_candidates": len(result.evaluations),
                                   "chosen": idx,
                                   "q_values": [float(e.q_value) for e in result.evaluations]})
            e = result.evaluations[idx]
            return type(result)(e.candidate, e.q_value, result.evaluations,
                                result.horizon, result.outcome_mode)
        except Exception as exc:
            self.last_error = f'{type(exc).__name__}: {exc}'
            self.watchdog.fallback = True
            return result

class TorchCandidatePolicy:
    """Deployment policy for a trained CandidateActorCritic.

    ``feature_builder`` receives planner evaluations, belief and state and must
    return ``[n_candidates, feature_dim]``.  The returned value is always an
    index into that evaluation list.
    """
    def __init__(self, model, feature_builder, *, temperature=1.0, stochastic=False):
        self.model, self.feature_builder = model, feature_builder
        self.temperature, self.stochastic = float(temperature), bool(stochastic)
        self.records = []
    def __call__(self, evaluations, belief, state):
        import torch
        x = torch.as_tensor(self.feature_builder(evaluations, belief, state), dtype=torch.float32)
        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        if x.ndim == 2: x = x[None, ...]
        self.model.eval()
        with torch.no_grad():
            logits, value = self.model(x)
        logp = torch.log_softmax(logits[0], -1)
        probs = torch.softmax(logits[0] / max(self.temperature, 1e-6), -1)
        action = int(torch.multinomial(probs, 1).item()) if self.stochastic else int(torch.argmax(logits[0]).item())
        self.records.append({'observation': x[0].tolist(), 'action': action,
                             'log_prob': float(logp[action]), 'value': float(value[0])})
        return action
    def reset(self): self.records = []

def legacy_evaluation_features(evaluations, belief, state):
    """Adapt existing math evaluations to the frozen 10+20x10 PPO schema."""
    from .features import feature_matrix
    rows = feature_matrix(evaluations, belief, state)
    out=[]; n=getattr(belief, 'n_channels', 20)
    for row, ev in zip(rows, evaluations):
        c=ev.candidate; base=list(row[:10]); interactions=[]
        active=set(getattr(c, 'scan_channels', ()) or ())
        for ch in range(1,n+1):
            interactions.extend([float(ch in active), float(ch in active and getattr(belief[ch], 'status', None).value == 'DETECTED'),
                                 float(c.refinement_gain if ch in active else 0), float(getattr(belief[ch], 'mec_radius', 0)),
                                 float(c.action_type.value == 'CLEAR' and ch == getattr(c,'clear_channel',None)),
                                 float(c.certificate_gain if ch in active else 0), 0., 0., 0., 0.])
        out.append(base+interactions)
    return out

def load_candidate_policy(checkpoint, *, hidden=128, n_channels=20):
    """Load a Candidate-PPO checkpoint; raises clearly on unavailable torch."""
    import torch
    from .candidate_ppo import CandidateActorCritic
    dim = 10 + n_channels * 10
    model = CandidateActorCritic(dim, hidden=hidden)
    blob = torch.load(checkpoint, map_location='cpu')
    if blob.get('input_dim', dim) != dim or blob.get('problem', 4) != 4:
        raise ValueError('checkpoint schema/problem mismatch')
    model.load_state_dict(blob.get('state_dict', blob)); model.eval()
    return TorchCandidatePolicy(model, legacy_evaluation_features), model
