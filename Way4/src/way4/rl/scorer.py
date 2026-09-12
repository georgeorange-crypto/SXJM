"""Residual MLP scorer ``ΔQ_θ`` (DESIGN.md §12).

A plain feed-forward net over the flat per-candidate feature vector
(``features.FEATURE_DIM``). It outputs **one scalar per candidate** — a residual
added to the planner's analytical ``Q_math`` — and *nothing else*:

    Q(B, a) = Q_math(B, a) + ΔQ_θ(B, a)

Hard constraints from the design, enforced structurally here:

  * **禁止8 (no arbitrary coordinates).** The scorer's output is a scalar per
    *existing* candidate; it can only reorder the set the math generator already
    produced. It never emits a waypoint. The public API takes candidates and
    returns residuals of the same length — there is no coordinate output path.
  * **禁止9 (no Dreamer/Mamba/GNN/LSTM).** This is an MLP — ``Linear → ReLU``
    stacks — over a vector the belief already summarises. No recurrence, no graph.
  * **Safe default = no-op.** A freshly-constructed scorer (``zero_init=True`` by
    default) outputs exactly ``0`` for every candidate, so wrapping the planner in
    it reproduces the pure-math ranking bit-for-bit. Training must *earn* any
    deviation. If torch is missing, construction still succeeds and the scorer
    reports ``available == False`` so the planner wrapper falls back to math (§11:
    "RL 不可用 → 退回 Way3 式保证完成策略").

The residual output is optionally squashed to ``±residual_clip`` seconds
(``tanh``-scaled) so a wild untrained/adversarial activation can perturb the
ranking but can never dominate a large ``Q_math`` gap — the math stays
authoritative (§12: 数学负责正确/安全/几何/覆盖/清除保证).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .features import FEATURE_DIM, FEATURE_SCHEMA_HASH, FEATURE_SCHEMA_VERSION

try:  # torch is optional; absence must not break import or the math pipeline.
    import torch
    from torch import nn

    _TORCH_OK = True
except Exception:  # pragma: no cover - exercised only in torch-free envs
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_OK = False


def _build_mlp(in_dim: int, hidden: Sequence[int], zero_init: bool):
    """A ``Linear→ReLU`` stack ending in a scalar head. When ``zero_init`` the
    final layer is zeroed so the whole net outputs 0 (the safe no-op default)."""
    layers: List["nn.Module"] = []
    d = in_dim
    for h in hidden:
        layers.append(nn.Linear(d, h))
        layers.append(nn.ReLU())
        d = h
    head = nn.Linear(d, 1)
    if zero_init:
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)


class ResidualScorer:
    """MLP residual ``ΔQ_θ`` over ``FEATURE_DIM`` features → one scalar/candidate.

    Parameters
    ----------
    hidden:
        Hidden layer widths (default ``(64, 64)`` — small; the input is already a
        compressed belief summary).
    zero_init:
        If True (default) the head is zeroed → the scorer is an exact no-op until
        trained. Set False only for training from a random init.
    residual_clip:
        If > 0, the raw output is squashed to ``±residual_clip`` seconds via
        ``residual_clip * tanh(raw / residual_clip)`` so it stays a *bounded*
        correction to ``Q_math``. ``0`` disables the squash (raw linear output).
    """

    def __init__(
        self,
        hidden: Sequence[int] = (64, 64),
        *,
        zero_init: bool = True,
        residual_clip: float = 60.0,
        in_dim: int = FEATURE_DIM,
    ) -> None:
        self.in_dim = int(in_dim)
        self.hidden = tuple(int(h) for h in hidden)
        self.residual_clip = float(residual_clip)
        self._zero_init = bool(zero_init)
        self.net = None
        if _TORCH_OK:
            self.net = _build_mlp(self.in_dim, self.hidden, zero_init)
            # Candidate index is a valid policy coordinate: rows in one
            # candidate set are distinct actions even when their engineered
            # features coincide. This learned tie-breaker fixes the degenerate
            # contextual-bandit case without emitting coordinates.
            self.action_bias = nn.Parameter(torch.zeros(128))
            self.net.eval()

    # -- capability --------------------------------------------------------

    @property
    def available(self) -> bool:
        """True iff a torch net is live. When False, callers must treat the
        residual as 0 (the RLResidualPlanner does this automatically)."""
        return self.net is not None

    # -- inference ---------------------------------------------------------

    def _forward(self, x):
        """Grad-respecting forward: raw net output → clipped residual (seconds).
        Shared by ``residuals`` (no-grad inference) and the trainer (with grad)."""
        raw = self.net(x).squeeze(-1)
        raw = raw + self.action_bias[: raw.shape[0]]
        if self.residual_clip > 0:
            raw = self.residual_clip * torch.tanh(raw / self.residual_clip)
        return raw

    def residuals(self, features: Sequence[Sequence[float]]) -> List[float]:
        """Score a batch of feature rows → one residual (seconds) per row.

        Torch-free / unavailable → all zeros (no-op). A zero-initialised net also
        returns all zeros, so an untrained scorer never changes the ranking."""
        n = len(features)
        if n == 0:
            return []
        if not self.available:
            return [0.0] * n
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32)
            return [float(v) for v in self._forward(x).tolist()]

    # -- persistence -------------------------------------------------------

    def state_dict(self):
        if not self.available: return {}
        sd = dict(self.net.state_dict()); sd["__action_bias__"] = self.action_bias.detach().clone(); return sd

    def load_state_dict(self, sd) -> None:
        if self.available:
            sd = dict(sd)
            bias = sd.pop("__action_bias__", None)
            self.net.load_state_dict(sd)
            if bias is not None: self.action_bias.data.copy_(bias)
            self.net.eval()

    def save(self, path: str) -> None:
        if not self.available:
            raise RuntimeError("torch unavailable; cannot save ResidualScorer")
        torch.save(
            {
                "in_dim": self.in_dim,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "feature_schema_hash": FEATURE_SCHEMA_HASH,
                "hidden": list(self.hidden),
                "residual_clip": self.residual_clip,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "ResidualScorer":
        if not _TORCH_OK:
            raise RuntimeError("torch unavailable; cannot load ResidualScorer")
        blob = torch.load(path, map_location="cpu")
        if blob.get("feature_schema_version") != FEATURE_SCHEMA_VERSION or blob.get("feature_schema_hash") != FEATURE_SCHEMA_HASH:
            raise ValueError("incompatible ResidualScorer feature schema; retraining is required")
        scorer = cls(
            hidden=blob.get("hidden", (64, 64)),
            zero_init=False,
            residual_clip=blob.get("residual_clip", 60.0),
            in_dim=blob.get("in_dim", FEATURE_DIM),
        )
        scorer.load_state_dict(blob["state_dict"])
        return scorer
