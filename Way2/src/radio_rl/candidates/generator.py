"""Candidate generator — the only place actions are *created*.

Architecture principle C: the learnable agent never emits coordinates; it selects
an index into the :class:`CandidateSet` this module produces. Candidates come
from five sources (clear a localized source, triangulate a detected one, follow
up a fresh bearing, search for unknown sources, or a safe fallback), are
annotated by the world model, then pruned to ``k_max``. An EXIT fallback is
always present so the set is never empty and the agent can always stop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.constants import CONSTANTS
from ..core.datatypes import (
    Action,
    ActionType,
    CandidateAction,
    CandidateSource,
)
from ..geometry.belief import BeliefState
from ..world_model.analytical import AnalyticalWorldModel
from ..world_model.base import WorldModel

# ranking priority for pruning (lower = kept first)
_SOURCE_PRIORITY = {
    int(CandidateSource.CLEAR): 0,
    int(CandidateSource.FOLLOWUP): 1,
    int(CandidateSource.LOCALIZE): 1,
    int(CandidateSource.SEARCH): 2,
    int(CandidateSource.SAFETY): 3,
}


@dataclass
class CandidateSet:
    """A padded, fixed-capacity set of candidates offered to the agent."""

    candidates: list[CandidateAction] = field(default_factory=list)
    k_max: int = 32
    n_real: int = 0

    def __post_init__(self) -> None:
        if self.n_real == 0:
            self.n_real = len(self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)

    @property
    def mask(self) -> list[bool]:
        """True for real candidates, False for padding."""
        return [i < self.n_real for i in range(len(self.candidates))]

    def real(self) -> list[CandidateAction]:
        return self.candidates[: self.n_real]

    def pad_to(self, k: Optional[int] = None) -> "CandidateSet":
        """Pad with invalid EXIT no-ops up to ``k`` (default ``k_max``)."""
        k = self.k_max if k is None else k
        if not self.candidates:
            return self
        filler = self.candidates[self.n_real - 1]
        while len(self.candidates) < k:
            pad = CandidateAction(
                candidate_id=len(self.candidates),
                action_type=int(ActionType.EXIT),
                channel=filler.channel,
                target_x=filler.target_x,
                target_y=filler.target_y,
                source=int(CandidateSource.SAFETY),
            )
            self.candidates.append(pad)
        return self


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    try:
        v = cfg.get(key) if hasattr(cfg, "get") else cfg[key]
    except Exception:
        return default
    return default if v is None else v


class CandidateGenerator:
    """Builds a :class:`CandidateSet` from the mathematical belief."""

    def __init__(self, cfg: Any = None, world_model: Optional[WorldModel] = None) -> None:
        search = _cfg_get(cfg, "search", {}) or {}
        loc = _cfg_get(cfg, "localization", {}) or {}
        clear = _cfg_get(cfg, "clear", {}) or {}
        safety = _cfg_get(cfg, "safety", {}) or {}

        self.k_max = int(_cfg_get(cfg, "k_max", 32))
        self.ring_radii = [float(r) for r in _cfg_get(search, "ring_radii",
                                                      [600.0, 1200.0, 1650.0])]
        self.n_angles = int(_cfg_get(search, "n_angles", 8))
        self.n_offsets = int(_cfg_get(loc, "n_offsets", 6))
        self.baseline_m = float(_cfg_get(loc, "baseline_m", 400.0))
        self.step_toward_m = float(_cfg_get(loc, "step_toward_m", 500.0))
        self.max_localize_scans = int(_cfg_get(loc, "max_scans", 30))
        self.clear_enabled = bool(_cfg_get(clear, "enabled", True))
        self.clear_margin_m = float(_cfg_get(clear, "margin_m", 1.0))
        self.safety_enabled = bool(_cfg_get(safety, "enabled", True))

        self.arena_radius = CONSTANTS.region_radius
        self.range_max = CONSTANTS.receive_radius_max
        self.wm = world_model or AnalyticalWorldModel()

    # -- helpers -----------------------------------------------------------
    def _clip_to_arena(self, x: float, y: float) -> tuple[float, float]:
        r = math.hypot(x, y)
        if r <= self.arena_radius:
            return x, y
        s = self.arena_radius / r
        return x * s, y * s

    # -- generation --------------------------------------------------------
    def generate(self, belief: BeliefState) -> CandidateSet:
        raw: list[CandidateAction] = []

        def add(action_type: ActionType, channel: int, x: float, y: float,
                source: CandidateSource) -> None:
            x, y = self._clip_to_arena(x, y)
            cand = CandidateAction(
                candidate_id=len(raw),
                action_type=int(action_type),
                channel=int(channel),
                target_x=float(x),
                target_y=float(y),
                source=int(source),
            )
            self.wm.annotate(belief, cand)
            raw.append(cand)

        self._add_clears(belief, add)
        self._add_localizers(belief, add)
        self._add_search(belief, add)
        if self.safety_enabled or not raw:
            add(ActionType.EXIT, belief.current_channel,
                belief.pose_x, belief.pose_y, CandidateSource.SAFETY)

        pruned = self._prune(raw)
        for i, c in enumerate(pruned):
            c.candidate_id = i
        return CandidateSet(candidates=pruned, k_max=self.k_max, n_real=len(pruned))

    def _add_clears(self, belief: BeliefState, add) -> None:
        if not self.clear_enabled:
            return
        # Clear any channel whose feasible region is small enough that a clear at
        # its centre is guaranteed to hit (covers 'localized' plus the 18-19 m band).
        for ch in belief.detected_channels():
            cb = belief.belief(ch)
            if cb.is_cleared or cb.estimate is None:
                continue
            if belief.clearable(ch, margin=self.clear_margin_m):
                add(ActionType.CLEAR, ch, cb.estimate[0], cb.estimate[1],
                    CandidateSource.CLEAR)

    def _add_localizers(self, belief: BeliefState, add) -> None:
        for ch in belief.detected_channels():
            cb = belief.belief(ch)
            if cb.is_localized or belief.clearable(ch, margin=self.clear_margin_m):
                continue
            # Give up triangulating a channel that will not shrink further.
            if belief.scans_on(ch) >= self.max_localize_scans:
                continue
            cx, cy = cb.region.centroid()
            # a spread of vantage points around the region for a crossing bearing
            for k in range(self.n_offsets):
                ang = 2.0 * math.pi * k / max(1, self.n_offsets)
                px = cx + self.baseline_m * math.cos(ang)
                py = cy + self.baseline_m * math.sin(ang)
                add(ActionType.SCAN, ch, px, py, CandidateSource.LOCALIZE)
            # follow-ups: step toward the region, and probe the centroid itself
            dx, dy = cx - belief.pose_x, cy - belief.pose_y
            d = math.hypot(dx, dy)
            if d > 1e-6:
                step = min(self.step_toward_m, d)
                add(ActionType.SCAN, ch, belief.pose_x + dx / d * step,
                    belief.pose_y + dy / d * step, CandidateSource.FOLLOWUP)
            add(ActionType.SCAN, ch, cx, cy, CandidateSource.FOLLOWUP)

    def _add_search(self, belief: BeliefState, add) -> None:
        undetected = belief.undetected_channels()
        if not undetected:
            return
        idx = 0
        # cheap in-place probes first (no travel, just switch + detect)
        for ch in undetected[: min(len(undetected), self.n_angles)]:
            add(ActionType.SCAN, ch, belief.pose_x, belief.pose_y,
                CandidateSource.SEARCH)
        # ring sweep, cycling channels so different rings probe different channels
        for ri, r in enumerate(self.ring_radii):
            for k in range(self.n_angles):
                ang = 2.0 * math.pi * k / self.n_angles + (math.pi / self.n_angles) * ri
                ch = undetected[idx % len(undetected)]
                idx += 1
                x, y = r * math.cos(ang), r * math.sin(ang)
                if belief.excluded(x, y, channel=ch):
                    continue
                add(ActionType.SCAN, ch, x, y, CandidateSource.SEARCH)

    def _prune(self, raw: list[CandidateAction]) -> list[CandidateAction]:
        if len(raw) <= self.k_max:
            return raw

        def key(c: CandidateAction):
            pr = _SOURCE_PRIORITY.get(c.source, 3)
            # within a priority band, higher info gain first, cheaper first
            return (pr, -c.heuristic_information_gain, c.expected_time)

        ordered = sorted(raw, key=key)
        kept = ordered[: self.k_max - 1]
        # guarantee an EXIT fallback survives pruning
        exit_c = next((c for c in raw if c.action_type == int(ActionType.EXIT)), None)
        if exit_c is not None and exit_c not in kept:
            kept.append(exit_c)
        else:
            kept = ordered[: self.k_max]
        return kept
