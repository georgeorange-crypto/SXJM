"""CandidateGenerator — propose the macro actions the planner ranks (DESIGN.md §7–8).

Turns the current (belief, certificate, robot state) into a set of ``MacroCandidate``
objects, each carrying a decomposed gain vector and a cost-model time estimate. It
proposes; it never marks absent (禁止6) or bypasses the clear guard (禁止7) — those
live in the certificate/safety layers. The families (§7):

  * **CLEAR**   — one per clearable channel, at its MEC centre (§3.3 blind clear).
  * **REFINE**  — one per DETECTED-but-not-clearable channel, at its minimax NBV
    viewpoint (§8), batch-scanning that channel + free-rider channels.
  * **EXPLORE** — multipurpose coverage/discovery waypoints (fallback anchors + a
    coarse ring), scored by joint UNKNOWN coverage gain; the scheduler picks the
    batch. Under VERIFICATION mode these become **VERIFY** waypoints scanning only
    still-UNKNOWN, un-certified channels (§6.7).
  * **COMPLETION** — a last-resort VERIFY fallback (§11 Safe fallback / §6.5
    Invariant C). Fires only when CLEAR/REFINE/EXPLORE are all empty yet a channel
    is unresolved: active scanning has saturated the 40 m heuristic map (no waypoint
    gains new coverage) but the hard verifier still can't certify a sub-grid seam and
    the legacy backbone isn't complete. Proposes the nearest unvisited Way3 anchors —
    the guaranteed-completion template — *not* gated on heuristic gain, since their
    progress is backbone completion, a different §6.5 source than disc-cover.
  * **EXIT**    — proposed once every channel is resolved (the EXIT guard in the
    safety layer has the final say).

Robot scan points are NOT clamped to the arena (源/机器人域分离, §7).
"""

from __future__ import annotations

from math import cos, hypot, radians, sin
from typing import List, Optional, Sequence, Tuple

from ..belief import ChannelStatus
from ..channels import ChannelScheduler, SchedulerMode
from ..core import AnalyticalCostModel, MacroActionType, MacroCandidate, RobotState
from ..sensing import MinimaxNBV

Point = Tuple[float, float]


class CandidateGenerator:
    def __init__(
        self,
        nbv: Optional[MinimaxNBV] = None,
        scheduler: Optional[ChannelScheduler] = None,
        cost_model: Optional[AnalyticalCostModel] = None,
        explore_ring_radii: Sequence[float] = (600.0, 1200.0),
        explore_ring_angles: int = 12,
        max_explore: int = 6,
        max_refine: int = 16,
        include_origin_explore: bool = True,
        min_coverage_gain: float = 1e-9,
    ) -> None:
        self.nbv = nbv or MinimaxNBV()
        self.scheduler = scheduler or ChannelScheduler()
        self.cost = cost_model or AnalyticalCostModel()
        self.explore_ring_radii = tuple(float(r) for r in explore_ring_radii)
        self.explore_ring_angles = int(explore_ring_angles)
        self.max_explore = int(max_explore)
        self.max_refine = int(max_refine)
        self.include_origin_explore = bool(include_origin_explore)
        self.min_coverage_gain = float(min_coverage_gain)

    # -- public -------------------------------------------------------------

    def generate(
        self,
        belief,
        certificate,
        state: RobotState,
        scan_mode: SchedulerMode = SchedulerMode.EARLY,
    ) -> List[MacroCandidate]:
        cands: List[MacroCandidate] = []
        cands.extend(self._clear_candidates(belief, state))
        cands.extend(self._refine_candidates(belief, certificate, state, scan_mode))
        cands.extend(self._explore_candidates(belief, certificate, state, scan_mode))
        if not cands and not belief.all_resolved():
            # §11 Safe fallback ("无候选 → 退回 Way3 式保证完成策略"): the active
            # families are exhausted but a channel is still unresolved — walk the
            # guaranteed-completion backbone (Invariant C) so the loop can never abort
            # no_candidate while work remains.
            cands.extend(self._completion_candidates(belief, certificate, state))
        if belief.all_resolved():
            cands.append(self._exit_candidate(state))
        return cands

    # -- CLEAR --------------------------------------------------------------

    def _clear_candidates(self, belief, state: RobotState) -> List[MacroCandidate]:
        out: List[MacroCandidate] = []
        for c in belief.clearable_channels():
            tgt = belief[c].clear_target
            if tgt is None:
                continue
            m = MacroCandidate(MacroActionType.CLEAR, tgt, clear_channel=c)
            # a clearable channel's MEC guarantees a hit at the centre (§3.3)
            m.expected_time = self.cost.clear_time_s(state, tgt, hit=True)
            m.refinement_gain = 0.0
            m.meta["clear_channel"] = c
            m.meta["mec_radius"] = belief[c].mec_radius
            out.append(m)
        return out

    # -- REFINE -------------------------------------------------------------

    def _refine_candidates(
        self, belief, certificate, state: RobotState, scan_mode: SchedulerMode
    ) -> List[MacroCandidate]:
        detected = [
            c for c in range(1, belief.n_channels + 1)
            if belief[c].status == ChannelStatus.DETECTED
        ]
        # Refine the largest (least-localised) regions first, but propose a candidate
        # for EVERY detected channel (≤16 by the cardinality bound, so max_refine=16
        # never truncates a real set). Capping at a handful starved the almost-clearable
        # channels of any REFINE candidate whenever many sources were detected at once,
        # so they could never reach LOCALIZED and never clear (the seed-1017 clr=0 stall).
        detected.sort(key=lambda c: -belief[c].mec_radius)
        out: List[MacroCandidate] = []
        for c in detected[: self.max_refine]:
            res = self.nbv.choose(belief[c], state.pos)
            if res is None:
                continue
            q = res.point
            plan = self.scheduler.select(
                q, belief, certificate, state.channel, mode=scan_mode, must_include=[c]
            )
            if plan.is_empty:
                plan_channels = (c,)
            else:
                plan_channels = tuple(plan.channels)
            m = MacroCandidate(MacroActionType.REFINE, q, scan_channels=plan_channels)
            m.refinement_gain = res.expected_shrink
            m.certificate_gain = self._unknown_coverage_gain(belief, certificate, plan_channels, q)
            m.exploration_gain = plan.value
            m.expected_time = self.cost.batch_scan_time_s(state, q, plan_channels)
            m.meta["refine_channel"] = c
            m.meta["worst_case_diameter"] = res.worst_case_diameter
            out.append(m)
        return out

    # -- EXPLORE / VERIFY ---------------------------------------------------

    def _explore_candidates(
        self, belief, certificate, state: RobotState, scan_mode: SchedulerMode
    ) -> List[MacroCandidate]:
        unknown = belief.unknown_channels()
        if not unknown:
            return []
        pool = self._explore_pool(certificate, state)
        # score each waypoint by joint UNKNOWN coverage gain (multipurpose, §8)
        scored: List[Tuple[float, Point]] = []
        for q in pool:
            g = self._unknown_coverage_gain(belief, certificate, unknown, q)
            scored.append((g, q))
        scored.sort(key=lambda t: -t[0])

        verify = scan_mode == SchedulerMode.VERIFICATION
        action = MacroActionType.VERIFY if verify else MacroActionType.EXPLORE
        out: List[MacroCandidate] = []
        for g, q in scored:
            if len(out) >= self.max_explore:
                break
            plan = self.scheduler.select(q, belief, certificate, state.channel, mode=scan_mode)
            if plan.is_empty:
                continue
            channels = tuple(plan.channels)
            cov_gain = self._unknown_coverage_gain(belief, certificate, channels, q)
            # An explore's whole job is UNKNOWN coverage (§8). A waypoint whose batch
            # gains no new coverage is a no-op — at an already-scanned spot the only
            # eligible channels are DETECTED re-scans (covers nothing new, doesn't
            # refine). Left in, the cheapest such no-op (stay put, C≈5s) beats every
            # progress action that must first move, and the loop livelocks (the seed
            # 1003 stall). Dropping zero-gain explores can never strand us: while any
            # hole remains it lies within detection range of a backbone anchor
            # (Invariant C), so a positive-gain explore always exists; once none does,
            # coverage is complete and the channels certify absent instead (§6.5).
            if cov_gain <= self.min_coverage_gain:
                continue
            m = MacroCandidate(action, q, scan_channels=channels)
            m.certificate_gain = cov_gain
            m.exploration_gain = plan.value
            m.expected_time = self.cost.batch_scan_time_s(state, q, channels)
            out.append(m)
        return out

    def _explore_pool(self, certificate, state: RobotState) -> List[Point]:
        pts: List[Point] = []
        if certificate is not None:
            pts.extend((float(a[0]), float(a[1])) for a in certificate.anchors)
        if self.include_origin_explore:
            pts.append((0.0, 0.0))
        for r in self.explore_ring_radii:
            for k in range(self.explore_ring_angles):
                ang = radians(360.0 * k / self.explore_ring_angles)
                pts.append((r * cos(ang), r * sin(ang)))
        # dedup
        out: List[Point] = []
        seen = set()
        for p in pts:
            key = (round(p[0], 3), round(p[1], 3))
            if key not in seen:
                seen.add(key)
                out.append(p)
        return out

    # -- COMPLETION (Way3 backbone fallback, §11 / Invariant C) -------------

    def _completion_candidates(
        self, belief, certificate, state: RobotState
    ) -> List[MacroCandidate]:
        """Backbone-completion fallback for the ``no_candidate`` corner (§11, §6.5).

        Reached only from ``generate`` when CLEAR/REFINE/EXPLORE all came back empty
        while a channel is still unresolved. That happens when active scanning has
        driven the 40 m heuristic coverage map to 100 % (every waypoint gains zero
        new coverage) yet the hard disc-cover verifier still can't certify a channel
        (a sub-grid seam its scan discs don't quite tile) and the legacy backbone is
        incomplete — the seed-1013/1015/1027/1048 abort.

        The Way3 omni-scan anchors are the guaranteed-completion template: their discs
        cover D_1800 with margin (Invariant C), so scanning NO_SIGNAL at the remaining
        unvisited anchors drives ``legacy_backbone_complete`` → ABSENT_CERTIFIED for a
        truly-absent channel (or a positive return promotes a still-hidden source to
        DETECTED). Candidates are the nearest unvisited anchors, batch-scanning the
        pending channels; they are deliberately NOT gated on heuristic coverage gain
        (their progress is a *different* §6.5 source than disc-cover). This proposes
        viewpoints only — it marks nothing and certifies nothing (禁止6)."""
        if certificate is None:
            return []
        pending = [
            c for c in belief.unknown_channels()
            if not certificate.is_absent_certified(c, force=False)
        ]
        if not pending:
            return []
        anchors = certificate.anchors
        # anchors still unvisited by at least one pending channel, nearest-first
        scored: List[Tuple[float, int, Point, List[int]]] = []
        for j, a in enumerate(anchors):
            ap = (float(a[0]), float(a[1]))
            chans = [
                c for c in pending
                if j not in certificate.certs[c].visited_anchor_idx
            ]
            if chans:
                scored.append((hypot(state.pos[0] - ap[0], state.pos[1] - ap[1]), j, ap, chans))
        if not scored:
            return []
        scored.sort(key=lambda t: t[0])
        out: List[MacroCandidate] = []
        for _d, j, ap, chans in scored[: self.max_explore]:
            plan = self.scheduler.select(
                ap, belief, certificate, state.channel,
                mode=SchedulerMode.VERIFICATION, must_include=chans,
            )
            channels = tuple(plan.channels) if not plan.is_empty else tuple(chans)
            m = MacroCandidate(MacroActionType.VERIFY, ap, scan_channels=channels)
            # backbone-completion progress (metadata only — never enters Q, §10); a
            # positive value keeps it out of any zero-gain no-op filter.
            m.certificate_gain = float(len(chans))
            m.exploration_gain = plan.value
            m.expected_time = self.cost.batch_scan_time_s(state, ap, channels)
            m.meta["completion_anchor"] = j
            out.append(m)
        return out

    # -- EXIT ---------------------------------------------------------------

    def _exit_candidate(self, state: RobotState) -> MacroCandidate:
        m = MacroCandidate(MacroActionType.EXIT, state.pos)
        m.expected_time = 0.0
        return m

    # -- helpers ------------------------------------------------------------

    def _unknown_coverage_gain(self, belief, certificate, channels, q: Point) -> float:
        if certificate is None:
            return 0.0
        unknown = [c for c in channels if belief[c].status == ChannelStatus.UNKNOWN]
        if not unknown:
            return 0.0
        return certificate.batch_coverage_gain(unknown, q)
