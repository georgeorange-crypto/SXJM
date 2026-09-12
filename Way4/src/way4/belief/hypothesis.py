"""Q4 coarse **hypothesis layer** ``H_c`` — the SOFT half of the two-layer belief
(DESIGN.md §3.1, §3.2). Milestone M8.

The HARD layer (``belief.channel.ChannelBelief``) owns the convex feasible set
``F_c`` and is the *only* thing that may drive clear / certificate / absent. This
module is its opposite number: a coarse, discrete, directional hypothesis grid
whose sole job is to give the planner a cheap, *sound-elimination* picture of
"where could an (as-yet-unlocalised) source of channel c still be, and which way
could it be pointing" — so EXPLORE/REFINE waypoints can be scored by how much
*hypothesis mass* a scan there would kill (the §3.2 "directional-visibility
gain").

It is deliberately incapable of certifying anything (Invariant B, 禁止6): it
exposes alive **counts**, **fractions** and elimination **gains** only — never an
``is_complete`` / ``is_absent`` / ``mark_*``. Absence is the hard layer's job.

Soundness contract (the M8 §14 gate)
------------------------------------
A hypothesis ``h = (cell, heading_bin, type)`` is eliminated by an observation
*only when every source configuration it represents is provably inconsistent with
that observation*. The elimination therefore never removes the hypothesis the
true source actually occupies. Concretely, on NO_SIGNAL @ S we eliminate ``h``
only if the source **would definitely have been detected** — using the guaranteed
lower bound ``R_lb = 1000`` (never the 1500 upper bound; 禁止5), the *farthest*
point of the cell for range, and the *whole* heading bin for the arc. Because the
detection predicate is convex in the source position and monotone over the bin,
the worst case is attained at the cell's corners / bin edges, so the corner test
is exact.

numpy is used purely for speed (§6.8), mirroring ``certificate.coverage_gain``;
the reasoning stays stdlib-simple.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
from math import log

import numpy as np

from way4.core.observation import Observation, ObservationKind

Point = Tuple[float, float]

# --- heading discretisation (DESIGN.md §3.1: 36 bins × 10°) -------------------

HEADING_BINS: int = 36
BIN_WIDTH_DEG: float = 360.0 / HEADING_BINS  # 10.0

_bin_lo_deg = np.arange(HEADING_BINS, dtype=float) * BIN_WIDTH_DEG      # [0,10,..,350]
_bin_hi_deg = _bin_lo_deg + BIN_WIDTH_DEG
_bin_center_deg = _bin_lo_deg + 0.5 * BIN_WIDTH_DEG
# Unit vectors of each bin's *edge* headings. Directional detection of S from a
# source at p with heading φ is the half-plane (S-p)·u_φ ≥ 0 (arc = φ ± 90°); for
# the whole bin the binding constraints are the two edge headings.
_U_LO = np.column_stack([np.cos(np.radians(_bin_lo_deg)), np.sin(np.radians(_bin_lo_deg))])  # (B,2)
_U_HI = np.column_stack([np.cos(np.radians(_bin_hi_deg)), np.sin(np.radians(_bin_hi_deg))])  # (B,2)


class HypothesisGrid:
    """Shared, immutable spatial discretisation for ``H_c`` (DESIGN.md §3.1).

    A uniform ``spacing`` (default 60 m) grid of square cells. A cell is kept iff
    its *box* overlaps the arena — i.e. its centre is within ``arena_radius +
    half-diagonal`` — so that **every** legal in-arena source lies inside some
    kept cell's box. That total-coverage property is what makes the corner-based
    elimination tests sound right up to the arena rim (a centre-in-arena filter
    would drop rim cells and break the guarantee there).
    """

    def __init__(self, arena_radius: float = 1800.0, spacing: float = 60.0) -> None:
        self.arena_radius = float(arena_radius)
        self.spacing = float(spacing)
        self.half = 0.5 * self.spacing                      # cell half-width
        self.half_diag = self.half * np.sqrt(2.0)           # cell half-diagonal

        # Generous square lattice, then keep boxes overlapping the arena disc.
        keep_r = self.arena_radius + self.half_diag
        n = int(np.ceil(keep_r / self.spacing)) + 1
        coords = np.arange(-n, n + 1, dtype=float) * self.spacing
        gx, gy = np.meshgrid(coords, coords)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        inside = (pts[:, 0] ** 2 + pts[:, 1] ** 2) <= keep_r * keep_r
        self.centers: np.ndarray = pts[inside]              # (N, 2)
        self.n: int = int(self.centers.shape[0])

        # Four corners of every cell box: (N, 4, 2). Order is arbitrary — every
        # test below reduces over the corner axis (min / max), so order is moot.
        offs = np.array(
            [[-self.half, -self.half], [-self.half, self.half],
             [self.half, -self.half], [self.half, self.half]],
            dtype=float,
        )                                                   # (4, 2)
        self.corners: np.ndarray = self.centers[:, None, :] + offs[None, :, :]  # (N,4,2)

    def cell_index(self, p: Point) -> int:
        """Index of the cell whose box contains ``p`` (nearest centre), or ``-1``
        if ``p`` falls outside every kept box. Used by tests to locate the true
        source's "conservative cell"."""
        if self.n == 0:
            return -1
        d2 = (self.centers[:, 0] - p[0]) ** 2 + (self.centers[:, 1] - p[1]) ** 2
        i = int(np.argmin(d2))
        cx, cy = self.centers[i]
        if abs(p[0] - cx) <= self.half + 1e-9 and abs(p[1] - cy) <= self.half + 1e-9:
            return i
        return -1

    def _min_box_dist2(self, S: Point) -> np.ndarray:
        """Squared distance from ``S`` to the *nearest* point of each cell box
        (0 when ``S`` is inside the box). Convex ⇒ clip ``S`` onto the box."""
        cx = self.centers[:, 0]
        cy = self.centers[:, 1]
        nx = np.clip(S[0], cx - self.half, cx + self.half)
        ny = np.clip(S[1], cy - self.half, cy + self.half)
        return (nx - S[0]) ** 2 + (ny - S[1]) ** 2

    def _max_box_dist2(self, S: Point) -> np.ndarray:
        """Squared distance from ``S`` to the *farthest* point of each cell box.
        Convex, so the max is at a corner."""
        diff = self.corners - np.asarray(S, dtype=float)[None, None, :]   # (N,4,2)
        return np.max(diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2, axis=1)    # (N,)


class ChannelHypotheses:
    """The alive hypothesis set for one channel over a shared :class:`HypothesisGrid`.

    Two boolean masks: ``omni_alive`` (N,) — "an omni source could sit in cell i"
    — and ``dir_alive`` (N, 36) — "a directional source could sit in cell i with
    heading in bin j". Everything starts alive; observations only ever clear bits
    (monotone), and never for the cell/bin the true source occupies.
    """

    def __init__(
        self,
        grid: HypothesisGrid,
        channel: int = 0,
        detect_lower_bound: float = 1000.0,   # R_eff guaranteed lower bound (禁止5)
        range_radius: float = 1500.0,         # R_eff upper bound (detected ⇒ within this)
        near_radius: float = 5.0,             # 'near' ⇒ source within 5 m (§1)
        wedge_half_deg: float = 1.0,          # bearing measurement half-window (§1)
        eps_deg: float = 1e-6,
    ) -> None:
        self.grid = grid
        self.channel = int(channel)
        self.R_lb = float(detect_lower_bound)
        self.R_up = float(range_radius)
        self.near_radius = float(near_radius)
        self.wedge_half_deg = float(wedge_half_deg)
        self.eps_deg = float(eps_deg)
        self._r_lb2 = self.R_lb * self.R_lb

        self.omni_alive: np.ndarray = np.ones(grid.n, dtype=bool)
        self.dir_alive: np.ndarray = np.ones((grid.n, HEADING_BINS), dtype=bool)
        self.scan_count: int = 0

    # -- observation intake (DESIGN.md §3.2) -------------------------------

    def record_observation(self, point: Point, obs: Observation) -> None:
        """Fold one channel outcome at ``point`` into the hypothesis masks."""
        if obs.kind is ObservationKind.NO_SIGNAL:
            self.record_no_signal(point)
        elif obs.kind is ObservationKind.BEARING:
            assert obs.svd_deg is not None
            self.record_bearing(point, float(obs.svd_deg))
        elif obs.kind is ObservationKind.NEAR:
            self.record_near(point)

    def record_no_signal(self, S: Point) -> None:
        """NO_SIGNAL @ ``S`` — the negative rule, and the M8 §14 soundness gate.

        Eliminate a hypothesis only if the source it represents would have been
        **definitely detected** here (so a genuine NO_SIGNAL rules it out):

        * omni cell — every point of the box is within the guaranteed range::

              max_corner ‖S − corner‖ ≤ R_lb

        * directional (cell, bin) — the above range condition **and** the whole
          bin's arc definitely covers ``S`` from every point of the box. Using
          the half-plane form of the ±90° arc, ``(S − p)·u_φ ≥ 0`` for every edge
          heading and every corner::

              min_corner (S − corner)·u_lo ≥ 0  AND  min_corner (S − corner)·u_hi ≥ 0

        Both reductions sit at corners because the distance and the dot product
        are convex / linear in the source position; the two edge headings bind
        because ``(S − p)·u_φ`` over a 10° bin is minimised at an endpoint (a
        10°-wide interval cannot straddle the antipodal trough without both
        endpoints already going negative). Hence the test is exact and never
        fires on the true source's cell/bin. R_lb (not R_up) is mandatory: a
        directional source may be well inside 1500 m yet facing away, so only the
        guaranteed-detect region 1000 m may be subtracted (禁止5)."""
        self.scan_count += 1
        S_arr = np.asarray(S, dtype=float)

        far2 = self.grid._max_box_dist2(S)                 # (N,)
        range_ok = far2 <= self._r_lb2                     # (N,) definitely in range

        # omni: range alone certifies definite detection.
        self.omni_alive &= ~range_ok

        if not range_ok.any():
            return  # nothing within guaranteed range ⇒ no directional kill either

        # directional: range_ok AND both bin-edge half-planes cover S at every corner.
        diff = S_arr[None, None, :] - self.grid.corners    # (N,4,2) = S - corner
        dot_lo = np.tensordot(diff, _U_LO, axes=([2], [1]))  # (N,4,B)
        dot_hi = np.tensordot(diff, _U_HI, axes=([2], [1]))  # (N,4,B)
        min_lo = dot_lo.min(axis=1)                         # (N,B)
        min_hi = dot_hi.min(axis=1)                         # (N,B)
        arc_ok = (min_lo >= 0.0) & (min_hi >= 0.0)          # (N,B)
        dir_elim = range_ok[:, None] & arc_ok               # (N,B)
        self.dir_alive &= ~dir_elim

    def record_near(self, S: Point) -> None:
        """NEAR @ ``S`` — source provably within ``near_radius`` (≤ 5 m). Keep
        only cells whose box reaches within that radius of ``S``; drop the rest
        (both omni and every directional bin). Heading is left unconstrained: at
        ≤ 5 m the arc is effectively unrestricted, so no bin is pruned."""
        self.scan_count += 1
        near2 = self.grid._min_box_dist2(S)
        keep = near2 <= (self.near_radius + self.grid.half_diag) ** 2
        self.omni_alive &= keep
        self.dir_alive &= keep[:, None]

    def record_bearing(self, S: Point, svd_deg: float) -> None:
        """BEARING ``svd_deg`` @ ``S`` — a positive detection. The source is
        within ``R_up`` of ``S`` and its true bearing from ``S`` lies within the
        measurement window ``svd ± wedge_half``. We *eliminate* only cells (and
        directional bins) that are **entirely** inconsistent — a conservative
        keep, so the true cell/bin always survives:

        * range — drop cells whose nearest point is beyond ``R_up``.
        * wedge — drop cells whose whole angular extent (as seen from ``S``, a
          cone of half-width ``asin(half_diag / min_box_dist)`` about the centre
          bearing) misses the ``svd ± wedge_half`` window.
        * directional bins — a source at ``p`` detected from ``S`` must have a
          heading within 90° of ``bearing(p → S)``. Drop bins outside
          ``bearing(centre → S) ± (90° + cone half-width)``.

        Positive filtering is a planning refinement, not the M8 gate; it is kept
        deliberately conservative (over-approximates the kept set)."""
        self.scan_count += 1
        S_arr = np.asarray(S, dtype=float)
        centers = self.grid.centers                        # (N,2)

        # Angular half-width of each cell as seen from S. Bound the cell by its
        # circumscribing circle (radius half_diag, centred at the cell centre):
        # the box ⊂ that circle, so the circle's angular extent about the centre
        # bearing over-covers the cell's. Half-width = asin(half_diag / d_centre)
        # when S is outside the circle; when S is *inside* it (d_centre ≤
        # half_diag, e.g. S sits in the box) the source may lie in ANY direction,
        # so the half-width is 180° and no angular pruning happens for that cell.
        dc2 = (centers[:, 0] - S[0]) ** 2 + (centers[:, 1] - S[1]) ** 2   # (N,)
        hd = self.grid.half_diag
        outside = dc2 > hd * hd
        ratio = np.where(outside, hd / np.sqrt(np.where(outside, dc2, 1.0)), 1.0)
        cone_half_deg = np.where(outside, np.degrees(np.arcsin(np.clip(ratio, 0.0, 1.0))), 180.0)

        # range: keep cells whose nearest box point is within R_up (a positive
        # detection means the source — hence its box's nearest point — is ≤ R_up).
        min2 = self.grid._min_box_dist2(S)                 # (N,)
        range_keep = min2 <= (self.R_up + 1e-6) ** 2       # (N,)

        # wedge: bearing S -> cell centre within svd window + cone half-width.
        b_s_to_c = _bearing_arr(S_arr, centers)            # (N,) deg, S -> centre
        sep_c = _angle_sep_arr(b_s_to_c, float(svd_deg))   # (N,)
        wedge_keep = sep_c <= (self.wedge_half_deg + cone_half_deg + self.eps_deg)

        cell_keep = range_keep & wedge_keep                # (N,)
        self.omni_alive &= cell_keep
        self.dir_alive &= cell_keep[:, None]

        # directional bins: heading must fall within 90° of bearing(centre -> S),
        # widened by the cone half-width. Applied only to still-alive cells.
        b_c_to_s = (b_s_to_c + 180.0) % 360.0              # (N,) centre -> S
        sep_bins = _angle_sep_arr(
            b_c_to_s[:, None], _bin_center_deg[None, :]
        )                                                  # (N,B)
        # 90° arc + cell-cone half-width + bin half-width (the true heading may sit
        # up to half a bin from the bin centre): keep generously so the true bin
        # of a detected source is never pruned.
        bin_slack = 90.0 + 0.5 * BIN_WIDTH_DEG + cone_half_deg[:, None] + self.eps_deg
        bin_keep = sep_bins <= bin_slack
        self.dir_alive &= bin_keep

    # -- planner-facing queries (NEVER certify — Invariant B) ---------------

    @property
    def n_alive_omni(self) -> int:
        return int(self.omni_alive.sum())

    @property
    def n_alive_dir(self) -> int:
        return int(self.dir_alive.sum())

    @property
    def n_alive(self) -> int:
        """Total alive hypotheses (omni + directional)."""
        return self.n_alive_omni + self.n_alive_dir

    def alive_fraction(self) -> float:
        """Fraction of the initial hypothesis population still alive (1 → 0).
        A shrinking planning signal, **not** a completion certificate."""
        total = self.grid.n * (1 + HEADING_BINS)
        return 0.0 if total == 0 else self.n_alive / total

    def elimination_gain(self, S: Point) -> float:
        """Fraction of the **currently alive** hypotheses that a NO_SIGNAL scan at
        ``S`` would eliminate — the §3.2 directional-visibility gain the planner
        maximises when choosing EXPLORE / REFINE waypoints. Pure lookahead: masks
        are not mutated."""
        alive = self.n_alive
        if alive == 0:
            return 0.0

        far2 = self.grid._max_box_dist2(S)
        range_ok = far2 <= self._r_lb2
        killed_omni = int((self.omni_alive & range_ok).sum())

        killed_dir = 0
        if range_ok.any():
            diff = np.asarray(S, dtype=float)[None, None, :] - self.grid.corners
            dot_lo = np.tensordot(diff, _U_LO, axes=([2], [1]))
            dot_hi = np.tensordot(diff, _U_HI, axes=([2], [1]))
            arc_ok = (dot_lo.min(axis=1) >= 0.0) & (dot_hi.min(axis=1) >= 0.0)
            killed_dir = int((self.dir_alive & range_ok[:, None] & arc_ok).sum())

        return (killed_omni + killed_dir) / alive

    def alive_cells_mask(self) -> np.ndarray:
        """Boolean (N,) — cells that could still hold *some* source (omni or any
        directional heading). Handy for visualising / seeding VERIFY targets;
        never a proof of the complement (Invariant B)."""
        return self.omni_alive | self.dir_alive.any(axis=1)

    def posterior_entropy(self) -> float:
        """Entropy of the conservative uniform soft posterior over alive states.

        This is deliberately a planning posterior, not a certificate: every
        alive hypothesis receives mass and every eliminated hypothesis receives
        zero mass.  Thus ``support(p)`` remains a subset of the hard hypothesis
        set, and no probability estimate can remove a mathematically possible
        source.
        """
        n = self.n_alive
        return 0.0 if n <= 1 else float(log(n))

    def posterior_mass(self) -> np.ndarray:
        """Normalised per-cell mass, marginalising omni and directional states."""
        mass = self.omni_alive.astype(float) + self.dir_alive.sum(axis=1).astype(float)
        z = float(mass.sum())
        return mass / z if z > 0 else mass

    def information_gain(self, S: Point) -> float:
        """Entropy reduction proxy for a guaranteed NO_SIGNAL observation at S."""
        before = self.posterior_entropy()
        killed = self.elimination_gain(S)
        remaining = max(1, int(round(self.n_alive * (1.0 - killed))))
        return max(0.0, before - (0.0 if remaining <= 1 else log(remaining)))


class HypothesisLayer:
    """Per-channel :class:`ChannelHypotheses` over one shared grid (DESIGN.md
    §3.1). The soft, planning-only companion to the hard ``ChannelBelief`` map;
    it never certifies absence or clear (Invariant B, 禁止6)."""

    def __init__(self, arena_radius: float = 1800.0, spacing: float = 60.0, **params) -> None:
        self.grid = HypothesisGrid(arena_radius=arena_radius, spacing=spacing)
        self._params = params
        self.channels: Dict[int, ChannelHypotheses] = {}

    def channel(self, c: int) -> ChannelHypotheses:
        h = self.channels.get(c)
        if h is None:
            h = ChannelHypotheses(self.grid, channel=c, **self._params)
            self.channels[c] = h
        return h

    def record_observation(self, c: int, point: Point, obs: Observation) -> None:
        self.channel(c).record_observation(point, obs)

    def elimination_gain(self, c: int, point: Point) -> float:
        """Directional-visibility gain of a NO_SIGNAL scan at ``point`` for a
        single channel (0 for a never-seen channel — nothing alive to kill yet)."""
        h = self.channels.get(c)
        return 0.0 if h is None else h.elimination_gain(point)

    def batch_elimination_gain(self, channels, point: Point, weights=None) -> float:
        """Weighted directional-visibility gain over a batch of channels at
        ``point`` (§7 batch-scan scoring)."""
        total = 0.0
        for c in channels:
            w = 1.0 if weights is None else float(weights.get(c, 1.0))
            total += w * self.elimination_gain(c, point)
        return total

    def information_gain(self, c: int, point: Point) -> float:
        h = self.channels.get(c)
        return 0.0 if h is None else h.information_gain(point)

    def batch_information_gain(self, channels, point: Point, weights=None) -> float:
        return sum((1.0 if weights is None else float(weights.get(c, 1.0))) *
                   self.information_gain(c, point) for c in channels)


# --- small vectorised angle helpers (deg, matching sxjm_core convention) ------


def _bearing_arr(origin: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Bearing(s) (deg CCW from +x, [0,360)) from a single ``origin`` (2,) to each
    row of ``targets`` (M,2). Mirrors ``sxjm_core.geometry.bearing_deg``."""
    d = targets - origin[None, :]
    return np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360.0


def _angle_sep_arr(a, b) -> np.ndarray:
    """Smallest absolute angular separation (deg, [0,180]) — broadcasts.
    Mirrors ``sxjm_core.geometry.angle_sep_deg``."""
    d = np.abs((np.asarray(a, dtype=float) % 360.0) - (np.asarray(b, dtype=float) % 360.0)) % 360.0
    return np.where(d > 180.0, 360.0 - d, d)
