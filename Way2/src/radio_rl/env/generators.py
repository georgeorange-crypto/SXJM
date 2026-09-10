"""Case & jammer model plus the case generators — ported from the reference
oracle so LocalEnv reproduces the official problem distribution faithfully.

Hard constraints enforced (CUMCM 2026 Problem B):
- circular arena radius 1800 m, sources strictly inside;
- channels are distinct integers in 1..20, one source per channel, 10..16 total;
- effective receive radius R_eff in [1000, 1500] m (may repeat across sources);
- directional sources cover a 180 deg arc (+/-90 deg about an unknown heading);
- P3 -> all omnidirectional; P4 -> at least one omni AND at least one directional.

This module is import-clean of torch and of the geometry package (principle A/B):
it defines its own tiny angle helpers rather than reaching into ``geometry``.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Optional

from .error_field import BaseErrorField, SmoothErrorField, make_error_field

# ---- global constants (problem statement) ---------------------------------
ARENA_RADIUS_M = 1800.0
R_EFF_MIN, R_EFF_MAX = 1000.0, 1500.0
DIRECTIONAL_HALF_ANGLE_DEG = 90.0
JAMMER_COUNT_MIN, JAMMER_COUNT_MAX = 10, 16
CHANNEL_MIN, CHANNEL_MAX = 1, 20


def norm_deg(a: float) -> float:
    """Normalise an angle to [0, 360)."""
    a = math.fmod(a, 360.0)
    if a < 0:
        a += 360.0
    if a >= 360.0:
        a -= 360.0
    return a


def ang_diff(a: float, b: float) -> float:
    """Smallest separation between two bearings, in [0, 180]."""
    d = abs(norm_deg(a) - norm_deg(b))
    return d if d <= 180.0 else 360.0 - d


@dataclass
class Jammer:
    channel: int                 # channel 1..20, unique
    x: float
    y: float
    r_eff: float                 # effective receive radius, [1000, 1500]
    kind: str                    # "omni" | "dir"
    direction_deg: Optional[float] = None   # heading for "dir"; None for "omni"
    cleared: bool = False        # a source can be cleared at most once

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("cleared", None)
        return d


class Case:
    """One test case: jammer layout + error field + time limits + mode."""

    def __init__(
        self,
        jammers: list[Jammer],
        field: BaseErrorField,
        seed: int,
        mode: str = "practice",      # practice (may reveal truth at end) | formal
        max_virtual_duration_s: float = 360000.0,
        max_real_duration_s: int = 1200,
    ):
        self.jammers = jammers
        self.field = field
        self.seed = seed
        self.mode = mode
        self.max_virtual_duration_s = float(max_virtual_duration_s)
        self.max_real_duration_s = int(max_real_duration_s)
        self._by_channel = {j.channel: j for j in jammers}

    def jammer_on_channel(self, channel: int) -> Optional[Jammer]:
        return self._by_channel.get(channel)

    @property
    def total(self) -> int:
        return len(self.jammers)

    @property
    def n_omni(self) -> int:
        return sum(1 for j in self.jammers if j.kind == "omni")

    @property
    def n_dir(self) -> int:
        return sum(1 for j in self.jammers if j.kind == "dir")

    @property
    def cleared_count(self) -> int:
        return sum(1 for j in self.jammers if j.cleared)

    def reveal(self) -> dict:
        """Ground truth, only for practice-mode evaluation (never shown to the
        agent, never in formal mode)."""
        return {
            "total": self.total,
            "n_omni": self.n_omni,
            "n_dir": self.n_dir,
            "jammers": [{**j.to_dict(), "cleared": j.cleared} for j in self.jammers],
        }

    def to_json(self) -> str:
        return json.dumps(
            {
                "seed": self.seed,
                "mode": self.mode,
                "max_virtual_duration_s": self.max_virtual_duration_s,
                "max_real_duration_s": self.max_real_duration_s,
                "field": self.field.config(),
                "jammers": [j.to_dict() for j in self.jammers],
            },
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def from_json(text: str) -> "Case":
        d = json.loads(text)
        fc = d.get("field", {})
        field = make_error_field(
            kind=fc.get("kind", "smooth"),
            seed=fc.get("seed", 0),
            params=fc.get("params"),
        )
        jammers = [
            Jammer(
                channel=int(j["channel"]),
                x=float(j["x"]),
                y=float(j["y"]),
                r_eff=float(j["r_eff"]),
                kind=str(j["kind"]),
                direction_deg=(None if j.get("direction_deg") is None
                               else float(j["direction_deg"])),
            )
            for j in d["jammers"]
        ]
        return Case(
            jammers=jammers,
            field=field,
            seed=int(d.get("seed", 0)),
            mode=str(d.get("mode", "practice")),
            max_virtual_duration_s=float(d.get("max_virtual_duration_s", 360000.0)),
            max_real_duration_s=int(d.get("max_real_duration_s", 1200)),
        )


def _sample_point_in_disk(rng: random.Random, radius: float) -> tuple[float, float]:
    r = radius * math.sqrt(rng.random())
    t = rng.uniform(0.0, 2.0 * math.pi)
    return r * math.cos(t), r * math.sin(t)


def _build_field(rng: random.Random, seed: Optional[int],
                 field_kind: str, field_params: Optional[dict]) -> BaseErrorField:
    fseed = rng.randrange(1 << 30) if seed is None else (seed * 2654435761 & 0x7FFFFFFF)
    return make_error_field(kind=field_kind, seed=fseed, params=field_params)


def generate_case(
    seed: Optional[int] = None,
    problem: int = 3,
    n_jammers: Optional[int] = None,
    n_directional: Optional[int] = None,
    mode: str = "practice",
    margin_m: float = 30.0,
    field_kind: str = "smooth",
    field_params: Optional[dict] = None,
) -> Case:
    """Generate a random case satisfying every hard constraint. P3 -> all omni;
    P4 -> at least one omni and at least one directional (heading random/unknown)."""
    rng = random.Random(seed if seed is not None else random.randrange(1 << 30))

    if n_jammers is None:
        n_jammers = rng.randint(JAMMER_COUNT_MIN, JAMMER_COUNT_MAX)
    n_jammers = max(JAMMER_COUNT_MIN, min(JAMMER_COUNT_MAX, int(n_jammers)))

    channels = rng.sample(range(CHANNEL_MIN, CHANNEL_MAX + 1), n_jammers)

    if problem == 4:
        if n_directional is None:
            n_directional = rng.randint(1, n_jammers - 1)
        n_directional = max(1, min(n_jammers - 1, int(n_directional)))
    else:
        n_directional = 0

    dir_flags = [True] * n_directional + [False] * (n_jammers - n_directional)
    rng.shuffle(dir_flags)

    jammers: list[Jammer] = []
    for ch, is_dir in zip(channels, dir_flags):
        x, y = _sample_point_in_disk(rng, ARENA_RADIUS_M - margin_m)
        r_eff = rng.uniform(R_EFF_MIN, R_EFF_MAX)
        if is_dir:
            jammers.append(Jammer(ch, x, y, r_eff, "dir",
                                  direction_deg=rng.uniform(0.0, 360.0)))
        else:
            jammers.append(Jammer(ch, x, y, r_eff, "omni", None))

    field = _build_field(rng, seed, field_kind, field_params)
    return Case(jammers=jammers, field=field, seed=(seed or 0), mode=mode,
                max_virtual_duration_s=360000.0, max_real_duration_s=1200)


# ---- stress / worst-case generators ---------------------------------------
STRESS_TYPES = (
    "edge_cluster",   # all near the r=1800 boundary
    "min_reff",       # every R_eff = 1000 (hardest to detect)
    "tiny_cluster",   # sources packed into a small area
    "collinear",      # sources on a line (near-parallel bearings -> ill-conditioned)
    "far_pair",       # tightly-spaced pairs (hard to resolve/clear)
    "max_count",      # 16 sources
    "min_count",      # 10 sources
    "dir_outward",    # P4: directional sources face away from the origin
    "dir_boundary",   # P4: origin sits near the 90 deg coverage boundary
    "dir_evasive",    # P4: headings chosen to dodge a given scan-point set
)


def generate_stress_case(
    stress_type: str,
    seed: Optional[int] = None,
    problem: int = 3,
    field_kind: str = "adversarial",
    field_params: Optional[dict] = None,
    scan_points: Optional[list[tuple[float, float]]] = None,
    mode: str = "practice",
) -> Case:
    """Generate an adversarial case for worst-case robustness evaluation, still
    obeying every hard constraint."""
    if stress_type not in STRESS_TYPES:
        raise ValueError(f"unknown stress_type {stress_type!r}; valid: {STRESS_TYPES}")
    rng = random.Random(seed if seed is not None else random.randrange(1 << 30))
    R = ARENA_RADIUS_M

    if stress_type == "max_count":
        n = JAMMER_COUNT_MAX
    elif stress_type == "min_count":
        n = JAMMER_COUNT_MIN
    else:
        n = rng.randint(JAMMER_COUNT_MIN, JAMMER_COUNT_MAX)

    channels = rng.sample(range(CHANNEL_MIN, CHANNEL_MAX + 1), n)

    is_dir_type = stress_type.startswith("dir_")
    if problem == 4 or is_dir_type:
        n_dir = max(1, min(n - 1, rng.randint(max(1, n // 2), n - 1)))
    else:
        n_dir = 0

    positions: list[tuple[float, float]] = []
    if stress_type == "edge_cluster":
        for _ in range(n):
            t = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(R - 60.0, R - 5.0)
            positions.append((r * math.cos(t), r * math.sin(t)))
    elif stress_type == "tiny_cluster":
        cx, cy = _sample_point_in_disk(rng, R - 300.0)
        for _ in range(n):
            positions.append((cx + rng.uniform(-40, 40), cy + rng.uniform(-40, 40)))
    elif stress_type == "collinear":
        ang = rng.uniform(0, math.pi)
        ux, uy = math.cos(ang), math.sin(ang)
        for _ in range(n):
            t = rng.uniform(-(R - 100.0), R - 100.0)
            jitter = rng.uniform(-3.0, 3.0)
            positions.append((ux * t - uy * jitter, uy * t + ux * jitter))
    elif stress_type == "far_pair":
        while len(positions) < n:
            cx, cy = _sample_point_in_disk(rng, R - 100.0)
            positions.append((cx, cy))
            if len(positions) < n:
                positions.append((cx + rng.uniform(-6, 6), cy + rng.uniform(-6, 6)))
    else:
        for _ in range(n):
            positions.append(_sample_point_in_disk(rng, R - 30.0))

    if stress_type == "min_reff":
        reffs = [R_EFF_MIN] * n
    else:
        reffs = [rng.uniform(R_EFF_MIN, R_EFF_MAX) for _ in range(n)]

    dir_flags = [True] * n_dir + [False] * (n - n_dir)
    rng.shuffle(dir_flags)

    def _dir_for(idx: int, x: float, y: float) -> float:
        if stress_type == "dir_outward":
            return norm_deg(math.degrees(math.atan2(y, x)))
        if stress_type == "dir_boundary":
            to_origin = norm_deg(math.degrees(math.atan2(-y, -x)))
            return norm_deg(to_origin + (90.0 if idx % 2 == 0 else -90.0))
        if stress_type == "dir_evasive" and scan_points:
            best = min(scan_points, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
            to_scan = math.degrees(math.atan2(best[1] - y, best[0] - x))
            return norm_deg(to_scan + 180.0)
        return rng.uniform(0.0, 360.0)

    jammers: list[Jammer] = []
    for i, (ch, (x, y), reff, is_dir) in enumerate(
        zip(channels, positions, reffs, dir_flags)
    ):
        d = math.hypot(x, y)
        if d > R - 3.0:
            s = (R - 3.0) / d
            x, y = x * s, y * s
        if is_dir:
            jammers.append(Jammer(ch, x, y, reff, "dir",
                                  direction_deg=_dir_for(i, x, y)))
        else:
            jammers.append(Jammer(ch, x, y, reff, "omni", None))

    field = _build_field(rng, seed, field_kind, field_params)
    return Case(jammers=jammers, field=field, seed=(seed or 0), mode=mode,
                max_virtual_duration_s=360000.0, max_real_duration_s=1200)
