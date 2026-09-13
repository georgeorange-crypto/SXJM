"""Responsibility-aware geometric backbone (P01--P04)."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional, Sequence


class BackboneStatus(str, Enum):
    UNSATISFIED = "UNSATISFIED"
    PARTIALLY_SATISFIED = "PARTIALLY_SATISFIED"
    SATISFIED_BY_VISIT = "SATISFIED_BY_VISIT"
    SATISFIED_BY_OTHER_OBSERVATION = "SATISFIED_BY_OTHER_OBSERVATION"
    REPLACED = "REPLACED"
    REDUNDANT = "REDUNDANT"
    SKIPPED = "SKIPPED"
    VISITED = "VISITED"


@dataclass(frozen=True)
class BackboneNode:
    node_id: str
    point: tuple[float, float]
    coverage_cells: tuple = ()
    directional_triangles: tuple = ()
    boundary_caps: tuple = ()
    certificate_holes: tuple = ()
    channels_if_needed: tuple[int, ...] = ()
    status: BackboneStatus = BackboneStatus.UNSATISFIED


@dataclass(frozen=True)
class BackboneEdge:
    source: str
    target: str
    cost_s: float


@dataclass(frozen=True)
class CoverageResponsibility:
    node_id: str
    coverage_cells: tuple = ()
    directional_triangles: tuple = ()
    boundary_caps: tuple = ()
    certificate_holes: tuple = ()
    channels_if_needed: tuple[int, ...] = ()


@dataclass(frozen=True)
class DirectionalTriangle:
    triangle_id: str
    vertices: tuple
    covered_region: tuple = ()
    directional_guarantee: str = "convex_hull_surrounds_source"
    certificate_status: BackboneStatus = BackboneStatus.UNSATISFIED

    @property
    def detector_a(self): return self.vertices[0]

    @property
    def detector_b(self): return self.vertices[1]

    @property
    def detector_c(self): return self.vertices[2]


@dataclass(frozen=True)
class BoundaryCap:
    cap_id: str
    endpoints: tuple
    detector_points: tuple = ()
    covered_arc: tuple = ()
    outward_directions_deg: tuple = ()
    certificate_status: BackboneStatus = BackboneStatus.UNSATISFIED


@dataclass(frozen=True)
class CoverageDebt:
    node_id: str
    remaining_cells: int
    remaining_holes: int


@dataclass(frozen=True)
class BackboneGeometryAudit:
    side_lengths_ok: bool
    boundary_support_ok: bool
    triangle_coverage_ok: bool
    cap_coverage_ok: bool
    worst_outward_direction_ok: bool
    full_arena_inclusion_ok: bool

    @property
    def passed(self) -> bool:
        return all(self.__dict__.values())


class BackboneManager:
    """Planner-only responsibility graph; it never mutates belief/certificate."""
    def __init__(self, nodes: Sequence[BackboneNode] = ()):
        self.nodes = {n.node_id: n for n in nodes}
        self.edges: list[BackboneEdge] = []
        self._satisfied: dict[str, dict[str, set]] = {}
        self.route: list[str] = []
        self.route_cost_s: float = 0.0

    def add_node(self, node: BackboneNode) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate backbone node: {node.node_id}")
        self.nodes[node.node_id] = node

    def set_status(self, node_id: str, status: BackboneStatus) -> None:
        old = self.nodes[node_id]
        self.nodes[node_id] = BackboneNode(**{**old.__dict__, "status": BackboneStatus(status)})

    def satisfy_by_observation(self, node_ids: Iterable[str], *, source: Optional[str] = None) -> None:
        """Mark future nodes satisfied by a remote observation and remove them from route."""
        for node_id in node_ids:
            if node_id not in self.nodes:
                raise KeyError(node_id)
            self.set_status(node_id, BackboneStatus.SATISFIED_BY_OTHER_OBSERVATION)

    def mark_visited(self, node_id: str) -> None:
        self.set_status(node_id, BackboneStatus.VISITED)

    def apply_observation(self, *, coverage_cells=(), directional_triangles=(),
                          boundary_caps=(), certificate_holes=(), channels=()) -> list[str]:
        """Use a dynamic observation to repay overlapping backbone responsibilities."""
        observed = {"coverage_cells": set(coverage_cells),
                    "directional_triangles": set(directional_triangles),
                    "boundary_caps": set(boundary_caps),
                    "certificate_holes": set(certificate_holes),
                    "channels_if_needed": set(channels)}
        changed = []
        for node_id, node in self.nodes.items():
            state = self._satisfied.setdefault(node_id, {k: set() for k in observed})
            for key, values in observed.items():
                state[key].update(values.intersection(set(getattr(node, key))))
            total = sum(len(getattr(node, key)) for key in observed)
            done = sum(len(state[key]) for key in observed)
            if total and done == total and node.status in (BackboneStatus.UNSATISFIED,
                                                            BackboneStatus.PARTIALLY_SATISFIED):
                self.set_status(node_id, BackboneStatus.SATISFIED_BY_OTHER_OBSERVATION)
                changed.append(node_id)
            elif done and node.status == BackboneStatus.UNSATISFIED:
                self.set_status(node_id, BackboneStatus.PARTIALLY_SATISFIED)
                changed.append(node_id)
        return changed

    def remaining_route_nodes(self) -> list[BackboneNode]:
        return [n for n in self.nodes.values() if n.status == BackboneStatus.UNSATISFIED]

    def initialize_open_route(self, start: tuple[float, float] = (0.0, 0.0),
                              speed_mps: float = 5.0) -> tuple[str, ...]:
        """Build a deterministic nearest-neighbour open route over safe nodes."""
        from math import hypot
        if speed_mps <= 0.0:
            raise ValueError("speed_mps must be positive")
        remaining = list(self.remaining_route_nodes())
        current = (float(start[0]), float(start[1]))
        order = []
        length = 0.0
        while remaining:
            node = min(remaining, key=lambda n: (hypot(n.point[0] - current[0],
                                                       n.point[1] - current[1]), n.node_id))
            length += hypot(node.point[0] - current[0], node.point[1] - current[1])
            order.append(node.node_id)
            current = node.point
            remaining.remove(node)
        self.route = order
        self.route_cost_s = length / float(speed_mps)
        return tuple(order)

    def prune_route(self, *, start: tuple[float, float] = (0.0, 0.0),
                    speed_mps: float = 5.0) -> tuple[str, ...]:
        """Remove satisfied nodes from the existing route, preserving order."""
        from math import hypot
        if speed_mps <= 0.0:
            raise ValueError("speed_mps must be positive")
        current = list(self.route)
        self.route = [node_id for node_id in current
                      if node_id in self.nodes and self.nodes[node_id].status == BackboneStatus.UNSATISFIED]
        points = [self.nodes[node_id].point for node_id in self.route]
        prev = (float(start[0]), float(start[1]))
        length = 0.0
        for point in points:
            length += hypot(point[0] - prev[0], point[1] - prev[1])
            prev = point
        self.route_cost_s = length / float(speed_mps)
        return tuple(self.route)

    def accept_shortened_route(self, proposed: Sequence[str], proposed_cost_s: float) -> bool:
        """Accept observation-driven plans only when they retire work and shorten cost."""
        proposed = tuple(proposed)
        old = tuple(self.route)
        if any(node_id not in old for node_id in proposed):
            return False
        positions = [old.index(node_id) for node_id in proposed]
        if positions != sorted(positions):
            return False
        if float(proposed_cost_s) > self.route_cost_s + 1e-9:
            return False
        self.route = list(proposed)
        self.route_cost_s = float(proposed_cost_s)
        return True

    def debt(self, node_id: str) -> CoverageDebt:
        n = self.nodes[node_id]
        state = self._satisfied.get(node_id, {})
        return CoverageDebt(
            node_id,
            max(0, len(n.coverage_cells) - len(state.get("coverage_cells", set()))),
            max(0, len(n.certificate_holes) - len(state.get("certificate_holes", set()))),
        )


def build_p4_triangle_mesh(*, arena_radius: float = 1800.0,
                           n_sectors: int = 12) -> tuple[list[DirectionalTriangle], tuple[tuple[float, float], ...]]:
    """Build a deterministic center/ring triangle mesh for P4 planning."""
    from math import cos, pi, sin
    if arena_radius <= 0 or n_sectors < 3:
        raise ValueError("arena_radius must be positive and n_sectors >= 3")
    center = (0.0, 0.0)
    ring = tuple((float(arena_radius * cos(2 * pi * i / n_sectors)),
                  float(arena_radius * sin(2 * pi * i / n_sectors)))
                 for i in range(n_sectors))
    triangles = [DirectionalTriangle(
        f"T{i:03d}", (center, ring[i], ring[(i + 1) % n_sectors]),
        covered_region=(center, ring[i], ring[(i + 1) % n_sectors]),
    ) for i in range(n_sectors)]
    return triangles, (center,) + ring


def build_boundary_caps(*, arena_radius: float = 1800.0,
                        detector_radius: float = 1000.0,
                        n_caps: int = 12,
                        outward_offset: float = 250.0) -> tuple[BoundaryCap, ...]:
    """Create explicit outward detector caps for the arena boundary."""
    from math import cos, pi, sin
    if arena_radius <= 0 or detector_radius <= 0 or n_caps < 3 or outward_offset < 0:
        raise ValueError("invalid boundary-cap parameters")
    caps = []
    for i in range(n_caps):
        a0, a1 = 2 * pi * i / n_caps, 2 * pi * (i + 1) / n_caps
        mid = (a0 + a1) / 2
        cap = (float(arena_radius * cos(mid)), float(arena_radius * sin(mid)))
        detector = (float((arena_radius + outward_offset) * cos(mid)),
                    float((arena_radius + outward_offset) * sin(mid)))
        caps.append(BoundaryCap(f"BC{i:03d}", (cap,), (detector,), (cap,),
                                (mid * 180 / pi,), BackboneStatus.UNSATISFIED))
    return tuple(caps)


def audit_backbone_geometry(triangles, caps=(), *, arena_radius: float = 1800.0,
                            detector_radius: float = 1000.0,
                            samples: Sequence[tuple[float, float]] = ()) -> BackboneGeometryAudit:
    """Deterministic geometry audit for triangle/cap backbone artifacts."""
    from math import hypot
    def d(a, b): return hypot(a[0] - b[0], a[1] - b[1])
    side_ok = all(d(t.vertices[i], t.vertices[(i + 1) % 3]) <= detector_radius + 1e-9
                  for t in triangles for i in range(3))
    boundary_ok = bool(caps) and all(c.detector_points and c.covered_arc for c in caps)
    triangle_ok = all(len(t.vertices) == 3 and t.covered_region for t in triangles)
    cap_ok = bool(caps) and all(any(d(p, c.covered_arc[0]) <= detector_radius + 1e-9
                                    for p in c.detector_points) for c in caps)
    outward_ok = bool(caps) and all(c.outward_directions_deg for c in caps)
    inclusion_ok = True
    if samples:
        from ..certificate.directional_certificate import point_in_triangle
        inclusion_ok = all(any(point_in_triangle(p, t.covered_region) for t in triangles)
                           for p in samples)
    return BackboneGeometryAudit(side_ok, boundary_ok, triangle_ok, cap_ok,
                                 outward_ok, inclusion_ok)
