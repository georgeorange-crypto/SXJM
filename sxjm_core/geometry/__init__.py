"""Authoritative geometry core (Way4/DESIGN.md §13, M1).

Stdlib-only for determinism and portability (matching Way3's ethos; numpy
vectorisation is a later, behaviour-preserving optimisation). The public surface
is a superset of the four legacy geometry copies listed in DESIGN.md §2:

    norm_deg, wedge_halfplanes, Halfplane, clip_polygon, halfplane_intersection,
    convex_hull, polygon_diameter, min_enclosing_circle  (estimator names)

plus the primitives Way4's belief/certificate layers need (bearings, wedge
membership, disc/arena membership, circle-circle intersection, arena superset
polygon).
"""

from .primitives import (
    Point,
    EPS,
    add,
    sub,
    scale,
    dot,
    cross,
    norm,
    dist,
    dist2,
    deg2rad,
    rad2deg,
    norm_deg,
    bearing_deg,
    angle_sep_deg,
    in_disc,
    in_arena,
    point_in_wedge,
    circle_circle_intersections,
)
from .convex import (
    Halfplane,
    wedge_halfplanes,
    arena_polygon,
    disc_superset_halfplanes,
    clip_polygon,
    halfplane_intersection,
    convex_hull,
    polygon_area,
    polygon_centroid,
    polygon_diameter,
)
from .mec import (
    min_enclosing_circle,
    circle_from_two,
    circumcircle,
)

__all__ = [
    "Point",
    "EPS",
    "add",
    "sub",
    "scale",
    "dot",
    "cross",
    "norm",
    "dist",
    "dist2",
    "deg2rad",
    "rad2deg",
    "norm_deg",
    "bearing_deg",
    "angle_sep_deg",
    "in_disc",
    "in_arena",
    "point_in_wedge",
    "circle_circle_intersections",
    "Halfplane",
    "wedge_halfplanes",
    "arena_polygon",
    "disc_superset_halfplanes",
    "clip_polygon",
    "halfplane_intersection",
    "convex_hull",
    "polygon_area",
    "polygon_centroid",
    "polygon_diameter",
    "min_enclosing_circle",
    "circle_from_two",
    "circumcircle",
]
