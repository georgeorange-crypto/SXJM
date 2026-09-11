"""sxjm_core — shared infrastructure for the SXJM CUMCM-B solvers.

Currently exposes the single authoritative geometry core (``sxjm_core.geometry``,
Way4/DESIGN.md §13 / M1). This replaces the four divergent geometry copies
(estimator / Way2 / Way3 / Way1) that DESIGN.md §2 and forbidden-item #2 warn
against. The port of those four packages onto this core is a per-branch refactor
(each lives on a different branch); Way4 depends on this core directly.
"""
